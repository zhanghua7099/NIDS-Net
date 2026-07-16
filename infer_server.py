#!/usr/bin/env python
# coding: utf-8
#
# Persistent inference server for the with-adapter pipeline (see
# infer_with_adapter.py for the one-shot script this is derived from).
#
# The one-shot script spends ~9.3s of its ~11s runtime just loading DINOv2 +
# GroundingDINO + SAM from disk -- only ~1.4s is the actual per-image
# inference. This server loads all three models once at startup and keeps
# them resident in GPU memory, so each HTTP request only pays the ~1.4s.
#
# Usage:
#   conda activate nids
#   python infer_server.py                 # listens on 0.0.0.0:8000 by default
#
#   curl -F "file=@example_dataset/inference/scene_018.jpg" \
#        http://127.0.0.1:8000/infer
#
# Config (all optional, read at startup):
#   INFER_HOST  -- bind address (default: 0.0.0.0)
#   INFER_PORT  -- bind port (default: 8000)
#   INFER_ADAPTER_DIR -- adapter artifact directory (default: rs_dataset/train/adapter)
#   INFER_SCORE_THRESHOLD -- minimum detection score to return (default: 0.5)
#
# Reaching it from other machines on your LAN:
#   This process only controls its own bind address -- if it's running
#   inside a container (e.g. a code-server/devcontainer setup), 0.0.0.0 only
#   gets you reachability *within* that container. To reach it from another
#   machine on the LAN, the container's port must be published to the host,
#   the same way code-server's own web port already is, e.g. add
#   "-p 8000:8000" to the `docker run`/`docker-compose.yml` that starts this
#   container and recreate it. After that, any device on the LAN can hit:
#     curl -F "file=@img.jpg" http://<host-LAN-ip>:8000/infer
#   directly -- no proxy, no TLS cert, no browser login involved.
#
# There's no authentication on /infer -- only expose this on a network you
# trust (LAN), not the public internet.
#
# NOTE: single process, single GPU model set -- do not run this under
# multiple uvicorn workers or with --reload, that would load the (multi-GB)
# model stack more than once.

import itertools
import json
import os
import tempfile
import threading

# Every relative path in this codebase (robokit/ObjDetection.py's config and
# checkpoint paths, ADAPTER_DIR below, ...) assumes cwd == this repo's root.
# That's true if you cd here and run "python infer_server.py" by hand, but
# process managers (systemd, a boot-time nohup wrapper, ...) don't reliably
# set the working directory the same way -- so pin it explicitly, regardless
# of how this process gets launched.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Skip the network round-trip huggingface_hub normally does to revalidate a
# cached file on every load (robokit/ObjDetection.py's hf_hub_download call)
# -- checkpoints are already downloaded by setup_nids_env.sh, so this is a
# pure latency/reliability win for a long-running server.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as Fnn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pycocotools import mask as maskUtils

from utils.inference_utils import (
    compute_similarity, stableMatching, get_bbox_masks_from_gdino_sam,
    get_object_proposal, get_features, FFA_preprocess, get_foreground_mask,
)
from adapter import WeightAdapter
from robokit.ObjDetection import GroundingDINOObjectPredictor, SegmentAnythingPredictor

ADAPTER_DIR = os.environ.get("INFER_ADAPTER_DIR", "rs_dataset/train/adapter")
SCORE_THRESHOLD = float(os.environ.get("INFER_SCORE_THRESHOLD", "0.5"))
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load everything once, at import time (i.e. once per server process, not
# once per request).
# ---------------------------------------------------------------------------
if not os.path.isdir(ADAPTER_DIR):
    raise FileNotFoundError(
        f"adapter directory not found: {ADAPTER_DIR!r}. Set INFER_ADAPTER_DIR to the directory containing "
        "meta.json, weights.pth, and adapted_features.json for your dataset."
    )

with open(os.path.join(ADAPTER_DIR, "meta.json")) as f:
    meta = json.load(f)
object_names = meta["object_names"]
num_object = meta["num_object"]
img_size = meta["img_size"]

# Per-object template counts (adapted_object_features is laid out as one
# contiguous block of templates per object, in object_names order). Older
# adapters trained before this field existed had a uniform count per object --
# fall back to that so they still load.
num_example_per_object = meta.get("num_example_per_object", [meta["num_example"]] * num_object)
assert len(num_example_per_object) == num_object

with open(os.path.join(ADAPTER_DIR, "adapted_features.json")) as f:
    adapted_object_features = torch.Tensor(json.load(f)["features"]).to(device)

assert sum(num_example_per_object) == adapted_object_features.shape[0], (
    f"meta.json's num_example_per_object sums to {sum(num_example_per_object)} but "
    f"adapted_features.json has {adapted_object_features.shape[0]} rows -- retrain the adapter"
)
# (start, end) row bounds of each object's contiguous template block within
# adapted_object_features, used to take a per-object max similarity below
# without assuming every object has the same number of templates.
_bounds = list(itertools.accumulate(num_example_per_object, initial=0))
template_blocks = list(zip(_bounds[:-1], _bounds[1:]))

adapter = WeightAdapter(adapted_object_features.shape[1], reduction=meta["reduction"]).to(device)
adapter.load_state_dict(torch.load(os.path.join(ADAPTER_DIR, "weights.pth"), map_location=device))
adapter.eval()

print(f"[server] adapter loaded: {num_object} objects: {object_names}")

print("[server] loading DINOv2 encoder...")
encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', skip_validation=True)
encoder.to(device)
encoder.eval()
# Permanent fp16 instead of the usual autocast(fp16) wrapper -- skips the
# per-op cast overhead. Verified this moves cosine similarity between
# fp32+autocast and fp16 features by <0.0005 across a real scene's crops,
# i.e. well below anything that would change a match/threshold decision.
if device.type == "cuda":
    encoder = encoder.half()

print("[server] loading GroundingDINO...")
gdino = GroundingDINOObjectPredictor(use_vitb=False, threshold=0.15)

print("[server] loading SAM (vit_b)...")
SAM = SegmentAnythingPredictor(vit_model="vit_b")

print("[server] all models loaded, ready to serve")

# get_object_proposal() parses an "_<digits>" image id out of the filename
# (matching the "scene_018.jpg" dataset convention) -- a random tempfile
# suffix breaks that, so requests get a predictable numeric name instead.
_req_counter = itertools.count()

# SAM's predictor (predictor.set_image / predict_torch) holds mutable
# per-call state, so two requests running the pipeline concurrently would
# corrupt each other's masks. Serialize inference through one lock rather
# than making SAM's predictor thread-safe.
_infer_lock = threading.Lock()

app = FastAPI(title="NIDS-Net inference server")


def _get_features_fp16(images, masks, img_size):
    """Same math as utils.inference_utils.get_features, but feeds the
    permanently-fp16 `encoder` directly instead of wrapping a fp32 model in
    autocast(fp16) -- see the comment where `encoder` is loaded above.
    CUDA only: `encoder` is only cast to half() when device.type == "cuda"."""
    with torch.no_grad():
        preprocessed_imgs = FFA_preprocess(images, img_size).to(device).half()
        mask_size = img_size // 14
        fg_masks = get_foreground_mask(masks, mask_size).to(device)
        emb = encoder.forward_features(preprocessed_imgs)
        grid = emb["x_norm_patchtokens"].float().view(len(images), mask_size, mask_size, -1)
        avg_feature = (grid * fg_masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / fg_masks.sum(dim=(1, 2, 3)).unsqueeze(-1)
        return avg_feature


def _run_pipeline(image_path: str):
    accurate_bboxs, masks = get_bbox_masks_from_gdino_sam(
        image_path, gdino, SAM, text_prompt="objects", visualize=False
    )
    if len(masks) == 0:
        return []

    accurate_bboxs_np = accurate_bboxs.cpu().numpy()
    # save_segm=False: RLE-encoding every proposal's mask costs ~6ms each --
    # with dozens of proposals per scene but only num_object results ever
    # returned, that's mostly wasted work. Encode below, only for the
    # handful of proposals that survive matching + the score threshold.
    _, sel_rois, cropped_imgs, cropped_masks = get_object_proposal(
        image_path, accurate_bboxs_np, masks, tag="mask", ratio=1.0,
        save_rois=False, output_dir=tempfile.gettempdir(), save_proposal=False, save_segm=False,
    )

    scene_features = []
    for i in range(0, len(cropped_imgs), BATCH_SIZE):
        batch_imgs = cropped_imgs[i:i + BATCH_SIZE]
        batch_masks = cropped_masks[i:i + BATCH_SIZE]
        if device.type == "cuda":
            ffa_feature = _get_features_fp16(batch_imgs, batch_masks, img_size)
        else:
            ffa_feature = get_features(batch_imgs, batch_masks, encoder, device=device, img_size=img_size)
        with torch.no_grad():
            ffa_feature = adapter(ffa_feature)
        scene_features.append(ffa_feature)
    scene_features = torch.cat(scene_features, dim=0)
    scene_features = Fnn.normalize(scene_features, dim=1, p=2)

    # [num_proposals, total_templates] -- template columns are grouped by
    # object (see template_blocks), but blocks aren't all the same width, so
    # this can't be reshaped into a rectangular [num_proposals, num_object,
    # num_example] tensor; take each object's max over its own column range.
    sim_mat = compute_similarity(adapted_object_features, scene_features)
    sims = torch.stack(
        [sim_mat[:, start:end].max(dim=1).values for start, end in template_blocks],
        dim=1,
    )

    num_proposals = len(sel_rois)
    sel_obj_ids = [str(v) for v in range(num_object)]
    sel_roi_ids = [str(v) for v in range(len(scene_features))]
    max_len = max(len(sel_roi_ids), len(sel_obj_ids))
    sel_sims_symmetric = torch.ones((max_len, max_len)) * -1
    sel_sims_symmetric[:len(sel_roi_ids), :len(sel_obj_ids)] = sims.clone()
    pad_len = abs(len(sel_roi_ids) - len(sel_obj_ids))
    if len(sel_roi_ids) > len(sel_obj_ids):
        sel_obj_ids += [str(i) for i in range(num_object, num_object + pad_len)]
    elif len(sel_roi_ids) < len(sel_obj_ids):
        sel_roi_ids += [str(i) for i in range(len(sel_roi_ids), len(sel_roi_ids) + pad_len)]

    matchedMat = stableMatching(sel_sims_symmetric.detach().cpu().numpy())
    matches = {}
    for i in range(matchedMat.shape[0]):
        a = matchedMat[i, :].argmax()
        matches[sel_roi_ids[i]] = sel_obj_ids[int(a)]

    results = []
    for k, v in matches.items():
        if int(k) >= num_proposals or int(v) >= num_object:
            continue
        score = float(sims[int(k), int(v)])
        if score < SCORE_THRESHOLD:
            continue
        rle = maskUtils.encode(np.asfortranarray(masks[int(k)].cpu().numpy().astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        results.append({
            "roi_id": int(k),
            "bbox": sel_rois[int(k)]["bbox"],
            "category_id": int(v),
            "category_name": object_names[int(v)],
            "score": score,
            # COCO RLE: {"size": [h, w], "counts": "<ascii-encoded RLE string>"}
            # decode with pycocotools.mask.decode(segmentation) -> HxW bool array
            "segmentation": rle,
        })
    results.sort(key=lambda r: -r["score"])
    return results


@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": str(device), "objects": object_names}


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, f"expected an image upload, got content-type={file.content_type!r}")

    data = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    tmp_path = os.path.join(tempfile.gettempdir(), f"nidsreq_{next(_req_counter):06d}{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(data)

    try:
        with _infer_lock:
            results = _run_pipeline(tmp_path)
    except Exception as e:
        raise HTTPException(500, f"inference failed: {e}")
    finally:
        os.remove(tmp_path)

    return {"detections": results}


if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0: accept connections from other machines on the LAN, not just
    # localhost. There's no auth on /infer -- only do this on a network you
    # trust (e.g. published to a LAN, not the public internet).
    host = os.environ.get("INFER_HOST", "0.0.0.0")
    port = int(os.environ.get("INFER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, workers=1)
