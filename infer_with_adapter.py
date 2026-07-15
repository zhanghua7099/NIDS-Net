#!/usr/bin/env python
# coding: utf-8
#
# Inference using a pre-trained adapter (see train_adapter.py).
# Only reads artifacts from example_dataset/train/adapter/ -- does not touch
# example_dataset/train/Objects/ again.
#
#   example_dataset/train/adapter/{weights.pth,adapted_features.json,meta.json}
#   example_dataset/inference/*.jpg  --GroundingDINO+SAM+FFA+adapter-->  proposals
#   cosine similarity + stable matching -> detections

import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as Fnn
from tqdm import trange

sys.path.append(".")
from utils.inference_utils import (
    compute_similarity, stableMatching, get_bbox_masks_from_gdino_sam,
    get_object_proposal, get_features,
)
from adapter import WeightAdapter
from robokit.ObjDetection import GroundingDINOObjectPredictor, SegmentAnythingPredictor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ADAPTER_DIR = "example_dataset/train/adapter"
INFERENCE_DIR = "example_dataset/inference"
OUTPUT_DIR = "exps/example_with_adapter"
SCORE_THRESHOLD = 0.5
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 0. Load the trained adapter + its artifacts (no access to raw template
#    images needed from here on)
# ---------------------------------------------------------------------------
with open(os.path.join(ADAPTER_DIR, "meta.json")) as f:
    meta = json.load(f)
object_names = meta["object_names"]
num_object = meta["num_object"]
num_example = meta["num_example"]
img_size = meta["img_size"]
print(f"[adapter] {num_object} objects, trained for {meta['epochs']} epochs, "
      f"final_loss={meta['final_loss']:.4f}")
print(f"[adapter] objects: {object_names}")

with open(os.path.join(ADAPTER_DIR, "adapted_features.json")) as f:
    adapted_object_features = torch.Tensor(json.load(f)["features"]).to(device)

adapter = WeightAdapter(adapted_object_features.shape[1], reduction=meta["reduction"]).to(device)
adapter.load_state_dict(torch.load(os.path.join(ADAPTER_DIR, "weights.pth"), map_location=device))
adapter.eval()
print(f"[adapter] loaded weights from {ADAPTER_DIR}/weights.pth")

test_images = sorted(
    p for p in glob.glob(os.path.join(INFERENCE_DIR, "*"))
    if p.lower().endswith((".jpg", ".jpeg", ".png"))
)
assert test_images, f"No inference images found in {INFERENCE_DIR}"
print(f"[data] {len(test_images)} inference image(s): {test_images}")

# ---------------------------------------------------------------------------
# 1. DINOv2 encoder (needed to featurize scene proposals, not for templates)
# ---------------------------------------------------------------------------
print("[step 1/3] Loading DINOv2...")
encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', skip_validation=True)
encoder.to(device)
encoder.eval()

# ---------------------------------------------------------------------------
# 2. GroundingDINO + SAM proposals -> FFA -> adapter -> features
# ---------------------------------------------------------------------------
print("[step 2/3] Loading GroundingDINO + SAM and running detection...")
gdino = GroundingDINOObjectPredictor(use_vitb=False, threshold=0.15)
SAM = SegmentAnythingPredictor(vit_model="vit_t")

all_results = {}
for test_path in test_images:
    image_name = os.path.splitext(os.path.basename(test_path))[0]
    accurate_bboxs, masks = get_bbox_masks_from_gdino_sam(test_path, gdino, SAM, text_prompt='objects', visualize=False)
    print(f"  {image_name}: {len(masks)} proposals")

    accurate_bboxs = accurate_bboxs.cpu().numpy()
    masks_np = masks.cpu().numpy()
    rois, sel_rois, cropped_imgs, cropped_masks = get_object_proposal(
        test_path, accurate_bboxs, masks, tag="mask", ratio=1.0,
        save_rois=False, output_dir=OUTPUT_DIR, save_proposal=False, save_segm=True,
    )

    scene_features = []
    for i in trange(0, len(cropped_imgs), BATCH_SIZE, desc=f"  FFA[{image_name}]"):
        ffa_feature = get_features(cropped_imgs[i:i + BATCH_SIZE], cropped_masks[i:i + BATCH_SIZE],
                                    encoder, device=device, img_size=img_size)
        with torch.no_grad():
            ffa_feature = adapter(ffa_feature)
        scene_features.append(ffa_feature)
    scene_features = torch.cat(scene_features, dim=0)
    scene_features = Fnn.normalize(scene_features, dim=1, p=2)

    # -----------------------------------------------------------------------
    # 3. Similarity + stable matching (adapted embeddings on both sides)
    # -----------------------------------------------------------------------
    sim_mat = compute_similarity(adapted_object_features, scene_features)
    sim_mat = sim_mat.view(len(scene_features), num_object, num_example)
    sims, _ = torch.max(sim_mat, dim=2)

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
    Matches = dict()
    for i in range(matchedMat.shape[0]):
        tmp = matchedMat[i, :]
        a = tmp.argmax()
        Matches[sel_roi_ids[i]] = sel_obj_ids[int(a)]

    results = []
    for k, v in Matches.items():
        if int(k) >= num_proposals or int(v) >= num_object:
            continue
        score = float(sims[int(k), int(v)])
        if score < SCORE_THRESHOLD:
            continue
        results.append({
            "roi_id": int(k),
            "bbox": sel_rois[int(k)]["bbox"],
            "category_id": int(v),
            "category_name": object_names[int(v)],
            "score": score,
            "segmentation": sel_rois[int(k)]["segmentation"],
        })
    results.sort(key=lambda r: -r["score"])
    all_results[image_name] = results

    print(f"  {image_name}: {len(results)} detections above score {SCORE_THRESHOLD}")
    for r in results:
        print(f"    {r['category_name']:35s} score={r['score']:.3f} bbox={r['bbox']}")

    # visualization (instance masks + bbox + label)
    img_cv = cv2.imread(test_path)
    overlay = img_cv.copy()
    rng = np.random.default_rng(42)
    colors = {name: tuple(int(c) for c in rng.integers(40, 255, size=3)) for name in object_names}
    for r in results:
        color = colors[r["category_name"]]
        overlay[masks_np[r["roi_id"]]] = color
    img_cv = cv2.addWeighted(overlay, 0.5, img_cv, 0.5, 0)
    for r in results:
        x, y, w, h = r["bbox"]
        color = colors[r["category_name"]]
        mask = masks_np[r["roi_id"]].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_cv, contours, -1, color, 2)
        cv2.putText(img_cv, f"{r['category_name']} {r['score']:.2f}", (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    out_path = os.path.join(OUTPUT_DIR, f"{image_name}_pred.jpg")
    cv2.imwrite(out_path, img_cv)
    print(f"  saved -> {out_path}")

with open(os.path.join(OUTPUT_DIR, "predictions.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n[done] With-adapter inference results written to {OUTPUT_DIR}/")
