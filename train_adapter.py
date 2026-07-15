#!/usr/bin/env python
# coding: utf-8
#
# Train the weight adapter on the example dataset's template features.
#
#   example_dataset/train/Objects/<obj>/{images,masks}  --FFA-->  raw template features
#   raw template features                                --InfoNCE-->  WeightAdapter
#
# Output (everything the inference script needs, self-contained):
#   example_dataset/train/adapter/weights.pth        -- adapter state_dict
#   example_dataset/train/adapter/adapted_features.json -- adapter(raw template features)
#   example_dataset/train/adapter/raw_features.json  -- raw FFA template features (cache)
#   example_dataset/train/adapter/meta.json          -- object_names + hyperparams

import glob
import json
import os
import sys

import torch
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader
from tqdm import trange

sys.path.append(".")
from utils.inference_utils import get_features
from utils.instance_det_dataset import InstanceDataset
from utils.adapter_dataset import FeatureDataset
from adapter import WeightAdapter, InfoNCELoss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_DIR = "example_dataset/train/Objects"
ADAPTER_DIR = "example_dataset/train/adapter"
IMG_SIZE = 448
REDUCTION = 4
TEMPERATURE = 0.05
LEARNING_RATE = 1e-3
EPOCHS = 200

os.makedirs(ADAPTER_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

object_names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(TRAIN_DIR, "*")))
num_object = len(object_names)
print(f"[data] {num_object} template objects: {object_names}")


def compute_ffa_template_features(out_path, object_dataset, model, img_size=448):
    """Foreground Feature Averaging over the template dataset. No training --
    a single forward pass through the frozen DINOv2 encoder, cached to disk."""
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            feat_dict = json.load(f)
        return torch.Tensor(feat_dict["features"]).to(device)

    batch_size = 32
    features, batch_images, batch_masks = [], [], []
    for i in trange(len(object_dataset), desc="FFA templates"):
        img, _, mask = object_dataset[i]
        mask = mask.convert("L")
        batch_images.append(img)
        batch_masks.append(mask)
        if len(batch_images) == batch_size or i == len(object_dataset) - 1:
            features.append(get_features(batch_images, batch_masks, model, device=device, img_size=img_size))
            batch_images, batch_masks = [], []
    features = torch.cat(features, dim=0)
    with open(out_path, "w") as f:
        json.dump({"features": features.detach().cpu().tolist()}, f)
    return features


# ---------------------------------------------------------------------------
# 1. DINOv2 encoder + raw template features
# ---------------------------------------------------------------------------
print("[step 1/2] Loading DINOv2 and computing raw template (FFA) features...")
encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', skip_validation=True)
encoder.to(device)
encoder.eval()

object_dataset = InstanceDataset(data_dir=TRAIN_DIR, dataset="Object", transform=None, imsize=IMG_SIZE)
raw_features_path = os.path.join(ADAPTER_DIR, "raw_features.json")
raw_object_features = compute_ffa_template_features(raw_features_path, object_dataset, encoder, img_size=IMG_SIZE)
num_example = len(raw_object_features) // num_object
print(f"[step 1/2] raw template features: {tuple(raw_object_features.shape)} "
      f"({num_object} objects x {num_example} views)")

# ---------------------------------------------------------------------------
# 2. Train the weight adapter (InfoNCE contrastive loss, ~seconds on GPU)
# ---------------------------------------------------------------------------
print(f"[step 2/2] Training WeightAdapter for {EPOCHS} epochs...")
input_features = raw_object_features.shape[1]
adapter = WeightAdapter(input_features, reduction=REDUCTION).to(device)
optimizer = torch.optim.Adam(adapter.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = InfoNCELoss(temperature=TEMPERATURE).to(device)

feature_dataset = FeatureDataset(data_json=raw_features_path, num_object=num_object)
dataloader = DataLoader(feature_dataset, batch_size=len(feature_dataset), shuffle=False)

adapter.train()
for epoch in range(EPOCHS):
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = adapter(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"  epoch {epoch + 1}/{EPOCHS}  loss={loss.item():.4f}")

weights_path = os.path.join(ADAPTER_DIR, "weights.pth")
torch.save(adapter.state_dict(), weights_path)
print(f"[step 2/2] adapter weights saved -> {weights_path}")

adapter.eval()
with torch.no_grad():
    adapted_object_features = adapter(raw_object_features.to(device))
adapted_object_features = Fnn.normalize(adapted_object_features, dim=1, p=2)
adapted_path = os.path.join(ADAPTER_DIR, "adapted_features.json")
with open(adapted_path, "w") as f:
    json.dump({"features": adapted_object_features.detach().cpu().tolist()}, f)
print(f"[step 2/2] adapted template features saved -> {adapted_path}")

meta = {
    "object_names": object_names,
    "num_object": num_object,
    "num_example": num_example,
    "img_size": IMG_SIZE,
    "reduction": REDUCTION,
    "temperature": TEMPERATURE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "final_loss": float(loss.item()),
}
meta_path = os.path.join(ADAPTER_DIR, "meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"[step 2/2] metadata saved -> {meta_path}")

print(f"\n[done] Trained adapter artifacts in {ADAPTER_DIR}/")
