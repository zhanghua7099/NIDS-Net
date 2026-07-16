#!/usr/bin/env python
# coding: utf-8
#
# Client for infer_server.py: sends one image over HTTP, draws the returned
# bounding boxes + instance segmentation masks on it, and saves/shows the
# result. Runs on any machine that can reach the server -- it does NOT need
# the "nids" conda env, torch, or a GPU.
#
# Install (client machine only):
#   pip install requests opencv-python numpy pycocotools
#
# Usage:
#   python infer_client.py scene_018.jpg --url http://<server-ip>:8000
#   python infer_client.py scene_018.jpg --url http://<server-ip>:8000 --show
#   python infer_client.py scene_018.jpg --out annotated.jpg

import argparse
import mimetypes
import os

import cv2
import numpy as np
import requests
from pycocotools import mask as maskUtils


def infer(url, image_path):
    content_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, content_type)}
        resp = requests.post(f"{url.rstrip('/')}/infer", files=files)
    if not resp.ok:
        detail = resp.text.strip()
        if detail:
            raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}", response=resp)
    resp.raise_for_status()
    return resp.json()["detections"]


def visualize(image_path, detections):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    overlay = img.copy()

    rng = np.random.default_rng(42)
    colors = {}

    def color_for(name):
        if name not in colors:
            colors[name] = tuple(int(c) for c in rng.integers(40, 255, size=3))
        return colors[name]

    for det in detections:
        if det.get("segmentation") is None:
            continue
        mask = maskUtils.decode(det["segmentation"]).astype(bool)
        overlay[mask] = color_for(det["category_name"])
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    for det in detections:
        color = color_for(det["category_name"])
        x, y, w, h = det["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"{det['category_name']} {det['score']:.2f}"
        cv2.putText(img, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


def main():
    parser = argparse.ArgumentParser(description="Send an image to infer_server.py and visualize the result.")
    parser.add_argument("image", help="path to the input image")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="inference server base URL")
    parser.add_argument("--out", default=None, help="output path (default: <image>_pred.jpg)")
    parser.add_argument("--show", action="store_true", help="also open a window to display the result")
    args = parser.parse_args()

    out_path = args.out or f"{os.path.splitext(args.image)[0]}_pred.jpg"

    print(f"[client] POST {args.image} -> {args.url}/infer")
    try:
        detections = infer(args.url, args.image)
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"[client] request failed: {e}")

    print(f"[client] {len(detections)} detection(s):")
    for d in sorted(detections, key=lambda d: -d["score"]):
        print(f"  {d['category_name']:35s} score={d['score']:.3f} bbox={d['bbox']}")

    img = visualize(args.image, detections)
    cv2.imwrite(out_path, img)
    print(f"[client] saved -> {out_path}")

    if args.show:
        cv2.imshow("detections", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
