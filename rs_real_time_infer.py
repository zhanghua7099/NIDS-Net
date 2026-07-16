#!/usr/bin/env python
# coding: utf-8
#
# Streams RealSense RGB frames and shows infer_server.py's segmentation
# result live. Inference (~1.4s/frame on the server) runs in a background
# thread so the preview window doesn't freeze between requests -- the
# overlay simply lags behind the live feed by however long the last
# request took.
#
# Usage:
#   python rs_capture.py --url http://<server-ip>:8000

import argparse
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from infer_client import infer_bytes, draw_detections


def main():
    parser = argparse.ArgumentParser(description="Show infer_server.py segmentation results on a live RealSense RGB feed.")
    parser.add_argument("--url", default="http://172.24.36.34:8000", help="inference server base URL")
    args = parser.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    lock = threading.Lock()
    latest_frame = None
    latest_detections = []
    stop_event = threading.Event()

    def infer_worker():
        nonlocal latest_detections
        while not stop_event.is_set():
            with lock:
                frame = None if latest_frame is None else latest_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            try:
                detections = infer_bytes(args.url, buf.tobytes())
            except Exception as e:
                print(f"[rs_capture] inference failed: {e}")
                time.sleep(0.5)
                continue
            with lock:
                latest_detections = detections

    worker = threading.Thread(target=infer_worker, daemon=True)
    worker.start()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            rgb_img = np.asanyarray(color_frame.get_data())

            with lock:
                latest_frame = rgb_img
                detections = latest_detections

            vis = draw_detections(rgb_img, detections)
            cv2.imshow("segmentation", vis)
            if cv2.waitKey(1) == ord('q'):
                print("q pressed. close the program.")
                break
    finally:
        stop_event.set()
        worker.join(timeout=2)
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
