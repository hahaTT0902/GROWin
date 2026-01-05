"""Probe local video capture indices and report which provide frames.

Usage:
  python scripts/list_cameras.py [max_index]

This is useful to find the index number of a virtual camera (e.g., OBS Virtual Camera)
that may not be index 0.
"""
import sys
import cv2
import time

max_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 10
print(f"Probing camera indices 0..{max_idx}...")
for i in range(max_idx + 1):
    print(f"Testing index {i}...", end=' ')
    cap = cv2.VideoCapture(i)
    if not cap or not cap.isOpened():
        print("not opened")
        continue
    ok = False
    start = time.time()
    while time.time() - start < 1.0:
        ret, frame = cap.read()
        if ret and frame is not None and getattr(frame, 'size', 0) > 0:
            print("frames OK")
            ok = True
            break
        time.sleep(0.05)
    if not ok:
        print("opened but no frames")
    cap.release()
print("Done.")