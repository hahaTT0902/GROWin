"""Simple helper to download a MediaPipe PoseLandmarker `.task` model.

Usage: python scripts/download_pose_model.py <url> [--out models/pose_landmarker.task]

If you don't have a direct URL, follow the guide in README.md to obtain an official model.
"""
import sys
import os
from urllib.request import urlopen, Request


def download(url, out_path):
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading {url} -> {out_path}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(out_path, 'wb') as f:
        f.write(r.read())
    print("Download complete")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/download_pose_model.py <url> [out_path]")
        sys.exit(2)
    url = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join('models', 'pose_landmarker.task')
    try:
        download(url, out)
    except Exception as e:
        print('Download failed:', type(e).__name__, e)
        sys.exit(1)
