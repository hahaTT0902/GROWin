import cv2
import os
import time


def _try_open_index(idx, api_preference=None, warmup_frames=10, warmup_delay=0.05):
    """Try opening a camera index with optional API preference and warm it up."""
    try:
        if api_preference is not None:
            cap = cv2.VideoCapture(idx, api_preference)
        else:
            cap = cv2.VideoCapture(idx)
    except Exception:
        return None

    if not cap or not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None

    # Warm up - attempt to read a few frames
    for _ in range(warmup_frames):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return cap
        time.sleep(warmup_delay)

    # nothing worked, release and return None
    try:
        cap.release()
    except Exception:
        pass
    return None


def setup_video_capture(source=None, max_probe_index=5):
    """Open a video capture robustly.

    Behavior:
    - If `source` is provided it is used (string path or integer index).
    - Otherwise uses `VIDEO_SOURCE` env var or defaults to '0'.
    - For integer indices, tries common Windows backends (CAP_DSHOW, CAP_MSMF) first.
    - If the given source doesn't produce frames, probes indices 0..max_probe_index to find a working device.
    """
    src = source if source is not None else os.getenv('VIDEO_SOURCE', '0')

    # Allow explicit API selection via env var VIDEO_API (e.g., 'CAP_DSHOW')
    api_name = os.getenv('VIDEO_API', '').upper() or None
    api_map = {
        'CAP_DSHOW': cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else None,
        'CAP_MSMF': cv2.CAP_MSMF if hasattr(cv2, 'CAP_MSMF') else None,
        'CAP_VFW': cv2.CAP_VFW if hasattr(cv2, 'CAP_VFW') else None,
    }
    api_pref = api_map.get(api_name)

    # If source looks like an int, try index open with possible backend preferences
    try:
        idx = int(src)
        print(f"Attempting camera index: {idx} (VIDEO_API={api_name})")
        # Try preferred API first
        if api_pref is not None:
            cap = _try_open_index(idx, api_pref)
            if cap:
                print(f"Opened camera index {idx} using API {api_name}")
                return cap
        # Try common Windows backends
        for api in (cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else None,
                    cv2.CAP_MSMF if hasattr(cv2, 'CAP_MSMF') else None,
                    None):
            cap = _try_open_index(idx, api)
            if cap:
                print(f"Opened camera index {idx} using API {api}")
                return cap
        # If index didn't work, fallthrough to probing
        print(f"Index {idx} did not produce frames, will probe other indices.")
    except Exception:
        # treat source as file path
        print(f"Attempting to open video file: {src}")
        try:
            cap = cv2.VideoCapture(src)
            if cap and cap.isOpened():
                print(f"Opened video file: {src}")
                return cap
            else:
                print(f"Failed to open video file: {src}")
        except Exception as e:
            print(f"Error opening video file {src}: {e}")

    # Probe indices to find a working camera
    print(f"Probing camera indices 0..{max_probe_index} to find a working source...")
    for i in range(0, int(max_probe_index) + 1):
        cap = _try_open_index(i)
        if cap:
            print(f"Found working camera at index {i}")
            return cap

    # As a last resort, try opening the original index without warmup
    try:
        fallback = cv2.VideoCapture(int(src)) if str(src).isdigit() else cv2.VideoCapture(src)
        if fallback and fallback.isOpened():
            print("Opened fallback capture, but it may not be returning frames yet.")
            return fallback
    except Exception:
        pass

    print("No working video source found.")
    return None


def release_video_capture(cap):
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
