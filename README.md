# GROWin

GROWin is an AI-based rowing technique analysis system that extracts and analyzes rowing mechanics from video input.

The system identifies rowing phases, tracks joint kinematics, and evaluates coordination between legs, trunk, and arms for technique analysis and performance feedback.

This project is under active development.

## Demo

### Indoor Rowing Machine Analysis
![12月30日 (4)(1)](https://github.com/user-attachments/assets/76647e75-3172-4630-8851-02c46bdbd62a)


### On-Water Rowing and Hardware Setup
![12月30日 (3)](https://github.com/user-attachments/assets/955144c1-0135-431f-a67a-53da5287058e)
![12月31日(2)](https://github.com/user-attachments/assets/5f3ec397-9fdb-4bdd-8f6b-7c7095152dfb)

## Features

- Rowing phase detection (Drive / Recovery)
- Joint angle tracking (knee, hip, elbow)
- Segment velocity analysis (legs, back, hands)
- Stroke-level segmentation and statistics
- Webcam and recorded video input support

## Tech Stack

- Python
- OpenCV
- MediaPipe Pose

## Troubleshooting ⚠️

If you see an error like:

```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

This means the installed MediaPipe package is a newer release that exposes the Tasks API (`mediapipe.tasks`) instead of the legacy `mediapipe.solutions` API used by this project.

Fix options:

1. Install a compatible MediaPipe version (recommended):

```bash
pip install mediapipe==0.8.10
```

2. Or update the project to use the new MediaPipe Tasks API (advanced).

---

Using the MediaPipe Tasks API (latest releases)

This project is fully migrated to the MediaPipe Tasks API and **requires**
MediaPipe >= 0.10 and a `pose_landmarker.task` model file. The project no longer
relies on `mediapipe.solutions`.

Steps to run with the Tasks API:

1. Install MediaPipe (>= 0.10):

```bash
pip install "mediapipe>=0.10.0"
```

2. Download an official `pose_landmarker.task` model and place it at:

```
models/pose_landmarker.task
```

Alternatively use the helper script if you have a direct download URL:

```bash
python scripts/download_pose_model.py <model_url>
```

3. Run the GUI or `main.py` (it expects the model at `models/pose_landmarker.task`):

```bash
python gui.py
```

Notes:
- The project now uses the Tasks `PoseLandmarker` model for better accuracy and
  tracking. If you prefer the old behavior you can still pin an older MediaPipe
  release (e.g., `mediapipe==0.8.10`) in `requirements.txt`.
- If you hit import errors, ensure `mediapipe` is installed in the same Python
  environment used to run the project.


Camera notes (OBS virtual camera)

- If you use a virtual camera (for example OBS Virtual Camera), it may not be device index `0`. To find which index corresponds to your virtual camera, run:

```bash
python scripts/list_cameras.py 10
```

- You can force which source to use by setting environment variables before launching the GUI:

```bash
# use camera index 1
set VIDEO_SOURCE=1
python gui.py

# or use a video file
set VIDEO_SOURCE=path\to\video.mp4
python gui.py
```

- If the app opens the camera but frames are not received, make sure the virtual camera is started in OBS (`Start Virtual Camera`) and that no other app is exclusively locking it. The app attempts backend fallbacks and will probe indices automatically if the configured source doesn't return frames.


- NumPy
- Matplotlib

## Status

Functional prototype.  
Currently optimized for indoor rowing environments.  
Outdoor rowing support is experimental.

## Website

https://hydro-drive.com/

## Contributions

This is a personal research and engineering project.  
Technical feedback and discussion are welcome.
