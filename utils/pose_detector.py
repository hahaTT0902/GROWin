"""MediaPipe Tasks-only Pose Landmarker helper.

This module implements a minimal Task-native `PoseDetector` that requires a
`pose_landmarker.task` model and MediaPipe >= 0.10.
"""
import os
import cv2
import numpy as np

# Simple skeleton pairs used for drawing
SKELETON_PAIRS = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 12),            # shoulders
    (11, 23), (12, 24),  # torso
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
    (23, 24)             # pelvis
]

class PoseDetector:
    """Tasks-native PoseLandmarker wrapper.

    Args:
        model_path: path to a `.task` pose landmarker model. Required.
        running_mode: one of 'IMAGE' or 'VIDEO'.
    """

    def __init__(self, model_path=None, running_mode='VIDEO'):
        if model_path is None:
            # default location
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'pose_landmarker.task')
            model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            raise RuntimeError(
                f"PoseLandmarker model not found at '{model_path}'.\n"
                "Download a MediaPipe pose_landmarker `.task` model and place it at that path or pass `model_path` explicitly."
            )

        try:
            from mediapipe.tasks.python.vision import (
                PoseLandmarker, PoseLandmarkerOptions, RunningMode
            )
            from mediapipe.tasks.python.core import base_options as mp_base_options
            import importlib
            mp_image_mod = importlib.import_module('mediapipe.tasks.python.vision.core.image')
        except Exception as e:
            raise RuntimeError(
                "Failed to import MediaPipe Tasks API. Ensure you have mediapipe>=0.10 installed."
            ) from e

        base_opts = mp_base_options.BaseOptions(model_asset_path=model_path)
        rm = RunningMode.VIDEO if running_mode.upper() == 'VIDEO' else RunningMode.IMAGE
        opts = PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=rm,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.pose_landmarker = PoseLandmarker.create_from_options(opts)
        self.Image = mp_image_mod.Image
        self.ImageFormat = mp_image_mod.ImageFormat
        self.mode = 'tasks'
        self.running_mode = rm

    def process(self, frame, timestamp_ms=None):
        """Process a BGR OpenCV frame and return a PoseLandmarkerResult.

        If `timestamp_ms` is supplied it will be used for VIDEO mode processing.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.Image(self.ImageFormat.SRGB, np.asarray(rgb, dtype=np.uint8))

        # For VIDEO running mode, call detect_for_video with timestamp
        ip_opts = None
        try:
            import importlib
            mp_ip = importlib.import_module('mediapipe.tasks.python.vision.core.image_processing_options')
            # image_processing_options currently supports region_of_interest and rotation_degrees
            # we don't need special options for normal operation, but keep the import available
        except Exception:
            mp_ip = None

        if getattr(self, 'running_mode', None) is not None and getattr(self, 'running_mode').name == 'VIDEO':
            # timestamp required for video mode; if not provided, use current time
            if timestamp_ms is None:
                import time
                timestamp_ms = int(time.time() * 1000)
            result = self.pose_landmarker.detect_for_video(image, int(timestamp_ms), ip_opts)
        else:
            # IMAGE mode
            result = self.pose_landmarker.detect(image, ip_opts)

        return result

    def draw_landmarks(self, frame, result, color=(0, 255, 0), radius=3, thickness=2):
        """Draw landmarks using Tasks result objects."""
        if result is None or not getattr(result, 'pose_landmarks', None):
            return
        h, w = frame.shape[:2]
        # PoseLandmarkerResult.pose_landmarks is a list of poses, each a list of normalized landmarks
        points = {}
        for pose_landmarks in result.pose_landmarks:
            for idx, lm in enumerate(pose_landmarks):
                try:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                except Exception:
                    continue
                points[idx] = (x, y)
                cv2.circle(frame, (x, y), radius, color, -1)

        for a, b in SKELETON_PAIRS:
            if a in points and b in points:
                cv2.line(frame, points[a], points[b], color, thickness)

    def close(self):
        try:
            self.pose_landmarker.close()
        except Exception:
            pass
