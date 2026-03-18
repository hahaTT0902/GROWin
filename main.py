import os
import sys
import cv2
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from utils.pose_utils import get_relevant_angles
from utils.video_stream import setup_video_capture, release_video_capture

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.pose_detector import PoseDetector

# Instantiate a Task-native pose landmarker. Require a model at 'models/pose_landmarker.task'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'pose_landmarker.task')
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Required model not found: {MODEL_PATH}\nPlease download a MediaPipe `pose_landmarker.task` model and place it there. See README.md for details."
    )
pose_detector = PoseDetector(model_path=MODEL_PATH, running_mode='VIDEO')

# No legacy drawing utils used; use `pose_detector.draw_landmarks` for visualization
mp_drawing = None

skeleton_pairs = [
    (11, 13), (13, 15),  # 左臂
    (12, 14), (14, 16),  # 右臂
    (11, 12),            # 双肩
    (11, 23), (12, 24),  # 躯干
    (23, 25), (25, 27),  # 左腿
    (24, 26), (26, 28),  # 右腿
    (23, 24)             # 骨盆
]

def smooth_append(series, value, alpha=0.3):
    if not series:
        series.append(value)
    else:
        smoothed = alpha * value + (1 - alpha) * series[-1]
        series.append(smoothed)

# 理想角度区间（用于反馈）
angle_ranges = {
    'leg_drive_angle': (60, 110),
    'back_angle': (20, 50),
    'arm_angle': (150, 170)
}

# 标准切换角度区间
switch_angle_ranges = {
    "Drive→Recovery": {
        "leg_drive_angle": (190, 220),
        "back_angle": (105, 135),
        "arm_angle": (80, 110),
    },
    "Recovery→Drive": {
        "leg_drive_angle": (275, 300),
        "back_angle": (20, 45),
        "arm_angle": (160, 180),
    }
}

# 阶段追踪，检测状态切换点
toggle_angles = []
# 右侧角度：仅在phase切换时更新，之后定格显示
frozen_angles = {}  # {name: angle_value} — snapshot at phase switch

class StrokeStateTracker:
    def __init__(self):
        self.state = "Unknown"
        self.previous_wrist_x = None
        self.last_state = "Unknown"
        self.stroke_count = 0
        self.stroke_timestamps = deque(maxlen=30)
        self.last_angles = {}
        self.stable_counter = 0
        self.stable_required = 3  # 连续帧数确认切换
        self.pending_state = None
        self.motion_threshold = 2.0

    def update(self, wrist_x, current_time, angles, hip_x=None):
        driver_x = wrist_x if wrist_x is not None else hip_x
        if driver_x is None:
            return self.state, self.stroke_count, 0.0, None

        if self.previous_wrist_x is None:
            self.previous_wrist_x = driver_x
            return self.state, self.stroke_count, 0.0, None

        dx = driver_x - self.previous_wrist_x
        self.previous_wrist_x = driver_x

        # 判断新状态
        new_state = self.state
        if dx < -self.motion_threshold:
            candidate_state = "Drive"
        elif dx > self.motion_threshold:
            candidate_state = "Recovery"
        else:
            candidate_state = self.state

        switch = None
        # 防抖动切换
        if candidate_state != self.state:
            if self.pending_state == candidate_state:
                self.stable_counter += 1
            else:
                self.pending_state = candidate_state
                self.stable_counter = 1

            required_count = 1 if self.state == "Unknown" else self.stable_required
            if self.stable_counter >= required_count:
                new_state = candidate_state
                self.stable_counter = 0
                self.pending_state = None
        else:
            self.stable_counter = 0
            self.pending_state = None

        if new_state != self.last_state:
            if self.last_state == "Drive" and new_state == "Recovery":
                self.stroke_count += 1
                self.stroke_timestamps.append(current_time)
                switch = "Drive→Recovery"
            elif self.last_state == "Recovery" and new_state == "Drive":
                switch = "Recovery→Drive"

            # 记录切换时的角度
            toggle_angles.append((current_time, switch, angles.copy()))
            self.last_state = new_state

        self.state = new_state

        # SPM
        spm = 0.0
        if len(self.stroke_timestamps) >= 2:
            durations = [self.stroke_timestamps[i] - self.stroke_timestamps[i - 1] for i in range(1, len(self.stroke_timestamps))]
            avg_duration = sum(durations) / len(durations)
            spm = 60.0 / avg_duration if avg_duration > 0 else 0.0

        return new_state, self.stroke_count, spm, switch

# 相对位移
def relative_movement(p1, p2):
    if p1 is None or p2 is None:
        return 0.0
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.sqrt(dx ** 2 + dy ** 2)

# 初始化缓存
time_series = deque(maxlen=100)
leg_series = deque(maxlen=100)
back_series = deque(maxlen=100)
arm_series = deque(maxlen=100)
phase_labels = deque(maxlen=100)
phase_spans = []

# 新增：判断可见性函数
def get_joint_if_visible(joints, idx, threshold=0.5):
    vis = joints.get(f"{idx}_vis", 1.0)
    return joints[idx] if vis > threshold else None

# 主程序
def main(data_callback=None, running_flag=lambda: True, get_mirror=lambda: False):
    cap = setup_video_capture()
    if not cap.isOpened():
        print("Camera failed to open.")
        return
    else:
        print("Camera opened successfully; starting processing loop.")

    tracker = StrokeStateTracker()
    prev_hip = prev_shoulder = prev_wrist = None
    start_time = time.time()
    frame_count = 0
    last_print = start_time

    # 统一日志文件
    log_f = open('log.csv', 'w', newline='')
    log_writer = csv.writer(log_f)
    
    # 关节索引列表（不含头部）
    joint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # 右/左肩、肘、腕、髋、膝、踝

    # 写入csv表头
    log_writer.writerow([
        'Time', 'Phase', 'SPM', 'Switch',
        *[f'joint_{idx}_x' for idx in joint_indices],
        *[f'joint_{idx}_y' for idx in joint_indices],
        *[f'joint_{idx}_z' for idx in joint_indices],
        *[f'vec_{a}_{b}_dx' for a, b in skeleton_pairs],
        *[f'vec_{a}_{b}_dy' for a, b in skeleton_pairs],
        *[f'vec_{a}_{b}_dz' for a, b in skeleton_pairs],
        'Leg Movement', 'Back Movement', 'Arm Movement',
        'leg_drive_angle', 'back_angle', 'arm_angle'
    ])

    last_feedback_msgs = []

    while cap.isOpened() and running_flag():
        ret, frame = cap.read()
        if not ret:
            print('Frame read failed or end of stream; attempting to reconnect...')
            # Try to reconnect a few times
            reconnected = False
            try:
                cap.release()
            except Exception:
                pass
            for attempt in range(3):
                time.sleep(0.5)
                cap = setup_video_capture()
                if cap and cap.isOpened():
                    r, f = cap.read()
                    if r and f is not None and getattr(f, 'size', 0) > 0:
                        ret, frame = r, f
                        reconnected = True
                        print(f'Reconnected to video source on attempt {attempt+1}')
                        break
            if not reconnected:
                # Probe for any working device
                cap = setup_video_capture(source=None, max_probe_index=int(os.getenv('VIDEO_PROBE_MAX', 5)))
                if not cap:
                    print('Reconnection/probe failed; exiting loop.')
                    break
                # try one read
                r, f = cap.read()
                if not r or f is None or getattr(f, 'size', 0) == 0:
                    print('Frame read still failing after probe; exiting loop.')
                    break
                ret, frame = r, f
        frame_count += 1
        # periodic progress print every 100 frames or every 5 seconds
        if frame_count % 100 == 0 or (time.time() - last_print) > 5:
            print(f'Processed frames: {frame_count}')
            last_print = time.time()

        # Mirror frame before algorithm if requested
        if callable(get_mirror) and get_mirror():
            frame = cv2.flip(frame, 1)

        # Use compatibility detector
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = pose_detector.process(frame, timestamp_ms=timestamp_ms)
        joints = {}
        stroke_phase = stroke_count = spm = 0
        angles = {}
        switch = None

        if result and getattr(result, 'pose_landmarks', None):
            # Prefer the legacy drawing util when available
            if mp_drawing is not None and hasattr(mp_drawing, 'draw_landmarks'):
                try:
                    mp_drawing.draw_landmarks(frame, result.pose_landmarks, None)
                except Exception:
                    # fallback to simple drawing
                    pose_detector.draw_landmarks(frame, result)
            else:
                pose_detector.draw_landmarks(frame, result)

            # Normalize landmark access: Tasks returns a list of poses (each a list of landmarks).
            landmark_list = []
            if hasattr(result.pose_landmarks, 'landmark'):
                # legacy-like structure
                landmark_list = result.pose_landmarks.landmark
            elif isinstance(result.pose_landmarks, (list, tuple)):
                # take first detected pose
                landmark_list = result.pose_landmarks[0] if len(result.pose_landmarks) > 0 else []

            for idx, landmark in enumerate(landmark_list):
                # landmark.x/y are normalized
                joints[idx] = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                # attempt to find visibility/score
                vis = getattr(landmark, 'visibility', None)
                if vis is None:
                    vis = getattr(landmark, 'score', 1.0)
                joints[f"{idx}_vis"] = vis

            angles = get_relevant_angles(joints)

            for name, joint_ids in [
                ('back_angle', (12, 24, 26)),
                ('leg_drive_angle', (24, 26, 28)),
                ('arm_angle', (12, 14, 16))
            ]:
                if name in angles and all(j in joints for j in joint_ids):
                    p1, p2, p3 = joints[joint_ids[0]], joints[joint_ids[1]], joints[joint_ids[2]]
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
                    cv2.line(frame, p2, p3, (0, 255, 0), 2)
                    # 仅显示phase切换时定格的角度
                    if name in frozen_angles:
                        cv2.putText(frame, f"{int(frozen_angles[name])}°", (p2[0] + 10, p2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

            shoulder = get_joint_if_visible(joints, 12)
            hip = get_joint_if_visible(joints, 24)
            wrist = get_joint_if_visible(joints, 16)

            leg_move = relative_movement(prev_hip, hip)
            back_move = relative_movement(prev_shoulder, shoulder)
            arm_move = relative_movement(prev_wrist, wrist)

            if hip is not None:
                prev_hip = hip
            if shoulder is not None:
                prev_shoulder = shoulder
            if wrist is not None:
                prev_wrist = wrist

            t = time.time() - start_time
            time_series.append(t)
            smooth_append(leg_series, leg_move)
            smooth_append(back_series, back_move)
            smooth_append(arm_series, arm_move)

            wrist_x = wrist[0] if wrist is not None else None
            hip_x = hip[0] if hip is not None else None

            stroke_phase, stroke_count, spm, switch = tracker.update(
                wrist_x,
                t,
                angles,
                hip_x=hip_x,
            )
            # phase切换时定格角度
            if switch is not None:
                frozen_angles.clear()
                frozen_angles.update(angles)

            phase_labels.append(stroke_phase)
            if len(phase_spans) == 0 or phase_spans[-1][1] != stroke_phase:
                phase_spans.append((t, stroke_phase))
            if len(phase_spans) > 200:
                phase_spans.pop(0)

            log_writer.writerow([
                t, stroke_phase, spm, switch if switch is not None else '',
                *[landmark_list[idx].x for idx in joint_indices],
                *[landmark_list[idx].y for idx in joint_indices],
                *[landmark_list[idx].z for idx in joint_indices],
                *[joints[b][0] - joints[a][0] for a, b in skeleton_pairs],
                *[joints[b][1] - joints[a][1] for a, b in skeleton_pairs],
                *[landmark_list[b].z - landmark_list[a].z for a, b in skeleton_pairs],
                leg_move, back_move, arm_move,
                angles.get('leg_drive_angle', 0),
                angles.get('back_angle', 0),
                angles.get('arm_angle', 0)
            ])

            if switch in switch_angle_ranges:
                phase_name = "Finish" if switch == "Drive→Recovery" else "Catch"
                feedback_msgs = []
                for name in ['leg_drive_angle', 'back_angle', 'arm_angle']:
                    angle = angles.get(name)
                    minv, maxv = switch_angle_ranges[switch][name]
                    if angle is None:
                        msg = f"{phase_name}: {name} Unknown"
                    elif minv <= angle <= maxv:
                        msg = f"{phase_name}: {name} OK"
                    elif angle < minv:
                        msg = f"{phase_name}: {name} Too Small"
                    else:
                        msg = f"{phase_name}: {name} Too Large"
                    feedback_msgs.append(msg)
                last_feedback_msgs = feedback_msgs

        # Text overlays removed — data sent via callback

        gui_toggle_angles = list(toggle_angles)
        if data_callback:
            data_callback({
                'frame': frame.copy(),
                'time_series': list(time_series),
                'leg_series': list(leg_series),
                'back_series': list(back_series),
                'arm_series': list(arm_series),
                'phase_spans': list(phase_spans),
                'phases': list(phase_labels),
                'toggle_angles': gui_toggle_angles,
                'stroke_phase': stroke_phase,
                'stroke_count': stroke_count,
                'spm': spm,
                'feedback_msgs': list(last_feedback_msgs),
                'angles': dict(angles),
            })

    release_video_capture(cap)
    log_f.close()
    print(f"\n程序结束 — total frames processed: {frame_count}")

    # 不要直接运行 main.py !，只作为模块被GUI调用!