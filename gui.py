import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QFrame, QSplitter, QScrollArea, QSizePolicy
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib

# ---- Matplotlib 中文字体支持 ----
_CN_FONTS = ['PingFang HK', 'STHeiti', 'Heiti TC', 'SimHei', 'Arial Unicode MS']
for _f in _CN_FONTS:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        matplotlib.rcParams['font.sans-serif'] = [_f] + matplotlib.rcParams.get('font.sans-serif', [])
        break
    except Exception:
        continue
matplotlib.rcParams['axes.unicode_minus'] = False

# ---- 多语言翻译 ----
TEXTS = {
    'zh': {
        'window_title': 'AiRowing 多视图界面',
        'pc_left': '电脑在左侧', 'pc_right': '电脑在右侧',
        'lang_btn': 'English',
        'phase': '阶段', 'strokes': '划桨次数', 'spm': '桨频',
        'phase_direction': '阶段方向',
        'drive': '驱动', 'recovery': '恢复',
        'finish': '出水', 'catch': '入水',
        'leg_drive': '腿驱动角度', 'back_angle': '背部角度', 'arm_angle': '手臂角度',
        'no_suggestions': '暂无建议', 'good_form': '动作良好', 'no_metrics': '暂无指标数据',
        'leg_small': '腿驱动角度偏小，建议加大腿部发力',
        'back_small': '背部角度偏小，建议后倾更多',
        'arm_small': '手臂角度偏小，建议手臂再靠近身体',
        'leg_large': '腿驱动角度偏大，注意避免过度伸展',
        'back_large': '背部角度偏大，注意避免过度后仰',
        'arm_large': '手臂角度偏大，注意手臂不要过度打开',
        'plot1_title': '实时运动轨迹', 'plot1_x': '时间 (秒)', 'plot1_y': '运动量 (像素)',
        'plot2_title': '阶段切换角度', 'plot2_x': '时间 (秒)', 'plot2_y': '角度 (°)',
        'p1_l1': '臀部', 'p1_l2': '背部', 'p1_l3': '手臂',
        'p2_l1': '腿驱角度', 'p2_l2': '背部角度', 'p2_l3': '手臂角度',
    },
    'en': {
        'window_title': 'AiRowing Multi-View GUI',
        'pc_left': 'PC on Left', 'pc_right': 'PC on Right',
        'lang_btn': '中文',
        'phase': 'Phase', 'strokes': 'Strokes', 'spm': 'SPM',
        'phase_direction': 'Phase Direction',
        'drive': 'Drive', 'recovery': 'Recovery',
        'finish': 'Finish', 'catch': 'Catch',
        'leg_drive': 'Leg Drive Angle', 'back_angle': 'Back Angle', 'arm_angle': 'Arm Angle',
        'no_suggestions': 'No suggestions yet', 'good_form': 'Good form', 'no_metrics': 'No metrics data',
        'leg_small': 'Leg drive angle too small, increase leg power',
        'back_small': 'Back angle too small, lean back more',
        'arm_small': 'Arm angle too small, bring arms closer',
        'leg_large': 'Leg drive angle too large, avoid overextension',
        'back_large': 'Back angle too large, avoid leaning back too far',
        'arm_large': 'Arm angle too large, avoid opening arms too wide',
        'plot1_title': 'Real-Time Movement', 'plot1_x': 'Time (s)', 'plot1_y': 'Movement (px)',
        'plot2_title': 'Angle at Phase Switch', 'plot2_x': 'Time (s)', 'plot2_y': 'Angle (°)',
        'p1_l1': 'Buttocks', 'p1_l2': 'Back', 'p1_l3': 'Arms',
        'p2_l1': 'Leg Drive', 'p2_l2': 'Back Angle', 'p2_l3': 'Arm Angle',
    },
}

# ---- 视频显示控件 ----
class VideoWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setScaledContents(False)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(self.size(), aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
        self.setPixmap(pixmap)

# ---- Matplotlib 曲线控件 ----
class PlotWidget(FigureCanvas):
    def __init__(self, title, xlabel, ylabel, lines_info):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.lines = []
        for color, label in lines_info:
            line, = self.ax.plot([], [], color=color, label=label)
            self.lines.append(line)
        self._style_axes()
        self.ax.legend()
        self.fig.tight_layout()

    def _style_axes(self):
        self.fig.patch.set_facecolor('#0f1a2b')
        self.ax.set_facecolor('#101e31')
        self.ax.grid(color='#27415e', linestyle='--', linewidth=0.8, alpha=0.45)
        self.ax.tick_params(colors='#b9d8f2')
        for spine in self.ax.spines.values():
            spine.set_color('#365676')
        self.ax.set_title(self._title, color='#9fe2ff')
        self.ax.set_xlabel(self._xlabel, color='#b9d8f2')
        self.ax.set_ylabel(self._ylabel, color='#b9d8f2')

    def update_plot(self, x, ys_list, phase_spans=None, phases=None):
        self.ax.clear()
        self._style_axes()
        if x and phase_spans:
            t_min = x[0]
            t_max = x[-1]
            current_bg = "#e6f2ff"
            if phases:
                current_bg = "#ffe6cc" if phases[-1] == "Drive" else "#e6f2ff"
            last_span_time = t_min
            for span_time, phase in phase_spans:
                if span_time < t_min:
                    last_span_time = span_time
                    continue
                if last_span_time > t_max:
                    break
                draw_start = max(last_span_time, t_min)
                draw_end = min(span_time, t_max)
                color = '#ffe6cc' if phase == 'Drive' else '#e6f2ff'
                if draw_start < draw_end:
                    self.ax.axvspan(draw_start, draw_end, facecolor=color, alpha=0.3, edgecolor='none')
                last_span_time = span_time
            if last_span_time < t_max:
                self.ax.axvspan(last_span_time, t_max, facecolor=current_bg, alpha=0.3, edgecolor='none')
        for line, y in zip(self.lines, ys_list):
            line, = self.ax.plot(x, y, color=line.get_color(), label=line.get_label())
        legend = self.ax.legend(facecolor='#101e31', edgecolor='#365676')
        if legend is not None:
            for text in legend.get_texts():
                text.set_color('#b9d8f2')
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw()

    def set_labels(self, title, xlabel, ylabel, line_labels=None):
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        if line_labels:
            for line, lbl in zip(self.lines, line_labels):
                line.set_label(lbl)


# ---- 视频右侧实时信息面板 ----
class InfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._lang = 'zh'
        self.setObjectName('infoPanel')
        self.setMinimumWidth(180)
        self.setStyleSheet("""
            QWidget#infoPanel {
                background: #0e1b2d;
                border: 1px solid #274766;
                border-radius: 10px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.phase_label = QLabel("")
        self.phase_label.setStyleSheet("font-size:22px; font-weight:700; color:#00d4ff;")
        self.strokes_label = QLabel("")
        self.strokes_label.setStyleSheet("font-size:18px; color:#d7e9ff;")
        self.spm_label = QLabel("")
        self.spm_label.setStyleSheet("font-size:18px; color:#7cffb2;")
        self.feedback_label = QLabel("")
        self.feedback_label.setStyleSheet("font-size:18px; color:#f0c060;")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setMinimumHeight(60)

        layout.addWidget(self.phase_label)
        layout.addWidget(self.strokes_label)
        layout.addWidget(self.spm_label)
        layout.addWidget(self.feedback_label)
        layout.addStretch(1)
        self._update_static_text()

    def _update_static_text(self):
        T = TEXTS[self._lang]
        self.phase_label.setText(f"{T['phase']}: —")
        self.strokes_label.setText(f"{T['strokes']}: 0")
        self.spm_label.setText(f"{T['spm']}: 0.0")

    def set_language(self, lang):
        self._lang = lang
        self._update_static_text()

    def update_info(self, data):
        T = TEXTS[self._lang]
        phase = data.get('stroke_phase', 0)
        if phase == 'Drive':
            color = '#00d4ff'
            phase_display = T['drive']
        elif phase == 'Recovery':
            color = '#7cffb2'
            phase_display = T['recovery']
        else:
            color = '#76889a'
            phase_display = str(phase)
        self.phase_label.setText(f"{T['phase']}: {phase_display}")
        self.phase_label.setStyleSheet(f"font-size:22px; font-weight:700; color:{color};")
        self.strokes_label.setText(f"{T['strokes']}: {data.get('stroke_count', 0)}")
        self.spm_label.setText(f"{T['spm']}: {data.get('spm', 0.0):.1f}")
        msgs = data.get('feedback_msgs', [])
        self.feedback_label.setText('\n'.join(msgs) if msgs else '')


class PhaseIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase = "Unknown"
        self.setMinimumHeight(88)

        self.title = QLabel(TEXTS['zh']['phase_direction'])
        self.title.setObjectName("panelTitle")
        self.left_label = QLabel(TEXTS['zh']['drive'])
        self.right_label = QLabel(TEXTS['zh']['recovery'])
        self.icon = QLabel("◉")
        self.icon.setObjectName("phaseIcon")
        self.icon.setAlignment(Qt.AlignCenter)
        self.icon.setFixedSize(36, 36)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 6, 10, 6)
        root.setSpacing(6)
        root.addWidget(self.title)

        self.track = QFrame()
        self.track.setObjectName("phaseTrack")
        self.track.setMinimumHeight(42)
        track_layout = QHBoxLayout(self.track)
        track_layout.setContentsMargins(12, 10, 12, 10)
        track_layout.addWidget(self.left_label, 0, Qt.AlignLeft | Qt.AlignVCenter)
        track_layout.addStretch(1)
        track_layout.addWidget(self.right_label, 0, Qt.AlignRight | Qt.AlignVCenter)
        root.addWidget(self.track)

        self._apply_phase_style()

    def _apply_phase_style(self):
        if self.phase == 'Drive':
            self.icon.setText('◀')
            self.icon.setStyleSheet("background:#00d4ff; color:#08131f; border-radius:18px; font-size:20px; font-weight:700;")
        elif self.phase == 'Recovery':
            self.icon.setText('▶')
            self.icon.setStyleSheet("background:#7cffb2; color:#08131f; border-radius:18px; font-size:20px; font-weight:700;")
        else:
            self.icon.setText('●')
            self.icon.setStyleSheet("background:#76889a; color:#e8f4ff; border-radius:18px; font-size:16px; font-weight:700;")

    def _phase_ratio(self):
        if self.phase == 'Drive':
            return 0.25
        if self.phase == 'Recovery':
            return 0.75
        return 0.5

    def _update_icon_position(self):
        if not self.track:
            return
        icon_w = self.icon.width()
        y = self.track.y() + (self.track.height() - self.icon.height()) // 2
        left_bound = self.track.x() + 8
        right_bound = self.track.x() + self.track.width() - icon_w - 8
        ratio = self._phase_ratio()
        x = int(left_bound + ratio * max(0, right_bound - left_bound))
        x = max(left_bound, min(right_bound, x))
        self.icon.move(x, y)
        self.icon.raise_()

    def set_language(self, lang):
        T = TEXTS[lang]
        self.title.setText(T['phase_direction'])
        self.left_label.setText(T['drive'])
        self.right_label.setText(T['recovery'])

    def set_phase(self, phase):
        self.phase = phase if phase in ('Drive', 'Recovery') else 'Unknown'
        self._apply_phase_style()
        self._update_icon_position()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_icon_position()

# ---- 后台线程：运行 main.py 的主循环 ----
class WorkerThread(QThread):
    data_signal = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self._running = True
        self._mirror = False

    def set_mirror(self, on):
        self._mirror = bool(on)

    def run(self):
        from main import main
        main(
            data_callback=self.data_signal.emit,
            running_flag=lambda: self._running,
            get_mirror=lambda: self._mirror,
        )

    def stop(self):
        self._running = False

# ---- 主窗口 ----
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._lang = 'zh'
        T = TEXTS[self._lang]
        self.setWindowTitle(T['window_title'])
        screen = QApplication.primaryScreen()
        size = screen.size()
        base_w, base_h = 1920, 1080
        scale_w = size.width() / base_w
        scale_h = size.height() / base_h
        self._ui_scale = min(scale_w, scale_h)

        self.video_widget = VideoWidget()
        self.info_panel = InfoPanel()
        self.metrics_widget = MetricsWidget()
        self.phase_indicator = PhaseIndicator()
        self.camera_side = 'left'
        self.mirror_on = False

        self.lang_btn = QPushButton(T['lang_btn'])
        self.lang_btn.setCheckable(False)
        self.lang_btn.clicked.connect(self._toggle_language)

        self.side_left_btn = QPushButton(T['pc_left'])
        self.side_right_btn = QPushButton(T['pc_right'])
        self.side_left_btn.setCheckable(True)
        self.side_right_btn.setCheckable(True)
        self.side_left_btn.setChecked(True)
        self.side_left_btn.clicked.connect(lambda: self._set_camera_side('left'))
        self.side_right_btn.clicked.connect(lambda: self._set_camera_side('right'))

        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)
        top_bar.addWidget(self.lang_btn)
        top_bar.addStretch(1)
        top_bar.addWidget(self.side_left_btn)
        top_bar.addWidget(self.side_right_btn)
        top_container = QWidget()
        top_container.setLayout(top_bar)

        self.suggestion_label = QLabel("")
        sug_font_size = max(16, int(18 * self._ui_scale))
        self.suggestion_label.setStyleSheet(
            f"font-size: {sug_font_size}px; color: #b6ffcc; background: #0e1b2d; border: 1px solid #274766; border-radius: 10px; padding: 10px;"
        )
        self.suggestion_label.setWordWrap(True)
        self.suggestion_label.setMinimumHeight(int(60 * self._ui_scale))

        # ---- 左侧：视频 + 曲线（可拖拽分割） ----
        lines_info1 = [('green', T['p1_l1']), ('blue', T['p1_l2']), ('magenta', T['p1_l3'])]
        self.plot1 = PlotWidget(T['plot1_title'], T['plot1_x'], T['plot1_y'], lines_info1)
        lines_info2 = [('lime', T['p2_l1']), ('cyan', T['p2_l2']), ('orange', T['p2_l3'])]
        self.plot2 = PlotWidget(T['plot2_title'], T['plot2_x'], T['plot2_y'], lines_info2)

        left_splitter = QSplitter(Qt.Vertical)
        video_container = QWidget()
        video_row = QHBoxLayout(video_container)
        video_row.setContentsMargins(0, 0, 0, 0)
        video_row.addWidget(self.video_widget, 3)
        video_row.addWidget(self.info_panel, 1)
        left_splitter.addWidget(video_container)

        plots_container = QWidget()
        plots_hbox = QHBoxLayout(plots_container)
        plots_hbox.setContentsMargins(0, 0, 0, 0)
        plots_hbox.addWidget(self.plot1)
        plots_hbox.addWidget(self.plot2)
        left_splitter.addWidget(plots_container)
        left_splitter.setStretchFactor(0, 2)
        left_splitter.setStretchFactor(1, 1)

        # ---- 右侧：控制 + 指标（滚动） ----
        right_inner = QWidget()
        right_vbox = QVBoxLayout(right_inner)
        right_vbox.setContentsMargins(4, 4, 4, 4)
        right_vbox.addWidget(self.phase_indicator)
        right_vbox.addWidget(self.metrics_widget, 1)
        right_vbox.addWidget(self.suggestion_label)

        right_scroll = QScrollArea()
        right_scroll.setWidget(right_inner)
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        # ---- 主分割 ----
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_scroll)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(4, 4, 4, 4)
        central_layout.addWidget(top_container)
        central_layout.addWidget(main_splitter, 1)
        self.setCentralWidget(central)

        self.resize(int(base_w * self._ui_scale * 0.8), int(base_h * self._ui_scale * 0.8))

        self.worker = WorkerThread()
        self.worker.data_signal.connect(self.update_all)
        self.worker.start()

        self._latest_data = None
        self._last_metrics = {'finish': [], 'catch': []}
        self._last_suggestions = T['no_suggestions']
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._refresh_plots)
        self.timer.start()

        self._apply_tech_theme()

    def _apply_tech_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0b1220;
                color: #d7e9ff;
                font-family: 'SF Pro Display', 'PingFang SC', 'Microsoft YaHei';
            }
            QLabel {
                color: #d7e9ff;
            }
            QLabel#panelTitle {
                font-size: 14px;
                color: #7cd6ff;
                font-weight: 700;
            }
            QFrame#phaseTrack {
                border: 1px solid #20364d;
                border-radius: 14px;
                background: #111b2d;
            }
            QPushButton {
                border: 1px solid #265e88;
                border-radius: 10px;
                padding: 8px 12px;
                background: #0f2137;
                color: #9ddcff;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #153150;
            }
            QPushButton:checked {
                border: 1px solid #00c8ff;
                background: #123f5f;
                color: #dff7ff;
            }
            QSplitter::handle {
                background: #274766;
            }
            QSplitter::handle:horizontal {
                width: 4px;
            }
            QSplitter::handle:vertical {
                height: 4px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

    def _toggle_language(self):
        self._lang = 'en' if self._lang == 'zh' else 'zh'
        self._update_all_text()

    def _update_all_text(self):
        T = TEXTS[self._lang]
        self.setWindowTitle(T['window_title'])
        self.lang_btn.setText(T['lang_btn'])
        self.side_left_btn.setText(T['pc_left'])
        self.side_right_btn.setText(T['pc_right'])
        self.info_panel.set_language(self._lang)
        self.phase_indicator.set_language(self._lang)
        self.plot1.set_labels(T['plot1_title'], T['plot1_x'], T['plot1_y'],
                              [T['p1_l1'], T['p1_l2'], T['p1_l3']])
        self.plot2.set_labels(T['plot2_title'], T['plot2_x'], T['plot2_y'],
                              [T['p2_l1'], T['p2_l2'], T['p2_l3']])
        self._last_metrics = {'finish': [], 'catch': []}
        self._last_suggestions = T['no_suggestions']
        self._update_metrics_and_suggestion()

    def _set_camera_side(self, side):
        self.camera_side = 'right' if side == 'right' else 'left'
        self.side_left_btn.setChecked(self.camera_side == 'left')
        self.side_right_btn.setChecked(self.camera_side == 'right')
        self.mirror_on = (self.camera_side == 'left')
        self.worker.set_mirror(self.mirror_on)

    def update_all(self, data):
        self.video_widget.update_frame(data['frame'])
        self.info_panel.update_info(data)
        self._latest_data = data
        phase_list = data.get('phases', [])
        self.phase_indicator.set_phase(phase_list[-1] if phase_list else 'Unknown')
        self._update_metrics_and_suggestion()

    def _refresh_plots(self):
        data = self._latest_data
        has_metrics = bool(self._last_metrics['finish'] or self._last_metrics['catch'])
        if data is None:
            if not has_metrics:
                self.metrics_widget.show_nodata(self._lang)
                self.suggestion_label.setText(TEXTS[self._lang]['no_suggestions'])
            else:
                self.metrics_widget.update_metrics(self._last_metrics['finish'], self._last_metrics['catch'], self._lang)
                self.suggestion_label.setText(self._last_suggestions)
            return
        x = data['time_series']
        phase_spans = data.get('phase_spans', None)
        phases = data.get('phases', None)
        if x:
            t_now = x[-1]
            t_min = max(x[0], t_now - 10)
            indices = [i for i, t in enumerate(x) if t >= t_min]
            x10 = [x[i] for i in indices]
            leg10 = [data['leg_series'][i] for i in indices]
            back10 = [data['back_series'][i] for i in indices]
            arm10 = [data['arm_series'][i] for i in indices]
        else:
            x10, leg10, back10, arm10 = [], [], [], []
        self.plot1.update_plot(x10, [leg10, back10, arm10], phase_spans=phase_spans, phases=phases)
        if data['toggle_angles']:
            filtered = [a for a in data['toggle_angles'] if a[0] >= t_min]
            if filtered:
                times = [a[0] for a in filtered]
                leg_angle = [a[2].get('leg_drive_angle', 0) for a in filtered]
                back_angle = [a[2].get('back_angle', 0) for a in filtered]
                arm_angle = [a[2].get('arm_angle', 0) for a in filtered]
                self.plot2.update_plot(times, [leg_angle, back_angle, arm_angle], phase_spans=phase_spans, phases=phases)
            else:
                self.plot2.update_plot([], [[], [], []], phase_spans=phase_spans, phases=phases)
        else:
            self.plot2.update_plot([], [[], [], []], phase_spans=phase_spans, phases=phases)
        self._update_metrics_and_suggestion()

    def _update_metrics_and_suggestion(self):
        T = TEXTS[self._lang]
        data = self._latest_data
        finish_metrics = self._last_metrics['finish']
        catch_metrics = self._last_metrics['catch']
        suggestions = self._last_suggestions
        if data is not None:
            toggle_angles = data.get('toggle_angles', [])
            new_finish = []
            new_catch = []
            new_suggestions = None
            if toggle_angles:
                finish = [a for a in toggle_angles if a[1] == 'Drive→Recovery']
                catch = [a for a in toggle_angles if a[1] == 'Recovery→Drive']
                if finish:
                    last = finish[-1]
                    angles = last[2]
                    new_finish = [
                        (T['leg_drive'], angles.get('leg_drive_angle', 0), 190, 220, "°"),
                        (T['back_angle'], angles.get('back_angle', 0), 105, 135, "°"),
                        (T['arm_angle'], angles.get('arm_angle', 0), 80, 110, "°")
                    ]
                if catch:
                    last = catch[-1]
                    angles = last[2]
                    new_catch = [
                        (T['leg_drive'], angles.get('leg_drive_angle', 0), 275, 300, "°"),
                        (T['back_angle'], angles.get('back_angle', 0), 20, 45, "°"),
                        (T['arm_angle'], angles.get('arm_angle', 0), 160, 180, "°")
                    ]
            if new_finish or new_catch:
                finish_metrics = new_finish
                catch_metrics = new_catch
                self._last_metrics['finish'] = finish_metrics
                self._last_metrics['catch'] = catch_metrics
                sug_list = []
                sug_checks = []
                if new_finish:
                    for key, m in zip(['leg', 'back', 'arm'], new_finish):
                        sug_checks.append((T['finish'], key, m[1], m[2], m[3]))
                if new_catch:
                    for key, m in zip(['leg', 'back', 'arm'], new_catch):
                        sug_checks.append((T['catch'], key, m[1], m[2], m[3]))
                for phase_name, key, val, lo, hi in sug_checks:
                    if val < lo:
                        sug_list.append(f"{phase_name}: {T[key + '_small']}")
                    elif val > hi:
                        sug_list.append(f"{phase_name}: {T[key + '_large']}")
                if not sug_list:
                    new_suggestions = T['good_form']
                else:
                    new_suggestions = "\n".join(sug_list)
                self._last_suggestions = new_suggestions
                suggestions = new_suggestions
        if finish_metrics or catch_metrics:
            self.metrics_widget.update_metrics(finish_metrics, catch_metrics, self._lang)
        else:
            self.metrics_widget.show_nodata(self._lang)
        self.suggestion_label.setText(suggestions)

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait(2000)
        event.accept()

class MetricsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(12)
        self.groups = {}
        self.setStyleSheet("background: #0e1b2d; border: 1px solid #274766; border-radius: 10px;")

    def clear(self):
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.groups = {}

    def add_group(self, title):
        label = QLabel(f"<b>{title}</b>")
        label.setStyleSheet("font-size: 20px; margin-bottom: 0px; padding-bottom: 0px; color:#7cd6ff;")
        self.layout.addWidget(label)
        group_widget = QWidget()
        group_layout = QVBoxLayout(group_widget)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(6)
        self.layout.addWidget(group_widget)
        self.groups[title] = group_layout
        return group_layout

    def add_metric(self, group, name, value, low, high, unit, min_val=None, max_val=None):
        hbox = QHBoxLayout()
        label = QLabel(name)
        label.setFixedWidth(220)
        label.setStyleSheet("font-size: 14px;")
        hbox.addWidget(label)
        if min_val is None:
            min_val = low - 20
        if max_val is None:
            max_val = high + 20
        bar = MetricBar(low, high, float(value), unit, min_val=min_val, max_val=max_val)
        hbox.addWidget(bar, 1)
        group.addLayout(hbox)

    def update_metrics(self, finish_metrics, catch_metrics, lang='zh'):
        if not hasattr(self, '_last_finish'):
            self._last_finish = []
            self._last_catch = []
        if not finish_metrics and not catch_metrics:
            return
        if finish_metrics == self._last_finish and catch_metrics == self._last_catch:
            return
        T = TEXTS[lang]
        self._last_finish = finish_metrics.copy()
        self._last_catch = catch_metrics.copy()
        self.clear()
        finish_group = self.add_group(T['finish'])
        for m in finish_metrics:
            self.add_metric(finish_group, *m)
        catch_group = self.add_group(T['catch'])
        for m in catch_metrics:
            self.add_metric(catch_group, *m)

    def show_nodata(self, lang='zh'):
        self.clear()
        label = QLabel(TEXTS[lang]['no_metrics'])
        label.setStyleSheet("font-size: 16px; color: #84a6c8;")
        self.layout.addWidget(label)

class MetricBar(QWidget):
    def __init__(self, low, high, value, unit, min_val=0, max_val=100, parent=None):
        super().__init__(parent)
        self.low = low
        self.high = high
        self.value = value
        self.unit = unit
        self.min_val = min_val
        self.max_val = max_val
        self.setFixedHeight(60)
        self.setMinimumWidth(300)

    def set_value(self, value):
        self.value = value
        self.update()

    def paintEvent(self, event):
        from PyQt5 import QtGui, QtCore
        painter = QtGui.QPainter(self)
        rect = self.rect()
        margin = 32
        bar_rect = QtCore.QRect(margin, rect.height()//2-8, rect.width()-2*margin, 16)
        painter.setBrush(QtGui.QColor("#14263c"))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(bar_rect, 8, 8)
        if self.max_val > self.min_val:
            ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
            ratio = max(0, min(1, ratio))
            green_rect = QtCore.QRect(bar_rect.left(), bar_rect.top(), int(bar_rect.width()*ratio), bar_rect.height())
            painter.setBrush(QtGui.QColor("#00d4ff"))
            painter.drawRoundedRect(green_rect, 8, 8)
        font = painter.font()
        font.setPointSize(11)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#8fd8ff"))
        left_ratio = (self.low - self.min_val) / (self.max_val - self.min_val)
        right_ratio = (self.high - self.min_val) / (self.max_val - self.min_val)
        left_x = bar_rect.left() + int(left_ratio * bar_rect.width())
        right_x = bar_rect.left() + int(right_ratio * bar_rect.width())
        left_label = f"{self.low}{self.unit}>"
        right_label = f"<{self.high}{self.unit}"
        left_label_width = painter.fontMetrics().width(left_label)
        painter.drawText(left_x - left_label_width + 2, bar_rect.top()-12, left_label)
        painter.drawText(right_x + 2, bar_rect.top()-12, right_label)
        if self.max_val > self.min_val:
            x = bar_rect.left() + int(ratio * bar_rect.width())
            painter.setPen(QtGui.QColor("#7cd6ff"))
            painter.drawLine(x, bar_rect.top()-2, x, bar_rect.bottom()+2)
            painter.setPen(QtGui.QColor("#d9f2ff"))
            font.setPointSize(13)
            painter.setFont(font)
            value_label = f"{self.value:.0f}{self.unit}"
            value_width = painter.fontMetrics().width(value_label)
            painter.drawText(x-value_width//2, bar_rect.bottom()+16, value_label)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())