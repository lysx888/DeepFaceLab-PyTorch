import json
import shutil
import sys
import threading
import subprocess
from collections import OrderedDict, deque
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, QSize
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton, QLineEdit,
    QGroupBox, QProgressBar, QFileDialog, QScrollArea, QTextEdit,
    QMessageBox, QDialog, QDialogButtonBox, QButtonGroup,
    QSplitter, QSlider, QFrame, QGridLayout,
)

from faceswap.shared.image_utils import bgr_to_rgb
from faceswap.shared.logger import get_logger
from faceswap.gui_app.gui_utils import install_no_wheel
from faceswap.gui_app.panels.base import (
    StepPanel, AutoScaleLabel, _PreviewSignal, _LogSignal, _ProgressSignal,
    _ClosePreviewSignal, _Worker, _StreamBridge,
)

_logger = get_logger("gui")
from faceswap.gui_app.param_defs import ParamGroup, GROUP_COLORS
from faceswap.setting import (
    WORKSPACE_DIR, MODEL_DIR, XSEG_MODEL_DIR,
    AMP_MODEL_DIR, QUICK96_MODEL_DIR,
    DATA_SRC_DIR, DATA_DST_DIR,
    DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR,
    DATA_DST_SWAPPED_DIR, DATA_DST_MERGED_DIR, DATA_DST_MERGED_MASK_DIR,
    IF_LANDMARK_MODEL_DIR, INSIGHTFACE_MANUAL_ANNOTATED_DIR,
    SCRFD_MODEL_DIR, PRETRAIN_DATA_DIR,
    FaceType,
)

class _VideoExtractHalf(QWidget):
    _sig_running = pyqtSignal(bool)

    def __init__(self, label: str, video_default: str, is_src: bool,
                 stream_callback=None, parent=None):
        super().__init__(parent)
        self._is_src = is_src
        self._running = False
        self._thread = None
        self._stream_callback = stream_callback
        self._sig_running.connect(self._apply_running)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        grp = QGroupBox(label)
        grp_layout = QVBoxLayout(grp)
        grp_layout.setSpacing(4)

        row0 = QHBoxLayout()
        lbl0 = QLabel("视频路径:")
        lbl0.setFixedWidth(80)
        self._video_path = QLineEdit(video_default)
        btn0 = QPushButton("浏览")
        btn0.setFixedWidth(80)
        btn0.setProperty("outline", True)
        btn0.clicked.connect(lambda: self._browse_video())
        row0.addWidget(lbl0)
        row0.addWidget(self._video_path, 1)
        row0.addWidget(btn0)
        grp_layout.addLayout(row0)

        row1 = QHBoxLayout()
        lbl1 = QLabel("FPS:")
        lbl1.setFixedWidth(80)
        self._fps = QSpinBox()
        self._fps.setRange(0, 120)
        self._fps.setValue(0)
        self._fps.setSpecialValueText("原始")
        if not is_src:
            self._fps.setReadOnly(True)
            self._fps.setEnabled(False)
        row1.addWidget(lbl1)
        row1.addWidget(self._fps)
        grp_layout.addLayout(row1)

        self._limit_threads_cb = QCheckBox("限制解码线程（HDD磁盘建议勾选）")
        grp_layout.addWidget(self._limit_threads_cb)

        self._run_btn = QPushButton("开始提取")
        self._run_btn.setStyleSheet(
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        grp_layout.addWidget(self._run_btn)

        layout.addWidget(grp)

    def _browse_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择视频文件", self._video_path.text())
        if p:
            self._video_path.setText(p)

    def _apply_running(self, running: bool):
        self._running = running
        self._run_btn.setEnabled(not running)
        self._run_btn.setText("提取中..." if running else "开始提取")
        win = self.window()
        if hasattr(win, 'set_busy'):
            win.set_busy(running)

    def _set_running(self, running: bool):
        self._sig_running.emit(running)

    def _on_run(self):
        from faceswap.business.video_processor import VideoProcessor
        vp = VideoProcessor()
        video = Path(self._video_path.text())

        self._set_running(True)

        def _task():
            try:
                fps = self._fps.value() or None
                limit_threads = self._limit_threads_cb.isChecked()
                if self._is_src:
                    vp.extract_frames_src(video, DATA_SRC_DIR, fps=fps,
                                          output_format="png",
                                          limit_threads=limit_threads,
                                          stream_callback=self._stream_callback)
                else:
                    vp.extract_frames_dst(video, DATA_DST_DIR, fps=fps,
                                          limit_threads=limit_threads,
                                          stream_callback=self._stream_callback)
                if self._stream_callback:
                    tag = "源视频" if self._is_src else "目标视频"
                    self._stream_callback(f"========== {tag}提取完成 ==========")
            except Exception as e:
                _logger.error(str(e))
            finally:
                self._set_running(False)

        self._thread = threading.Thread(target=_task, daemon=True)
        self._thread.start()


class _VideoCutHalf(QWidget):
    _sig_running = pyqtSignal(bool)

    def __init__(self, label: str, video_default: str, stream_callback=None, parent=None):
        super().__init__(parent)
        self._stream_callback = stream_callback
        self._sig_running.connect(self._apply_running)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        grp = QGroupBox(label)
        grp_layout = QVBoxLayout(grp)
        grp_layout.setSpacing(4)

        r1 = QHBoxLayout()
        lbl1 = QLabel("视频路径:")
        lbl1.setFixedWidth(80)
        self._video_path = QLineEdit(video_default)
        btn_browse = QPushButton("浏览")
        btn_browse.setFixedWidth(80)
        btn_browse.setProperty("outline", True)
        btn_browse.clicked.connect(self._browse_video)
        r1.addWidget(lbl1)
        r1.addWidget(self._video_path, 1)
        r1.addWidget(btn_browse)
        grp_layout.addLayout(r1)

        r2 = QHBoxLayout()
        lbl2 = QLabel("开始时间:")
        lbl2.setFixedWidth(80)
        self._cut_start = QLineEdit("00:00:00.000")
        self._cut_start.setFixedWidth(120)
        lbl3 = QLabel("结束:")
        self._cut_end = QLineEdit("00:00:00.000")
        self._cut_end.setFixedWidth(120)
        self._run_btn = QPushButton("开始切割")
        self._run_btn.setProperty("warning", True)
        self._run_btn.setStyleSheet(
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        r2.addWidget(lbl2)
        r2.addWidget(self._cut_start)
        r2.addWidget(lbl3)
        r2.addWidget(self._cut_end)
        r2.addStretch()
        r2.addWidget(self._run_btn)
        grp_layout.addLayout(r2)

        r3 = QHBoxLayout()
        lbl4 = QLabel("均分段数:")
        lbl4.setFixedWidth(80)
        self._seg_count = QSpinBox()
        self._seg_count.setRange(2, 100)
        self._seg_count.setValue(2)
        self._seg_count.setFixedWidth(80)
        self._seg_btn = QPushButton("开始分割")
        self._seg_btn.setProperty("warning", True)
        self._seg_btn.clicked.connect(self._on_segment_run)
        r3.addWidget(lbl4)
        r3.addWidget(self._seg_count)
        r3.addStretch()
        r3.addWidget(self._seg_btn)
        grp_layout.addLayout(r3)

        layout.addWidget(grp)

    def _browse_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择视频文件", self._video_path.text())
        if p:
            self._video_path.setText(p)

    def _apply_running(self, running: bool):
        self._run_btn.setText("切割中..." if running else "开始切割")
        self._seg_btn.setText("分割中..." if running else "开始分割")
        win = self.window()
        if hasattr(win, 'set_busy'):
            win.set_busy(running)

    def _on_run(self):
        from faceswap.business.video_processor import VideoProcessor
        vp = VideoProcessor()
        video = Path(self._video_path.text())
        start = self._cut_start.text().strip()
        end = self._cut_end.text().strip()
        output = video.with_stem(video.stem + "_cut")

        self._sig_running.emit(True)

        def _task():
            try:
                vp.cut_video(video, output, start, end, stream_callback=self._stream_callback)
                if self._stream_callback:
                    self._stream_callback("========== 视频切割完成 ==========")
            except Exception as e:
                _logger.error(str(e))
            finally:
                self._sig_running.emit(False)

        threading.Thread(target=_task, daemon=True).start()

    def _on_segment_run(self):
        from faceswap.business.video_processor import VideoProcessor
        vp = VideoProcessor()
        video = Path(self._video_path.text())
        n_seg = self._seg_count.value()

        self._sig_running.emit(True)

        def _task():
            try:
                info = vp.get_video_info(video)
                duration = float(info.get("duration", 0))
                if duration <= 0:
                    if self._stream_callback:
                        self._stream_callback(f"无法获取视频时长: {video.name}")
                    return
                seg_dur = duration / n_seg
                for i in range(n_seg):
                    start_s = i * seg_dur
                    end_s = (i + 1) * seg_dur if i < n_seg - 1 else duration
                    start_ts = f"{int(start_s // 3600):02d}:{int((start_s % 3600) // 60):02d}:{start_s % 60:06.3f}"
                    end_ts = f"{int(end_s // 3600):02d}:{int((end_s % 3600) // 60):02d}:{end_s % 60:06.3f}"
                    out = video.with_stem(f"{video.stem}_seg{i + 1}")
                    if self._stream_callback:
                        self._stream_callback(f"分割第 {i + 1}/{n_seg} 段: {start_ts} -> {end_ts}")
                    vp.cut_video(video, out, start_ts, end_ts, stream_callback=self._stream_callback)
                if self._stream_callback:
                    self._stream_callback(f"========== 视频分段完成: {n_seg} 段 ==========")
            except Exception as e:
                _logger.error(str(e))
            finally:
                self._sig_running.emit(False)

        threading.Thread(target=_task, daemon=True).start()


class Step1VideoExtract(StepPanel):
    step_title = "1. 视频提取帧"
    step_desc = "从源视频/目标视频中提取帧画面（PNG无损）。支持按时间段切割视频。"
    show_run_buttons = False

    def _build_params(self):
        self._stream_bridge = _StreamBridge(self)

        self._stream_output = QTextEdit()
        self._stream_output.setReadOnly(True)
        self._stream_output.setMaximumHeight(180)
        self._stream_output.setStyleSheet(
            "QTextEdit { background-color: #0C0C0C; color: #CCCCCC; "
            "font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 12px; "
            "border: 1px solid #D6D6D6; border-radius: 4px; padding: 4px; }"
        )

        self._stream_bridge.stream_signal.connect(self._append_stream)

        self._src_half = _VideoExtractHalf(
            "源视频 (SRC)", str(WORKSPACE_DIR / "data_src.mp4"),
            is_src=True, stream_callback=self._stream_bridge.emit_stream)
        self._dst_half = _VideoExtractHalf(
            "目标视频 (DST)", str(WORKSPACE_DIR / "data_dst.mp4"),
            is_src=False, stream_callback=self._stream_bridge.emit_stream)
        row = QHBoxLayout()
        row.addWidget(self._src_half)
        row.addWidget(self._dst_half)
        self._params_area.addLayout(row)

        self._params_area.addWidget(self._stream_output)

        cut_label = QLabel("视频切割 (按时间段截取)")
        cut_label.setObjectName("stepDesc")
        self._params_area.addWidget(cut_label)

        self._src_cut = _VideoCutHalf(
            "源视频切割", str(WORKSPACE_DIR / "data_src.mp4"),
            stream_callback=self._stream_bridge.emit_stream)
        self._dst_cut = _VideoCutHalf(
            "目标视频切割", str(WORKSPACE_DIR / "data_dst.mp4"),
            stream_callback=self._stream_bridge.emit_stream)
        cut_row = QHBoxLayout()
        cut_row.addWidget(self._src_cut)
        cut_row.addWidget(self._dst_cut)
        self._params_area.addLayout(cut_row)

    def _append_stream(self, line: str, overwrite: bool = False):
        te = self._stream_output
        if overwrite:
            cursor = te.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            te.setTextCursor(cursor)
        te.moveCursor(te.textCursor().MoveOperation.End)
        te.insertPlainText(line + "\n")
        te.ensureCursorVisible()

