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

class _OutputLogSignal(QObject):
    log_signal = pyqtSignal(str, bool)
    done_signal = pyqtSignal()


class Step6Output(StepPanel):
    step_title = "6. 导出视频"
    step_desc = "将合成后的帧转换为带音频的输出视频。"
    show_run_buttons = False

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        title = QLabel(self.step_title)
        title.setObjectName("stepTitle")
        layout.addWidget(title)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("输出格式:"))
        self._fmt = QComboBox()
        self._fmt.addItems(["mp4", "avi", "mov"])
        param_row.addWidget(self._fmt)
        param_row.addWidget(QLabel("帧率:"))
        self._out_fps = QSpinBox()
        self._out_fps.setRange(0, 120)
        self._out_fps.setValue(0)
        self._out_fps.setToolTip("0=自动检测(从源视频)")
        param_row.addWidget(self._out_fps)
        self._lossless = QCheckBox("无损输出")
        param_row.addWidget(self._lossless)
        param_row.addStretch()
        self._run_btn = QPushButton("开始")
        self._run_btn.setFixedWidth(100)
        self._run_btn.clicked.connect(self._on_run)
        param_row.addWidget(self._run_btn)
        self._stop_btn = QPushButton("停止")
        self._stop_btn.setProperty("busy_keep", True)
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #C42B1C; color: white; font-weight: bold; "
            "padding: 6px 14px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #D43B2C; }"
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
        self._stop_btn.setFixedWidth(70)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(lambda: setattr(self, '_stop_requested', True))
        param_row.addWidget(self._stop_btn)
        layout.addLayout(param_row)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; background: #1a1a1a; color: #cccccc;")
        layout.addWidget(self._log_text)

        self._log_signal = _OutputLogSignal()
        self._log_signal.log_signal.connect(self._on_log)
        self._log_signal.done_signal.connect(self._on_done)
        self._running = False
        self._stop_requested = False

    def _build_params(self):
        pass

    def _on_log(self, text, overwrite):
        if overwrite:
            cursor = self._log_text.textCursor()
            cursor.beginEditBlock()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.movePosition(cursor.MoveOperation.StartOfBlock, cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(text)
            cursor.endEditBlock()
        else:
            self._log_text.append(text)

    def _on_done(self):
        self._running = False
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._set_busy(False)

    def _on_run(self):
        if self._running:
            return
        from faceswap.business.video_output import VideoOutput, OutputFormat
        from faceswap.business.video_processor import VideoProcessor
        from faceswap.business.workspace_manager import WorkspaceManager

        self._running = True
        self._stop_requested = False
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._log_text.clear()
        self._set_busy(True)

        vp = VideoProcessor()
        vo = VideoOutput(vp)
        ws = WorkspaceManager()
        fmt = OutputFormat(self._fmt.currentText())
        ref = ws.find_dst_video()
        output_path = WORKSPACE_DIR / f"result.{fmt.value}"
        override_fps = self._out_fps.value() or None
        sig = self._log_signal

        def _stream_cb(line, overwrite=False):
            sig.log_signal.emit(line, overwrite)

        def _task():
            try:
                vo.merged_to_video(DATA_DST_MERGED_DIR, output_path, fmt,
                                   reference_video=ref, include_audio=True,
                                   lossless=self._lossless.isChecked(),
                                   override_fps=override_fps,
                                   stream_callback=_stream_cb,
                                   stop_check=lambda: self._stop_requested)
                if self._stop_requested:
                    sig.log_signal.emit("已停止", False)
                else:
                    sig.log_signal.emit("合成完毕", False)
            except Exception as e:
                sig.log_signal.emit(f"错误: {e}", False)
            finally:
                sig.done_signal.emit()

        self._thread = threading.Thread(target=_task, daemon=True)
        self._thread.start()

