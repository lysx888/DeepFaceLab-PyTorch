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


class _PreviewSignal(QObject):
    preview_ready = pyqtSignal(np.ndarray)


class AutoScaleLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._raw_pixmap = None

    def setRawPixmap(self, pixmap):
        self._raw_pixmap = pixmap
        self._update_scaled()

    def _update_scaled(self):
        if self._raw_pixmap is None:
            return
        lw = self.width()
        lh = self.height()
        if lw > 10 and lh > 10:
            scaled = self._raw_pixmap.scaled(
                QSize(lw, lh),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            super().setPixmap(scaled)
        else:
            super().setPixmap(self._raw_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled()


class _LogSignal(QObject):
    log_ready = pyqtSignal(str, bool)


class _ProgressSignal(QObject):
    progress_ready = pyqtSignal(str)
    done_ready = pyqtSignal(str, str)
    error_ready = pyqtSignal(str, str)


class _ClosePreviewSignal(QObject):
    close_ready = pyqtSignal()


class _Worker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            self._fn(*self._args, **self._kwargs)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


def exec_modal(widget, dlg):
    """打开模态对话框：期间全局禁用主窗口，关闭后恢复（不依赖 exec 模态行为）。

    用于所有打开独立对话框的按钮，保证一次只能运行一个操作。
    """
    win = widget.window()
    if hasattr(win, 'set_busy'):
        # 先临时禁用对话框整棵子树，使 set_busy 遍历时其控件 effective=False 被跳过；
        # 再恢复，各控件自身 enabled 状态保持原样（含对话框初始禁用的控件）。
        dlg.setEnabled(False)
        win.set_busy(True)
        dlg.setEnabled(True)
    try:
        return dlg.exec()
    finally:
        if hasattr(win, 'set_busy'):
            win.set_busy(False)


class StepPanel(QWidget):
    step_title = ""
    step_desc = ""
    show_run_buttons = True

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[_Worker] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_requested = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        title = QLabel(self.step_title)
        title.setObjectName("stepTitle")
        layout.addWidget(title)

        desc = QLabel(self.step_desc)
        desc.setObjectName("stepDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self._params_area = QVBoxLayout()
        self._params_area.setSpacing(6)
        layout.addLayout(self._params_area)

        self._build_params()

        if self.show_run_buttons:
            btn_row = QHBoxLayout()
            btn_row.addStretch()
            self._run_btn = QPushButton("开始")
            self._run_btn.setFixedWidth(120)
            self._run_btn.setStyleSheet(
                "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
            )
            self._run_btn.clicked.connect(self._on_run)
            btn_row.addWidget(self._run_btn)
            self._stop_btn = QPushButton("停止")
            self._stop_btn.setProperty("busy_keep", True)
            self._stop_btn.setStyleSheet(
                "QPushButton { background-color: #C42B1C; color: white; font-weight: bold; "
                "padding: 6px 16px; border-radius: 4px; font-size: 13px; }"
                "QPushButton:hover { background-color: #D43B2C; }"
                "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
            )
            self._stop_btn.setFixedWidth(80)
            self._stop_btn.setEnabled(False)
            self._stop_btn.clicked.connect(self._on_stop)
            btn_row.addWidget(self._stop_btn)
            layout.addLayout(btn_row)

            self._progress = QProgressBar()
            self._progress.setVisible(False)
            layout.addWidget(self._progress)

            self._progress_label = QLabel("")
            self._progress_label.setStyleSheet("font-size: 11px; color: #666666;")
            self._progress_label.setVisible(False)
            layout.addWidget(self._progress_label)

        layout.addStretch()

    def _build_params(self):
        pass

    def _add_param_row(self, label_text: str, widget: QWidget, tooltip: str = None) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(180)
        if tooltip:
            lbl.setToolTip(tooltip)
        row.addWidget(lbl)
        row.addWidget(widget)
        self._params_area.addLayout(row)
        return row

    def _add_combo(self, label: str, items: list, default: str = "") -> QComboBox:
        cb = QComboBox()
        cb.addItems(items)
        if default:
            idx = cb.findText(default)
            if idx >= 0:
                cb.setCurrentIndex(idx)
        self._add_param_row(label, cb)
        return cb

    def _add_spin(self, label: str, min_v: int, max_v: int, default: int, step: int = 1, tooltip: str = None) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(min_v, max_v)
        sb.setValue(default)
        sb.setSingleStep(step)
        if tooltip:
            sb.setToolTip(tooltip)
        self._add_param_row(label, sb, tooltip=tooltip)
        return sb

    def _add_dspin(self, label: str, min_v: float, max_v: float, default: float, step: float = 0.01, decimals: int = 4) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(min_v, max_v)
        sb.setValue(default)
        sb.setSingleStep(step)
        sb.setDecimals(decimals)
        self._add_param_row(label, sb)
        return sb

    def _add_check(self, label: str, default: bool = False) -> QCheckBox:
        cb = QCheckBox(label)
        cb.setChecked(default)
        self._params_area.addWidget(cb)
        return cb

    def _add_path_input(self, label: str, default: str = "", is_dir: bool = False) -> tuple[QLineEdit, QPushButton]:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(180)
        le = QLineEdit(default)
        btn = QPushButton("浏览")
        btn.setFixedWidth(80)
        btn.setProperty("outline", True)

        def _browse():
            if is_dir:
                p = QFileDialog.getExistingDirectory(self, label, le.text())
            else:
                p, _ = QFileDialog.getOpenFileName(self, label, le.text())
            if p:
                le.setText(p)

        btn.clicked.connect(_browse)
        row.addWidget(lbl)
        row.addWidget(le, 1)
        row.addWidget(btn)
        self._params_area.addLayout(row)
        return le, btn

    def _set_running(self, running: bool):
        self._running = running
        # run 按钮的禁用由全局 set_busy 统一管理（:disabled 样式负责变灰）
        self._stop_btn.setEnabled(running)
        self._progress.setVisible(running)
        self._progress_label.setVisible(running)
        if not running:
            self._stop_requested = False
            self._progress_label.setText("")

    def showEvent(self, event):
        super().showEvent(event)
        try:
            self.on_panel_shown()
        except Exception as e:  # 刷新失败不影响界面显示
            _logger.error(f"on_panel_shown failed: {e}")

    def on_panel_shown(self):
        """面板变为可见时调用（界面切换/首次显示/窗口还原），子类可覆盖。"""

    def _set_busy(self, busy: bool):
        win = self.window()
        if hasattr(win, 'set_busy'):
            win.set_busy(busy)

    def _exec_modal(self, dlg):
        return exec_modal(self, dlg)

    def _on_run(self):
        pass

    def _on_stop(self):
        self._stop_requested = True
        self._stop_btn.setEnabled(False)
        self._progress_label.setText("正在停止...")

    def _request_training_stop(self, trainer, source: str, log_signal=None):
        if trainer is None:
            return
        already_stopped = getattr(trainer, '_stop_requested', False)
        if not already_stopped and hasattr(trainer, '_stop_event'):
            already_stopped = trainer._stop_event.is_set()
        if already_stopped:
            return
        trainer.request_stop()
        msg = f"[{source}] 正在保存并停止训练..."
        if log_signal is not None:
            log_signal.log_ready.emit(msg, False)
        else:
            self._append_log(msg, False)
        self._log_need_newline = False

    def _update_progress(self, current: int, total: int, text: str = ""):
        if total > 0:
            self._progress.setMaximum(total)
            self._progress.setValue(current)
        if text:
            self._progress_label.setText(text)

    def _run_in_thread(self, fn, *args, **kwargs):
        if self._running:
            return
        self._set_running(True)
        self._set_busy(True)
        self._stop_requested = False

        def _wrapped():
            try:
                fn(*args, **kwargs)
            except Exception as e:
                _logger.error(str(e))

        self._worker = _Worker(_wrapped)
        self._worker.finished.connect(lambda: (self._set_running(False), self._set_busy(False)))
        self._worker.error.connect(lambda e: _logger.error(str(e)))
        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()


class _StreamBridge(QObject):
    stream_signal = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)

    def emit_stream(self, line: str, overwrite: bool = False):
        self.stream_signal.emit(line, overwrite)
