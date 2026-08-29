import json
import shutil
import sys
import threading
import subprocess
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
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
    DATA_SRC_DIR, DATA_DST_DIR,
    DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR,
    DATA_DST_SWAPPED_DIR, DATA_DST_MERGED_DIR, DATA_DST_MERGED_MASK_DIR,
    IF_LANDMARK_MODEL_DIR, INSIGHTFACE_MANUAL_ANNOTATED_DIR,
    SCRFD_MODEL_DIR,
    FaceType,
)


class _PreviewSignal(QObject):
    preview_ready = pyqtSignal(np.ndarray)


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
            self._run_btn.clicked.connect(self._on_run)
            btn_row.addWidget(self._run_btn)
            self._stop_btn = QPushButton("停止")
            self._stop_btn.setProperty("danger", True)
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
        self._run_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._progress.setVisible(running)
        self._progress_label.setVisible(running)
        if not running:
            self._stop_requested = False
            self._progress_label.setText("")

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
        self._set_running(True)
        self._stop_requested = False

        def _wrapped():
            try:
                fn(*args, **kwargs)
            except Exception as e:
                _logger.error(str(e))

        self._worker = _Worker(_wrapped)
        self._worker.finished.connect(lambda: self._set_running(False))
        self._worker.error.connect(lambda e: _logger.error(str(e)))
        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()


class _StreamBridge(QObject):
    stream_signal = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)

    def emit_stream(self, line: str, overwrite: bool = False):
        self.stream_signal.emit(line, overwrite)


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

        self._run_btn = QPushButton("开始提取")
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
                if self._is_src:
                    vp.extract_frames_src(video, DATA_SRC_DIR, fps=fps,
                                          output_format="png",
                                          stream_callback=self._stream_callback)
                else:
                    vp.extract_frames_dst(video, DATA_DST_DIR, fps=fps,
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
    _sig_btn_enabled = pyqtSignal(bool)

    def __init__(self, label: str, video_default: str, stream_callback=None, parent=None):
        super().__init__(parent)
        self._stream_callback = stream_callback

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
        self._run_btn.clicked.connect(self._on_run)
        r2.addWidget(lbl2)
        r2.addWidget(self._cut_start)
        r2.addWidget(lbl3)
        r2.addWidget(self._cut_end)
        r2.addStretch()
        r2.addWidget(self._run_btn)
        grp_layout.addLayout(r2)

        layout.addWidget(grp)

    def _browse_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择视频文件", self._video_path.text())
        if p:
            self._video_path.setText(p)

    def _on_run(self):
        from faceswap.business.video_processor import VideoProcessor
        vp = VideoProcessor()
        video = Path(self._video_path.text())
        start = self._cut_start.text().strip()
        end = self._cut_end.text().strip()
        output = video.with_stem(video.stem + "_cut")

        self._run_btn.setEnabled(False)

        def _task():
            try:
                vp.cut_video(video, output, start, end, stream_callback=self._stream_callback)
                if self._stream_callback:
                    self._stream_callback("========== 视频切割完成 ==========")
            except Exception as e:
                _logger.error(str(e))
            finally:
                self._sig_btn_enabled.emit(True)

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


class _FaceExtractHalf(QWidget):
    _sig_running = pyqtSignal(bool)

    def __init__(self, label: str, is_src: bool, get_config=None, progress_callback=None,
                 check_aligned=None, del_face_cb=None, analyze_cb=None, parent=None):
        super().__init__(parent)
        self._is_src = is_src
        self._get_config = get_config
        self._progress_callback = progress_callback
        self._check_aligned = check_aligned
        self._sig_running.connect(self._apply_running)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        grp = QGroupBox(label)
        grp_layout = QVBoxLayout(grp)
        grp_layout.setSpacing(4)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("开始提取")
        self._run_btn.clicked.connect(self._on_run)
        self._preview_btn = QPushButton("预览生成")
        self._preview_btn.setStyleSheet("QPushButton { background-color: #D45500; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
        self._preview_btn.clicked.connect(self._on_preview)
        btn_row.addWidget(self._run_btn, 3)
        btn_row.addWidget(self._preview_btn, 3)
        if del_face_cb is not None:
            del_btn = QPushButton("删除头像")
            del_btn.setStyleSheet("QPushButton { background-color: #D45500; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
            del_btn.clicked.connect(del_face_cb)
            btn_row.addWidget(del_btn, 2)
        if analyze_cb is not None:
            analyze_btn = QPushButton("分析头像")
            analyze_btn.setStyleSheet("QPushButton { background-color: #5B2D8E; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
            analyze_btn.clicked.connect(analyze_cb)
            btn_row.addWidget(analyze_btn, 2)
        grp_layout.addLayout(btn_row)

        layout.addWidget(grp)

    def _apply_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._run_btn.setText("提取中..." if running else "开始提取")

    def _on_preview(self):
        from faceswap.gui_app.manual_annotator import DebugPreviewDialog
        dlg = DebugPreviewDialog(self._is_src, self)
        dlg.exec()

    def _on_run(self):
        if self._check_aligned:
            if not self._check_aligned(self._is_src):
                return

        from faceswap.business.face_extractor import FaceExtractor

        config, gpu_ids = self._get_config()
        extractor = FaceExtractor(gpu_ids=gpu_ids)

        self._sig_running.emit(True)

        def _task():
            try:
                if self._is_src:
                    extractor.extract_src_faces(DATA_SRC_DIR, DATA_SRC_ALIGNED_DIR, config,
                                                progress_callback=self._progress_callback)
                else:
                    extractor.extract_dst_faces(DATA_DST_DIR, DATA_DST_ALIGNED_DIR, config,
                                                progress_callback=self._progress_callback)
            except Exception as e:
                _logger.error(str(e))
            finally:
                self._sig_running.emit(False)

        threading.Thread(target=_task, daemon=True).start()


class _ProgressArea(QWidget):
    _sig_progress = pyqtSignal(int, int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sig_progress.connect(self._apply_progress)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        self._progress_bar = QProgressBar()
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("font-size: 11px; color: #666666;")
        layout.addWidget(self._progress_label)

    def _apply_progress(self, current: int, total: int, text: str):
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._progress_label.setText(text)

    def on_progress(self, current: int, total: int, elapsed: float, remaining: float, speed: str):
        def _fmt_time(seconds: float) -> str:
            m = int(seconds) // 60
            s = int(seconds) % 60
            return f"{m}:{s:02d}"
        elapsed_str = _fmt_time(elapsed)
        eta_str = _fmt_time(remaining) if remaining > 0 else "--:--"
        text = f"{current}/{total}  [{elapsed_str}<{eta_str}, {speed}]"
        self._sig_progress.emit(current, total, text)


_FACE_TYPE_DEFAULT_SIZE = {
    "whole_face": "512",
    "head": "768",
}


class Step2FaceExtract(StepPanel):
    step_title = "2. 人脸提取"
    step_desc = "从提取的帧画面中检测并对齐人脸。"
    show_run_buttons = False

    def __init__(self, parent=None):
        super().__init__(parent)

    def _build_params(self):
        self._sort_progress_sig = _ProgressSignal()
        self._sort_progress_sig.done_ready.connect(lambda t, m: QMessageBox.information(self, t, m))
        self._sort_progress_sig.error_ready.connect(lambda t, m: QMessageBox.warning(self, t, m))
        import torch
        self._gpu_items = ["CPU"]
        self._num_gpus = 0
        if torch.cuda.is_available():
            self._num_gpus = torch.cuda.device_count()
            self._gpu_items += [f"[CUDA:{i}] {torch.cuda.get_device_name(i)}" for i in range(self._num_gpus)]
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            xpu_count = torch.xpu.device_count()
            self._gpu_items += [f"[XPU:{i}] {torch.xpu.get_device_name(i)}" for i in range(xpu_count)]
            self._num_gpus += xpu_count
        gpu_row = QHBoxLayout()
        lbl_gpu = QLabel("GPU:")
        lbl_gpu.setFixedWidth(70)
        self._gpu = QComboBox()
        self._gpu.addItems(self._gpu_items)
        if len(self._gpu_items) > 1:
            self._gpu.setCurrentIndex(1)
        gpu_row.addWidget(lbl_gpu)
        gpu_row.addWidget(self._gpu, 1)
        self._params_area.addLayout(gpu_row)
        self._multi_gpu = QCheckBox("多GPU并行（使用所有可用GPU）")
        self._multi_gpu.setChecked(False)
        if self._num_gpus < 2:
            self._multi_gpu.setEnabled(False)
            self._multi_gpu.setToolTip("需要2个以上GPU才能启用")
        self._multi_gpu.setVisible(False)
        self._params_area.addWidget(self._multi_gpu)

        row1 = QHBoxLayout()
        lbl1 = QLabel("人脸类型:")
        lbl1.setFixedWidth(70)
        self._face_type = QComboBox()
        self._face_type.addItems(["whole_face", "head"])
        self._face_type.setCurrentText("whole_face")
        self._face_type.currentTextChanged.connect(self._on_face_type_changed)
        self._face_type.setFixedWidth(120)
        row1.addWidget(lbl1)
        row1.addWidget(self._face_type)
        row1.addSpacing(20)
        lbl2 = QLabel("最大人脸数:")
        lbl2.setFixedWidth(80)
        self._max_faces = QSpinBox()
        self._max_faces.setRange(0, 100)
        self._max_faces.setValue(0)
        self._max_faces.setToolTip("0=不限")
        self._max_faces.setFixedWidth(80)
        row1.addWidget(lbl2)
        row1.addWidget(self._max_faces)
        row1.addSpacing(20)
        lbl3 = QLabel("检测阈值:")
        lbl3.setFixedWidth(70)
        self._det_thresh = QDoubleSpinBox()
        self._det_thresh.setRange(0.1, 1.0)
        self._det_thresh.setValue(0.5)
        self._det_thresh.setSingleStep(0.05)
        self._det_thresh.setDecimals(2)
        self._det_thresh.setFixedWidth(80)
        row1.addWidget(lbl3)
        row1.addWidget(self._det_thresh)
        row1.addStretch()
        self._params_area.addLayout(row1)

        row2 = QHBoxLayout()
        lbl4 = QLabel("图像尺寸:")
        lbl4.setFixedWidth(70)
        self._output_size = QComboBox()
        self._output_size.addItems(["512", "768", "384", "256", "640", "1024", "128", "896"])
        self._output_size.setCurrentText("512")
        self._output_size.setFixedWidth(120)
        row2.addWidget(lbl4)
        row2.addWidget(self._output_size)
        row2.addSpacing(20)
        lbl5 = QLabel("输出格式:")
        lbl5.setFixedWidth(80)
        self._output_format = QComboBox()
        self._output_format.addItems(["jpg", "png"])
        self._output_format.setCurrentText("jpg")
        self._output_format.setFixedWidth(80)
        row2.addWidget(lbl5)
        row2.addWidget(self._output_format)
        row2.addSpacing(20)
        lbl6 = QLabel("JPEG质量:")
        lbl6.setFixedWidth(70)
        self._jpg_quality = QSpinBox()
        self._jpg_quality.setRange(1, 100)
        self._jpg_quality.setValue(100)
        self._jpg_quality.setFixedWidth(80)
        row2.addWidget(lbl6)
        row2.addWidget(self._jpg_quality)
        row2.addStretch()
        self._params_area.addLayout(row2)

        self._debug = self._add_check("输出调试图像到 aligned_debug", True)

        self._progress = _ProgressArea()

        row = QHBoxLayout()
        self._src_half = _FaceExtractHalf(
            "源人脸 (SRC)", is_src=True, get_config=self._get_config,
            progress_callback=self._progress.on_progress,
            check_aligned=self._check_aligned_dir,
            del_face_cb=lambda: self._on_delete_faces(True),
            analyze_cb=lambda: self._on_analyze_faces(True))
        self._dst_half = _FaceExtractHalf(
            "目标人脸 (DST)", is_src=False, get_config=self._get_config,
            progress_callback=self._progress.on_progress,
            check_aligned=self._check_aligned_dir,
            del_face_cb=lambda: self._on_delete_faces(False),
            analyze_cb=lambda: self._on_analyze_faces(False))
        row.addWidget(self._src_half)
        row.addWidget(self._dst_half)
        self._params_area.addLayout(row)

        self._params_area.addWidget(self._progress)

        src_tool_grp = QGroupBox("源目标 (SRC) 工具栏")
        src_tool_lay = QHBoxLayout(src_tool_grp)
        src_tool_lay.setContentsMargins(8, 14, 8, 4)
        src_tool_lay.setSpacing(8)
        dedup_btn = QPushButton("去重过滤")
        dedup_btn.setStyleSheet("QPushButton { background-color: #5B2D8E; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
        dedup_btn.clicked.connect(self._on_dedup)
        src_tool_lay.addWidget(dedup_btn)
        rename_btn = QPushButton("批量重命名")
        rename_btn.setMinimumWidth(100)
        rename_btn.clicked.connect(self._on_rename)
        src_tool_lay.addWidget(rename_btn)
        src_tool_lay.addStretch()
        rename_desc = QLabel("去重: 3DDFA 状态去冗余  |  重命名: 连续编号")
        rename_desc.setStyleSheet("color: #666; font-size: 11px;")
        src_tool_lay.addWidget(rename_desc)
        self._params_area.addWidget(src_tool_grp)

        tool_grp = QGroupBox("标注与训练")
        tool_lay = QHBoxLayout(tool_grp)
        tool_lay.setContentsMargins(8, 14, 8, 4)
        tool_lay.setSpacing(8)
        manual_btn = QPushButton("手动标注")
        manual_btn.setStyleSheet("QPushButton { background-color: #5B2D8E; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
        manual_btn.clicked.connect(self._on_manual_annotate)
        tool_lay.addWidget(manual_btn)
        xseg_train_btn = QPushButton("IF训练")
        xseg_train_btn.setStyleSheet("QPushButton { background-color: #D45500; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
        xseg_train_btn.clicked.connect(self._on_goto_if_train)
        tool_lay.addWidget(xseg_train_btn)
        tool_lay.addStretch()
        tool_desc = QLabel("手动标注 XSeg 遮罩 / 跳转到 IF (InsightFace) 训练页")
        tool_desc.setStyleSheet("color: #666; font-size: 11px;")
        tool_lay.addWidget(tool_desc)
        self._params_area.addWidget(tool_grp)

        sort_grp = QGroupBox("排序筛选")
        sort_outer = QVBoxLayout(sort_grp)
        sort_outer.setContentsMargins(8, 14, 8, 4)
        sort_outer.setSpacing(4)

        target_row = QHBoxLayout()
        target_row.setSpacing(8)
        self._sort_src_rb = QRadioButton("源人脸 (SRC)")
        self._sort_src_rb.setChecked(True)
        self._sort_dst_rb = QRadioButton("目标人脸 (DST)")
        sort_run_btn = QPushButton("排序")
        sort_run_btn.setMinimumWidth(60)
        sort_run_btn.clicked.connect(self._on_sort)
        target_row.addWidget(self._sort_src_rb)
        target_row.addWidget(self._sort_dst_rb)
        target_row.addStretch()
        target_row.addWidget(sort_run_btn)
        sort_outer.addLayout(target_row)

        algo_row = QHBoxLayout()
        algo_row.setSpacing(2)
        self._sort_algo_btns = []
        _ALGO_BTN_STYLE = (
            "QPushButton { background-color: #FFFFFF; border: 1px solid #0078D4; border-radius: 3px; padding: 4px 8px; font-size: 11px; color: #0078D4; }"
            "QPushButton:hover { background-color: #E8F0FE; }"
            "QPushButton:checked { background-color: #0078D4; color: white; border: 1px solid #0078D4; }"
        )
        for algo in ["blur", "hist", "yaw", "pitch", "brightness", "hue", "oneface", "final"]:
            btn = QPushButton(algo)
            btn.setCheckable(True)
            btn.setStyleSheet(_ALGO_BTN_STYLE)
            btn.clicked.connect(lambda checked, a=algo: self._set_sort_algo(a))
            self._sort_algo_btns.append((algo, btn))
            algo_row.addWidget(btn, 1)
        self._sort_algo_btns[0][1].setChecked(True)
        algo_row.addStretch()
        sort_outer.addLayout(algo_row)

        self._sort_desc = QLabel()
        self._sort_desc.setStyleSheet("color: #555; font-size: 13px; padding-left: 4px;")
        self._sort_desc.setWordWrap(True)
        sort_outer.addWidget(self._sort_desc)
        self._on_sort_algo_changed("blur")

        self._sort_progress = QLabel()
        self._sort_progress.setStyleSheet("color: #0078D4; font-size: 12px; padding-left: 4px;")
        sort_outer.addWidget(self._sort_progress)
        self._sort_progress_sig.progress_ready.connect(self._sort_progress.setText)

        self._params_area.addWidget(sort_grp)

    def _on_face_type_changed(self, text: str):
        default_size = _FACE_TYPE_DEFAULT_SIZE.get(text, "512")
        idx = self._output_size.findText(default_size)
        if idx >= 0:
            self._output_size.setCurrentIndex(idx)
        else:
            self._output_size.setCurrentText(default_size)

    def _on_manual_annotate(self):
        from faceswap.gui_app.manual_annotator import ManualAnnotatorDialog
        dlg = ManualAnnotatorDialog(self)
        dlg.exec()

    def _on_goto_if_train(self):
        w = self.window()
        if hasattr(w, '_switch'):
            w._switch(3)

    def _on_dedup(self):
        from faceswap.gui_app.dedup_dialog import SrcDedupDialog
        dlg = SrcDedupDialog(DATA_SRC_ALIGNED_DIR, parent=self)
        dlg.exec()

    def _set_sort_algo(self, algo: str):
        for a, btn in self._sort_algo_btns:
            btn.setChecked(a == algo)
        self._on_sort_algo_changed(algo)

    _SORT_DESCRIPTIONS = {
        "blur": "按模糊度排序：将模糊的图像排到后面，方便删除低质量人脸",
        "hist": "按直方图相似度排序：将相似的人脸排在一起，方便批量筛选",
        "yaw": "按偏航角排序：按人脸左右旋转角度排列",
        "pitch": "按俯仰角排序：按人脸上下旋转角度排列",
        "brightness": "按亮度排序：将过暗或过亮的人脸排到后面",
        "hue": "按色调排序：按肤色色调分组排列",
        "oneface": "单人筛选：只保留每张图中最大的人脸，删除多余人脸",
        "final": "最终排序：综合多种因素进行最终排序筛选",
    }

    def _on_sort_algo_changed(self, text: str):
        desc = self._SORT_DESCRIPTIONS.get(text, "")
        self._sort_desc.setText(desc)

    def _on_delete_faces(self, is_src: bool):
        from faceswap.gui_app.face_delete_dialog import FaceDeleteDialog
        from faceswap.shared.file_manager import FileManager
        target_dir = DATA_SRC_ALIGNED_DIR if is_src else DATA_DST_ALIGNED_DIR
        if not target_dir.exists() or not any(FileManager.find_images(target_dir)):
            QMessageBox.information(self, "提示", f"目录中没有图片:\n{target_dir}")
            return
        dlg = FaceDeleteDialog(target_dir, parent=self)
        dlg.exec()

    def _on_analyze_faces(self, is_src: bool):
        from faceswap.gui_app.dedup_dialog import StateAnalysisDialog
        target_dir = DATA_SRC_ALIGNED_DIR if is_src else DATA_DST_ALIGNED_DIR
        if not target_dir.exists():
            QMessageBox.information(self, "提示", f"目录不存在:\n{target_dir}")
            return
        dlg = StateAnalysisDialog(target_dir, yaw_grid=5.0, pitch_grid=5.0, parent=self)
        dlg.exec()

    def _on_sort(self):
        from faceswap.business.face_sorter import FaceSorter, SortAlgorithm

        target_dir = DATA_SRC_ALIGNED_DIR if self._sort_src_rb.isChecked() else DATA_DST_ALIGNED_DIR
        algo_name = "blur"
        for a, btn in self._sort_algo_btns:
            if btn.isChecked():
                algo_name = a
                break
        algo = SortAlgorithm(algo_name)
        sorter = FaceSorter()
        sig = self._sort_progress_sig
        self._sort_progress.setText("Sorting: 0%")
        self._set_sort_buttons_enabled(False)

        def _progress(current, total, elapsed, remaining=0, speed=""):
            if total > 0:
                pct = int(current / total * 100)
                m_e = int(elapsed) // 60
                s_e = int(elapsed) % 60
                m_r = int(remaining) // 60 if remaining > 0 else 0
                s_r = int(remaining) % 60 if remaining > 0 else 0
                sig.progress_ready.emit(f"Sorting: {pct}% | {current}/{total} [{m_e}:{s_e:02d}<{m_r}:{s_r:02d}, {speed}]")

        def _task():
            try:
                count = sorter.sort_aligned(target_dir, algo, progress_callback=_progress)
                sig.done_ready.emit("完成", f"排序完成，共处理 {count} 张人脸")
            except Exception as e:
                sig.error_ready.emit("错误", str(e))
            finally:
                sig.progress_ready.emit("")
                self._set_sort_buttons_enabled(True)

        threading.Thread(target=_task, daemon=True).start()

    def _set_sort_buttons_enabled(self, enabled: bool):
        for _, btn in self._sort_algo_btns:
            btn.setEnabled(enabled)

    def _on_rename(self):
        from faceswap.shared.file_manager import FileManager
        target_dir = DATA_SRC_ALIGNED_DIR
        if not target_dir.exists():
            QMessageBox.warning(self, "错误", f"目录不存在: {target_dir}")
            return

        all_paths = FileManager.find_images(target_dir)
        if not all_paths:
            QMessageBox.information(self, "提示", "目录中没有图像文件")
            return

        all_paths.sort(key=lambda p: p.name)
        count = len(all_paths)
        sig = self._sort_progress_sig

        def _task():
            import os
            try:
                temp_dir = target_dir / "_rename_tmp"
                temp_dir.mkdir(exist_ok=True)
                for i, img_path in enumerate(all_paths):
                    ext = img_path.suffix
                    new_name = f"{i + 1:05d}_0{ext}"
                    os.rename(str(img_path), str(temp_dir / new_name))
                    meta_path = img_path.with_suffix(".json")
                    if meta_path.exists():
                        os.rename(str(meta_path), str(temp_dir / f"{i + 1:05d}_0.json"))
                for f in temp_dir.iterdir():
                    os.rename(str(f), str(target_dir / f.name))
                temp_dir.rmdir()
                sig.done_ready.emit("完成", f"已重命名 {count} 个源人脸文件")
            except Exception as e:
                sig.error_ready.emit("重命名错误", str(e))

        threading.Thread(target=_task, daemon=True).start()

    def _check_aligned_dir(self, is_src: bool) -> bool:
        aligned_dir = DATA_SRC_ALIGNED_DIR if is_src else DATA_DST_ALIGNED_DIR
        if aligned_dir.exists():
            from faceswap.shared.file_manager import FileManager
            files = FileManager.find_images(aligned_dir)
            if files:
                tag = "源" if is_src else "目标"
                reply = QMessageBox.warning(
                    self, "目录非空",
                    f"{tag}人脸目录 ({aligned_dir}) 中已有 {len(files)} 个文件。\n"
                    f"继续提取将覆盖已有文件，是否继续？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return False
        return True

    def _get_config(self):
        from faceswap.business.face_extractor import ExtractConfig

        gpu_ids = []
        if self._multi_gpu.isChecked():
            import torch
            if torch.cuda.is_available():
                gpu_ids = list(range(torch.cuda.device_count()))
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                gpu_ids += list(range(torch.xpu.device_count()))
        else:
            gpu_text = self._gpu.currentText()
            if gpu_text.startswith("[CUDA:") or gpu_text.startswith("[XPU:"):
                try:
                    bracket_end = gpu_text.index("]")
                    prefix_end = gpu_text.index(":") + 1
                    gpu_ids = [int(gpu_text[prefix_end:bracket_end])]
                except (ValueError, IndexError):
                    gpu_ids = [-1]
            else:
                gpu_ids = [-1]

        ft_map = {"whole_face": FaceType.WHOLE_FACE, "head": FaceType.HEAD}
        ft = ft_map.get(self._face_type.currentText(), FaceType.WHOLE_FACE)

        config = ExtractConfig(
            face_type=ft,
            max_faces=self._max_faces.value(),
            det_thresh=self._det_thresh.value(),
            output_size=int(self._output_size.currentText()),
            jpg_quality=self._jpg_quality.value(),
            output_format=self._output_format.currentText(),
            debug_output=self._debug.isChecked(),
        )
        return config, gpu_ids


class Step3XSeg(StepPanel):
    step_title = "3. 遮罩 XSeg"
    step_desc = "编辑、训练和应用XSeg人脸遮罩。"
    show_run_buttons = False

    def _build_params(self):
        grp_edit = QGroupBox("遮罩编辑")
        edit_lay = QVBoxLayout(grp_edit)
        edit_lay.setSpacing(4)

        target_row = QHBoxLayout()
        self._edit_src_btn = QRadioButton("源人脸（data_src）")
        self._edit_src_btn.setChecked(True)
        self._edit_dst_btn = QRadioButton("目标人脸（data_dst）")
        target_row.addWidget(self._edit_src_btn)
        target_row.addWidget(self._edit_dst_btn)
        target_row.addStretch()
        edit_lay.addLayout(target_row)

        edit_btn_row = QHBoxLayout()
        edit_btn = QPushButton("打开遮罩编辑器")
        edit_btn.setStyleSheet(
            "QPushButton { background-color: #5B2D8E; color: white; font-weight: bold; padding: 6px 20px; border-radius: 4px; }"
        )
        edit_btn.clicked.connect(self._on_edit)
        fetch_btn = QPushButton("提取已标注")
        fetch_btn.setProperty("outline", True)
        fetch_btn.clicked.connect(self._on_fetch)
        remove_btn = QPushButton("清除标注")
        remove_btn.setProperty("danger", True)
        remove_btn.clicked.connect(self._on_remove_annotations)
        edit_btn_row.addWidget(edit_btn, 1)
        edit_btn_row.addWidget(fetch_btn, 1)
        edit_btn_row.addWidget(remove_btn, 1)
        edit_lay.addLayout(edit_btn_row)

        self._params_area.addWidget(grp_edit)

        grp_train = QGroupBox("遮罩训练")
        train_lay = QVBoxLayout(grp_train)
        train_lay.setSpacing(4)

        train_row = QHBoxLayout()
        lbl_ft = QLabel("脸部类型:")
        self._xseg_face_type = QComboBox()
        self._xseg_face_type.addItems(["wf", "head"])
        self._xseg_face_type.setCurrentText("wf")
        self._xseg_face_type.setFixedWidth(70)
        _tip_ft = "人脸裁切类型：wf=全脸(256×256)，head=含额头头发(384×384)。应与提取时一致"
        lbl_ft.setToolTip(_tip_ft)
        self._xseg_face_type.setToolTip(_tip_ft)
        lbl3 = QLabel("批次:")
        self._xseg_batch = QSpinBox()
        self._xseg_batch.setRange(1, 64)
        self._xseg_batch.setValue(4)
        self._xseg_batch.setFixedWidth(50)
        _tip_bs = "每步训练的样本数。越大训练越稳定但显存占用越高。RTX 4090建议4-8"
        lbl3.setToolTip(_tip_bs)
        self._xseg_batch.setToolTip(_tip_bs)
        lbl4 = QLabel("迭代:")
        self._xseg_iters = QSpinBox()
        self._xseg_iters.setRange(1000, 10000000)
        self._xseg_iters.setValue(100000)
        self._xseg_iters.setSingleStep(10000)
        self._xseg_iters.setFixedWidth(90)
        _tip_it = "总训练迭代次数。遮罩训练通常5-10万次即可收敛"
        lbl4.setToolTip(_tip_it)
        self._xseg_iters.setToolTip(_tip_it)
        lbl5 = QLabel("混合精度:")
        self._xseg_amp = QComboBox()
        self._xseg_amp.addItems(["fp32", "fp16", "bf16"])
        self._xseg_amp.setCurrentText("fp16")
        self._xseg_amp.setFixedWidth(75)
        _tip_amp = "混合精度训练模式:\nfp16: 半精度+GradScaler, 省显存加速(推荐)\nbf16: BF16半精度, 精度不足易导致梯度爆炸, 不推荐\nfp32: 全精度, 最稳定但最慢"
        lbl5.setToolTip(_tip_amp)
        self._xseg_amp.setToolTip(_tip_amp)
        lbl_lr = QLabel("学习率:")
        self._xseg_lr = QDoubleSpinBox()
        self._xseg_lr.setRange(1e-6, 1e-2)
        self._xseg_lr.setValue(1e-4)
        self._xseg_lr.setSingleStep(1e-5)
        self._xseg_lr.setDecimals(6)
        self._xseg_lr.setFixedWidth(90)
        _tip_lr = "优化器学习率。DFL默认1e-4。值太大训练不稳定，太小收敛慢。通常1e-4~5e-5"
        lbl_lr.setToolTip(_tip_lr)
        self._xseg_lr.setToolTip(_tip_lr)
        lbl6 = QLabel("预训练:")
        self._xseg_pretrain = QCheckBox()
        self._xseg_pretrain.setChecked(False)
        _tip_pt = "先启用DSSIM+MSE灰度图自重建预训练(skip=zeros_like)，让模型学会人脸结构后再用BCE微调mask分割。建议首次训练开启"
        lbl6.setToolTip(_tip_pt)
        self._xseg_pretrain.setToolTip(_tip_pt)
        lbl7 = QLabel("lr_dropout:")
        self._xseg_lr_dropout = QDoubleSpinBox()
        self._xseg_lr_dropout.setRange(0.0, 1.0)
        self._xseg_lr_dropout.setValue(0.3)
        self._xseg_lr_dropout.setSingleStep(0.1)
        self._xseg_lr_dropout.setFixedWidth(65)
        _tip_ld = "学习率随机丢弃率。每步以该概率将梯度置零，相当于随机正则化。0=关闭，0.3=DFL默认。训练后期开启可获更锐利结果"
        lbl7.setToolTip(_tip_ld)
        self._xseg_lr_dropout.setToolTip(_tip_ld)
        train_row.addWidget(lbl_ft)
        train_row.addWidget(self._xseg_face_type)
        train_row.addSpacing(10)
        train_row.addWidget(lbl3)
        train_row.addWidget(self._xseg_batch)
        train_row.addSpacing(10)
        train_row.addWidget(lbl4)
        train_row.addWidget(self._xseg_iters)
        train_row.addSpacing(10)
        train_row.addWidget(lbl5)
        train_row.addWidget(self._xseg_amp)
        train_row.addSpacing(10)
        train_row.addWidget(lbl_lr)
        train_row.addWidget(self._xseg_lr)
        train_row.addSpacing(10)
        train_row.addWidget(lbl7)
        train_row.addWidget(self._xseg_lr_dropout)
        train_row.addSpacing(10)
        train_row.addWidget(lbl6)
        train_row.addWidget(self._xseg_pretrain)
        train_row.addStretch()
        train_lay.addLayout(train_row)

        for w in [self._xseg_face_type, self._xseg_batch, self._xseg_iters,
                  self._xseg_amp, self._xseg_lr, self._xseg_lr_dropout]:
            install_no_wheel(w)

        train_btn_row = QHBoxLayout()
        self._train_btn = QPushButton("开始训练")
        self._train_btn.setStyleSheet(
            "QPushButton { background-color: #D45500; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background-color: #E06010; }"
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
        self._train_btn.clicked.connect(self._on_train)
        self._stop_train_btn = QPushButton("停止训练")
        self._stop_train_btn.setProperty("danger", True)
        self._stop_train_btn.setEnabled(False)
        self._stop_train_btn.clicked.connect(self._on_stop_train)
        train_btn_row.addStretch()
        train_btn_row.addWidget(self._train_btn)
        train_btn_row.addWidget(self._stop_train_btn)
        train_lay.addLayout(train_btn_row)

        self._train_log = QTextEdit()
        self._train_log.setReadOnly(True)
        self._train_log.setMinimumHeight(100)
        self._train_log.setMaximumHeight(200)
        self._train_log.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; background-color: #1e1e1e; color: #d4d4d4;")
        train_lay.addWidget(self._train_log)

        self._params_area.addWidget(grp_train)

        grp_apply = QGroupBox("遮罩应用")
        apply_lay = QVBoxLayout(grp_apply)
        apply_lay.setSpacing(4)

        apply_target_row = QHBoxLayout()
        self._apply_src_btn = QRadioButton("源人脸（data_src）")
        self._apply_src_btn.setChecked(True)
        self._apply_dst_btn = QRadioButton("目标人脸（data_dst）")
        apply_target_row.addWidget(self._apply_src_btn)
        apply_target_row.addWidget(self._apply_dst_btn)
        apply_target_row.addStretch()
        apply_lay.addLayout(apply_target_row)

        apply_row = QHBoxLayout()
        apply_trained_btn = QPushButton("应用训练遮罩")
        apply_trained_btn.clicked.connect(self._on_apply_trained)
        remove_trained_btn = QPushButton("移除训练遮罩")
        remove_trained_btn.setProperty("danger", True)
        remove_trained_btn.clicked.connect(self._on_remove_trained)
        apply_generic_btn = QPushButton("应用通用遮罩")
        apply_generic_btn.setProperty("outline", True)
        apply_generic_btn.clicked.connect(self._on_apply_generic)
        apply_row.addWidget(apply_trained_btn)
        apply_row.addWidget(remove_trained_btn)
        apply_row.addWidget(apply_generic_btn)
        apply_lay.addLayout(apply_row)

        self._apply_status = QLabel("")
        self._apply_status.setStyleSheet("color: #888; font-size: 11px; font-family: Consolas, monospace;")
        apply_lay.addWidget(self._apply_status)

        self._params_area.addWidget(grp_apply)

        self._xseg_trainer = None
        self._preview_win = None
        self._preview_signal = _PreviewSignal()
        self._preview_signal.preview_ready.connect(self._show_preview)
        self._log_signal = _LogSignal()
        self._log_signal.log_ready.connect(self._append_log)
        self._progress_signal = _ProgressSignal()
        self._progress_signal.progress_ready.connect(self._apply_status.setText)
        self._progress_signal.done_ready.connect(lambda t, m: QMessageBox.information(self, t, m))
        self._progress_signal.error_ready.connect(lambda t, m: QMessageBox.critical(self, t, m))
        self._close_preview_signal = _ClosePreviewSignal()
        self._close_preview_signal.close_ready.connect(self._close_preview)
        self._load_xseg_config()

    def _load_xseg_config(self):
        config_path = Path(XSEG_MODEL_DIR) / "XSeg_config.json"
        if not config_path.exists():
            return
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            self._xseg_face_type.setCurrentText(cfg.get("face_type", "wf"))
            self._xseg_batch.setValue(cfg.get("batch_size", 4))
            self._xseg_iters.setValue(cfg.get("target_iter", 100000))
            self._xseg_amp.setCurrentText(cfg.get("amp_mode", "fp16"))
            self._xseg_lr.setValue(cfg.get("learning_rate", 1e-4))
            self._xseg_lr_dropout.setValue(cfg.get("lr_dropout", 0.3))
            pretrain = cfg.get("pretrain", None)
            if pretrain is not None:
                self._xseg_pretrain.setChecked(pretrain)
                if pretrain is False:
                    self._xseg_pretrain.setEnabled(False)
                    self._xseg_pretrain.setToolTip("已退出预训练模式，不可回退")
        except Exception:
            pass

    def _close_preview(self):
        if self._preview_win is not None:
            self._preview_win.close()
            self._preview_win = None

    def _set_xseg_controls_enabled(self, enabled: bool):
        self._train_btn.setEnabled(enabled)
        self._stop_train_btn.setEnabled(not enabled)
        for w in [self._xseg_face_type, self._xseg_batch, self._xseg_iters,
                  self._xseg_amp, self._xseg_lr, self._xseg_lr_dropout,
                  self._xseg_pretrain]:
            w.setEnabled(enabled)
        if enabled:
            config_path = Path(XSEG_MODEL_DIR) / "XSeg_config.json"
            if config_path.exists():
                try:
                    cfg = json.loads(config_path.read_text(encoding="utf-8"))
                    if cfg.get("pretrain") is False:
                        self._xseg_pretrain.setEnabled(False)
                        self._xseg_pretrain.setToolTip("已退出预训练模式，不可回退")
                except Exception:
                    pass

    def _set_running(self, running: bool):
        self._running = running

    def _get_edit_target_dir(self) -> Path:
        return DATA_SRC_ALIGNED_DIR if self._edit_src_btn.isChecked() else DATA_DST_ALIGNED_DIR

    def _get_apply_target_dir(self) -> Path:
        return DATA_SRC_ALIGNED_DIR if self._apply_src_btn.isChecked() else DATA_DST_ALIGNED_DIR

    def _show_preview(self, preview_bgr: np.ndarray):
        import cv2
        from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout, QSizePolicy
        from PyQt6.QtGui import QImage, QPixmap
        from PyQt6.QtCore import Qt

        if self._preview_win is None:
            self._preview_win = QDialog(self)
            self._preview_win.setWindowTitle("XSeg 训练预览 | [space]:切换 [p]:刷新 [s]:保存 [l]:历史范围 [Enter]:保存并停止")
            self._preview_win.setWindowFlags(self._preview_win.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint)
            lay = QVBoxLayout(self._preview_win)
            lay.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            lay.addWidget(lbl)
            self._preview_win._label = lbl
            self._preview_win._orig_pixmap = None
            self._preview_win.resize(600, 800)
            self._preview_win.keyPressEvent = lambda e: self._on_preview_key(e)
            original_close = self._preview_win.close
            self._preview_win.closeEvent = lambda e: self._on_preview_close(e, original_close)
            original_resize = self._preview_win.resizeEvent
            def _on_resize(e, orig=original_resize):
                orig(e)
                self._rescale_xseg_preview_pixmap()
            self._preview_win.resizeEvent = _on_resize
            self._preview_win.show()

        if not self._preview_win.isVisible():
            self._preview_win.show()

        rgb = bgr_to_rgb(preview_bgr)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self._preview_win._orig_pixmap = pix

        try:
            screen = self._preview_win.screen()
            max_w = int(screen.availableGeometry().width() * 0.85)
            max_h = int(screen.availableGeometry().height() * 0.85)
        except Exception:
            max_w, max_h = 1400, 900
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            win_w = int(w * scale)
            win_h = int(h * scale)
        else:
            win_w = w
            win_h = h
        self._preview_win.resize(win_w, win_h)

        self._rescale_xseg_preview_pixmap()

    def _append_log(self, line: str, overwrite: bool = False):
        te = self._train_log
        if overwrite:
            cursor = te.textCursor()
            cursor.beginEditBlock()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.movePosition(cursor.MoveOperation.StartOfBlock, cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(line)
            cursor.endEditBlock()
            te.setTextCursor(cursor)
        else:
            te.append(line)
        sb = te.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _rescale_xseg_preview_pixmap(self):
        if self._preview_win is None:
            return
        pix = self._preview_win._orig_pixmap
        if pix is None:
            return
        lbl_size = self._preview_win._label.size()
        if lbl_size.width() < 1 or lbl_size.height() < 1:
            return
        from PyQt6.QtCore import Qt
        scaled = pix.scaled(lbl_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._preview_win._label.setPixmap(scaled)

    def _on_preview_key(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            if self._xseg_trainer is not None:
                self._xseg_trainer._preview_page += 1
                self._xseg_trainer.request_preview()
        elif key == Qt.Key.Key_P:
            if self._xseg_trainer is not None:
                self._xseg_trainer.request_preview()
        elif key == Qt.Key.Key_S:
            if self._xseg_trainer is not None:
                self._xseg_trainer.request_save()
                self._log_signal.log_ready.emit("[save] Checkpoint saved.", False)
        elif key == Qt.Key.Key_L:
            if self._xseg_trainer is not None:
                self._xseg_trainer.cycle_loss_range()
                self._xseg_trainer.request_preview()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._stop_and_close_preview()

    def _on_preview_close(self, event, original_close):
        self._stop_and_close_preview()
        event.accept()

    def _stop_and_close_preview(self):
        self._request_training_stop(self._xseg_trainer, "enter", self._log_signal)

    def _on_edit(self):
        from faceswap.gui_app.xseg_editor_dialog import XSegEditorDialog
        d = self._get_edit_target_dir()
        dlg = XSegEditorDialog(d, self)
        dlg.exec()

    def _on_fetch(self):
        from faceswap.business.xseg_editor import XSegEditor
        d = self._get_edit_target_dir()
        editor = XSegEditor()
        count = editor.fetch_annotated(d)
        QMessageBox.information(self, "完成", f"已提取 {count} 个已标注人脸")

    def _on_remove_annotations(self):
        d = self._get_edit_target_dir()
        reply = QMessageBox.warning(
            self, "确认", f"确定要清除 {d} 中所有XSeg标注吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        from faceswap.business.xseg_editor import XSegEditor
        editor = XSegEditor()
        sig = self._progress_signal
        sig.progress_ready.emit("Processing...")

        def _on_progress(current, total, elapsed):
            if total > 0:
                eta = elapsed / current * (total - current) if current > 0 else 0
                rate = current / elapsed if elapsed > 0 else 0
                sig.progress_ready.emit(f"Processing: {current}/{total} [{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}it/s]")

        def _task():
            try:
                count = editor.remove_annotations(d, progress_callback=_on_progress)
                sig.progress_ready.emit(f"Done: {count} faces")
                sig.done_ready.emit("完成", f"已清除 {count} 张人脸的XSeg标注")
            except Exception as e:
                sig.progress_ready.emit(f"Error: {e}")
                sig.error_ready.emit("错误", str(e))

        self._run_in_thread(_task)

    def _on_train(self):
        from faceswap.business.xseg_trainer import XSegTrainer
        self._xseg_trainer = XSegTrainer()
        self._set_xseg_controls_enabled(False)
        self._train_log.clear()

        self._log_throttle_time = 0.0
        self._log_throttle_interval = 0.5
        self._log_need_newline = True
        self._loss_smooth_buffer = deque(maxlen=100)

        def _on_iter(iter_count, loss_val, iter_ms):
            import time as _time
            now = _time.time()
            if now - self._log_throttle_time < self._log_throttle_interval:
                return
            self._log_throttle_time = now
            self._loss_smooth_buffer.append(loss_val)
            smoothed = sum(self._loss_smooth_buffer) / len(self._loss_smooth_buffer)
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            if iter_ms >= 1000:
                time_str = f"{iter_ms/1000:.1f}s"
            else:
                time_str = f"{int(iter_ms)}ms"
            line = f"[{ts}][#{iter_count}][{time_str}][loss {smoothed:.5f}]"

            overwrite = not self._log_need_newline
            self._log_need_newline = False
            self._log_signal.log_ready.emit(line, overwrite)

        def _on_preview(preview_bgr):
            self._preview_signal.preview_ready.emit(preview_bgr)

        def _on_save(iter_count):
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            self._log_signal.log_ready.emit(f"[{ts}][#{iter_count}] saved", False)
            self._log_need_newline = False

        def _task():
            try:
                self._xseg_trainer.train(
                    DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR, XSEG_MODEL_DIR,
                    batch_size=self._xseg_batch.value(),
                    target_iter=self._xseg_iters.value(),
                    face_type=self._xseg_face_type.currentText(),
                    learning_rate=self._xseg_lr.value(),
                    amp_mode=self._xseg_amp.currentText(),
                    pretrain=self._xseg_pretrain.isChecked(),
                    lr_dropout=self._xseg_lr_dropout.value(),
                    pretrain_data_dir=Path(DATA_SRC_ALIGNED_DIR),
                    on_iter=_on_iter,
                    on_preview=_on_preview,
                    on_save=_on_save,
                )
            finally:
                self._set_xseg_controls_enabled(True)
                self._log_signal.log_ready.emit("已停止", True)
                self._close_preview_signal.close_ready.emit()

        self._run_in_thread(_task)

    def _on_stop_train(self):
        self._stop_train_btn.setEnabled(False)
        self._request_training_stop(self._xseg_trainer, "stop", self._log_signal)

    def _on_apply_trained(self):
        from faceswap.business.xseg_trainer import XSegTrainer
        d = self._get_apply_target_dir()
        trainer = XSegTrainer()
        sig = self._progress_signal
        sig.progress_ready.emit("Processing...")

        def _on_progress(current, total, elapsed):
            if total > 0:
                eta = elapsed / current * (total - current) if current > 0 else 0
                rate = current / elapsed if elapsed > 0 else 0
                sig.progress_ready.emit(f"Processing: {current}/{total} [{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}it/s]")

        def _task():
            try:
                count = trainer.apply_trained_mask(d, XSEG_MODEL_DIR, progress_callback=_on_progress)
                sig.progress_ready.emit(f"Done: {count} faces")
                sig.done_ready.emit("完成", f"已应用训练遮罩到 {count} 张人脸")
            except Exception as e:
                sig.progress_ready.emit(f"Error: {e}")
                sig.error_ready.emit("错误", str(e))

        self._run_in_thread(_task)

    def _on_remove_trained(self):
        d = self._get_apply_target_dir()
        reply = QMessageBox.warning(
            self, "确认", f"确定要移除 {d} 中所有训练遮罩吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        from faceswap.business.xseg_trainer import XSegTrainer
        trainer = XSegTrainer()
        sig = self._progress_signal
        sig.progress_ready.emit("Processing...")

        def _on_progress(current, total, elapsed):
            if total > 0:
                eta = elapsed / current * (total - current) if current > 0 else 0
                rate = current / elapsed if elapsed > 0 else 0
                sig.progress_ready.emit(f"Processing: {current}/{total} [{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}it/s]")

        def _task():
            try:
                count = trainer.remove_trained_mask(d, progress_callback=_on_progress)
                sig.progress_ready.emit(f"Done: {count} faces")
                sig.done_ready.emit("完成", f"已移除 {count} 张人脸的训练遮罩")
            except Exception as e:
                sig.progress_ready.emit(f"Error: {e}")
                sig.error_ready.emit("错误", str(e))

        self._run_in_thread(_task)

    def _on_apply_generic(self):
        from faceswap.business.xseg_trainer import XSegTrainer

        d = self._get_apply_target_dir()
        dialog = _AutoMaskSelectionDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        method = dialog.get_method()
        if method is None:
            return

        trainer = XSegTrainer()
        sig = self._progress_signal
        sig.progress_ready.emit("Processing...")

        def _on_progress(current, total, elapsed):
            if total > 0:
                eta = elapsed / current * (total - current) if current > 0 else 0
                rate = current / elapsed if elapsed > 0 else 0
                sig.progress_ready.emit(f"Processing: {current}/{total} [{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}it/s]")

        method_names = {"face_parsing": "Face Parsing", "dfl": "DFL遮罩", "sam3": "SAM3"}

        def _task():
            try:
                count = trainer.apply_auto_mask(d, method, progress_callback=_on_progress)
                sig.progress_ready.emit(f"Done: {count} faces")
                sig.done_ready.emit("完成", f"已应用{method_names.get(method, method)}遮罩到 {count} 张人脸")
            except Exception as e:
                sig.progress_ready.emit(f"Error: {e}")
                sig.error_ready.emit("错误", str(e))

        self._run_in_thread(_task)


class _AutoMaskSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择自动遮罩方法")
        self.setFixedWidth(380)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel("选择要应用的自动遮罩方法："))

        self._dfl_btn = QRadioButton("DFL遮罩（推荐！）")
        self._dfl_btn.setChecked(True)
        self._face_parsing_btn = QRadioButton("Face Parsing（速度很慢！）")
        self._sam3_btn = QRadioButton("SAM3自动分割（速度很慢！）")
        layout.addWidget(self._dfl_btn)
        layout.addWidget(self._face_parsing_btn)
        layout.addWidget(self._sam3_btn)

        hint = QLabel("将对目标目录中每张人脸运行所选方法，\n生成多边形遮罩并保存到元数据。")
        hint.setStyleSheet("color: #666666; font-size: 12px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_method(self) -> Optional[str]:
        if self._dfl_btn.isChecked():
            return "dfl"
        if self._face_parsing_btn.isChecked():
            return "face_parsing"
        if self._sam3_btn.isChecked():
            return "sam3"
        return None


class _RenameConflictDialog(QDialog):
    def __init__(self, existing_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型名称冲突")
        self.setFixedWidth(420)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel(f"已存在名称为 '{existing_name}' 的模型，请重命名。"))

        rename_row = QHBoxLayout()
        rename_row.addWidget(QLabel("重命名:"))
        self._name_edit = QLineEdit()
        rename_row.addWidget(self._name_edit)
        layout.addLayout(rename_row)

        hint = QLabel("若不重新命名，则点击保存后将使用原有模型继续训练。")
        hint.setStyleSheet("color: #666666; font-size: 12px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_name(self) -> str:
        return self._name_edit.text().strip()


class Step4Train(StepPanel):
    step_title = "4. 训练"
    step_desc = "换脸模型训练：选择模型类型(SAEHD/AMP/Quick96)和参数，训练特定人物对模型。"
    show_run_buttons = False

    _MODEL_DIRS = {
        "SAEHD": "SAEHD_MODEL_DIR",
        "AMP": "AMP_MODEL_DIR",
        "Quick96": "QUICK96_MODEL_DIR",
    }
    _MODEL_CONFIG_NAMES = {
        "SAEHD": "SAEHD_training_config.json",
        "AMP": "AMP_training_config.json",
        "Quick96": "Quick96_training_config.json",
    }

    def _build_params(self):
        from faceswap.gui_app.param_defs import (
            ParamGroupWidget, CompositeParamGroup, ConfigManager,
            TrainingSignals, TrainingStatusBar,
        )
        from faceswap.gui_app.model_name_selector import ModelNameSelector

        mt_row = QHBoxLayout()
        mt_label = QLabel("模型类型:")
        mt_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        mt_label.setFixedWidth(80)
        mt_row.addWidget(mt_label)

        self._model_type_group = QButtonGroup(self)
        self._model_type_btns: dict[str, QPushButton] = {}
        _mt_style_checked = (
            "QPushButton { background-color: #0078D4; color: white; "
            "font-weight: bold; padding: 6px 20px; border-radius: 4px; font-size: 13px; }"
        )
        _mt_style_unchecked = (
            "QPushButton { background-color: #E0E0E0; color: #333; "
            "font-weight: bold; padding: 6px 20px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background-color: #C8E0F4; }"
        )
        for mt in ("SAEHD", "AMP", "Quick96"):
            btn = QPushButton(mt)
            btn.setCheckable(True)
            btn.setMinimumWidth(100)
            btn.setStyleSheet(_mt_style_unchecked)
            self._model_type_btns[mt] = btn
            self._model_type_group.addButton(btn)
            mt_row.addWidget(btn)
        mt_row.addStretch()
        self._params_area.addLayout(mt_row)

        self._model_type_btns["SAEHD"].setChecked(True)
        self._model_type_btns["SAEHD"].setStyleSheet(_mt_style_checked)
        self._current_model_type = "SAEHD"
        self._model_type_group.buttonClicked.connect(
            lambda btn: self._on_model_type_changed(btn, _mt_style_checked, _mt_style_unchecked)
        )

        self._model_container = QWidget()
        self._model_container_layout = QVBoxLayout(self._model_container)
        self._model_container_layout.setContentsMargins(0, 0, 0, 0)
        self._model_container_layout.setSpacing(6)
        self._params_area.addWidget(self._model_container)

        self._build_model_params("SAEHD")

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._run_btn = QPushButton("开始训练")
        self._run_btn.setFixedWidth(140)
        self._run_btn.setStyleSheet(
            "QPushButton { background-color: #D45500; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background-color: #E06010; }"
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)
        self._stop_btn = QPushButton("停止")
        self._stop_btn.setFixedWidth(100)
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #C42B1C; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background-color: #D43B2C; }"
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)
        self._params_area.addLayout(btn_row)

        self._reconnect_model_selector_buttons()

        self._status_bar = TrainingStatusBar()
        self._params_area.addWidget(self._status_bar)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMinimumHeight(100)
        self._log_text.setMaximumHeight(200)
        self._log_text.setStyleSheet(
            "QTextEdit { font-family: Consolas, 'Cascadia Code', monospace; font-size: 11px; "
            "background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #3C3C3C; "
            "border-radius: 4px; padding: 4px; }"
        )
        self._params_area.addWidget(self._log_text)

        from faceswap.shared.logger import attach_gui_handler
        attach_gui_handler(
            self._log_text,
            overwrite_getter=lambda: not self._log_need_newline,
            need_newline_setter=lambda v: setattr(self, '_log_need_newline', v),
        )

        self._signals = TrainingSignals()
        self._signals.iter_signal.connect(self._on_iter_status)
        self._signals.save_signal.connect(self._on_save_notify)
        self._signals.error_signal.connect(self._on_training_error)
        self._signals.finished_signal.connect(self._on_training_finished)
        self._signals.log_signal.connect(self._append_log)

        self._preview_signal = _PreviewSignal()
        self._preview_signal.preview_ready.connect(self._show_preview)

        self._trainer = None
        self._training_thread = None
        self._preview_win = None
        self._log_need_newline = True
        self._log_throttle_time = 0.0
        self._log_throttle_interval = 0.5

    def _on_model_type_changed(self, btn, style_checked, style_unchecked):
        model_type = btn.text()
        for mt, b in self._model_type_btns.items():
            b.setStyleSheet(style_checked if mt == model_type else style_unchecked)
        self._build_model_params(model_type)
        self._reconnect_model_selector_buttons()

    def _reconnect_model_selector_buttons(self):
        try:
            self._model_name_selector.train_btn.clicked.disconnect()
            self._model_name_selector.stop_btn.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        self._model_name_selector.train_btn.clicked.connect(self._on_run)
        self._model_name_selector.stop_btn.clicked.connect(self._on_stop)

    def _build_model_params(self, model_type: str):
        while self._model_container_layout.count():
            item = self._model_container_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self._current_model_type = model_type

        if model_type == "SAEHD":
            self._build_saehd_params()
        elif model_type in ("AMP", "Quick96"):
            self._build_generic_params(model_type)

    def _build_saehd_params(self):
        from faceswap.gui_app.param_defs import (
            ParamGroupWidget, CompositeParamGroup, ConfigManager,
        )
        from faceswap.gui_app.saehd_param_defs import (
            SAEHD_SUBGROUPS, get_saehd_params_by_keys,
        )
        from faceswap.gui_app.model_name_selector import ModelNameSelector
        from faceswap.setting import SAEHD_MODEL_DIR

        self._model_name_selector = ModelNameSelector(SAEHD_MODEL_DIR)
        self._model_container_layout.addWidget(self._model_name_selector)

        self._param_groups: dict[ParamGroup, CompositeParamGroup] = {}
        self._subgroup_widgets: list[ParamGroupWidget] = []
        for sg_def in SAEHD_SUBGROUPS:
            sg_params = get_saehd_params_by_keys(sg_def.param_keys)
            if not sg_params:
                continue
            parent_group = sg_params[0].group
            group_color = GROUP_COLORS.get(parent_group, "#0078D4")
            pw = ParamGroupWidget(
                parent_group, sg_params,
                title=sg_def.subgroup_name, color=group_color,
            )
            self._subgroup_widgets.append(pw)
            composite = self._param_groups.get(parent_group)
            if composite is None:
                composite = CompositeParamGroup(parent_group)
                self._param_groups[parent_group] = composite
            composite.add_sub_widget(pw)
            self._model_container_layout.addWidget(pw)

        self._config_manager = ConfigManager(
            self._model_name_selector.current_dir(), "SAEHD_training_config.json")
        self._model_name_selector.model_changed.connect(self._on_model_changed)
        self._load_model_config()

        self._link_face_type_resolution()
        self._link_archi_filter()

    def _build_generic_params(self, model_type: str):
        from faceswap.gui_app.param_defs import (
            ParamGroupWidget, CompositeParamGroup, ConfigManager,
        )
        from faceswap.gui_app.model_name_selector import ModelNameSelector
        from faceswap.gui_app.multi_model_param_defs import (
            AMP_PARAM_DEFS, QUICK96_PARAM_DEFS,
        )
        from faceswap.setting import (
            AMP_MODEL_DIR, QUICK96_MODEL_DIR,
        )

        dirs_map = {"AMP": AMP_MODEL_DIR, "Quick96": QUICK96_MODEL_DIR}
        defs_map = {"AMP": AMP_PARAM_DEFS, "Quick96": QUICK96_PARAM_DEFS}
        config_name = self._MODEL_CONFIG_NAMES[model_type]

        self._model_name_selector = ModelNameSelector(
            dirs_map[model_type], config_filename=config_name)
        self._model_container_layout.addWidget(self._model_name_selector)

        param_defs = defs_map[model_type]
        groups_order = []
        groups: dict[ParamGroup, list] = {}
        for p in param_defs:
            if p.group not in groups:
                groups[p.group] = []
                groups_order.append(p.group)
            groups[p.group].append(p)

        self._param_groups: dict[ParamGroup, CompositeParamGroup] = {}
        self._subgroup_widgets: list[ParamGroupWidget] = []
        for group in groups_order:
            params = groups[group]
            pw = ParamGroupWidget(group, params)
            self._subgroup_widgets.append(pw)
            composite = CompositeParamGroup(group)
            composite.add_sub_widget(pw)
            self._param_groups[group] = composite
            self._model_container_layout.addWidget(pw)

        self._config_manager = ConfigManager(
            self._model_name_selector.current_dir(), config_name)
        self._model_name_selector.model_changed.connect(self._on_model_changed)
        self._load_model_config()

    def _load_model_config(self) -> None:
        saved = self._config_manager.load_config()
        if saved:
            for pw in self._param_groups.values():
                pw.set_values(saved)
            if self._current_model_type in ("SAEHD", "AMP"):
                pretrain_val = saved.get("pretrain", None)
                if pretrain_val is False:
                    basic_pw = self._param_groups.get(ParamGroup.BASIC)
                    if basic_pw and "pretrain" in basic_pw._param_widgets:
                        pretrain_w = basic_pw._param_widgets["pretrain"]
                        pretrain_w.setEnabled(False)
                        pretrain_w.setToolTip("已退出预训练模式，不可回退")
            self._lock_architecture_params()
        else:
            for pw in self._param_groups.values():
                defaults = {p.key: p.default for p in pw._params}
                pw.set_values(defaults)
                pw.set_editable(True)

    def _on_model_changed(self, model_name: str) -> None:
        self._config_manager.set_model_dir(self._model_name_selector.current_dir())
        self._load_model_config()

    def _collect_params(self) -> dict:
        params = {}
        for pw in self._param_groups.values():
            params.update(pw.get_values())
        return params

    def _link_face_type_resolution(self):
        ft_res = {'wf': 256, 'head': 384}
        basic = self._param_groups.get(ParamGroup.BASIC)
        if basic is None:
            return
        ft_w = basic._param_widgets.get('face_type')
        res_w = basic._param_widgets.get('resolution')
        if ft_w is None or res_w is None:
            return

        def _update_res(text):
            r = ft_res.get(text, 128)
            res_w.setValue(r)

        ft_w.currentTextChanged.connect(_update_res)

    def _link_archi_filter(self):
        basic = self._param_groups.get(ParamGroup.BASIC)
        if basic is None:
            return
        from faceswap.gui_app.param_defs import ArchiSelector
        archi_w = basic.get_widget('archi')
        if not isinstance(archi_w, ArchiSelector):
            return
        base = archi_w.value().split('-', 1)[0]
        self._on_archi_changed(base)
        archi_w.archi_changed.connect(self._on_archi_changed)
        d_cb = archi_w._opts.get('d')
        if d_cb is not None:
            d_cb.stateChanged.connect(lambda: self._update_resolution_alignment(archi_w))
        self._update_resolution_alignment(archi_w)

    def _on_archi_changed(self, base: str) -> None:
        for pw in self._subgroup_widgets:
            for key, pdef in pw._param_defs.items():
                if pdef.archi_filter is not None and base not in pdef.archi_filter:
                    pw.set_param_visible(key, False)
                else:
                    pw.set_param_visible(key, True)

    def _update_resolution_alignment(self, archi_w) -> None:
        full_archi = archi_w.value()
        needs_32 = 'd' in full_archi.split('-', 1)[-1] if '-' in full_archi else False
        align = 32 if needs_32 else 16
        for pw in self._subgroup_widgets:
            res_w = pw._param_widgets.get('resolution')
            if res_w is not None:
                res_w.setSingleStep(align)
                cur = res_w.value()
                aligned = (cur // align) * align
                if aligned < res_w.minimum():
                    aligned = res_w.minimum()
                if aligned != cur:
                    res_w.setValue(aligned)
                if needs_32:
                    res_w.setToolTip("分辨率（像素）。\n当前架构含'd'（分辨率倍增），必须是32的倍数。")
                else:
                    res_w.setToolTip("分辨率（像素）。必须是16的倍数。")
                break

    def _lock_architecture_params(self):
        if self._current_model_type == "SAEHD":
            _lock_keys = {
                'resolution', 'face_type', 'archi', 'pretrain', 'amp_mode',
                'ae_dims', 'e_dims', 'd_dims', 'd_mask_dims',
                'gan_patch_size', 'gan_dims',
            }
        elif self._current_model_type == "AMP":
            _lock_keys = {
                'resolution', 'face_type', 'ae_dims', 'inter_dims',
                'e_dims', 'd_dims', 'd_mask_dims', 'morph_factor', 'pretrain',
                'gan_patch_size', 'gan_dims', 'amp_mode',
            }
        else:
            _lock_keys = {'amp_mode'}
        _lock_tooltip = "模型已初始化，此参数不可修改（改变会导致模型结构或计算方式不兼容）"
        for pw in self._subgroup_widgets:
            for key in _lock_keys:
                w = pw._param_widgets.get(key)
                if w is not None:
                    w.setEnabled(False)
                    w.setToolTip(_lock_tooltip)

    def _on_run(self):
        from faceswap.setting import DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR
        from faceswap.shared.file_manager import FileManager
        import time as _time

        if not DATA_SRC_ALIGNED_DIR.exists() or not DATA_DST_ALIGNED_DIR.exists():
            QMessageBox.warning(self, "数据目录不存在",
                f"源人脸目录: {DATA_SRC_ALIGNED_DIR}\n"
                f"目标人脸目录: {DATA_DST_ALIGNED_DIR}\n\n"
                "请先完成步骤1-3（视频提取、人脸提取、XSeg标注）。")
            return

        src_images = FileManager.find_images(DATA_SRC_ALIGNED_DIR)
        dst_images = FileManager.find_images(DATA_DST_ALIGNED_DIR)
        if len(src_images) < 1 or len(dst_images) < 1:
            QMessageBox.warning(self, "训练数据不足",
                f"源人脸: {len(src_images)} 张\n"
                f"目标人脸: {len(dst_images)} 张\n\n"
                "每个目录至少需要1张对齐后的人脸图像才能开始训练。")
            return

        if len(src_images) < 4 or len(dst_images) < 4:
            reply = QMessageBox.question(self, "训练数据较少",
                f"源人脸: {len(src_images)} 张\n"
                f"目标人脸: {len(dst_images)} 张\n\n"
                "数据较少可能影响训练效果。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        params = self._collect_params()
        model_type = self._current_model_type

        if model_type == "SAEHD":
            from faceswap.business.saehd_trainer import SAEHDTrainer, TrainingConfig
            from faceswap.setting import SAEHD_MODEL_DIR as model_base_dir
            config = TrainingConfig(**params)
            archi = params.get('archi', 'df')
            trainer_cls = SAEHDTrainer
        elif model_type == "AMP":
            from faceswap.business.amp_trainer import AMPTrainer, AMPTrainingConfig
            from faceswap.setting import AMP_MODEL_DIR as model_base_dir
            config = AMPTrainingConfig(**params)
            archi = None
            trainer_cls = AMPTrainer
        elif model_type == "Quick96":
            from faceswap.business.quick96_trainer import Quick96Trainer, Quick96TrainingConfig
            from faceswap.setting import QUICK96_MODEL_DIR as model_base_dir
            config = Quick96TrainingConfig(**params)
            archi = None
            trainer_cls = Quick96Trainer

        base_name = self._model_name_selector.resolve_dir_name()

        if model_type == "SAEHD":
            config_file = model_base_dir / base_name / "training_config.json"
            if '_' in base_name and config_file.exists():
                final_dir_name = base_name
            else:
                final_dir_name = f"{base_name}_{archi}"
        else:
            final_dir_name = base_name

        final_dir = model_base_dir / final_dir_name

        is_new_action = self._model_name_selector.is_new_action
        if is_new_action and final_dir.exists() and any(final_dir.iterdir()):
            reply = QMessageBox.question(
                self, "模型已存在",
                f"已经有相同名字的权重文件了，请重新命名！\n"
                f"若坚持用这个名字，则会清空原来的权重文件！\n\n"
                f"模型名称: {final_dir_name}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                import shutil
                for item in final_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(str(item))
                    else:
                        item.unlink()
            else:
                self._model_name_selector._on_new()
                return

        final_dir.mkdir(parents=True, exist_ok=True)

        is_resume = any(final_dir.iterdir())
        if is_resume:
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            self._append_log(f"[{ts}] 检测到已有模型，继续训练: {final_dir_name}\n")

        self._config_manager.set_model_dir(final_dir)
        self._config_manager.save_config(params)

        self._set_controls_enabled(False)
        self._status_bar.start_pulse()
        self._log_text.clear()
        self._log_need_newline = True
        self._log_throttle_time = 0.0
        self._loss_smooth_buffer = deque(maxlen=100)
        self._src_loss_smooth = deque(maxlen=100)
        self._dst_loss_smooth = deque(maxlen=100)
        self._d_gan_loss_smooth = deque(maxlen=100)
        self._loss_since_save_src: list[float] = []
        self._loss_since_save_dst: list[float] = []
        self._loss_since_save_d_gan: list[float] = []
        self._save_interval_start_iter: int = -1
        self._save_interval_time_sum: float = 0.0

        signals = self._signals
        panel = self

        def _on_progress(iter_num, src_loss, dst_loss, iter_ms, lr, converged=False, d_gan_loss=0.0):
            if iter_num == -1:
                if panel._loss_since_save_src:
                    from datetime import datetime
                    ts = datetime.now().strftime("%H:%M:%S")
                    n = len(panel._loss_since_save_src)
                    interval_src = sum(panel._loss_since_save_src) / n
                    interval_dst = sum(panel._loss_since_save_dst) / n
                    interval_avg = (interval_src + interval_dst) / 2
                    start_iter = panel._save_interval_start_iter
                    end_iter = start_iter + n
                    avg_ms = panel._save_interval_time_sum / n if n > 0 else 0
                    panel._loss_since_save_src.clear()
                    panel._loss_since_save_dst.clear()
                    panel._save_interval_start_iter = -1
                    panel._save_interval_time_sum = 0.0
                    if avg_ms >= 1000:
                        time_str = f"{avg_ms/1000:.1f}s"
                    else:
                        time_str = f"{int(avg_ms)}ms"
                    line = f"[{ts}][#{start_iter}-{end_iter}][{time_str}][src {interval_src:.5f} dst {interval_dst:.5f}]"
                    if panel._loss_since_save_d_gan:
                        interval_d_gan = sum(panel._loss_since_save_d_gan) / n
                        line += f" D_gan={interval_d_gan:.5f}"
                    panel._loss_since_save_d_gan.clear()
                    overwrite = not panel._log_need_newline
                    signals.log_signal.emit(line, overwrite)
                    panel._loss_smooth_buffer.append(interval_avg)
                    panel._src_loss_smooth.append(interval_src)
                    panel._dst_loss_smooth.append(interval_dst)
                    smoothed = sum(panel._loss_smooth_buffer) / len(panel._loss_smooth_buffer)
                    signals.iter_signal.emit(iter_num, smoothed, 0)
                signals.log_signal.emit("[...] 正在保存模型...", False)
                panel._log_need_newline = False
                return
            panel._loss_since_save_src.append(src_loss)
            panel._loss_since_save_dst.append(dst_loss)
            if d_gan_loss > 0:
                panel._loss_since_save_d_gan.append(d_gan_loss)
            if panel._save_interval_start_iter < 0:
                panel._save_interval_start_iter = iter_num
            panel._save_interval_time_sum += iter_ms
            now = _time.time()
            if now - panel._log_throttle_time < panel._log_throttle_interval:
                return
            panel._log_throttle_time = now
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            loss_val = (src_loss + dst_loss) / 2
            panel._loss_smooth_buffer.append(loss_val)
            panel._src_loss_smooth.append(src_loss)
            panel._dst_loss_smooth.append(dst_loss)
            smoothed = sum(panel._loss_smooth_buffer) / len(panel._loss_smooth_buffer)
            src_smoothed = sum(panel._src_loss_smooth) / len(panel._src_loss_smooth)
            dst_smoothed = sum(panel._dst_loss_smooth) / len(panel._dst_loss_smooth)
            if iter_ms >= 1000:
                time_str = f"{iter_ms/1000:.1f}s"
            else:
                time_str = f"{int(iter_ms)}ms"
            extra = ""
            if d_gan_loss > 0:
                panel._d_gan_loss_smooth.append(d_gan_loss)
                d_gan_smoothed = sum(panel._d_gan_loss_smooth) / len(panel._d_gan_loss_smooth)
                extra += f" D_gan={d_gan_smoothed:.5f}"
            line = f"[{ts}][#{iter_num}][{time_str}][src {src_smoothed:.5f} dst {dst_smoothed:.5f}]{extra}"

            overwrite = not panel._log_need_newline
            panel._log_need_newline = False
            signals.log_signal.emit(line, overwrite)
            signals.iter_signal.emit(iter_num, smoothed, iter_ms)

        def _on_preview(preview_bgr):
            panel._preview_signal.preview_ready.emit(preview_bgr)

        def _task():
            try:
                if panel._pending_stop_event.is_set():
                    signals.finished_signal.emit()
                    return
                trainer = trainer_cls(
                    config=config,
                    model_dir=final_dir,
                    src_aligned_dir=DATA_SRC_ALIGNED_DIR,
                    dst_aligned_dir=DATA_DST_ALIGNED_DIR,
                    progress_callback=_on_progress,
                    preview_callback=_on_preview,
                )
                if panel._pending_stop_event.is_set():
                    trainer.request_stop()
                panel._trainer = trainer
                trainer.train()
                if not trainer._stop_requested:
                    signals.finished_signal.emit()
                    from datetime import datetime
                    ts = datetime.now().strftime("%H:%M:%S")
                    signals.log_signal.emit(f"[{ts}] 训练完成，共 {trainer._iter_count} 次迭代", True)
                else:
                    signals.finished_signal.emit()
                    from datetime import datetime
                    ts = datetime.now().strftime("%H:%M:%S")
                    signals.log_signal.emit(f"[{ts}] 训练已停止，共 {trainer._iter_count} 次迭代", True)
            except Exception as e:
                signals.error_signal.emit(str(e))

        self._pending_stop_event = threading.Event()
        self._training_thread = threading.Thread(target=_task, daemon=True)
        self._training_thread.start()

    def _on_stop(self):
        if hasattr(self, '_pending_stop_event'):
            self._pending_stop_event.set()
        if self._trainer is not None:
            self._request_training_stop(self._trainer, "stop")
        self._stop_btn.setEnabled(False)
        self._model_name_selector.stop_btn.setEnabled(False)

    def _set_controls_enabled(self, enabled: bool):
        self._model_name_selector.set_enabled(enabled)
        self._run_btn.setEnabled(enabled)
        self._stop_btn.setEnabled(not enabled)
        self._model_name_selector.train_btn.setEnabled(enabled)
        self._model_name_selector.stop_btn.setEnabled(not enabled)
        for pw in self._param_groups.values():
            pw.set_editable(enabled)
        if enabled:
            saved = self._config_manager.load_config()
            if self._current_model_type in ("SAEHD", "AMP"):
                if saved and saved.get("pretrain") is False:
                    basic_pw = self._param_groups.get(ParamGroup.BASIC)
                    if basic_pw and "pretrain" in basic_pw._param_widgets:
                        pretrain_w = basic_pw._param_widgets["pretrain"]
                        pretrain_w.setEnabled(False)
                        pretrain_w.setToolTip("已退出预训练模式，不可回退")
            if saved:
                self._lock_architecture_params()

    def _on_iter_status(self, iter_num: int, loss: float, ms: float):
        self._status_bar.update_status(iter_num, loss, ms)

    def _append_log(self, line: str, overwrite: bool = False):
        te = self._log_text
        if overwrite:
            cursor = te.textCursor()
            cursor.beginEditBlock()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.movePosition(cursor.MoveOperation.StartOfBlock, cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(line)
            cursor.endEditBlock()
            te.setTextCursor(cursor)
        else:
            te.append(line)
        sb = te.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _show_preview(self, preview_bgr: np.ndarray):
        import cv2
        from PyQt6.QtGui import QImage, QPixmap
        from PyQt6.QtWidgets import QSizePolicy

        if self._preview_win is None:
            self._preview_win = QDialog(self)
            self._preview_win.setWindowFlags(self._preview_win.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint)
            lay = QVBoxLayout(self._preview_win)
            lay.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            lay.addWidget(lbl)
            self._preview_win._label = lbl
            self._preview_win._orig_pixmap = None
            self._preview_win.resize(600, 800)
            self._preview_win.keyPressEvent = lambda e: self._on_preview_key(e)
            original_close = self._preview_win.close
            self._preview_win.closeEvent = lambda e: self._on_preview_close(e, original_close)
            original_resize = self._preview_win.resizeEvent
            def _on_resize(e, orig=original_resize):
                orig(e)
                self._rescale_preview_pixmap()
            self._preview_win.resizeEvent = _on_resize
            self._preview_win.show()

        if not self._preview_win.isVisible():
            self._preview_win.show()

        self._preview_win.setWindowTitle(
            f"{self._current_model_type} 训练预览 | "
            f"[space]:切换section [p]:刷新 [s]:保存 [l]:Loss范围 [Enter]:保存并停止")

        rgb = bgr_to_rgb(preview_bgr)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self._preview_win._orig_pixmap = pix

        try:
            screen = self._preview_win.screen()
            max_w = int(screen.availableGeometry().width() * 0.85)
            max_h = int(screen.availableGeometry().height() * 0.85)
        except Exception:
            max_w, max_h = 1400, 900
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            win_w = int(w * scale)
            win_h = int(h * scale)
        else:
            win_w = w
            win_h = h
        self._preview_win.resize(win_w, win_h)

        self._rescale_preview_pixmap()

    def _rescale_preview_pixmap(self):
        if self._preview_win is None:
            return
        pix = self._preview_win._orig_pixmap
        if pix is None:
            return
        lbl_size = self._preview_win._label.size()
        if lbl_size.width() < 1 or lbl_size.height() < 1:
            return
        scaled = pix.scaled(lbl_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._preview_win._label.setPixmap(scaled)

    def _on_preview_key(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            if self._trainer is not None:
                if hasattr(self._trainer, 'next_preview_page'):
                    self._trainer.next_preview_page()
                if hasattr(self._trainer, 'request_preview'):
                    self._trainer.request_preview()
        elif key == Qt.Key.Key_P:
            if self._trainer is not None and hasattr(self._trainer, 'request_preview'):
                self._trainer.request_preview()
        elif key == Qt.Key.Key_S:
            if self._trainer is not None and hasattr(self._trainer, 'request_save'):
                self._trainer.request_save()
                self._append_log("[save] Checkpoint saved.", False)
        elif key == Qt.Key.Key_L:
            if self._trainer is not None and hasattr(self._trainer, 'cycle_loss_range'):
                self._trainer.cycle_loss_range()
                self._trainer.request_preview()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._stop_and_close_preview()

    def _on_preview_close(self, event, original_close):
        self._stop_and_close_preview()
        event.accept()

    def _stop_and_close_preview(self):
        self._request_training_stop(self._trainer, "enter")

    def _on_save_notify(self, iter_num: int):
        current = self._status_bar._status_label.text()
        self._status_bar._status_label.setText(f"{current} | 已保存 #{iter_num}")

    def _on_training_error(self, error_msg: str):
        self._set_controls_enabled(True)
        self._status_bar.stop_pulse()
        QMessageBox.critical(self, "训练错误", error_msg)

    def _on_training_finished(self):
        self._set_controls_enabled(True)
        self._status_bar.stop_pulse()
        if self._preview_win is not None:
            self._preview_win.close()
            self._preview_win = None


class Step5Merge(StepPanel):
    step_title = "5. 合成融合"
    step_desc = "实时预览换脸效果，调整遮罩与颜色迁移参数。"

    def _build_ui(self):
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._params_area = QVBoxLayout()
        self._params_area.setContentsMargins(0, 0, 0, 0)
        self._params_area.setSpacing(0)
        layout.addLayout(self._params_area)
        self._build_params()

    def _build_params(self):
        import cv2
        self._cv2 = cv2
        self._trainer = None
        self._frames = []
        self._frame_idx = 0
        self._frame_to_faces = {}
        self._current_img = None
        self._stop_requested = False

        self._scan_frames()
        self._load_face_metadata()

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_scroll = QScrollArea()
        left_scroll.setFixedWidth(380)
        left_scroll.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(8, 8, 8, 8)

        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(6)
        row = QHBoxLayout()
        row.addWidget(QLabel("类型:"))
        self._model_type = QComboBox()
        self._model_type.addItems(["SAEHD", "AMP", "Quick96"])
        self._model_type.currentTextChanged.connect(self._on_model_type_changed)
        row.addWidget(self._model_type, 1)
        model_layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("名称:"))
        self._model_name = QComboBox()
        self._model_name.setMinimumWidth(180)
        row.addWidget(self._model_name, 1)
        model_layout.addLayout(row)
        self._load_model_btn = QPushButton("加载模型")
        self._load_model_btn.clicked.connect(self._on_load_model)
        model_layout.addWidget(self._load_model_btn)
        self._model_status = QLabel("未加载")
        self._model_status.setStyleSheet("font-size: 11px; color: #999;")
        model_layout.addWidget(self._model_status)
        left_layout.addWidget(model_group)

        merge_group = QGroupBox("合成参数")
        merge_layout = QVBoxLayout(merge_group)
        merge_layout.setSpacing(6)
        merge_grid = QGridLayout()
        merge_grid.setHorizontalSpacing(8)
        merge_grid.setVerticalSpacing(6)
        merge_grid.setColumnStretch(1, 1)
        _merge_row = [0]

        def _mr(label_text, widget):
            lbl = QLabel(label_text)
            merge_grid.addWidget(lbl, _merge_row[0], 0)
            merge_grid.addWidget(widget, _merge_row[0], 1)
            _merge_row[0] += 1

        self._mask_mode = QComboBox()
        self._mask_mode.addItems(["xseg", "dst", "learned", "learned-prd", "learned-dst", "learned-prd*dst"])
        self._mask_mode.setCurrentText("xseg")
        _mr("遮罩模式:", self._mask_mode)
        self._erode = QDoubleSpinBox()
        self._erode.setRange(-400, 400)
        self._erode.setValue(0)
        _mr("遮罩侵蚀:", self._erode)
        self._blur = QDoubleSpinBox()
        self._blur.setRange(0, 400)
        self._blur.setValue(20)
        _mr("遮罩模糊:", self._blur)
        self._ct_mode = QComboBox()
        self._ct_mode.addItems(["none", "rct", "lct", "mkl", "idt", "sot"])
        _mr("颜色迁移:", self._ct_mode)
        self._face_scale = QDoubleSpinBox()
        self._face_scale.setRange(-50, 50)
        self._face_scale.setValue(0)
        _mr("人脸缩放:", self._face_scale)
        self._enhancer = QComboBox()
        self._enhancer.addItems(["(无)", "gfpgan_1.4", "gpen_bfr_512", "gpen_bfr_1024", "restoreformer_pp"])
        _mr("人脸增强:", self._enhancer)
        self._enhancer_blend = QSpinBox()
        self._enhancer_blend.setRange(0, 100)
        self._enhancer_blend.setValue(80)
        _mr("增强混合:", self._enhancer_blend)
        merge_layout.addLayout(merge_grid)
        left_layout.addWidget(merge_group)

        nav_group = QGroupBox("帧导航")
        nav_layout = QVBoxLayout(nav_group)
        nav_layout.setSpacing(6)
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(max(0, len(self._frames) - 1))
        self._frame_slider.valueChanged.connect(self._on_frame_changed)
        nav_layout.addWidget(self._frame_slider)
        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("◀ 上一帧")
        self._prev_btn.clicked.connect(self._on_prev_frame)
        nav_row.addWidget(self._prev_btn)
        self._frame_info = QLabel("0/0")
        self._frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_row.addWidget(self._frame_info, 1)
        self._next_btn = QPushButton("下一帧 ▶")
        self._next_btn.clicked.connect(self._on_next_frame)
        nav_row.addWidget(self._next_btn)
        nav_layout.addLayout(nav_row)
        left_layout.addWidget(nav_group)

        self._merge_all_btn = QPushButton("合成全部帧")
        self._merge_all_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self._merge_all_btn.clicked.connect(self._on_merge_all)
        left_layout.addWidget(self._merge_all_btn)
        self._stop_btn = QPushButton("停止")
        self._stop_btn.setProperty("danger", True)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(lambda: setattr(self, '_stop_requested', True))
        left_layout.addWidget(self._stop_btn)
        self._batch_progress = QProgressBar()
        self._batch_progress.setVisible(False)
        left_layout.addWidget(self._batch_progress)
        left_layout.addStretch()
        left_scroll.setWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._preview_label = QLabel("请加载模型并选择帧")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet("background: #1a1a1a; color: #888; font-size: 14px;")
        self._preview_label.setMinimumSize(400, 300)
        right_layout.addWidget(self._preview_label, 1)
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("font-size: 11px; color: #666; padding: 4px;")
        right_layout.addWidget(self._status_label)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self._params_area.addWidget(splitter)

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._do_render)

        for w in [self._mask_mode, self._ct_mode, self._enhancer]:
            w.currentTextChanged.connect(self._on_param_changed)
        for w in [self._erode, self._blur, self._face_scale, self._enhancer_blend]:
            w.valueChanged.connect(self._on_param_changed)

        self._on_model_type_changed()
        if self._frames:
            self._frame_slider.setValue(0)

    def _scan_frames(self):
        from faceswap.shared.file_manager import FileManager
        self._frames = FileManager.find_images(DATA_DST_DIR) if DATA_DST_DIR.exists() else []

    def _load_face_metadata(self):
        from faceswap.core.metadata_manager import MetadataManager
        all_meta = MetadataManager.load_all(DATA_DST_ALIGNED_DIR) if DATA_DST_ALIGNED_DIR.exists() else {}
        self._frame_to_faces = {}
        for face_name, meta in all_meta.items():
            src_fn = meta.source_filename
            if src_fn not in self._frame_to_faces:
                self._frame_to_faces[src_fn] = []
            self._frame_to_faces[src_fn].append((face_name, meta))

    def _on_model_type_changed(self):
        from faceswap.setting import SAEHD_MODEL_DIR, AMP_MODEL_DIR, QUICK96_MODEL_DIR
        mt = self._model_type.currentText()
        base_dirs = {"SAEHD": SAEHD_MODEL_DIR, "AMP": AMP_MODEL_DIR, "Quick96": QUICK96_MODEL_DIR}
        base_dir = base_dirs.get(mt, SAEHD_MODEL_DIR)
        self._model_name.clear()
        if base_dir.exists():
            for d in sorted(base_dir.iterdir()):
                if d.is_dir() and (d / "training_config.json").exists():
                    self._model_name.addItem(d.name)
        self._trainer = None
        self._model_status.setText("未加载")
        self._model_status.setStyleSheet("font-size: 11px; color: #999;")

    def _on_load_model(self):
        mt = self._model_type.currentText()
        mn = self._model_name.currentText()
        if not mn:
            QMessageBox.warning(self, "未选择模型", "请先选择要加载的模型名称。")
            return
        from faceswap.setting import SAEHD_MODEL_DIR, AMP_MODEL_DIR, QUICK96_MODEL_DIR
        base_dirs = {"SAEHD": SAEHD_MODEL_DIR, "AMP": AMP_MODEL_DIR, "Quick96": QUICK96_MODEL_DIR}
        model_dir = base_dirs[mt] / mn
        config_path = model_dir / "training_config.json"
        if not config_path.exists():
            QMessageBox.warning(self, "配置不存在", f"找不到训练配置: {config_path}")
            return
        try:
            config_dict = json.loads(config_path.read_text(encoding="utf-8"))
            from faceswap.shared.config import auto_select_device
            device = auto_select_device()
            dummy = Path(".")
            if mt == "SAEHD":
                from faceswap.business.saehd_trainer import SAEHDTrainer, TrainingConfig
                config = TrainingConfig(**config_dict)
                self._trainer = SAEHDTrainer(config, model_dir, dummy, dummy, device=device)
            elif mt == "AMP":
                from faceswap.business.amp_trainer import AMPTrainer, AMPTrainingConfig
                config = AMPTrainingConfig(**config_dict)
                self._trainer = AMPTrainer(config, model_dir, dummy, dummy, device=device)
            elif mt == "Quick96":
                from faceswap.business.quick96_trainer import Quick96Trainer, Quick96TrainingConfig
                config = Quick96TrainingConfig(**config_dict)
                self._trainer = Quick96Trainer(config, model_dir, dummy, dummy, device=device)
            res = self._trainer.config.resolution
            ft = getattr(self._trainer.config, 'face_type', 'wf')
            self._model_status.setText(f"已加载: {mn} (res={res}, face={ft})")
            self._model_status.setStyleSheet("font-size: 11px; color: #0078D4;")
            self._do_render()
        except Exception as e:
            self._trainer = None
            self._model_status.setText(f"加载失败: {e}")
            self._model_status.setStyleSheet("font-size: 11px; color: #d32f2f;")
            QMessageBox.critical(self, "模型加载失败", str(e))

    def _collect_merge_config(self):
        from faceswap.business.model_merger import MergeConfig, MaskMode
        ct = self._ct_mode.currentText()
        enh = self._enhancer.currentText()
        return MergeConfig(
            mask_mode=MaskMode(self._mask_mode.currentText()),
            erode_mask_modifier=self._erode.value(),
            blur_mask_modifier=self._blur.value(),
            color_transfer=ct != "none",
            color_transfer_mode=ct,
            output_face_scale=self._face_scale.value(),
            enhancer_model="" if enh == "(无)" else enh,
            enhancer_blend=self._enhancer_blend.value(),
        )

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_W:
            self._erode.setValue(self._erode.value() + 1)
            self._on_param_changed()
        elif key == Qt.Key.Key_S:
            self._erode.setValue(self._erode.value() - 1)
            self._on_param_changed()
        elif key == Qt.Key.Key_E:
            self._blur.setValue(self._blur.value() + 1)
            self._on_param_changed()
        elif key == Qt.Key.Key_D:
            self._blur.setValue(self._blur.value() - 1)
            self._on_param_changed()
        elif key == Qt.Key.Key_T:
            self._face_scale.setValue(self._face_scale.value() + 1)
            self._on_param_changed()
        elif key == Qt.Key.Key_G:
            self._face_scale.setValue(self._face_scale.value() - 1)
            self._on_param_changed()
        else:
            super().keyPressEvent(event)

    def _on_param_changed(self):
        self._render_timer.start(300)

    def _on_frame_changed(self, idx):
        self._frame_idx = idx
        total = len(self._frames)
        name = self._frames[idx].name if 0 <= idx < total else "?"
        self._frame_info.setText(f"{idx + 1}/{total}  {name}")
        self._do_render()

    def _on_prev_frame(self):
        if self._frame_idx > 0:
            self._frame_slider.setValue(self._frame_idx - 1)

    def _on_next_frame(self):
        if self._frame_idx < len(self._frames) - 1:
            self._frame_slider.setValue(self._frame_idx + 1)

    def _do_render(self):
        if not self._frames:
            return
        idx = self._frame_idx
        if idx < 0 or idx >= len(self._frames):
            return
        frame_path = self._frames[idx]
        frame_img = self._cv2.imread(str(frame_path))
        if frame_img is None:
            return
        faces = self._frame_to_faces.get(frame_path.name, [])
        if self._trainer is not None and faces:
            try:
                from faceswap.business.model_merger import ModelMerger
                config = self._collect_merge_config()
                if not hasattr(self, '_merger') or self._merger is None:
                    self._merger = ModelMerger()
                result = self._merger.composite_single_frame(
                    frame_img, faces, self._trainer.predictor_func, config,
                    DATA_DST_ALIGNED_DIR,
                )
            except Exception as e:
                _logger.warning(f"Render failed: {e}")
                result = frame_img
                self._status_label.setText(f"渲染错误: {e}")
        else:
            result = frame_img
            if not faces:
                self._status_label.setText("此帧无人脸")
            elif self._trainer is None:
                self._status_label.setText("未加载模型（显示原图）")
        self._update_preview(result)

    def _update_preview(self, img):
        from PyQt6.QtGui import QImage, QPixmap
        self._current_img = img
        rgb = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self._preview_img_data = rgb
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label_size = self._preview_label.size()
        if label_size.width() > 10 and label_size.height() > 10:
            pixmap = pixmap.scaled(
                label_size, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._preview_label.setPixmap(pixmap)
        n_faces = len(self._frame_to_faces.get(self._frames[self._frame_idx].name, [])) if self._frames else 0
        self._status_label.setText(f"帧 {self._frame_idx + 1}/{len(self._frames)}  |  {n_faces} 人脸  |  {img.shape[1]}x{img.shape[0]}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current_img is not None:
            self._update_preview(self._current_img)

    def _on_merge_all(self):
        if not self._frames:
            QMessageBox.warning(self, "无帧", "未找到目标帧。请先完成视频提取。")
            return
        self._stop_requested = False
        self._merge_all_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._batch_progress.setVisible(True)
        self._batch_progress.setRange(0, len(self._frames))
        sig = _ProgressSignal()
        sig.done_ready.connect(self._on_merge_done)
        sig.error_ready.connect(self._on_merge_error)
        sig.progress_ready.connect(lambda t: self._batch_progress.setFormat(t))

        def _task():
            try:
                import cv2
                from faceswap.shared.file_manager import imwrite_auto
                from faceswap.business.model_merger import ModelMerger
                config = self._collect_merge_config()
                merger = ModelMerger()
                DATA_DST_MERGED_DIR.mkdir(parents=True, exist_ok=True)
                count = 0
                for i, fp in enumerate(self._frames):
                    if self._stop_requested:
                        break
                    frame_img = cv2.imread(str(fp))
                    if frame_img is None:
                        continue
                    faces = self._frame_to_faces.get(fp.name, [])
                    if self._trainer is not None and faces:
                        result = merger.composite_single_frame(
                            frame_img, faces, self._trainer.predictor_func, config,
                            DATA_DST_ALIGNED_DIR,
                        )
                    else:
                        result = frame_img
                    imwrite_auto(DATA_DST_MERGED_DIR / fp.name, result)
                    count += 1
                    sig.progress_ready.emit(f"{i + 1}/{len(self._frames)}")
                sig.done_ready.emit("完成", f"已合成 {count} 帧到 {DATA_DST_MERGED_DIR}")
            except Exception as e:
                sig.error_ready.emit("错误", str(e))

        self._thread = threading.Thread(target=_task, daemon=True)
        self._thread.start()

    def _on_merge_done(self, title, msg):
        self._merge_all_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._batch_progress.setVisible(False)
        QMessageBox.information(self, title, msg)

    def _on_merge_error(self, title, msg):
        self._merge_all_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._batch_progress.setVisible(False)
        QMessageBox.critical(self, title, msg)


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
        self._stop_btn.setProperty("danger", True)
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
                                   stream_callback=_stream_cb)
                sig.log_signal.emit("合成完毕", False)
            except Exception as e:
                sig.log_signal.emit(f"错误: {e}", False)
            finally:
                sig.done_signal.emit()

        self._thread = threading.Thread(target=_task, daemon=True)
        self._thread.start()


class Step7IFTrain(StepPanel):
    step_title = "7. IF训练"
    step_desc = ("同时训练SCRFD检测器和106点landmark模型，导出ONNX替换insightface预训练权重。\n"
                 "默认学习率0.001适合微调（勾选预训练权重）。从零训练时请调大至0.1。")
    show_run_buttons = False

    def _build_params(self):
        self._batch_size = self._add_spin("批次大小:", 1, 256, 32, 1)
        self._lr = self._add_dspin("学习率:", 0.0001, 1.0, 0.001, 0.001, decimals=4)
        self._max_epochs = self._add_spin("最大轮数:", 1, 300, 30, 1)
        self._augment = self._add_check("数据增强", True)
        self._train_scrfd = self._add_check("同时训练SCRFD检测器", True)
        self._load_lm_pretrained = self._add_check("加载Landmark预训练权重", True)
        self._load_pretrained = self._add_check("加载SCRFD预训练权重", True)

        btn_row = QHBoxLayout()
        self._train_btn = QPushButton("开始训练")
        self._train_btn.setFixedWidth(120)
        self._train_btn.setStyleSheet("QPushButton { background-color: #0078D4; color: white; font-weight: bold; }")
        self._train_btn.clicked.connect(self._on_train)
        btn_row.addWidget(self._train_btn)

        self._stop_btn = QPushButton("停止")
        self._stop_btn.setFixedWidth(80)
        self._stop_btn.setProperty("danger", True)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)

        self._lm_preview_btn = QPushButton("Landmark预览")
        self._lm_preview_btn.setFixedWidth(110)
        self._lm_preview_btn.setProperty("outline", True)
        self._lm_preview_btn.clicked.connect(self._on_lm_preview)
        btn_row.addWidget(self._lm_preview_btn)

        self._scrfd_preview_btn = QPushButton("SCRFD预览")
        self._scrfd_preview_btn.setFixedWidth(100)
        self._scrfd_preview_btn.setProperty("outline", True)
        self._scrfd_preview_btn.clicked.connect(self._on_scrfd_preview)
        btn_row.addWidget(self._scrfd_preview_btn)

        self._export_btn = QPushButton("导出ONNX")
        self._export_btn.setFixedWidth(100)
        self._export_btn.setProperty("outline", True)
        self._export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(self._export_btn)

        btn_row.addStretch()
        self._params_area.addLayout(btn_row)

        self._status_label = QLabel("就绪")
        self._status_label.setStyleSheet("font-size: 12px; color: #666666;")
        self._params_area.addWidget(self._status_label)

        lm_chart_label = QLabel("Landmark损失曲线:")
        lm_chart_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-top: 8px;")
        self._params_area.addWidget(lm_chart_label)

        self._lm_chart_label = QLabel()
        self._lm_chart_label.setMinimumHeight(150)
        self._lm_chart_label.setStyleSheet("border: 1px solid #ccc; background-color: #1e1e1e;")
        self._params_area.addWidget(self._lm_chart_label)

        self._scrfd_chart_label = QLabel()
        self._scrfd_chart_label.setMinimumHeight(150)
        self._scrfd_chart_label.setStyleSheet("border: 1px solid #ccc; background-color: #1e1e1e;")
        self._scrfd_chart_group = QWidget()
        scrfd_chart_layout = QVBoxLayout(self._scrfd_chart_group)
        scrfd_chart_label = QLabel("SCRFD损失曲线:")
        scrfd_chart_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-top: 4px;")
        scrfd_chart_layout.addWidget(scrfd_chart_label)
        scrfd_chart_layout.addWidget(self._scrfd_chart_label)
        self._params_area.addWidget(self._scrfd_chart_group)

        log_label = QLabel("训练日志:")
        log_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-top: 8px;")
        self._params_area.addWidget(log_label)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(150)
        self._log_text.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #cccccc; "
            "font-family: Consolas, monospace; font-size: 11px; }")
        self._params_area.addWidget(self._log_text)

        self._lm_trainer = None
        self._scrfd_trainer = None
        self._lm_thread = None
        self._scrfd_thread = None

        self._lm_signals = _ProgressSignal()
        self._lm_signals.progress_ready.connect(self._on_lm_epoch)
        self._lm_signals.error_ready.connect(self._on_lm_error)
        self._lm_signals.done_ready.connect(self._on_lm_done)
        self._lm_preview_signal = _PreviewSignal()
        self._lm_preview_signal.preview_ready.connect(self._show_lm_preview)

        self._scrfd_signals = _ProgressSignal()
        self._scrfd_signals.progress_ready.connect(self._on_scrfd_epoch)
        self._scrfd_signals.error_ready.connect(self._on_scrfd_error)
        self._scrfd_signals.done_ready.connect(self._on_scrfd_done)
        self._scrfd_preview_signal = _PreviewSignal()
        self._scrfd_preview_signal.preview_ready.connect(self._show_scrfd_preview)

        self._lm_preview_dialog = None
        self._scrfd_preview_dialog = None
        self._train_scrfd_checked = True

    def _on_train(self):
        data_dir = INSIGHTFACE_MANUAL_ANNOTATED_DIR
        if not data_dir.exists():
            QMessageBox.warning(self, "警告", f"数据目录不存在:\n{data_dir}")
            return

        image_count = len(list(data_dir.glob("*.jpg"))) + len(list(data_dir.glob("*.png")))
        if image_count == 0:
            QMessageBox.warning(self, "警告", f"数据目录中没有图像:\n{data_dir}")
            return

        lm_model_dir = IF_LANDMARK_MODEL_DIR
        lm_model_dir.mkdir(parents=True, exist_ok=True)

        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._export_btn.setEnabled(False)
        self._status_label.setText("训练中...")
        self._log_text.clear()

        batch_size = self._batch_size.value()
        lr = self._lr.value()
        max_epochs = self._max_epochs.value()
        augment = self._augment.isChecked()
        self._train_scrfd_checked = self._train_scrfd.isChecked()

        self._scrfd_chart_group.setVisible(self._train_scrfd_checked)

        from faceswap.business.if_landmark_trainer import IFLandmarkTrainer
        self._lm_trainer = IFLandmarkTrainer(device="auto")

        def _lm_on_epoch(epoch: int, loss: float, lr_val: float):
            self._lm_signals.progress_ready.emit(
                f"[LM] Epoch {epoch}/{max_epochs}  loss={loss:.6f}  lr={lr_val:.6f}")

        def _lm_on_preview(img: np.ndarray):
            self._lm_preview_signal.preview_ready.emit(img)

        def _lm_on_save(epoch: int):
            pass

        def _lm_task():
            try:
                lm_pretrained = None
                if self._load_lm_pretrained.isChecked():
                    from faceswap.setting import INSIGHTFACE_MODEL_DIR, INSIGHTFACE_MODEL_PACKAGE
                    lm_pretrained = str(INSIGHTFACE_MODEL_DIR / "models" / INSIGHTFACE_MODEL_PACKAGE / "2d106det.onnx")
                self._lm_trainer.train(
                    data_dir=data_dir,
                    model_dir=lm_model_dir,
                    batch_size=batch_size,
                    learning_rate=lr,
                    max_epochs=max_epochs,
                    augment=augment,
                    pretrained_onnx=lm_pretrained,
                    on_epoch=_lm_on_epoch,
                    on_preview=_lm_on_preview,
                    on_save=_lm_on_save,
                )
                self._lm_signals.done_ready.emit("Landmark训练完成", "")
            except Exception as e:
                self._lm_signals.error_ready.emit("Landmark训练错误", str(e))

        self._lm_thread = threading.Thread(target=_lm_task, daemon=True)
        self._lm_thread.start()

        if self._train_scrfd_checked:
            scrfd_model_dir = SCRFD_MODEL_DIR
            scrfd_model_dir.mkdir(parents=True, exist_ok=True)

            from faceswap.business.scrfd_trainer import SCRFDTrainer
            self._scrfd_trainer = SCRFDTrainer(device="auto")

            def _scrfd_on_epoch(epoch: int, loss: float, lr_val: float):
                self._scrfd_signals.progress_ready.emit(
                    f"[SCRFD] Epoch {epoch}/{max_epochs}  loss={loss:.6f}  lr={lr_val:.6f}")

            def _scrfd_on_preview(img: np.ndarray):
                self._scrfd_preview_signal.preview_ready.emit(img)

            def _scrfd_on_save(epoch: int):
                pass

            def _scrfd_task():
                try:
                    pretrained_path = None
                    if self._load_pretrained.isChecked():
                        from faceswap.setting import INSIGHTFACE_MODEL_DIR, INSIGHTFACE_MODEL_PACKAGE
                        pretrained_path = str(INSIGHTFACE_MODEL_DIR / "models" / INSIGHTFACE_MODEL_PACKAGE / "scrfd_10g_bnkps.onnx")
                    self._scrfd_trainer.train(
                        data_dir=data_dir,
                        model_dir=scrfd_model_dir,
                        batch_size=min(batch_size, 8),
                        learning_rate=lr,
                        max_epochs=max_epochs,
                        augment=augment,
                        pretrained_onnx=pretrained_path,
                        on_epoch=_scrfd_on_epoch,
                        on_preview=_scrfd_on_preview,
                        on_save=_scrfd_on_save,
                    )
                    self._scrfd_signals.done_ready.emit("SCRFD训练完成", "")
                except Exception as e:
                    self._scrfd_signals.error_ready.emit("SCRFD训练错误", str(e))

            self._scrfd_thread = threading.Thread(target=_scrfd_task, daemon=True)
            self._scrfd_thread.start()

    def _on_stop(self):
        if self._lm_trainer is not None:
            self._lm_trainer.request_stop()
        if self._scrfd_trainer is not None:
            self._scrfd_trainer.request_stop()
        self._stop_btn.setEnabled(False)
        self._status_label.setText("正在停止...")

    def _on_lm_preview(self):
        if self._lm_trainer is not None:
            self._lm_trainer.request_preview()

    def _on_scrfd_preview(self):
        if self._scrfd_trainer is not None:
            self._scrfd_trainer.request_preview()

    def _on_export(self):
        lm_pth = IF_LANDMARK_MODEL_DIR / "if_net.pth"
        if lm_pth.exists():
            from faceswap.models.if_landmark.if_landmark_arch import IFLandmarkNet
            import torch
            net = IFLandmarkNet()
            state = torch.load(lm_pth, map_location="cpu", weights_only=False)
            net.load_state_dict(state)
            onnx_path = IF_LANDMARK_MODEL_DIR / "if_landmark_2d106.onnx"
            net.export_onnx(str(onnx_path))
            self._log_text.append(f"[导出] Landmark ONNX: {onnx_path}")

        scrfd_pth = SCRFD_MODEL_DIR / "scrfd_net.pth"
        if scrfd_pth.exists():
            from faceswap.models.scrfd.scrfd_arch import SCRFDNet
            import torch
            net = SCRFDNet()
            state = torch.load(scrfd_pth, map_location="cpu", weights_only=False)
            net.load_state_dict(state)
            onnx_path = SCRFD_MODEL_DIR / "scrfd_custom.onnx"
            net.export_onnx(str(onnx_path), input_size=640)
            self._log_text.append(f"[导出] SCRFD ONNX: {onnx_path}")

        if not lm_pth.exists() and not scrfd_pth.exists():
            QMessageBox.warning(self, "警告", "模型文件不存在，请先训练。")
            return
        QMessageBox.information(self, "导出完成", "ONNX模型已导出。")

    def _on_lm_epoch(self, msg: str):
        self._log_text.append(msg)
        if self._lm_trainer is not None:
            chart = self._lm_trainer.generate_loss_chart()
            from PyQt6.QtGui import QImage, QPixmap
            h, w = chart.shape[:2]
            qimg = QImage(chart.data, w, h, w * 3, QImage.Format.Format_BGR888)
            self._lm_chart_label.setPixmap(QPixmap.fromImage(qimg))
        self._update_status()

    def _on_scrfd_epoch(self, msg: str):
        self._log_text.append(msg)
        if self._scrfd_trainer is not None:
            chart = self._scrfd_trainer.generate_loss_chart()
            from PyQt6.QtGui import QImage, QPixmap
            h, w = chart.shape[:2]
            qimg = QImage(chart.data, w, h, w * 3, QImage.Format.Format_BGR888)
            self._scrfd_chart_label.setPixmap(QPixmap.fromImage(qimg))
        self._update_status()

    def _update_status(self):
        lm_alive = self._lm_thread is not None and self._lm_thread.is_alive()
        scrfd_alive = self._scrfd_thread is not None and self._scrfd_thread.is_alive()
        if lm_alive and scrfd_alive:
            self._status_label.setText("训练中: Landmark + SCRFD")
        elif lm_alive:
            self._status_label.setText("训练中: Landmark")
        elif scrfd_alive:
            self._status_label.setText("训练中: SCRFD")
        elif self._stop_btn.isEnabled():
            self._status_label.setText("训练完成")

    def _on_lm_error(self, title: str, msg: str):
        self._log_text.append(f"[错误] {title}: {msg}")
        self._check_all_done()

    def _on_scrfd_error(self, title: str, msg: str):
        self._log_text.append(f"[错误] {title}: {msg}")
        self._check_all_done()

    def _on_lm_done(self, title: str, msg: str):
        self._log_text.append(f"[完成] {title}")
        self._check_all_done()

    def _on_scrfd_done(self, title: str, msg: str):
        self._log_text.append(f"[完成] {title}")
        self._check_all_done()

    def _check_all_done(self):
        lm_done = self._lm_thread is None or not self._lm_thread.is_alive()
        scrfd_done = (self._scrfd_thread is None or not self._scrfd_thread.is_alive()) if self._train_scrfd_checked else True
        if lm_done and scrfd_done:
            self._train_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._export_btn.setEnabled(True)
            self._status_label.setText("训练完成")

    def _show_lm_preview(self, img: np.ndarray):
        if self._lm_preview_dialog is None:
            self._lm_preview_dialog = QDialog(self)
            self._lm_preview_dialog.setWindowTitle("Landmark 预览")
            self._lm_preview_dialog.setMinimumSize(800, 600)
            layout = QVBoxLayout(self._lm_preview_dialog)
            self._lm_preview_image_label = QLabel()
            layout.addWidget(self._lm_preview_image_label)
            hint = QLabel("绿色=预测  红色=标注  [P]刷新  [Esc]关闭")
            hint.setStyleSheet("color: #666666; font-size: 11px;")
            layout.addWidget(hint)
            self._lm_preview_dialog.keyPressEvent = self._lm_preview_key_press

        from PyQt6.QtGui import QImage, QPixmap
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_BGR888)
        self._lm_preview_image_label.setPixmap(QPixmap.fromImage(qimg))
        if not self._lm_preview_dialog.isVisible():
            self._lm_preview_dialog.show()

    def _lm_preview_key_press(self, event):
        from PyQt6.QtCore import Qt
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self._lm_preview_dialog.close()
        elif key == Qt.Key.Key_P:
            if self._lm_trainer is not None:
                self._lm_trainer.request_preview()
        else:
            QDialog.keyPressEvent(self._lm_preview_dialog, event)

    def _show_scrfd_preview(self, img: np.ndarray):
        if self._scrfd_preview_dialog is None:
            self._scrfd_preview_dialog = QDialog(self)
            self._scrfd_preview_dialog.setWindowTitle("SCRFD 预览")
            self._scrfd_preview_dialog.setMinimumSize(800, 600)
            layout = QVBoxLayout(self._scrfd_preview_dialog)
            self._scrfd_preview_image_label = QLabel()
            layout.addWidget(self._scrfd_preview_image_label)
            hint = QLabel("绿色=标注  红色=检测  [P]刷新  [Esc]关闭")
            hint.setStyleSheet("color: #666666; font-size: 11px;")
            layout.addWidget(hint)
            self._scrfd_preview_dialog.keyPressEvent = self._scrfd_preview_key_press

        from PyQt6.QtGui import QImage, QPixmap
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_BGR888)
        self._scrfd_preview_image_label.setPixmap(QPixmap.fromImage(qimg))
        if not self._scrfd_preview_dialog.isVisible():
            self._scrfd_preview_dialog.show()

    def _scrfd_preview_key_press(self, event):
        from PyQt6.QtCore import Qt
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self._scrfd_preview_dialog.close()
        elif key == Qt.Key.Key_P:
            if self._scrfd_trainer is not None:
                self._scrfd_trainer.request_preview()
        else:
            QDialog.keyPressEvent(self._scrfd_preview_dialog, event)


class Step8Workspace(StepPanel):
    step_title = "8. 工作区"
    step_desc = "清理工作区、重建目录结构。"

    def _build_params(self):
        pass

    def _on_run(self):
        from faceswap.business.workspace_manager import WorkspaceManager
        ws = WorkspaceManager()

        def _task():
            ws.clear_workspace()

        self._run_in_thread(_task)
