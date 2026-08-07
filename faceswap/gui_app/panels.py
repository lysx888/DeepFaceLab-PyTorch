import json
import sys
import threading
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QObject, QMetaObject, Q_ARG
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton, QLineEdit,
    QGroupBox, QProgressBar, QFileDialog, QScrollArea, QTextEdit,
    QMessageBox, QDialog,
)

from faceswap.shared.image_utils import bgr_to_rgb
from faceswap.gui_app.gui_log import gui_log, gui_error
from faceswap.gui_app.gui_utils import install_no_wheel
from faceswap.gui_app.param_defs import ParamGroup
from faceswap.setting import (
    WORKSPACE_DIR, MODEL_DIR,
    DATA_SRC_DIR, DATA_DST_DIR,
    DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR,
    DATA_DST_SWAPPED_DIR, DATA_DST_MERGED_DIR, DATA_DST_MERGED_MASK_DIR,
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
                gui_error(str(e))

        self._worker = _Worker(_wrapped)
        self._worker.finished.connect(lambda: self._set_running(False))
        self._worker.error.connect(lambda e: gui_error(e))
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
                gui_error(str(e))
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
                gui_error(str(e))
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
                 check_aligned=None, manual_annotate_cb=None, xseg_train_cb=None, parent=None):
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
        btn_row.addWidget(self._run_btn, 4)
        btn_row.addWidget(self._preview_btn, 4)
        if manual_annotate_cb is not None:
            manual_btn = QPushButton("手动标注")
            manual_btn.setStyleSheet("QPushButton { background-color: #5B2D8E; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
            manual_btn.clicked.connect(manual_annotate_cb)
            btn_row.addWidget(manual_btn, 1)
        if xseg_train_cb is not None:
            xseg_train_btn = QPushButton("IF训练")
            xseg_train_btn.setStyleSheet("QPushButton { background-color: #D45500; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
            xseg_train_btn.clicked.connect(xseg_train_cb)
            btn_row.addWidget(xseg_train_btn, 1)
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
                gui_error(str(e))
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
    "full": "384",
    "mid_full": "256",
    "half": "128",
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
        self._face_type.addItems(["whole_face", "head", "full", "mid_full", "half"])
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
            manual_annotate_cb=self._on_manual_annotate,
            xseg_train_cb=self._on_goto_if_train)
        self._dst_half = _FaceExtractHalf(
            "目标人脸 (DST)", is_src=False, get_config=self._get_config,
            progress_callback=self._progress.on_progress,
            check_aligned=self._check_aligned_dir)
        row.addWidget(self._src_half)
        row.addWidget(self._dst_half)
        self._params_area.addLayout(row)

        self._params_area.addWidget(self._progress)

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
        del_face_btn = QPushButton("删除头像")
        del_face_btn.setMinimumWidth(70)
        del_face_btn.setStyleSheet(
            "QPushButton { background-color: #D45500; color: white; font-weight: bold; padding: 4px 10px; border-radius: 3px; }")
        del_face_btn.clicked.connect(self._on_delete_faces)
        target_row.addWidget(del_face_btn)
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

        rename_grp = QGroupBox("批量重命名")
        rename_lay = QHBoxLayout(rename_grp)
        rename_lay.setContentsMargins(8, 14, 8, 4)
        rename_lay.setSpacing(8)
        rename_btn = QPushButton("重命名源人脸")
        rename_btn.setMinimumWidth(100)
        rename_btn.clicked.connect(self._on_rename)
        rename_lay.addWidget(rename_btn)
        rename_desc = QLabel("将源人脸文件从 00001_0 开始连续编号")
        rename_desc.setStyleSheet("color: #666; font-size: 11px;")
        rename_lay.addWidget(rename_desc)
        rename_lay.addStretch()
        self._params_area.addWidget(rename_grp)

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

    def _on_delete_faces(self):
        from faceswap.gui_app.face_delete_dialog import FaceDeleteDialog
        from faceswap.shared.file_manager import FileManager
        target_dir = DATA_SRC_ALIGNED_DIR if self._sort_src_rb.isChecked() else DATA_DST_ALIGNED_DIR
        if not target_dir.exists() or not any(FileManager.find_images(target_dir)):
            QMessageBox.information(self, "提示", f"目录中没有图片:\n{target_dir}")
            return
        dlg = FaceDeleteDialog(target_dir, parent=self)
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

        ft_map = {"whole_face": FaceType.WHOLE_FACE, "head": FaceType.HEAD, "full": FaceType.FULL,
                  "mid_full": FaceType.MID_FULL, "half": FaceType.HALF}
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
        self._xseg_amp.setCurrentText("bf16")
        self._xseg_amp.setFixedWidth(75)
        _tip_amp = "混合精度模式：bf16=半精度(推荐，速度快显存省)，fp16=传统半精度，fp32=全精度(最慢最准)"
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
        config_path = Path(MODEL_DIR) / "XSeg_config.json"
        if not config_path.exists():
            return
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            self._xseg_face_type.setCurrentText(cfg.get("face_type", "wf"))
            self._xseg_batch.setValue(cfg.get("batch_size", 4))
            self._xseg_iters.setValue(cfg.get("target_iter", 100000))
            self._xseg_amp.setCurrentText(cfg.get("amp_mode", "bf16"))
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
            config_path = Path(MODEL_DIR) / "XSeg_config.json"
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
        if reply == QMessageBox.StandardButton.Yes:
            from faceswap.business.xseg_editor import XSegEditor
            editor = XSegEditor()
            editor.remove_annotations(d)

    def _on_train(self):
        from faceswap.business.xseg_trainer import XSegTrainer
        self._xseg_trainer = XSegTrainer()
        self._set_xseg_controls_enabled(False)
        self._train_log.clear()

        self._log_throttle_time = 0.0
        self._log_throttle_interval = 0.5
        self._log_need_newline = True

        def _on_iter(iter_count, loss_val, iter_ms):
            import time as _time
            now = _time.time()
            if now - self._log_throttle_time < self._log_throttle_interval:
                return
            self._log_throttle_time = now
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            if iter_ms >= 1000:
                time_str = f"{iter_ms/1000:.1f}s"
            else:
                time_str = f"{int(iter_ms)}ms"
            line = f"[{ts}][#{iter_count}][{time_str}][loss {loss_val:.5f}]"

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
                    DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR, MODEL_DIR,
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
                count = trainer.apply_trained_mask(d, MODEL_DIR, progress_callback=_on_progress)
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
        from faceswap.setting import _PROJECT_ROOT
        d = self._get_apply_target_dir()
        generic_dir = _PROJECT_ROOT / "_internal" / "model_generic_xseg"
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
                count = trainer.apply_generic_mask(d, generic_dir, progress_callback=_on_progress)
                sig.progress_ready.emit(f"Done: {count} faces")
                sig.done_ready.emit("完成", f"已应用通用遮罩到 {count} 张人脸")
            except Exception as e:
                sig.progress_ready.emit(f"Error: {e}")
                sig.error_ready.emit("错误", str(e))

        self._run_in_thread(_task)



class Step4Train(StepPanel):
    step_title = "4. 训练"
    step_desc = "SAEHD换脸模型训练：选择架构和参数，训练特定人物对模型。"
    show_run_buttons = False

    def _build_params(self):
        from faceswap.gui_app.param_defs import (
            ParamGroupWidget, ConfigManager,
            TrainingSignals, TrainingStatusBar,
        )
        from faceswap.gui_app.saehd_param_defs import get_saehd_params_by_group
        from faceswap.setting import SAEHD_MODEL_DIR

        self._param_groups: dict[ParamGroup, ParamGroupWidget] = {}
        for group in [ParamGroup.ARCHITECTURE, ParamGroup.BASIC, ParamGroup.FACE_DETAIL, ParamGroup.LOSS_SAMPLING, ParamGroup.OPTIMIZATION]:
            pw = ParamGroupWidget(group, get_saehd_params_by_group(group))
            self._param_groups[group] = pw
            self._params_area.addWidget(pw)

        self._config_manager = ConfigManager(SAEHD_MODEL_DIR, "SAEHD_training_config.json")
        saved = self._config_manager.load_config()
        if saved:
            for pw in self._param_groups.values():
                pw.set_values(saved)
            pretrain_val = saved.get("pretrain", None)
            if pretrain_val is False:
                basic_pw = self._param_groups.get(ParamGroup.BASIC)
                if basic_pw and "pretrain" in basic_pw._param_widgets:
                    pretrain_w = basic_pw._param_widgets["pretrain"]
                    pretrain_w.setEnabled(False)
                    pretrain_w.setToolTip("已退出预训练模式，不可回退")
            optimizer_val = saved.get("optimizer", None)
            if optimizer_val == "adabelief":
                basic_pw = self._param_groups.get(ParamGroup.BASIC)
                if basic_pw and "optimizer" in basic_pw._param_widgets:
                    opt_w = basic_pw._param_widgets["optimizer"]
                    opt_w.setEnabled(False)
                    opt_w.setToolTip("AdaBelief一旦启用不可关闭 (参考DFL官方建议)")

            self._lock_architecture_params()

        self._link_face_type_resolution()
        self._link_archi_true_face_power()

        self._ddp_check = QCheckBox("DDP多卡训练")
        self._ddp_check.setToolTip("启用DistributedDataParallel多GPU训练（需2+张NVIDIA GPU）")
        self._ddp_check.setEnabled(False)
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                self._ddp_check.setEnabled(True)
                self._ddp_check.setChecked(True)
        except Exception:
            pass

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._ddp_check)
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

    def _collect_params(self) -> dict:
        params = {}
        for pw in self._param_groups.values():
            params.update(pw.get_values())
        return params

    def _link_face_type_resolution(self):
        ft_res = {'h': 64, 'mf': 128, 'f': 128, 'wf': 256, 'head': 384}
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

    def _link_archi_true_face_power(self):
        from faceswap.gui_app.param_defs import ArchiButtonGroup
        arch_pw = self._param_groups.get(ParamGroup.ARCHITECTURE)
        loss_pw = self._param_groups.get(ParamGroup.LOSS_SAMPLING)
        basic_pw = self._param_groups.get(ParamGroup.BASIC)
        if arch_pw is None or loss_pw is None or basic_pw is None:
            return
        arch_w = arch_pw._param_widgets.get('archi')
        tfp_w = loss_pw._param_widgets.get('true_face_power')
        hsv_w = basic_pw._param_widgets.get('random_hsv_power')
        ct_w = basic_pw._param_widgets.get('ct_mode')
        if arch_w is None:
            return

        def _update_archi_link(archi_str: str):
            is_df = archi_str.startswith('df')
            is_liae = archi_str.startswith('liae')
            if tfp_w is not None:
                tfp_w.setEnabled(is_df)
                if not is_df:
                    tfp_w.setValue(0.0)
            if hsv_w is not None:
                hsv_w.setEnabled(is_df)
                if is_liae:
                    hsv_w.setValue(0.0)
            if ct_w is not None:
                ct_w.setEnabled(is_df)
                if is_liae:
                    ct_w.setCurrentText('none')

        if isinstance(arch_w, ArchiButtonGroup):
            arch_w.value_changed.connect(_update_archi_link)
            _update_archi_link(arch_w.value())

    def _lock_architecture_params(self):
        _lock_keys = {
            ParamGroup.BASIC: ['resolution', 'face_type', 'optimizer', 'lr', 'batch_size', 'amp_mode', 'gradient_checkpointing'],
            ParamGroup.ARCHITECTURE: ['archi', 'ae_dims', 'e_dims', 'd_dims', 'd_mask_dims'],
            ParamGroup.LOSS_SAMPLING: ['gan_patch_size', 'gan_dims'],
            ParamGroup.OPTIMIZATION: ['enable_torch_compile', 'use_ms_ssim'],
        }
        _lock_tooltip = "模型已初始化，此参数不可修改（改变会导致模型结构或计算方式不兼容）"
        for group, keys in _lock_keys.items():
            pw = self._param_groups.get(group)
            if pw is None:
                continue
            for key in keys:
                w = pw._param_widgets.get(key)
                if w is not None:
                    w.setEnabled(False)
                    w.setToolTip(_lock_tooltip)

    def _on_run(self):
        from faceswap.business.saehd_trainer import SAEHDTrainer, TrainingConfig
        from faceswap.setting import DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR, SAEHD_MODEL_DIR
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
        self._config_manager.save_config(params)

        config = TrainingConfig(**params)

        self._set_controls_enabled(False)
        self._status_bar.start_pulse()
        self._log_text.clear()
        self._log_need_newline = True
        self._log_throttle_time = 0.0

        signals = self._signals
        panel = self

        def _on_progress(iter_num, src_loss, dst_loss, iter_ms, lr, converged=False):
            if iter_num == -1:
                signals.log_signal.emit("[...] 正在保存模型...", False)
                panel._log_need_newline = False
                return
            now = _time.time()
            if now - panel._log_throttle_time < panel._log_throttle_interval:
                return
            panel._log_throttle_time = now
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            loss_val = (src_loss + dst_loss) / 2
            if iter_ms >= 1000:
                time_str = f"{iter_ms/1000:.1f}s"
            else:
                time_str = f"{int(iter_ms)}ms"
            line = f"[{ts}][#{iter_num}][{time_str}][src {src_loss:.5f} dst {dst_loss:.5f}][loss {loss_val:.5f}]"

            overwrite = not panel._log_need_newline
            panel._log_need_newline = False
            signals.log_signal.emit(line, overwrite)
            signals.iter_signal.emit(iter_num, loss_val, iter_ms)

        def _on_preview(preview_bgr):
            panel._preview_signal.preview_ready.emit(preview_bgr)

        def _task():
            try:
                trainer = SAEHDTrainer(
                    config=config,
                    model_dir=SAEHD_MODEL_DIR,
                    src_aligned_dir=DATA_SRC_ALIGNED_DIR,
                    dst_aligned_dir=DATA_DST_ALIGNED_DIR,
                    progress_callback=_on_progress,
                    preview_callback=_on_preview,
                )
                if panel._ddp_check.isChecked() and panel._ddp_check.isEnabled():
                    import torch
                    world_size = torch.cuda.device_count()
                    trainer.enable_ddp(world_size)
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

        self._training_thread = threading.Thread(target=_task, daemon=True)
        self._training_thread.start()

    def _on_stop(self):
        self._request_training_stop(self._trainer, "stop")
        self._stop_btn.setEnabled(False)

    def _set_controls_enabled(self, enabled: bool):
        self._run_btn.setEnabled(enabled)
        self._stop_btn.setEnabled(not enabled)
        if hasattr(self, '_ddp_check'):
            if enabled:
                try:
                    import torch
                    self._ddp_check.setEnabled(torch.cuda.is_available() and torch.cuda.device_count() > 1)
                except Exception:
                    self._ddp_check.setEnabled(False)
            else:
                self._ddp_check.setEnabled(False)
        for pw in self._param_groups.values():
            pw.set_editable(enabled)
        if enabled:
            saved = self._config_manager.load_config()
            if saved and saved.get("pretrain") is False:
                basic_pw = self._param_groups.get(ParamGroup.BASIC)
                if basic_pw and "pretrain" in basic_pw._param_widgets:
                    pretrain_w = basic_pw._param_widgets["pretrain"]
                    pretrain_w.setEnabled(False)
                    pretrain_w.setToolTip("已退出预训练模式，不可回退")

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
            self._preview_win.setWindowTitle("SAEHD 训练预览 | [space]:切换section [p]:刷新 [s]:保存 [l]:Loss范围 [Enter]:保存并停止")
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
    step_desc = "将替换后的人脸对齐回原始帧，使用遮罩混合，生成完整的合成帧。"

    def _build_params(self):
        self._mask_mode = self._add_combo("遮罩模式:", ["xseg", "dst", "learned"], "xseg")
        self._erode = self._add_dspin("遮罩侵蚀:", -400, 400, 0, 5, 0)
        self._blur = self._add_dspin("遮罩模糊:", 0, 400, 20, 5, 0)
        self._color_transfer = self._add_check("多尺度颜色均衡", True)
        self._enhancer_model = self._add_combo("人脸增强:", ["(无)", "gfpgan_1.4", "gpen_bfr_512", "gpen_bfr_1024", "restoreformer_pp"], "(无)")
        self._enhancer_blend = self._add_spin("增强混合:", 0, 100, 80, 5)

    def _on_run(self):
        from faceswap.business.model_merger import ModelMerger, MergeConfig, MaskMode

        mask_mode = MaskMode(self._mask_mode.currentText())
        enhancer_name = self._enhancer_model.currentText()
        config = MergeConfig(
            mask_mode=mask_mode,
            erode_mask_modifier=self._erode.value(),
            blur_mask_modifier=self._blur.value(),
            color_transfer=self._color_transfer.isChecked(),
            enhancer_model="" if enhancer_name == "(无)" else enhancer_name,
            enhancer_blend=self._enhancer_blend.value(),
        )
        merger = ModelMerger()
        sig = _ProgressSignal()
        sig.done_ready.connect(lambda t, m: (self._update_progress(0, 0, ""), QMessageBox.information(self, t, m)))
        sig.error_ready.connect(lambda t, m: (self._update_progress(0, 0, ""), QMessageBox.critical(self, t, m)))

        def _on_progress(current, total, elapsed, remaining, speed):
            if total > 0:
                pct = int(current / total * 100)
                m_e = int(elapsed) // 60
                s_e = int(elapsed) % 60
                m_r = int(remaining) // 60 if remaining > 0 else 0
                s_r = int(remaining) % 60 if remaining > 0 else 0
                text = f"合成: {pct}% | {current}/{total} [{m_e}:{s_e:02d}<{m_r}:{s_r:02d}, {speed}]"
                sig.progress_ready.emit(text)
        sig.progress_ready.connect(lambda t: self._update_progress(0, 0, t))

        def _task():
            try:
                count = merger.composite_to_frames(
                    DATA_DST_DIR, DATA_DST_ALIGNED_DIR,
                    DATA_DST_SWAPPED_DIR, DATA_DST_MERGED_DIR,
                    config, progress_callback=_on_progress,
                    stop_check=lambda: self._stop_requested,
                )
                if self._stop_requested:
                    sig.done_ready.emit("已停止", f"已合成 {count} 帧（用户停止）")
                else:
                    sig.done_ready.emit("完成", f"已合成 {count} 帧到 {DATA_DST_MERGED_DIR}")
            except Exception as e:
                sig.error_ready.emit("错误", str(e))

        self._run_in_thread(_task)


class Step6Output(StepPanel):
    step_title = "6. 导出视频"
    step_desc = "将合成后的帧转换为带音频的输出视频。"

    def _build_params(self):
        self._fmt = self._add_combo("输出格式:", ["mp4", "avi", "mov"], "mp4")
        self._lossless = self._add_check("无损输出", False)
        self._out_fps = self._add_spin("输出帧率:", 0, 120, 0, 1, tooltip="0=自动检测(从源视频)")

    def _on_run(self):
        from faceswap.business.video_output import VideoOutput, OutputFormat
        from faceswap.business.video_processor import VideoProcessor
        from faceswap.business.workspace_manager import WorkspaceManager

        vp = VideoProcessor()
        vo = VideoOutput(vp)
        ws = WorkspaceManager()
        fmt = OutputFormat(self._fmt.currentText())
        ref = ws.find_dst_video()
        output_path = WORKSPACE_DIR / f"result.{fmt.value}"
        override_fps = self._out_fps.value() or None

        def _task():
            vo.merged_to_video(DATA_DST_MERGED_DIR, output_path, fmt,
                               reference_video=ref, include_audio=True,
                               lossless=self._lossless.isChecked(),
                               override_fps=override_fps)

        self._run_in_thread(_task)


class Step7Tools(StepPanel):
    step_title = "7. 人脸工具"
    step_desc = "手动标注、增强、打包/解包、缩放、元数据管理、关键点调试。"
    show_run_buttons = False

    def _build_params(self):
        self._tool = self._add_combo("工具:", ["手动标注", "增强", "打包", "解包", "保存元数据",
                                                "恢复元数据", "缩放", "关键点调试"], "手动标注")
        self._target = self._add_combo("目标:", ["源人脸（data_src）", "目标人脸（data_dst）"], "源人脸（data_src）")
        self._resize_size = self._add_spin("缩放尺寸:", 64, 2048, 256, 128)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("执行")
        self._run_btn.setFixedWidth(120)
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch()
        self._params_area.addLayout(btn_row)

    def _on_run(self):
        t = self._tool.currentText()

        if t == "手动标注":
            from faceswap.gui_app.manual_annotator import ManualAnnotatorDialog
            is_src = self._target.currentIndex() == 0
            dlg = ManualAnnotatorDialog(self, is_src=is_src)
            dlg.exec()
            return

        from faceswap.business.face_tool import FaceTool

        tool = FaceTool()
        d = DATA_SRC_ALIGNED_DIR if self._target.currentIndex() == 0 else DATA_DST_ALIGNED_DIR

        def _task():
            if t == "增强":
                tool.enhance(d)
            elif t == "打包":
                tool.pack(d)
            elif t == "解包":
                tool.unpack(d / "faceset.pak", d)
            elif t == "保存元数据":
                tool.metadata_save(d)
            elif t == "恢复元数据":
                tool.metadata_restore(d)
            elif t == "缩放":
                tool.resize(d, self._resize_size.value())
            elif t == "关键点调试":
                tool.add_landmarks_debug_images(d)

        self._run_in_thread(_task)


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
