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

from DeepFaceLab.gui_app.gui_log import gui_log, gui_error
from DeepFaceLab.gui_app.param_defs import ParamGroup
from DeepFaceLab.setting import (
    WORKSPACE_DIR, MODEL_DIR,
    DATA_SRC_DIR, DATA_DST_DIR,
    DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR,
    DATA_DST_MERGED_DIR, DATA_DST_MERGED_MASK_DIR,
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

        layout.addStretch()

    def _build_params(self):
        pass

    def _add_param_row(self, label_text: str, widget: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(180)
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

    def _add_spin(self, label: str, min_v: int, max_v: int, default: int, step: int = 1) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(min_v, max_v)
        sb.setValue(default)
        sb.setSingleStep(step)
        self._add_param_row(label, sb)
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

    def _on_run(self):
        pass

    def _on_stop(self):
        pass

    def _run_in_thread(self, fn, *args, **kwargs):
        self._set_running(True)

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
        from DeepFaceLab.business.video_processor import VideoProcessor
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
        from DeepFaceLab.business.video_processor import VideoProcessor
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
        from DeepFaceLab.gui_app.manual_annotator import DebugPreviewDialog
        dlg = DebugPreviewDialog(self._is_src, self)
        dlg.exec()

    def _on_run(self):
        if self._check_aligned:
            if not self._check_aligned(self._is_src):
                return

        from DeepFaceLab.business.face_extractor import FaceExtractor

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
            self._gpu_items += [f"[{i}] {torch.cuda.get_device_name(i)}" for i in range(self._num_gpus)]
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
        from DeepFaceLab.gui_app.manual_annotator import ManualAnnotatorDialog
        dlg = ManualAnnotatorDialog(self)
        dlg.exec()

    def _on_goto_if_train(self):
        w = self.window()
        if hasattr(w, '_switch'):
            w._switch(8)

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

    def _on_sort(self):
        from DeepFaceLab.business.face_sorter import FaceSorter, SortAlgorithm

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

        def _progress(current, total, elapsed):
            if total > 0:
                pct = int(current / total * 100)
                eta = elapsed / current * (total - current) if current > 0 else 0
                rate = current / elapsed if elapsed > 0 else 0
                sig.progress_ready.emit(f"Sorting: {pct}% | {current}/{total} [{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}it/s]")

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
        from DeepFaceLab.shared.file_manager import FileManager
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
            from DeepFaceLab.shared.file_manager import FileManager
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
        from DeepFaceLab.business.face_extractor import ExtractConfig

        gpu_ids = []
        if self._multi_gpu.isChecked():
            import torch
            if torch.cuda.is_available():
                gpu_ids = list(range(torch.cuda.device_count()))
        else:
            gpu_text = self._gpu.currentText()
            if gpu_text.startswith("["):
                try:
                    gpu_ids = [int(gpu_text[1:gpu_text.index("]")])]
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
        lbl3 = QLabel("批次大小:")
        self._xseg_batch = QSpinBox()
        self._xseg_batch.setRange(1, 64)
        self._xseg_batch.setValue(4)
        self._xseg_batch.setFixedWidth(60)
        lbl4 = QLabel("迭代次数:")
        self._xseg_iters = QSpinBox()
        self._xseg_iters.setRange(1000, 10000000)
        self._xseg_iters.setValue(100000)
        self._xseg_iters.setSingleStep(10000)
        self._xseg_iters.setFixedWidth(90)
        self._train_btn = QPushButton("开始训练")
        self._train_btn.setStyleSheet(
            "QPushButton { background-color: #D45500; color: white; font-weight: bold; padding: 6px 20px; border-radius: 4px; }"
        )
        self._train_btn.clicked.connect(self._on_train)
        self._stop_train_btn = QPushButton("停止训练")
        self._stop_train_btn.setProperty("danger", True)
        self._stop_train_btn.setEnabled(False)
        self._stop_train_btn.clicked.connect(self._on_stop_train)
        train_row.addWidget(lbl3)
        train_row.addWidget(self._xseg_batch)
        train_row.addSpacing(12)
        train_row.addWidget(lbl4)
        train_row.addWidget(self._xseg_iters)
        train_row.addStretch()
        train_row.addWidget(self._train_btn)
        train_row.addWidget(self._stop_train_btn)
        train_lay.addLayout(train_row)

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
    def _set_running(self, running: bool):
        self._running = running

    def _get_edit_target_dir(self) -> Path:
        return DATA_SRC_ALIGNED_DIR if self._edit_src_btn.isChecked() else DATA_DST_ALIGNED_DIR

    def _get_apply_target_dir(self) -> Path:
        return DATA_SRC_ALIGNED_DIR if self._apply_src_btn.isChecked() else DATA_DST_ALIGNED_DIR

    def _show_preview(self, preview_bgr: np.ndarray):
        import cv2
        from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout
        from PyQt6.QtGui import QImage, QPixmap
        from PyQt6.QtCore import Qt

        if self._preview_win is None:
            self._preview_win = QDialog(self)
            self._preview_win.setWindowTitle("XSeg 训练预览 | [space]:切换 [p]:刷新 [s]:保存 [l]:历史范围 [Enter]:保存并停止")
            self._preview_win.setWindowFlags(self._preview_win.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint)
            lay = QVBoxLayout(self._preview_win)
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(lbl)
            self._preview_win._label = lbl
            self._preview_win.resize(800, 900)
            self._preview_win.keyPressEvent = lambda e: self._on_preview_key(e)
            original_close = self._preview_win.close
            self._preview_win.closeEvent = lambda e: self._on_preview_close(e, original_close)
            self._preview_win.show()

        if not self._preview_win.isVisible():
            self._preview_win.show()

        rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self._preview_win._label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._preview_win._label.setPixmap(scaled)

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

    def _on_preview_key(self, event):
        from PyQt6.QtCore import Qt
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
        if self._xseg_trainer is not None and not self._xseg_trainer._stop_event.is_set():
            self._xseg_trainer.request_stop()
            self._log_signal.log_ready.emit("[enter] Saving and stopping training...", False)
        if self._preview_win is not None and self._preview_win.isVisible():
            self._preview_win.hide()

    def _on_edit(self):
        from DeepFaceLab.gui_app.xseg_editor_dialog import XSegEditorDialog
        d = self._get_edit_target_dir()
        dlg = XSegEditorDialog(d, self)
        dlg.exec()

    def _on_fetch(self):
        from DeepFaceLab.business.xseg_editor import XSegEditor
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
            from DeepFaceLab.business.xseg_editor import XSegEditor
            editor = XSegEditor()
            editor.remove_annotations(d)

    def _on_train(self):
        from DeepFaceLab.business.xseg_trainer import XSegTrainer
        self._xseg_trainer = XSegTrainer()
        self._train_btn.setEnabled(False)
        self._stop_train_btn.setEnabled(True)
        self._train_log.clear()
        self._train_log.append(f"=== Model Options ===")
        self._train_log.append(f"  face_type: wf")
        self._train_log.append(f"  batch_size: {self._xseg_batch.value()}")
        self._train_log.append(f"  target_iter: {self._xseg_iters.value()}")
        self._train_log.append(f"  resolution: 256")
        self._train_log.append("")

        self._log_throttle_time = 0.0
        self._log_throttle_interval = 0.5
        self._log_need_newline = False

        def _on_iter(iter_count, loss_val, iter_ms):
            import time
            now = time.time()
            if now - self._log_throttle_time < self._log_throttle_interval:
                return
            self._log_throttle_time = now
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            if iter_ms >= 10000:
                line = f"[{ts}][#{iter_count}][{iter_ms/1000:.1f}s][{loss_val:.4f}]"
            else:
                line = f"[{ts}][#{iter_count}][{int(iter_ms)}ms][{loss_val:.4f}]"
            overwrite = not self._log_need_newline
            self._log_need_newline = False
            self._log_signal.log_ready.emit(line, overwrite)

        def _on_preview(preview_bgr):
            self._preview_signal.preview_ready.emit(preview_bgr)

        def _on_save(iter_count):
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            self._log_need_newline = True
            self._log_signal.log_ready.emit(f"[{ts}][#{iter_count}] saved", False)

        def _task():
            try:
                self._xseg_trainer.train(
                    DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR, MODEL_DIR,
                    batch_size=self._xseg_batch.value(),
                    target_iter=self._xseg_iters.value(),
                    on_iter=_on_iter,
                    on_preview=_on_preview,
                    on_save=_on_save,
                )
            finally:
                self._train_btn.setEnabled(True)
                self._stop_train_btn.setEnabled(False)
                self._log_signal.log_ready.emit("已停止", False)

        self._run_in_thread(_task)

    def _on_stop_train(self):
        if self._xseg_trainer is not None:
            self._xseg_trainer.request_stop()
            self._log_signal.log_ready.emit("正在停止...", False)

    def _on_apply_trained(self):
        from DeepFaceLab.business.xseg_trainer import XSegTrainer
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
        from DeepFaceLab.business.xseg_trainer import XSegTrainer
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
        from DeepFaceLab.business.xseg_trainer import XSegTrainer
        from DeepFaceLab.setting import _PROJECT_ROOT
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
    step_title = "4. 训练模型"
    step_desc = "选择模型类型进行训练。"
    show_run_buttons = False

    MODE_SAEHD = "SAEHD"
    MODE_TFM = "TFM"

    def _build_params(self):
        from DeepFaceLab.gui_app.param_defs import (
            ParamGroupWidget, ConfigManager,
            TrainingSignals, TrainingStatusBar,
        )
        from DeepFaceLab.gui_app.tfm_param_defs import (
            get_tfm_params_by_group, TFMPresetManager,
        )
        from DeepFaceLab.gui_app.saehd_param_defs import (
            get_saehd_params_by_group,
        )
        from PyQt6.QtWidgets import QStackedWidget

        # ---- Mode toggle buttons (SAEHD | TFM) ----
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        self._mode_btn_saehd = QPushButton("SAEHD")
        self._mode_btn_tfm = QPushButton("TFM")
        for btn in (self._mode_btn_saehd, self._mode_btn_tfm):
            btn.setFixedHeight(36)
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QPushButton { font-size: 14px; font-weight: 700; border-radius: 4px; padding: 0 20px; }"
                "QPushButton:checked { background-color: #0078D4; color: white; }"
                "QPushButton:not(:checked) { background-color: #E0E0E0; color: #333; }"
            )
        self._mode_btn_saehd.setChecked(True)
        self._mode_btn_tfm.setChecked(False)
        mode_row.addWidget(self._mode_btn_saehd)
        mode_row.addWidget(self._mode_btn_tfm)
        mode_row.addStretch(1)
        self.layout().addLayout(mode_row)

        self._mode_btn_saehd.clicked.connect(lambda: self._switch_mode(self.MODE_SAEHD))
        self._mode_btn_tfm.clicked.connect(lambda: self._switch_mode(self.MODE_TFM))
        self._current_mode = self.MODE_SAEHD

        # ---- Stacked area for SAEHD / TFM params ----
        self._stack = QStackedWidget()
        self.layout().addWidget(self._stack)

        # -- Page 0: SAEHD params (grouped style, same as TFM) --
        saehd_page = QWidget()
        saehd_layout = QVBoxLayout(saehd_page)
        saehd_layout.setContentsMargins(0, 4, 0, 0)
        saehd_layout.setSpacing(6)

        self._saehd_param_groups: dict[ParamGroup, ParamGroupWidget] = {}
        for group in [ParamGroup.BASIC, ParamGroup.ARCHITECTURE, ParamGroup.FACE_DETAIL]:
            pw = ParamGroupWidget(group, get_saehd_params_by_group(group))
            self._saehd_param_groups[group] = pw
            saehd_layout.addWidget(pw)
            spacer = QWidget()
            spacer.setFixedHeight(12)
            saehd_layout.addWidget(spacer)

        self._saehd_config_manager = ConfigManager(MODEL_DIR, "SAEHD_training_config.json")
        saved_saehd = self._saehd_config_manager.load_config()
        if saved_saehd:
            for pw in self._saehd_param_groups.values():
                pw.set_values(saved_saehd)

        self._stack.addWidget(saehd_page)  # index 0

        # -- Page 1: TFM params (grouped style) --
        tfm_page = QWidget()
        tfm_layout = QVBoxLayout(tfm_page)
        tfm_layout.setContentsMargins(0, 4, 0, 0)
        tfm_layout.setSpacing(6)

        self._tfm_param_groups: dict[ParamGroup, ParamGroupWidget] = {}
        for group in [ParamGroup.BASIC, ParamGroup.ARCHITECTURE, ParamGroup.FACE_DETAIL, ParamGroup.LOSS_SAMPLING]:
            pw = ParamGroupWidget(group, get_tfm_params_by_group(group))
            self._tfm_param_groups[group] = pw
            tfm_layout.addWidget(pw)
            if group != ParamGroup.LOSS_SAMPLING:
                spacer = QWidget()
                spacer.setFixedHeight(12)
                tfm_layout.addWidget(spacer)

        preset_combo = self._tfm_param_groups[ParamGroup.ARCHITECTURE].get_widget("model_preset")
        self._preset_manager = TFMPresetManager(self._tfm_param_groups[ParamGroup.ARCHITECTURE], preset_combo)

        self._tfm_config_manager = ConfigManager(MODEL_DIR, "TFM_training_config.json")
        saved_tfm = self._tfm_config_manager.load_config()
        if saved_tfm:
            for pw in self._tfm_param_groups.values():
                pw.set_values(saved_tfm)
            self._preset_manager.apply_saved_preset(saved_tfm)
        self._preset_manager.on_preset_changed(self._preset_manager.get_current_preset())

        self._stack.addWidget(tfm_page)  # index 1

        # ---- Shared: Run / Stop buttons ----
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
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
        self.layout().addLayout(btn_row)

        # ---- Shared: Status bar ----
        self._status_bar = TrainingStatusBar()
        self.layout().addWidget(self._status_bar)

        # ---- Shared: Log text ----
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMinimumHeight(100)
        self._log_text.setMaximumHeight(200)
        self._log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; background-color: #1e1e1e; color: #d4d4d4;")
        self.layout().addWidget(self._log_text)

        # ---- Shared: Signals ----
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

    def _switch_mode(self, mode: str):
        self._current_mode = mode
        self._mode_btn_saehd.setChecked(mode == self.MODE_SAEHD)
        self._mode_btn_tfm.setChecked(mode == self.MODE_TFM)
        self._stack.setCurrentIndex(0 if mode == self.MODE_SAEHD else 1)

    def _collect_saehd_params(self) -> dict:
        params = {}
        for pw in self._saehd_param_groups.values():
            params.update(pw.get_values())
        params.pop("face_type", None)
        return params

    def _collect_tfm_params(self) -> dict:
        params = {}
        for pw in self._tfm_param_groups.values():
            params.update(pw.get_values())
        if "depths" in params and isinstance(params["depths"], str):
            try:
                params["depths"] = json.loads(params["depths"])
            except Exception:
                params["depths"] = [2, 2, 6, 2]
        if "num_heads" in params and isinstance(params["num_heads"], str):
            try:
                params["num_heads"] = json.loads(params["num_heads"])
            except Exception:
                params["num_heads"] = [3, 6, 12, 24]
        if "window_size" in params:
            params["window_size"] = int(params["window_size"])
        for key in ["embed_dim", "depths", "num_heads", "base_channels", "w_dim"]:
            params.pop(key, None)
        return params

    # ---- Run / Stop ----

    def _on_run(self):
        if self._current_mode == self.MODE_SAEHD:
            self._run_saehd()
        else:
            self._run_tfm()

    def _run_saehd(self):
        from DeepFaceLab.business.saehd_trainer import SAEHDTrainer
        import threading
        import time as _time

        params = self._collect_saehd_params()
        self._saehd_config_manager.save_config(params)

        self._set_controls_enabled(False)
        self._status_bar.start_pulse()
        self._log_text.clear()
        self._log_need_newline = True
        self._log_throttle_time = 0.0

        trainer = SAEHDTrainer()
        self._trainer = trainer
        signals = self._signals
        panel = self

        def _on_iter(iter_count, loss_val, iter_ms):
            now = _time.time()
            if now - panel._log_throttle_time < panel._log_throttle_interval:
                return
            panel._log_throttle_time = now
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            if iter_ms >= 10000:
                line = f"[{ts}][#{iter_count}][{iter_ms/1000:.1f}s][{loss_val:.4f}]"
            else:
                line = f"[{ts}][#{iter_count}][{int(iter_ms)}ms][{loss_val:.4f}]"
            overwrite = not panel._log_need_newline
            panel._log_need_newline = False
            signals.log_signal.emit(line, overwrite)

        def _on_preview(preview_bgr):
            panel._preview_signal.preview_ready.emit(preview_bgr)

        def _on_save(iter_count):
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            panel._log_need_newline = True
            signals.log_signal.emit(f"[{ts}][#{iter_count}] saved", False)

        def _on_log(text, overwrite):
            signals.log_signal.emit(text, overwrite)

        def _task():
            try:
                trainer.train(
                    DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR, MODEL_DIR,
                    **params,
                    on_iter=_on_iter,
                    on_preview=_on_preview,
                    on_save=_on_save,
                    on_log=_on_log,
                )
            except Exception as e:
                signals.error_signal.emit(str(e))
            finally:
                signals.finished_signal.emit()

        self._training_thread = threading.Thread(target=_task, daemon=True)
        self._training_thread.start()

    def _run_tfm(self):
        from DeepFaceLab.business.tfm_trainer import TFMTrainer
        import threading
        import time as _time

        params = self._collect_tfm_params()
        self._tfm_config_manager.save_config(params)

        self._set_controls_enabled(False)
        self._status_bar.start_pulse()
        self._log_text.clear()
        self._log_need_newline = True
        self._log_throttle_time = 0.0

        trainer = TFMTrainer()
        self._trainer = trainer
        signals = self._signals
        panel = self

        def _on_iter(iter_count, loss_val, iter_ms):
            now = _time.time()
            if now - panel._log_throttle_time < panel._log_throttle_interval:
                return
            panel._log_throttle_time = now
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            if iter_ms >= 10000:
                line = f"[{ts}][#{iter_count}][{iter_ms/1000:.1f}s][{loss_val:.4f}]"
            else:
                line = f"[{ts}][#{iter_count}][{int(iter_ms)}ms][{loss_val:.4f}]"
            overwrite = not panel._log_need_newline
            panel._log_need_newline = False
            signals.log_signal.emit(line, overwrite)

        def _on_preview(preview_bgr):
            panel._preview_signal.preview_ready.emit(preview_bgr)

        def _on_save(iter_count):
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            panel._log_need_newline = True
            signals.log_signal.emit(f"[{ts}][#{iter_count}] saved", False)

        def _on_log(text, overwrite):
            signals.log_signal.emit(text, overwrite)

        def _task():
            try:
                trainer.train(
                    DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR, MODEL_DIR,
                    **params,
                    on_iter=_on_iter,
                    on_preview=_on_preview,
                    on_save=_on_save,
                    on_log=_on_log,
                )
            except Exception as e:
                signals.error_signal.emit(str(e))
            finally:
                signals.finished_signal.emit()

        self._training_thread = threading.Thread(target=_task, daemon=True)
        self._training_thread.start()

    def _on_stop(self):
        if self._trainer is not None and not self._trainer._stop_event.is_set():
            self._trainer.request_stop()
            self._append_log("正在停止...", False)

    def _set_controls_enabled(self, enabled: bool):
        self._run_btn.setEnabled(enabled)
        self._stop_btn.setEnabled(not enabled)
        self._mode_btn_saehd.setEnabled(enabled)
        self._mode_btn_tfm.setEnabled(enabled)
        for pw in self._saehd_param_groups.values():
            pw.set_editable(enabled)
        for pw in self._tfm_param_groups.values():
            pw.set_editable(enabled)
        preset_combo = self._tfm_param_groups[ParamGroup.ARCHITECTURE].get_widget("model_preset")
        if preset_combo:
            preset_combo.setEnabled(enabled)

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

        if self._preview_win is None:
            self._preview_win = QDialog(self)
            mode_name = "SAEHD" if self._current_mode == self.MODE_SAEHD else "TFM"
            self._preview_win.setWindowTitle(f"{mode_name} 训练预览 | [space]:切换 [p]:刷新 [s]:保存 [l]:历史范围 [Enter]:保存并停止")
            self._preview_win.setWindowFlags(self._preview_win.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint)
            lay = QVBoxLayout(self._preview_win)
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(lbl)
            self._preview_win._label = lbl
            self._preview_win.resize(900, 700)
            self._preview_win.keyPressEvent = lambda e: self._on_preview_key(e)
            original_close = self._preview_win.close
            self._preview_win.closeEvent = lambda e: self._on_preview_close(e, original_close)
            self._preview_win.show()

        if not self._preview_win.isVisible():
            self._preview_win.show()

        rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self._preview_win._label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._preview_win._label.setPixmap(scaled)

    def _on_preview_key(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            if self._trainer is not None:
                if hasattr(self._trainer, '_preview_page'):
                    self._trainer._preview_page += 1
                self._trainer.request_preview()
        elif key == Qt.Key.Key_P:
            if self._trainer is not None:
                self._trainer.request_preview()
        elif key == Qt.Key.Key_S:
            if self._trainer is not None:
                self._trainer.request_save()
                self._append_log("[save] Checkpoint saved.", False)
        elif key == Qt.Key.Key_L:
            if self._trainer is not None:
                self._trainer.cycle_loss_range()
                self._trainer.request_preview()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._stop_and_close_preview()

    def _on_preview_close(self, event, original_close):
        self._stop_and_close_preview()
        event.accept()

    def _stop_and_close_preview(self):
        if self._trainer is not None and not self._trainer._stop_event.is_set():
            self._trainer.request_stop()
            self._append_log("[enter] Saving and stopping training...", False)
        if self._preview_win is not None and self._preview_win.isVisible():
            self._preview_win.hide()

    def _on_save_notify(self, iter_num: int):
        current = self._status_bar._status_label.text()
        self._status_bar._status_label.setText(f"{current} | 已保存 #{iter_num}")

    def _on_training_error(self, error_msg: str):
        from PyQt6.QtWidgets import QMessageBox
        self._set_controls_enabled(True)
        self._status_bar.stop_pulse()
        QMessageBox.critical(self, "训练错误", error_msg)

    def _on_training_finished(self):
        self._set_controls_enabled(True)
        self._status_bar.stop_pulse()
        self._append_log("已停止", False)


class Step5Merge(StepPanel):
    step_title = "5. 合成人脸"
    step_desc = "使用训练好的模型将人脸合成到目标帧中。"

    def _build_params(self):
        self._model_type = self._add_combo("模型类型:", ["SAEHD", "Quick96", "AMP", "INSwapper", "TFM"], "SAEHD")
        self._mask_mode = self._add_combo("遮罩模式:", ["xseg", "dst", "learned"], "xseg")

    def _on_run(self):
        from DeepFaceLab.business.model_merger import ModelMerger, MergeConfig, MaskMode
        from DeepFaceLab.core.insightface_adapter import InsightFaceAdapter

        adapter = InsightFaceAdapter()
        merger = ModelMerger(adapter)
        mask_mode = MaskMode(self._mask_mode.currentText())
        config = MergeConfig(mask_mode=mask_mode)
        mt = self._model_type.currentText()

        def _task():
            if mt == "INSwapper":
                merger.merge_inswapper(DATA_DST_DIR, DATA_DST_MERGED_DIR, DATA_SRC_ALIGNED_DIR, DATA_DST_DIR)
            else:
                merger.merge_auto(MODEL_DIR, mt, DATA_DST_DIR, DATA_DST_MERGED_DIR,
                                  DATA_DST_MERGED_MASK_DIR, DATA_DST_ALIGNED_DIR, config,
                                  src_aligned_dir=DATA_SRC_ALIGNED_DIR)

        self._run_in_thread(_task)


class Step6Output(StepPanel):
    step_title = "6. 导出视频"
    step_desc = "将合成后的帧转换为带音频的输出视频。"

    def _build_params(self):
        self._fmt = self._add_combo("输出格式:", ["mp4", "avi", "mov"], "mp4")
        self._lossless = self._add_check("无损输出", False)

    def _on_run(self):
        from DeepFaceLab.business.video_output import VideoOutput, OutputFormat
        from DeepFaceLab.business.video_processor import VideoProcessor
        from DeepFaceLab.business.workspace_manager import WorkspaceManager

        vp = VideoProcessor()
        vo = VideoOutput(vp)
        ws = WorkspaceManager()
        fmt = OutputFormat(self._fmt.currentText())
        ref = ws.find_dst_video()
        output_path = WORKSPACE_DIR / f"result.{fmt.value}"

        def _task():
            vo.merged_to_video(DATA_DST_MERGED_DIR, output_path, fmt,
                               reference_video=ref, lossless=self._lossless.isChecked())

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
            from DeepFaceLab.gui_app.manual_annotator import ManualAnnotatorDialog
            is_src = self._target.currentIndex() == 0
            dlg = ManualAnnotatorDialog(self, is_src=is_src)
            dlg.exec()
            return

        from DeepFaceLab.business.face_tool import FaceTool

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
        from DeepFaceLab.business.workspace_manager import WorkspaceManager
        ws = WorkspaceManager()

        def _task():
            ws.clear_workspace()

        self._run_in_thread(_task)
