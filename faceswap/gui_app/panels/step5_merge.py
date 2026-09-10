import gc
import json
import shutil
import sys
import threading
import subprocess
from collections import OrderedDict, deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

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

class Step5Merge(StepPanel):
    step_title = "5. 合成融合"
    step_desc = "实时预览换脸效果，调整遮罩与颜色迁移参数。"
    _model_loaded_sig = pyqtSignal(object, str)
    _model_error_sig = pyqtSignal(str)
    _render_done_sig = pyqtSignal(object, str)

    def _build_ui(self):
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QVBoxLayout()
        header.setContentsMargins(20, 16, 20, 4)
        header.setSpacing(4)
        title = QLabel(self.step_title)
        title.setObjectName("stepTitle")
        header.addWidget(title)
        desc = QLabel(self.step_desc)
        desc.setObjectName("stepDesc")
        desc.setWordWrap(True)
        header.addWidget(desc)
        layout.addLayout(header, 0)

        self._params_area = QVBoxLayout()
        self._params_area.setContentsMargins(0, 0, 0, 0)
        self._params_area.setSpacing(0)
        layout.addLayout(self._params_area, 1)
        self._build_params()

    def _build_params(self):
        import cv2
        self._cv2 = cv2
        self._trainer = None
        self._merger = None
        self._frames = []
        self._frame_idx = 0
        # 载量前缀索引: 帧stem -> [aligned文件名]（仅 listdir 文件名, 不读 JSON 内容）
        self._frame_index: dict[str, list[str]] = {}
        # 按帧惰性加载的 [(face_name, meta)] LRU 缓存
        self._faces_cache = OrderedDict()
        self._aligned_dir_mtime = 0.0
        self._current_img = None
        self._current_pixmap = None
        self._stop_requested = False
        self._rendering = False
        self._pending_cleanup = False
        self._model_loading = False
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._do_resize_preview)

        self._scan_frames()
        self._build_frame_index()

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_scroll = QScrollArea()
        left_scroll.setFixedWidth(330)
        left_scroll.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

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
        self._load_model_btn.setStyleSheet(
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
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
        self._next_btn = QPushButton("下一帧 ▶")
        self._next_btn.clicked.connect(self._on_next_frame)
        nav_row.addWidget(self._next_btn)
        nav_layout.addLayout(nav_row)
        self._frame_info = QLabel("0/0")
        self._frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self._frame_info)
        left_layout.addWidget(nav_group)

        self._merge_all_btn = QPushButton("合成全部帧")
        self._merge_all_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self._merge_all_btn.clicked.connect(self._on_merge_all)
        left_layout.addWidget(self._merge_all_btn)
        self._stop_btn = QPushButton("停止")
        self._stop_btn.setProperty("busy_keep", True)
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #C42B1C; color: white; font-weight: bold; "
            "padding: 6px 14px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #D43B2C; }"
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
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
        self._model_loaded_sig.connect(self._on_model_loaded)
        self._model_error_sig.connect(self._on_model_error)
        self._render_done_sig.connect(self._on_render_done)

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

    def _build_frame_index(self):
        """构建 帧stem -> [aligned文件名] 索引。
        优先读 _source_index.json 映射文件，不存在或过期则重建。
        """
        self._frame_index = {}
        self._faces_cache = OrderedDict()
        if DATA_DST_ALIGNED_DIR.exists():
            from faceswap.core.metadata_manager import MetadataManager
            self._frame_index = MetadataManager.get_source_index(DATA_DST_ALIGNED_DIR)
            try:
                self._aligned_dir_mtime = DATA_DST_ALIGNED_DIR.stat().st_mtime
            except OSError:
                self._aligned_dir_mtime = 0.0

    def _get_faces_for_frame(self, frame_path, check_mtime: bool = False):
        """按需获取某一帧的 [(face_name, meta)]。

        只读该帧前缀匹配到的少数 JSON, 结果按帧名 LRU 缓存(默认 32 帧)。
        check_mtime=True 时(单帧预览路径)先检测 aligned 目录变更并重建索引;
        批量合成等已确定目录不变的路径传 False 避免每帧 stat 开销。
        """
        if check_mtime and DATA_DST_ALIGNED_DIR.exists():
            try:
                mt = DATA_DST_ALIGNED_DIR.stat().st_mtime
            except OSError:
                mt = self._aligned_dir_mtime
            if mt != self._aligned_dir_mtime:
                self._build_frame_index()
        name = Path(frame_path).name
        cached = self._faces_cache.get(name)
        if cached is not None:
            self._faces_cache.move_to_end(name)
            return cached
        face_names = self._frame_index.get(Path(frame_path).stem, [])
        faces: list = []
        if face_names:
            from faceswap.core.metadata_manager import MetadataManager
            for fn in face_names:
                meta = MetadataManager.load(DATA_DST_ALIGNED_DIR / fn)
                if meta is not None:
                    faces.append((fn, meta))
        if len(self._faces_cache) >= 32:
            self._faces_cache.popitem(last=False)
        self._faces_cache[name] = faces
        return faces

    def _refresh_model_names(self):
        """扫描当前类型目录刷新模型名称列表, 尽量保留当前选择 (不触发类型切换副作用)。"""
        from faceswap.setting import SAEHD_MODEL_DIR, AMP_MODEL_DIR, QUICK96_MODEL_DIR
        mt = self._model_type.currentText()
        base_dirs = {"SAEHD": SAEHD_MODEL_DIR, "AMP": AMP_MODEL_DIR, "Quick96": QUICK96_MODEL_DIR}
        base_dir = base_dirs.get(mt, SAEHD_MODEL_DIR)
        current = self._model_name.currentText()
        self._model_name.blockSignals(True)
        self._model_name.clear()
        if base_dir.exists():
            for d in sorted(base_dir.iterdir()):
                if d.is_dir() and ((d / "training_config.json").exists() or (d / "SAEHD_training_config.json").exists()):
                    self._model_name.addItem(d.name)
        if current:
            idx = self._model_name.findText(current)
            if idx >= 0:
                self._model_name.setCurrentIndex(idx)
        self._model_name.blockSignals(False)

    def on_panel_shown(self):
        # 切换到本界面时主动刷新模型/权重列表 (不打断已加载模型)
        self._refresh_model_names()

    def _on_model_type_changed(self):
        self._refresh_model_names()
        self._trainer = None
        self._merger = None
        self._model_status.setText("未加载")
        self._model_status.setStyleSheet("font-size: 11px; color: #999;")

    def _on_load_model(self):
        if self._model_loading:
            return
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
            config_path = model_dir / "SAEHD_training_config.json"
        if not config_path.exists():
            QMessageBox.warning(self, "配置不存在", f"找不到训练配置: {model_dir}")
            return
        self._model_loading = True
        self._set_busy(True)

        if self._trainer is not None:
            if self._rendering:
                self._pending_cleanup = True
            try:
                if hasattr(self._trainer, 'model'):
                    self._trainer.model.cpu()
            except Exception:
                pass
            self._trainer = None
            self._merger = None
            from faceswap.business.vram_manager import cleanup_memory, synchronize
            synchronize()
            cleanup_memory(aggressive=True)

        self._model_status.setText("正在加载模型...")
        self._model_status.setStyleSheet("font-size: 11px; color: #FAAD14;")

        def _task():
            try:
                config_dict = json.loads(config_path.read_text(encoding="utf-8"))
                from faceswap.shared.config import auto_select_device
                device = auto_select_device()
                dummy = Path(".")
                trainer = None
                if mt == "SAEHD":
                    from faceswap.business.saehd_trainer import SAEHDTrainer, TrainingConfig
                    config = TrainingConfig(**config_dict)
                    trainer = SAEHDTrainer(config, model_dir, dummy, dummy, device=device)
                elif mt == "AMP":
                    from faceswap.business.amp_trainer import AMPTrainer, AMPTrainingConfig
                    config = AMPTrainingConfig(**config_dict)
                    trainer = AMPTrainer(config, model_dir, dummy, dummy, device=device)
                elif mt == "Quick96":
                    from faceswap.business.quick96_trainer import Quick96Trainer, Quick96TrainingConfig
                    config = Quick96TrainingConfig(**config_dict)
                    trainer = Quick96Trainer(config, model_dir, dummy, dummy, device=device)
                self._model_loaded_sig.emit(trainer, mn)
            except Exception as e:
                self._model_error_sig.emit(str(e))

        threading.Thread(target=_task, daemon=True).start()

    def _on_model_loaded(self, trainer, mn):
        self._model_loading = False
        self._set_busy(False)
        self._trainer = trainer
        self._model_status.setText(f"已加载: {mn}")
        self._model_status.setStyleSheet("font-size: 11px; color: #0078D4;")
        self._do_render()

    def _on_model_error(self, err):
        self._model_loading = False
        self._set_busy(False)
        self._trainer = None
        self._model_status.setText(f"加载失败: {err}")
        self._model_status.setStyleSheet("font-size: 11px; color: #d32f2f;")
        QMessageBox.critical(self, "模型加载失败", err)

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
        elif key == Qt.Key.Key_S:
            self._erode.setValue(self._erode.value() - 1)
        elif key == Qt.Key.Key_E:
            self._blur.setValue(self._blur.value() + 1)
        elif key == Qt.Key.Key_D:
            self._blur.setValue(self._blur.value() - 1)
        elif key == Qt.Key.Key_T:
            self._face_scale.setValue(self._face_scale.value() + 1)
        elif key == Qt.Key.Key_G:
            self._face_scale.setValue(self._face_scale.value() - 1)
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
        if not self._frames or self._rendering:
            return
        idx = self._frame_idx
        if idx < 0 or idx >= len(self._frames):
            return
        frame_path = self._frames[idx]
        trainer = self._trainer
        merger = self._merger
        self._rendering = True

        def _task():
            try:
                frame_img = self._cv2.imread(str(frame_path))
                if frame_img is None:
                    self._render_done_sig.emit(None, "无法读取帧")
                    return
                faces = self._get_faces_for_frame(frame_path, check_mtime=True)
                if trainer is not None and faces:
                    try:
                        from faceswap.business.model_merger import ModelMerger
                        config = self._collect_merge_config()
                        m = self._merger
                        if m is None:
                            m = ModelMerger()
                        result = m.composite_single_frame(
                            frame_img, faces, trainer.predictor_func, config,
                            DATA_DST_ALIGNED_DIR,
                        )
                        self._merger = m
                    except Exception as e:
                        _logger.warning(f"Render failed: {e}")
                        result = frame_img
                        self._render_done_sig.emit(result, f"渲染错误: {e}")
                        return
                else:
                    result = frame_img
                    if not faces:
                        self._render_done_sig.emit(result, "此帧无人脸")
                        return
                    elif trainer is None:
                        self._render_done_sig.emit(result, "未加载模型（显示原图）")
                        return
                self._render_done_sig.emit(result, "")
            except Exception as e:
                self._render_done_sig.emit(None, f"渲染异常: {e}")

        threading.Thread(target=_task, daemon=True).start()

    def _on_render_done(self, img, status_msg):
        self._rendering = False
        if getattr(self, '_pending_cleanup', False):
            self._pending_cleanup = False
            from faceswap.business.vram_manager import cleanup_memory, synchronize
            synchronize()
            cleanup_memory(aggressive=True)
        if img is None:
            if status_msg:
                self._status_label.setText(status_msg)
            return
        self._update_preview(img)
        if status_msg:
            self._status_label.setText(status_msg)

    def _update_preview(self, img):
        from PyQt6.QtGui import QImage, QPixmap
        self._current_img = img
        rgb = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self._current_pixmap = QPixmap.fromImage(qimg)
        label_size = self._preview_label.size()
        if label_size.width() > 10 and label_size.height() > 10:
            pixmap = self._current_pixmap.scaled(
                label_size, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            pixmap = self._current_pixmap
        self._preview_label.setPixmap(pixmap)
        n_faces = len(self._frame_index.get(self._frames[self._frame_idx].stem, [])) if self._frames else 0
        self._status_label.setText(f"帧 {self._frame_idx + 1}/{len(self._frames)}  |  {n_faces} 人脸  |  {img.shape[1]}x{img.shape[0]}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_timer.start(100)

    def _do_resize_preview(self):
        if self._current_pixmap is not None:
            label_size = self._preview_label.size()
            if label_size.width() > 10 and label_size.height() > 10:
                pixmap = self._current_pixmap.scaled(
                    label_size, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._preview_label.setPixmap(pixmap)

    def _on_merge_all(self):
        if not self._frames:
            QMessageBox.warning(self, "无帧", "未找到目标帧。请先完成视频提取。")
            return
        self._stop_requested = False
        self._merge_all_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._batch_progress.setVisible(True)
        self._batch_progress.setRange(0, len(self._frames))
        self._set_busy(True)
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
                    faces = self._get_faces_for_frame(fp)
                    if self._trainer is not None and faces:
                        result = merger.composite_single_frame(
                            frame_img, faces, self._trainer.predictor_func, config,
                            DATA_DST_ALIGNED_DIR,
                        )
                    else:
                        result = frame_img
                    if not imwrite_auto(DATA_DST_MERGED_DIR / fp.name, result):
                        sig.error_ready.emit("写入失败", f"无法写入: {fp.name}\n磁盘可能已满或路径不可写。")
                        break
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
        self._set_busy(False)
        QMessageBox.information(self, title, msg)

    def _on_merge_error(self, title, msg):
        self._merge_all_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._batch_progress.setVisible(False)
        self._set_busy(False)
        QMessageBox.critical(self, title, msg)

