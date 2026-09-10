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
        self._stop_btn.setProperty("busy_keep", True)
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

    def showEvent(self, event):
        super().showEvent(event)
        if not self._running and hasattr(self, '_model_name_selector'):
            sel = self._model_name_selector
            current = sel.current_name()
            sel.refresh_models()
            if current:
                idx = sel._combo.findText(current)
                if idx >= 0 and idx != sel._combo.currentIndex():
                    sel._combo.setCurrentIndex(idx)

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
        opts = full_archi.split('-', 1)[-1] if '-' in full_archi else ''
        needs_32 = ('d' in opts) or ('t' in opts)
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
                    res_w.setToolTip("分辨率（像素）。\n当前架构含'd'（分辨率倍增）或't'（深层），必须是32的倍数。")
                else:
                    res_w.setToolTip("分辨率（像素）。必须是16的倍数。")
                break

    def _lock_architecture_params(self):
        if self._current_model_type == "SAEHD":
            _lock_keys = {
                'resolution', 'face_type', 'archi', 'amp_mode',
                'ae_dims', 'e_dims', 'd_dims', 'd_mask_dims',
                'gan_patch_size', 'gan_dims',
            }
        elif self._current_model_type == "AMP":
            _lock_keys = {
                'resolution', 'face_type', 'ae_dims', 'inter_dims',
                'e_dims', 'd_dims', 'd_mask_dims', 'morph_factor',
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

        try:
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
        except Exception as e:
            QMessageBox.critical(self, "参数错误", f"训练参数构建失败:\n{e}")
            return

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
        is_resume = final_dir.exists() and any(final_dir.iterdir())
        if is_new_action and is_resume:
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
                is_resume = False
            else:
                self._model_name_selector._on_new()
                return

        final_dir.mkdir(parents=True, exist_ok=True)

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
        self._loss_since_save_src_sum: float = 0.0
        self._loss_since_save_dst_sum: float = 0.0
        self._loss_since_save_d_gan_sum: float = 0.0
        self._loss_since_save_d_gan_count: int = 0
        self._loss_since_save_count: int = 0
        self._save_interval_start_iter: int = -1
        self._save_interval_time_sum: float = 0.0

        signals = self._signals
        panel = self

        def _on_progress(iter_num, src_loss, dst_loss, iter_ms, lr, converged=False, d_gan_loss=0.0):
            if iter_num == -1:
                if panel._loss_since_save_count > 0:
                    from datetime import datetime
                    ts = datetime.now().strftime("%H:%M:%S")
                    n = panel._loss_since_save_count
                    interval_src = panel._loss_since_save_src_sum / n
                    interval_dst = panel._loss_since_save_dst_sum / n
                    interval_avg = (interval_src + interval_dst) / 2
                    start_iter = panel._save_interval_start_iter
                    end_iter = start_iter + n
                    avg_ms = panel._save_interval_time_sum / n if n > 0 else 0
                    panel._loss_since_save_src_sum = 0.0
                    panel._loss_since_save_dst_sum = 0.0
                    panel._loss_since_save_count = 0
                    panel._save_interval_start_iter = -1
                    panel._save_interval_time_sum = 0.0
                    if avg_ms >= 1000:
                        time_str = f"{avg_ms/1000:.1f}s"
                    else:
                        time_str = f"{int(avg_ms)}ms"
                    line = f"[{ts}][#{start_iter}-{end_iter}][{time_str}][src {interval_src:.5f} dst {interval_dst:.5f}]"
                    if panel._loss_since_save_d_gan_count > 0:
                        interval_d_gan = panel._loss_since_save_d_gan_sum / panel._loss_since_save_d_gan_count
                        line += f" D_gan={interval_d_gan:.5f}"
                    panel._loss_since_save_d_gan_sum = 0.0
                    panel._loss_since_save_d_gan_count = 0
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
            panel._loss_since_save_src_sum += src_loss
            panel._loss_since_save_dst_sum += dst_loss
            panel._loss_since_save_count += 1
            if d_gan_loss > 0:
                panel._loss_since_save_d_gan_sum += d_gan_loss
                panel._loss_since_save_d_gan_count += 1
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
            trainer = None
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
            finally:
                if trainer is not None:
                    trainer.cleanup()

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
        self._set_busy(not enabled)
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
        if hasattr(self, '_pending_stop_event'):
            self._pending_stop_event.set()
        self._request_training_stop(self._trainer, "enter")

    def _on_save_notify(self, iter_num: int):
        current = self._status_bar._status_label.text()
        self._status_bar._status_label.setText(f"{current} | 已保存 #{iter_num}")

    def _on_training_error(self, error_msg: str):
        self._set_controls_enabled(True)
        self._status_bar.stop_pulse()
        self._trainer = None
        QMessageBox.critical(self, "训练错误", error_msg)

    def _on_training_finished(self):
        self._set_controls_enabled(True)
        self._status_bar.stop_pulse()
        self._trainer = None
        if self._preview_win is not None:
            self._preview_win.close()
            self._preview_win = None

