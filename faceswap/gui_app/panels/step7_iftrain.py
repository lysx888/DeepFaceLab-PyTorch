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
    QSplitter, QSlider, QFrame, QGridLayout, QListWidget,
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

class Step7IFTrain(StepPanel):
    step_title = "7. IF训练"
    step_desc = "先训练SCRFD检测器，再训练106点landmark模型。导出ONNX替换insightface预训练权重。"
    show_run_buttons = False

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

        main_row = QHBoxLayout()
        main_row.setSpacing(12)

        left_widget = QWidget()
        left_widget.setFixedWidth(300)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        self._params_area = left_layout
        self._build_params()

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        from PyQt6.QtWidgets import QSizePolicy
        preview_policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        chart_policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        scrfd_title = QLabel("SCRFD检测预览:")
        scrfd_title.setStyleSheet("font-size: 12px; font-weight: bold;")
        right_layout.addWidget(scrfd_title)
        self._scrfd_preview_label = AutoScaleLabel("等待训练...")
        self._scrfd_preview_label.setMinimumHeight(200)
        self._scrfd_preview_label.setMinimumWidth(200)
        self._scrfd_preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #1e1e1e; color: #888;")
        self._scrfd_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scrfd_preview_label.setSizePolicy(preview_policy)
        right_layout.addWidget(self._scrfd_preview_label)
        self._scrfd_chart_label = AutoScaleLabel()
        self._scrfd_chart_label.setFixedHeight(120)
        self._scrfd_chart_label.setMinimumWidth(200)
        self._scrfd_chart_label.setStyleSheet("border: 1px solid #ccc; background-color: #1e1e1e;")
        self._scrfd_chart_label.setSizePolicy(chart_policy)
        right_layout.addWidget(self._scrfd_chart_label)

        lm_title = QLabel("Landmark预览:")
        lm_title.setStyleSheet("font-size: 12px; font-weight: bold; margin-top: 4px;")
        right_layout.addWidget(lm_title)
        self._lm_preview_label = AutoScaleLabel("等待训练...")
        self._lm_preview_label.setMinimumHeight(200)
        self._lm_preview_label.setMinimumWidth(200)
        self._lm_preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #1e1e1e; color: #888;")
        self._lm_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lm_preview_label.setSizePolicy(preview_policy)
        right_layout.addWidget(self._lm_preview_label)
        self._lm_chart_label = AutoScaleLabel()
        self._lm_chart_label.setFixedHeight(120)
        self._lm_chart_label.setMinimumWidth(200)
        self._lm_chart_label.setStyleSheet("border: 1px solid #ccc; background-color: #1e1e1e;")
        self._lm_chart_label.setSizePolicy(chart_policy)
        right_layout.addWidget(self._lm_chart_label)

        main_row.addWidget(left_widget)
        main_row.addWidget(right_widget, 1)
        layout.addLayout(main_row)

    def _build_params(self):
        ds_label = QLabel("数据集:")
        self._params_area.addWidget(ds_label)
        self._dataset_combo = QComboBox()
        self._dataset_combo.addItem("manual_annotated", str(INSIGHTFACE_MANUAL_ANNOTATED_DIR))
        lapa_dir = str(PRETRAIN_DATA_DIR / "lapa_training")
        self._dataset_combo.addItem("LaPa", lapa_dir)
        self._params_area.addWidget(self._dataset_combo)

        extra_grp = QGroupBox("额外数据集（concat混合训练）")
        extra_lay = QVBoxLayout(extra_grp)
        extra_lay.setContentsMargins(8, 14, 8, 8)
        extra_lay.setSpacing(4)
        self._extra_dirs_list = QListWidget()
        self._extra_dirs_list.setMaximumHeight(80)
        extra_lay.addWidget(self._extra_dirs_list)
        extra_btn_row = QHBoxLayout()
        extra_add_btn = QPushButton("添加目录")
        extra_add_btn.clicked.connect(self._add_extra_dir)
        extra_rm_btn = QPushButton("移除")
        extra_rm_btn.clicked.connect(self._remove_extra_dir)
        extra_btn_row.addWidget(extra_add_btn)
        extra_btn_row.addWidget(extra_rm_btn)
        extra_btn_row.addStretch()
        extra_lay.addLayout(extra_btn_row)
        extra_desc = QLabel("可添加WFLW等数据集目录，自动映射到106点并屏蔽缺少点的损失")
        extra_desc.setStyleSheet("color: #666; font-size: 11px;")
        extra_desc.setWordWrap(True)
        extra_lay.addWidget(extra_desc)
        self._params_area.addWidget(extra_grp)

        model_grp = QGroupBox("训练选择")
        model_vl = QVBoxLayout(model_grp)
        model_vl.setContentsMargins(8, 12, 8, 8)
        model_row = QHBoxLayout()
        self._train_scrfd_check = QCheckBox("训练SCRFD")
        self._train_scrfd_check.setChecked(True)
        self._train_scrfd_check.setToolTip("勾选后训练SCRFD人脸检测模型")
        self._train_lm_check = QCheckBox("训练Landmark")
        self._train_lm_check.setChecked(True)
        self._train_lm_check.setToolTip("勾选后训练106点Landmark模型")
        model_row.addWidget(self._train_scrfd_check)
        model_row.addWidget(self._train_lm_check)
        model_vl.addLayout(model_row)
        self._params_area.addWidget(model_grp)

        scrfd_grp = QGroupBox("SCRFD参数")
        scrfd_vl = QVBoxLayout(scrfd_grp)
        scrfd_vl.setContentsMargins(8, 12, 8, 8)
        scrfd_row1 = QHBoxLayout()
        scrfd_row1.addWidget(QLabel("批次:"))
        self._scrfd_batch_size = QSpinBox()
        self._scrfd_batch_size.setRange(1, 64)
        self._scrfd_batch_size.setValue(8)
        self._scrfd_batch_size.setToolTip("每批训练图片数量。SCRFD输入640×640，显存占用大，建议4-16。")
        scrfd_row1.addWidget(self._scrfd_batch_size)
        scrfd_row1.addWidget(QLabel("学习率:"))
        self._scrfd_lr = QDoubleSpinBox()
        self._scrfd_lr.setRange(0.0001, 1.0)
        self._scrfd_lr.setDecimals(4)
        self._scrfd_lr.setSingleStep(0.001)
        self._scrfd_lr.setValue(0.001)
        self._scrfd_lr.setToolTip("学习率。微调(加载预训练)默认0.001，从头训练默认0.01。")
        scrfd_row1.addWidget(self._scrfd_lr)
        scrfd_vl.addLayout(scrfd_row1)
        scrfd_row2 = QHBoxLayout()
        scrfd_row2.addWidget(QLabel("轮数:"))
        self._scrfd_epochs = QSpinBox()
        self._scrfd_epochs.setRange(1, 300)
        self._scrfd_epochs.setValue(30)
        self._scrfd_epochs.setToolTip("额外训练轮数。每次点击训练都会训练这么多轮。\n续训时从上次结束的位置继续，lr_steps自动管理。")
        scrfd_row2.addWidget(self._scrfd_epochs)
        self._load_scrfd_pretrained = QCheckBox("微调")
        self._load_scrfd_pretrained.setChecked(True)
        self._load_scrfd_pretrained.setToolTip(
            "勾选后加载SCRFD预训练权重进行微调(分阶段训练)。\n"
            "勾选+非刚性形变增强=针对嘴巴变形/遮挡的微调。\n"
            "已有训练模型时自动续训。")
        self._load_scrfd_pretrained.toggled.connect(
            lambda checked: self._on_pretrain_toggled(checked, self._scrfd_lr, 0.001, 0.01))
        self._load_scrfd_pretrained.toggled.connect(
            lambda checked: self._deform_aug.setEnabled(checked or self._load_lm_pretrained.isChecked()))
        scrfd_row2.addWidget(self._load_scrfd_pretrained)
        scrfd_row2.addStretch()
        scrfd_vl.addLayout(scrfd_row2)
        self._params_area.addWidget(scrfd_grp)

        lm_grp = QGroupBox("Landmark参数")
        lm_vl = QVBoxLayout(lm_grp)
        lm_vl.setContentsMargins(8, 12, 8, 8)
        lm_row1 = QHBoxLayout()
        lm_row1.addWidget(QLabel("批次:"))
        self._lm_batch_size = QSpinBox()
        self._lm_batch_size.setRange(1, 256)
        self._lm_batch_size.setValue(32)
        self._lm_batch_size.setToolTip("每批训练图片数量。Landmark输入192×192，显存占用小，建议32-128。")
        lm_row1.addWidget(self._lm_batch_size)
        lm_row1.addWidget(QLabel("学习率:"))
        self._lm_lr = QDoubleSpinBox()
        self._lm_lr.setRange(0.0001, 1.0)
        self._lm_lr.setDecimals(4)
        self._lm_lr.setSingleStep(0.001)
        self._lm_lr.setValue(0.01)
        self._lm_lr.setToolTip("学习率。微调(加载预训练)默认0.01，从头训练默认0.1。")
        lm_row1.addWidget(self._lm_lr)
        lm_vl.addLayout(lm_row1)
        lm_row2 = QHBoxLayout()
        lm_row2.addWidget(QLabel("轮数:"))
        self._lm_epochs = QSpinBox()
        self._lm_epochs.setRange(1, 300)
        self._lm_epochs.setValue(30)
        self._lm_epochs.setToolTip("额外训练轮数。每次点击训练都会训练这么多轮。\n续训时从上次结束的位置继续，lr_steps自动管理。")
        lm_row2.addWidget(self._lm_epochs)
        self._load_lm_pretrained = QCheckBox("微调")
        self._load_lm_pretrained.setChecked(True)
        self._load_lm_pretrained.setToolTip(
            "三种训练模式:\n"
            "1. 不勾选=正常训练: 从零开始, 需要数十万~百万级数据集, 21张图会直接崩\n"
            "2. 勾选=微调(刚性): 加载预训练权重+分阶段训练, 适合LaPa等正常人脸数据\n"
            "3. 勾选+非刚性形变增强=微调(非刚性): 额外启用TPS形变/嘴部动作/Cutout遮挡增强,\n"
            "   专门针对咀嚼/鼓腮/遮挡等非刚性形变场景\n"
            "已有训练模型时自动续训, 此选项被忽略。")
        self._load_lm_pretrained.toggled.connect(
            lambda checked: self._on_pretrain_toggled(checked, self._lm_lr, 0.01, 0.1))
        lm_row2.addWidget(self._load_lm_pretrained)
        lm_row2.addStretch()
        lm_vl.addLayout(lm_row2)
        lm_row3 = QHBoxLayout()
        lm_row3.addWidget(QLabel("主Loss:"))
        self._lm_loss_type = QComboBox()
        self._lm_loss_type.addItem("Wing Loss", "wing")
        self._lm_loss_type.addItem("Adaptive Wing Loss", "awing")
        self._lm_loss_type.setToolTip("主损失函数类型。前20%轮数用Smooth L1预热，之后切换到主Loss。\nWing Loss: 对小误差敏感，适合精细定位。\nAdaptive Wing Loss: 自适应版本，对极端样本更鲁棒。")
        lm_row3.addWidget(self._lm_loss_type)
        lm_row3.addStretch()
        lm_vl.addLayout(lm_row3)
        lm_row4 = QHBoxLayout()
        self._lm_auto_batch = QCheckBox("自动批次探测")
        self._lm_auto_batch.setToolTip("启用后自动探测最大不OOM的batch size，探测结果不超过手动设置的批次大小。")
        lm_row4.addWidget(self._lm_auto_batch)
        lm_row4.addWidget(QLabel("梯度累积:"))
        self._lm_accum_steps = QSpinBox()
        self._lm_accum_steps.setRange(1, 16)
        self._lm_accum_steps.setValue(1)
        self._lm_accum_steps.setToolTip("梯度累积步数，等效增大batch size。1=不累积。")
        lm_row4.addWidget(self._lm_accum_steps)
        lm_row4.addStretch()
        lm_vl.addLayout(lm_row4)
        self._params_area.addWidget(lm_grp)

        common_grp = QGroupBox("通用")
        common_vl = QVBoxLayout(common_grp)
        common_vl.setContentsMargins(8, 12, 8, 8)
        self._augment = QCheckBox("数据增强")
        self._augment.setChecked(True)
        self._augment.setToolTip("启用数据增强(颜色抖动、模糊、翻转、仿射变换等)，提高泛化能力。")
        common_vl.addWidget(self._augment)
        self._deform_aug = QCheckBox("非刚性形变增强")
        self._deform_aug.setChecked(False)
        self._deform_aug.setEnabled(False)
        self._deform_aug.setToolTip(
            "非刚性形变增强(TPS形变+嘴部大动作+Cutout遮挡), 专门针对咀嚼/鼓腮/遮挡等场景。\n"
            "仅在微调模式下可用。勾选后自动启用时序跳变检测。")
        common_vl.addWidget(self._deform_aug)
        self._load_lm_pretrained.toggled.connect(
            lambda checked: self._deform_aug.setEnabled(checked or self._load_scrfd_pretrained.isChecked()))
        self._deform_aug.setEnabled(self._load_lm_pretrained.isChecked() or self._load_scrfd_pretrained.isChecked())
        self._fresh_start = QCheckBox("从头训练")
        self._fresh_start.toggled.connect(self._on_fresh_start_toggled)
        self._fresh_start.setToolTip(
            "删除已训练的模型文件，从零开始重新训练。\n"
            "警告: 从头训练需要数十万~百万级数据集, 少量数据(如21张)会直接崩溃。\n"
            "数据量少时请用微调模式。\n勾选时会弹出确认警告。")
        common_vl.addWidget(self._fresh_start)
        self._deploy_onnx = QCheckBox("训练后部署到antelopev2")
        self._deploy_onnx.setToolTip(
            "训练结束后自动将导出的ONNX模型部署到antelopev2权重目录。\n"
            "首次部署前会永久备份官方权重为.onnx.official(不被覆盖)，\n"
            "后续微调始终从.official加载，避免自训权重污染初始化。")
        common_vl.addWidget(self._deploy_onnx)
        self._params_area.addWidget(common_grp)

        btn_row = QHBoxLayout()
        self._train_btn = QPushButton("开始训练")
        self._train_btn.setStyleSheet("QPushButton { background-color: #0078D4; color: white; font-weight: bold; }"
                                      "QPushButton:disabled { background-color: #cccccc; color: #666666; }")
        self._train_btn.clicked.connect(self._on_train)
        btn_row.addWidget(self._train_btn)

        self._stop_btn = QPushButton("停止")
        self._stop_btn.setProperty("busy_keep", True)
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #C42B1C; color: white; font-weight: bold; "
            "padding: 6px 14px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #D43B2C; }"
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)
        self._params_area.addLayout(btn_row)

        self._params_area.addStretch()

        self._lm_trainer = None
        self._scrfd_trainer = None
        self._train_thread = None
        self._training = False
        self._train_error = None

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

        self._all_done_signal = _ProgressSignal()
        self._all_done_signal.done_ready.connect(self._on_all_done)

        self._preview_timer = QTimer()
        self._preview_timer.timeout.connect(self._request_previews)

        self._load_saved_configs()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._training and not getattr(self, '_configs_loaded', False):
            self._configs_loaded = True
            self._load_saved_configs()
        self._preview_timer.start(5000)

    def hideEvent(self, event):
        super().hideEvent(event)
        self._preview_timer.stop()

    def _load_saved_configs(self):
        import json
        scrfd_cfg = SCRFD_MODEL_DIR / "SCRFD_config.json"
        if scrfd_cfg.exists():
            try:
                d = json.loads(scrfd_cfg.read_text(encoding='utf-8'))
                self._scrfd_batch_size.setValue(d.get('batch_size', 8))
                self._scrfd_epochs.setValue(d.get('max_epochs', 30))
                had_pretrain = bool(d.get('pretrained_onnx', ''))
                self._load_scrfd_pretrained.blockSignals(True)
                self._load_scrfd_pretrained.setChecked(had_pretrain)
                self._load_scrfd_pretrained.blockSignals(False)
                self._scrfd_lr.setValue(d.get('learning_rate', 0.001 if had_pretrain else 0.01))
            except Exception:
                pass

        lm_cfg = IF_LANDMARK_MODEL_DIR / "IFLandmark_config.json"
        if lm_cfg.exists():
            try:
                d = json.loads(lm_cfg.read_text(encoding='utf-8'))
                self._lm_batch_size.setValue(d.get('batch_size', 32))
                self._lm_epochs.setValue(d.get('max_epochs', 30))
                had_pretrain = bool(d.get('pretrained_onnx', ''))
                self._load_lm_pretrained.blockSignals(True)
                self._load_lm_pretrained.setChecked(had_pretrain)
                self._load_lm_pretrained.blockSignals(False)
                self._lm_lr.setValue(d.get('learning_rate', 0.01 if had_pretrain else 0.1))
            except Exception:
                pass

        scrfd_ts = SCRFD_MODEL_DIR / "SCRFD_training_state.json"
        if scrfd_ts.exists():
            try:
                ts = json.loads(scrfd_ts.read_text(encoding='utf-8'))
                hist = ts.get('loss_history', [])
                if hist:
                    chart = self._generate_chart_from_history(hist)
                    self._set_scaled_pixmap(chart, self._scrfd_chart_label)
            except Exception:
                pass

        lm_ts = IF_LANDMARK_MODEL_DIR / "IFLandmark_training_state.json"
        if lm_ts.exists():
            try:
                ts = json.loads(lm_ts.read_text(encoding='utf-8'))
                hist = ts.get('loss_history', [])
                if hist:
                    chart = self._generate_chart_from_history(hist)
                    self._set_scaled_pixmap(chart, self._lm_chart_label)
            except Exception:
                pass

    @staticmethod
    def _generate_chart_from_history(loss_history, width: int = 600, height: int = 200) -> np.ndarray:
        canvas = np.full((height, width, 3), 30, dtype=np.uint8)
        if not loss_history or len(loss_history) < 2:
            return canvas
        epochs = [h[0] for h in loss_history]
        losses = [h[1] for h in loss_history]
        max_epoch = max(epochs)
        max_loss = max(losses) if losses else 1.0
        min_loss = min(losses) if losses else 0.0
        loss_range = max(max_loss - min_loss, 1e-6)
        margin = 40
        plot_w = width - 2 * margin
        plot_h = height - 2 * margin
        pts = []
        for ep, ls in zip(epochs, losses):
            x = margin + int((ep / max_epoch) * plot_w) if max_epoch > 0 else margin
            y = margin + plot_h - int(((ls - min_loss) / loss_range) * plot_h)
            pts.append((x, y))
        cv2.line(canvas, (margin, margin), (margin, height - margin), (100, 100, 100), 1)
        cv2.line(canvas, (margin, height - margin), (width - margin, height - margin), (100, 100, 100), 1)
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], (0, 200, 0), 2)
        cv2.putText(canvas, f"loss={losses[-1]:.6f}", (margin, margin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, f"epoch={epochs[-1]}", (width - margin - 80, height - margin + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        return canvas

    def _on_pretrain_toggled(self, checked: bool, lr_spin: QDoubleSpinBox,
                             finetune_lr: float, normal_lr: float):
        if checked:
            lr_spin.setValue(finetune_lr)
        else:
            lr_spin.setValue(normal_lr)

    def _on_fresh_start_toggled(self, checked: bool):
        if not checked:
            return
        reply = QMessageBox.warning(
            self, "从头训练",
            "将删除以前所有的训练文件，从零开始重新训练！\n\n"
            "注意: 从头训练会自动取消\"微调\"勾选(不加载预训练权重)。\n"
            "从头训练需要数十万~百万级数据集，少量数据会崩溃。\n\n"
            "确定请点击\"是\"，否则点击\"否\"。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._load_lm_pretrained.blockSignals(True)
            self._load_lm_pretrained.setChecked(False)
            self._load_lm_pretrained.blockSignals(False)
            self._load_scrfd_pretrained.blockSignals(True)
            self._load_scrfd_pretrained.setChecked(False)
            self._load_scrfd_pretrained.blockSignals(False)
            self._deform_aug.setEnabled(False)
            self._lm_lr.setValue(0.1)
            self._scrfd_lr.setValue(0.01)
        else:
            self._fresh_start.blockSignals(True)
            self._fresh_start.setChecked(False)
            self._fresh_start.blockSignals(False)

    def _request_previews(self):
        if not self._training:
            return
        if self._train_thread is not None and not self._train_thread.is_alive():
            self._check_all_done()
            return
        if self._scrfd_trainer is not None:
            self._scrfd_trainer.request_preview()
        if self._lm_trainer is not None:
            self._lm_trainer.request_preview()

    def _add_extra_dir(self):
        p = QFileDialog.getExistingDirectory(self, "选择额外数据集目录")
        if p:
            self._extra_dirs_list.addItem(p)

    def _remove_extra_dir(self):
        row = self._extra_dirs_list.currentRow()
        if row >= 0:
            self._extra_dirs_list.takeItem(row)

    def _on_train(self):
        if self._training:
            return
        do_scrfd = self._train_scrfd_check.isChecked()
        do_lm = self._train_lm_check.isChecked()
        if not do_scrfd and not do_lm:
            QMessageBox.warning(self, "警告", "请至少选择一个模型进行训练。")
            return

        self._scrfd_trainer = None
        self._lm_trainer = None
        data_dir = Path(self._dataset_combo.currentData())
        if not data_dir.exists():
            QMessageBox.warning(self, "警告", f"数据目录不存在:\n{data_dir}")
            return

        extra_data_dirs = []
        for i in range(self._extra_dirs_list.count()):
            d = Path(self._extra_dirs_list.item(i).text())
            if d.exists():
                extra_data_dirs.append(d)

        image_count = sum(1 for _ in data_dir.glob("*.jpg")) + sum(1 for _ in data_dir.glob("*.png"))
        if image_count == 0:
            QMessageBox.warning(self, "警告", f"数据目录中没有图像:\n{data_dir}")
            return

        lm_model_dir = IF_LANDMARK_MODEL_DIR
        scrfd_model_dir = SCRFD_MODEL_DIR

        if self._fresh_start.isChecked():
            import shutil
            dirs_to_clear = []
            if do_scrfd:
                dirs_to_clear.append(scrfd_model_dir)
            if do_lm:
                dirs_to_clear.append(lm_model_dir)
            for d in dirs_to_clear:
                d.mkdir(parents=True, exist_ok=True)
                if d.exists():
                    for f in d.iterdir():
                        if f.suffix in ('.pth', '.json', '.txt') or f.name == '.save_complete':
                            f.unlink()
                    bk = d / "autobackups"
                    if bk.exists():
                        shutil.rmtree(bk, ignore_errors=True)

        if do_lm:
            lm_model_dir.mkdir(parents=True, exist_ok=True)
        if do_scrfd:
            scrfd_model_dir.mkdir(parents=True, exist_ok=True)

        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        if do_scrfd and do_lm:
            self.window().statusBar().showMessage("训练中: SCRFD → Landmark...")
        elif do_scrfd:
            self.window().statusBar().showMessage("训练中: SCRFD...")
        else:
            self.window().statusBar().showMessage("训练中: Landmark...")
        self._training = True
        self._stop_requested = False
        self._set_busy(True)

        augment = self._augment.isChecked()

        scrfd_bs = self._scrfd_batch_size.value()
        scrfd_lr = self._scrfd_lr.value()
        scrfd_epochs = self._scrfd_epochs.value()
        load_scrfd_pretrained = self._load_scrfd_pretrained.isChecked()

        lm_bs = self._lm_batch_size.value()
        lm_lr = self._lm_lr.value()
        lm_epochs = self._lm_epochs.value()
        load_lm_pretrained = self._load_lm_pretrained.isChecked()
        lm_loss_type = self._lm_loss_type.currentData()
        lm_deform_aug = self._deform_aug.isChecked() and self._load_lm_pretrained.isChecked()
        lm_auto_batch = self._lm_auto_batch.isChecked()
        lm_accum_steps = self._lm_accum_steps.value()

        def _task():
            train_error = None
            try:
                if do_scrfd:
                    try:
                        from faceswap.business.scrfd_trainer import SCRFDTrainer
                        self._scrfd_trainer = SCRFDTrainer(device="auto")
                        pretrained_path = None
                        if load_scrfd_pretrained:
                            from faceswap.setting import INSIGHTFACE_MODEL_DIR, INSIGHTFACE_MODEL_PACKAGE
                            _scrfd_dir = INSIGHTFACE_MODEL_DIR / "models" / INSIGHTFACE_MODEL_PACKAGE
                            _scrfd_official = _scrfd_dir / "scrfd_10g_bnkps.onnx.official"
                            pretrained_path = str(_scrfd_official if _scrfd_official.exists() else _scrfd_dir / "scrfd_10g_bnkps.onnx")
                        scrfd_deform = self._deform_aug.isChecked() and load_scrfd_pretrained
                        self._scrfd_trainer.train(
                            data_dir=data_dir,
                            model_dir=scrfd_model_dir,
                            batch_size=scrfd_bs,
                            learning_rate=scrfd_lr,
                            max_epochs=scrfd_epochs,
                            augment=augment,
                            pretrained_onnx=pretrained_path,
                            deform_aug=scrfd_deform,
                            finetune_mode=load_scrfd_pretrained,
                            on_epoch=lambda e, l, r: self._scrfd_signals.progress_ready.emit(f"Epoch {e}/{scrfd_epochs}  loss={l:.6f}  lr={r:.6f}"),
                            on_preview=lambda img: self._scrfd_preview_signal.preview_ready.emit(img),
                            on_save=lambda e: None,
                        )
                        self._scrfd_signals.done_ready.emit("SCRFD训练完成", "")
                    except Exception as e:
                        train_error = e
                        self._scrfd_signals.error_ready.emit("SCRFD训练失败", str(e))
                    finally:
                        self._scrfd_trainer = None

                if train_error is not None or self._stop_requested:
                    return

                if do_lm:
                    try:
                        from faceswap.business.if_landmark_trainer import IFLandmarkTrainer
                        self._lm_trainer = IFLandmarkTrainer(device="auto")
                        lm_pretrained = None
                        if load_lm_pretrained:
                            from faceswap.setting import INSIGHTFACE_MODEL_DIR, INSIGHTFACE_MODEL_PACKAGE
                            _lm_dir = INSIGHTFACE_MODEL_DIR / "models" / INSIGHTFACE_MODEL_PACKAGE
                            _official = _lm_dir / "2d106det.onnx.official"
                            lm_pretrained = str(_official if _official.exists() else _lm_dir / "2d106det.onnx")
                        self._lm_trainer.train(
                            data_dir=data_dir,
                            model_dir=lm_model_dir,
                            batch_size=lm_bs,
                            learning_rate=lm_lr,
                            max_epochs=lm_epochs,
                            augment=augment,
                            pretrained_onnx=lm_pretrained,
                            loss_type=lm_loss_type,
                            deform_aug=lm_deform_aug,
                            finetune_mode=load_lm_pretrained,
                            extra_data_dirs=extra_data_dirs,
                            auto_batch=lm_auto_batch,
                            accum_steps=lm_accum_steps,
                            on_epoch=lambda e, l, r: self._lm_signals.progress_ready.emit(f"Epoch {e}/{lm_epochs}  loss={l:.6f}  lr={r:.6f}"),
                            on_preview=lambda img: self._lm_preview_signal.preview_ready.emit(img),
                            on_save=lambda e: None,
                        )
                        self._lm_signals.done_ready.emit("Landmark训练完成", "")
                    except Exception as e:
                        train_error = e
                        self._lm_signals.error_ready.emit("Landmark训练失败", str(e))
                    finally:
                        self._lm_trainer = None
            finally:
                self._train_error = train_error
                if self._fresh_start.isChecked():
                    self._fresh_start.blockSignals(True)
                    self._fresh_start.setChecked(False)
                    self._fresh_start.blockSignals(False)
                if self._deploy_onnx.isChecked() and not self._stop_requested and train_error is None:
                    try:
                        import shutil
                        from faceswap.setting import INSIGHTFACE_MODEL_DIR, INSIGHTFACE_MODEL_PACKAGE
                        antelope_dir = INSIGHTFACE_MODEL_DIR / "models" / INSIGHTFACE_MODEL_PACKAGE
                        deployed = []
                        lm_onnx = IF_LANDMARK_MODEL_DIR / "if_landmark_2d106.onnx"
                        if lm_onnx.exists():
                            dst = antelope_dir / "2d106det.onnx"
                            official = antelope_dir / "2d106det.onnx.official"
                            if dst.exists() and not official.exists():
                                shutil.copy2(str(dst), str(official))
                            if dst.exists():
                                bak = dst.with_suffix(".onnx.bak")
                                shutil.copy2(str(dst), str(bak))
                            shutil.copy2(str(lm_onnx), str(dst))
                            deployed.append(str(dst))
                        scrfd_onnx = SCRFD_MODEL_DIR / "scrfd_custom.onnx"
                        if scrfd_onnx.exists():
                            dst = antelope_dir / "scrfd_10g_bnkps.onnx"
                            official = antelope_dir / "scrfd_10g_bnkps.onnx.official"
                            if dst.exists() and not official.exists():
                                shutil.copy2(str(dst), str(official))
                            if dst.exists():
                                bak = dst.with_suffix(".onnx.bak")
                                shutil.copy2(str(dst), str(bak))
                            shutil.copy2(str(scrfd_onnx), str(dst))
                            deployed.append(str(dst))
                        self._deploy_msg = f"ONNX已部署到antelopev2:\n" + "\n".join(deployed) if deployed else ""
                    except Exception as e:
                        self._deploy_msg = f"部署失败: {e}"
                else:
                    self._deploy_msg = ""
                self._all_done_signal.done_ready.emit("", "")

        self._train_thread = threading.Thread(target=_task, daemon=True)
        self._train_thread.start()

    def _on_stop(self):
        self._stop_requested = True
        if self._scrfd_trainer is not None:
            self._scrfd_trainer.request_stop()
        if self._lm_trainer is not None:
            self._lm_trainer.request_stop()
        self._stop_btn.setEnabled(False)
        self.window().statusBar().showMessage("正在停止...")

        from PyQt6.QtCore import QTimer
        def _check_stop_timeout():
            if self._train_thread is not None and self._train_thread.is_alive():
                self.window().statusBar().showMessage("停止超时，训练线程仍在运行，请等待或关闭程序")
            else:
                self._on_all_done("", "")
                self.window().statusBar().showMessage("训练已停止")
        QTimer.singleShot(15000, _check_stop_timeout)

    def _set_scaled_pixmap(self, img: np.ndarray, label: AutoScaleLabel) -> None:
        from PyQt6.QtGui import QImage, QPixmap
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        qimg = QImage(img.tobytes(), w, h, w * 3, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        label.setRawPixmap(pixmap)

    def _on_scrfd_epoch(self, msg: str):
        if self._scrfd_trainer is not None:
            chart = self._scrfd_trainer.generate_loss_chart()
            self._set_scaled_pixmap(chart, self._scrfd_chart_label)

    def _on_lm_epoch(self, msg: str):
        if self._lm_trainer is not None:
            chart = self._lm_trainer.generate_loss_chart()
            self._set_scaled_pixmap(chart, self._lm_chart_label)

    def _on_scrfd_error(self, title: str, msg: str):
        QMessageBox.critical(self, title, msg)

    def _on_lm_error(self, title: str, msg: str):
        QMessageBox.critical(self, title, msg)

    def _on_scrfd_done(self, title: str, msg: str):
        pass

    def _on_lm_done(self, title: str, msg: str):
        pass

    def _on_all_done(self, title: str, msg: str):
        self._training = False
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._set_busy(False)
        train_error = getattr(self, '_train_error', None)
        if train_error is not None:
            self.window().statusBar().showMessage("训练失败")
            return
        self.window().statusBar().showMessage("训练完成")
        deploy_msg = getattr(self, '_deploy_msg', '')
        if deploy_msg:
            QMessageBox.information(self, "训练完成", f"训练已完成。\n{deploy_msg}")
        else:
            QMessageBox.information(self, "训练完成", "训练已完成。")

    def _check_all_done(self):
        if self._train_thread is None or not self._train_thread.is_alive():
            self._on_all_done("", "")

    def _show_scrfd_preview(self, img: np.ndarray):
        self._set_scaled_pixmap(img, self._scrfd_preview_label)

    def _show_lm_preview(self, img: np.ndarray):
        self._set_scaled_pixmap(img, self._lm_preview_label)

