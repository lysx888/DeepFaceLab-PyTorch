import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from DeepFaceLab.models.tfm_model import _TFM_PRESETS
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("tfm_param_defs")


class ParamType(str, Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"


class ParamGroup(str, Enum):
    BASIC = "basic"
    ARCHITECTURE = "arch"
    FACE_DETAIL = "face"
    LOSS_SAMPLING = "loss"


GROUP_COLORS = {
    ParamGroup.BASIC: "#0078D4",
    ParamGroup.ARCHITECTURE: "#CA5010",
    ParamGroup.FACE_DETAIL: "#0F8B4D",
    ParamGroup.LOSS_SAMPLING: "#7A7574",
}

GROUP_NAMES = {
    ParamGroup.BASIC: "基础训练",
    ParamGroup.ARCHITECTURE: "模型架构",
    ParamGroup.FACE_DETAIL: "人脸细节",
    ParamGroup.LOSS_SAMPLING: "损失函数与数据采样",
}


@dataclass
class ParamDef:
    key: str
    label: str
    type: ParamType
    default: Any
    min_val: Optional[Union[int, float]] = None
    max_val: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    choices: Optional[list] = None
    group: ParamGroup = ParamGroup.BASIC
    decimals: int = 4
    preset_controlled: bool = False
    align_multiple: Optional[int] = None


TFM_PARAM_DEFS: list[ParamDef] = [
    ParamDef(key="resolution", label="分辨率:", type=ParamType.INT, default=128, min_val=64, max_val=256, step=16, group=ParamGroup.BASIC, align_multiple=16),
    ParamDef(key="face_type", label="人脸类型:", type=ParamType.STR, default="whole_face", choices=["half", "mid_full", "full", "whole_face", "head"], group=ParamGroup.BASIC),
    ParamDef(key="batch_size", label="批次大小:", type=ParamType.INT, default=4, min_val=1, max_val=64, group=ParamGroup.BASIC),
    ParamDef(key="learning_rate", label="学习率:", type=ParamType.FLOAT, default=1e-4, min_val=1e-6, max_val=1e-2, step=1e-5, decimals=6, group=ParamGroup.BASIC),
    ParamDef(key="use_amp", label="混合精度", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="random_warp", label="随机变形", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="gan_power", label="GAN强度:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=5.0, step=0.1, decimals=2, group=ParamGroup.BASIC),
    ParamDef(key="random_hsv_power", label="HSV增强:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=0.3, step=0.01, decimals=3, group=ParamGroup.BASIC),
    ParamDef(key="lr_schedule", label="学习率调度:", type=ParamType.STR, default="constant", choices=["constant", "cosine_annealing"], group=ParamGroup.BASIC),
    ParamDef(key="save_interval_min", label="保存间隔(分):", type=ParamType.INT, default=15, min_val=1, max_val=120, group=ParamGroup.BASIC),
    ParamDef(key="preview_interval_sec", label="预览间隔(秒):", type=ParamType.INT, default=60, min_val=10, max_val=600, group=ParamGroup.BASIC),
    ParamDef(key="gradient_clip", label="梯度裁剪:", type=ParamType.FLOAT, default=1.0, min_val=0.0, max_val=10.0, step=0.1, decimals=2, group=ParamGroup.BASIC),
    ParamDef(key="random_flip", label="随机翻转", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="color_transfer", label="颜色迁移:", type=ParamType.STR, default="none", choices=["none", "rct", "mkl"], group=ParamGroup.BASIC),
    ParamDef(key="model_preset", label="模型预设:", type=ParamType.STR, default="medium", choices=["tiny", "small", "medium", "large"], group=ParamGroup.ARCHITECTURE),
    ParamDef(key="window_size", label="窗口大小:", type=ParamType.STR, default="8", choices=["4", "8", "16"], group=ParamGroup.ARCHITECTURE),
    ParamDef(key="skip_strength", label="Skip强度:", type=ParamType.FLOAT, default=0.5, min_val=0.0, max_val=1.0, step=0.1, decimals=2, group=ParamGroup.ARCHITECTURE),
    ParamDef(key="gradient_checkpoint", label="梯度检查点", type=ParamType.BOOL, default=False, group=ParamGroup.ARCHITECTURE),
    ParamDef(key="use_compile", label="torch.compile", type=ParamType.BOOL, default=False, group=ParamGroup.ARCHITECTURE),
    ParamDef(key="embed_dim", label="嵌入维度:", type=ParamType.INT, default=96, min_val=16, max_val=256, group=ParamGroup.ARCHITECTURE, preset_controlled=True),
    ParamDef(key="depths", label="块数:", type=ParamType.STR, default="[2,2,6,2]", group=ParamGroup.ARCHITECTURE, preset_controlled=True),
    ParamDef(key="num_heads", label="头数:", type=ParamType.STR, default="[3,6,12,24]", group=ParamGroup.ARCHITECTURE, preset_controlled=True),
    ParamDef(key="base_channels", label="基础通道:", type=ParamType.INT, default=512, min_val=32, max_val=1024, group=ParamGroup.ARCHITECTURE, preset_controlled=True),
    ParamDef(key="w_dim", label="W+维度:", type=ParamType.INT, default=512, min_val=64, max_val=1024, group=ParamGroup.ARCHITECTURE, preset_controlled=True),
    ParamDef(key="eye_priority", label="眼睛优先:", type=ParamType.FLOAT, default=1.0, min_val=0.5, max_val=5.0, step=0.1, decimals=2, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="mouth_priority", label="嘴巴优先:", type=ParamType.FLOAT, default=1.0, min_val=0.5, max_val=5.0, step=0.1, decimals=2, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="nose_priority", label="鼻子优先:", type=ParamType.FLOAT, default=1.0, min_val=0.5, max_val=3.0, step=0.1, decimals=2, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="jaw_priority", label="下颌优先:", type=ParamType.FLOAT, default=1.0, min_val=0.5, max_val=3.0, step=0.1, decimals=2, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="face_style_power", label="脸部风格:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=1.0, step=0.01, decimals=3, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="bg_style_power", label="背景风格:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=1.0, step=0.01, decimals=3, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="enable_mask", label="启用遮罩", type=ParamType.BOOL, default=True, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="perceptual_weight", label="感知损失:", type=ParamType.FLOAT, default=0.1, min_val=0.0, max_val=1.0, step=0.01, decimals=3, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="identity_weight", label="身份保持:", type=ParamType.FLOAT, default=0.1, min_val=0.0, max_val=1.0, step=0.01, decimals=3, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="uniform_yaw_sampling", label="均匀角度采样", type=ParamType.BOOL, default=False, group=ParamGroup.LOSS_SAMPLING),
]

OLD_MODEL_PARAMS = {
    "SAEHD": [
        ParamDef(key="resolution", label="分辨率:", type=ParamType.INT, default=128, min_val=64, max_val=640, step=16, align_multiple=16),
        ParamDef(key="face_type", label="人脸类型:", type=ParamType.STR, default="whole_face", choices=["half", "mid_full", "full", "whole_face", "head"]),
        ParamDef(key="architecture", label="架构:", type=ParamType.STR, default="df", choices=["df", "liae"]),
        ParamDef(key="ae_dims", label="自编码器维度:", type=ParamType.INT, default=256, min_val=32, max_val=1024),
        ParamDef(key="e_dims", label="编码器维度:", type=ParamType.INT, default=64, min_val=16, max_val=256),
        ParamDef(key="d_dims", label="解码器维度:", type=ParamType.INT, default=64, min_val=16, max_val=256),
        ParamDef(key="batch_size", label="批次大小:", type=ParamType.INT, default=4, min_val=1, max_val=64),
        ParamDef(key="learning_rate", label="学习率:", type=ParamType.FLOAT, default=1e-4, min_val=1e-6, max_val=1e-2, step=1e-5, decimals=6),
        ParamDef(key="use_amp", label="混合精度", type=ParamType.BOOL, default=True),
        ParamDef(key="random_warp", label="随机变形", type=ParamType.BOOL, default=True),
        ParamDef(key="gan_power", label="GAN强度:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=5.0, step=0.1, decimals=2),
        ParamDef(key="random_hsv_power", label="HSV增强:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=0.3, step=0.01, decimals=3),
    ],
    "Quick96": [
        ParamDef(key="batch_size", label="批次大小:", type=ParamType.INT, default=4, min_val=1, max_val=64),
        ParamDef(key="learning_rate", label="学习率:", type=ParamType.FLOAT, default=1e-4, min_val=1e-6, max_val=1e-2, step=1e-5, decimals=6),
        ParamDef(key="use_amp", label="混合精度", type=ParamType.BOOL, default=True),
    ],
    "AMP": [
        ParamDef(key="resolution", label="分辨率:", type=ParamType.INT, default=128, min_val=64, max_val=640, step=16),
        ParamDef(key="batch_size", label="批次大小:", type=ParamType.INT, default=4, min_val=1, max_val=64),
        ParamDef(key="learning_rate", label="学习率:", type=ParamType.FLOAT, default=1e-4, min_val=1e-6, max_val=1e-2, step=1e-5, decimals=6),
        ParamDef(key="use_amp", label="混合精度", type=ParamType.BOOL, default=True),
        ParamDef(key="src_src_mode", label="SRC-SRC模式", type=ParamType.BOOL, default=False),
    ],
}


def get_params_by_group(group: ParamGroup) -> list[ParamDef]:
    return [p for p in TFM_PARAM_DEFS if p.group == group]


class ConfigManager:
    def __init__(self, model_dir: Path) -> None:
        self._model_dir = Path(model_dir)
        self._config_path = self._model_dir / "TFM_training_config.json"

    def _ensure_dir(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, values: dict) -> None:
        try:
            self._ensure_dir()
            self._config_path.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            _logger.warning(f"Failed to save TFM config: {e}")

    def load_config(self) -> Optional[dict]:
        if not self._config_path.exists():
            return None
        try:
            return json.loads(self._config_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @property
    def config_path(self) -> Path:
        return self._config_path


class ParamGroupWidget(QGroupBox):
    def __init__(self, group: ParamGroup, params: Optional[list[ParamDef]] = None) -> None:
        group_color = GROUP_COLORS.get(group, "#0078D4")
        group_name = GROUP_NAMES.get(group, "")
        super().__init__(group_name)
        self._group = group
        self._params = params if params is not None else get_params_by_group(group)
        self._param_widgets: dict[str, QWidget] = {}
        self._param_defs: dict[str, ParamDef] = {}
        self.setStyleSheet(
            f"QGroupBox::title {{ color: {group_color}; }}"
        )
        self._build_rows()

    def _build_rows(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(12, 20, 12, 8)

        max_label_w = 0
        from PyQt6.QtGui import QFontMetrics, QFont
        font = QFont("Segoe UI", 13)
        fm = QFontMetrics(font)
        for p in self._params:
            tw = fm.horizontalAdvance(p.label) + 8
            if tw > max_label_w:
                max_label_w = tw

        for row_start in range(0, len(self._params), 3):
            row_params = self._params[row_start:row_start + 3]
            row_layout = QHBoxLayout()
            row_layout.setSpacing(12)
            for p in row_params:
                unit = QHBoxLayout()
                unit.setSpacing(4)
                lbl = QLabel(p.label)
                lbl.setFixedWidth(max_label_w)
                lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                widget = self._create_widget(p)
                unit.addWidget(lbl)
                unit.addWidget(widget, 1)
                self._param_widgets[p.key] = widget
                self._param_defs[p.key] = p
                row_layout.addLayout(unit, 1)
            remaining = 3 - len(row_params)
            for _ in range(remaining):
                row_layout.addStretch(1)
            layout.addLayout(row_layout)

    def _create_widget(self, param: ParamDef) -> QWidget:
        if param.type == ParamType.INT:
            w = QSpinBox()
            if param.min_val is not None:
                w.setMinimum(param.min_val)
            if param.max_val is not None:
                w.setMaximum(param.max_val)
            if param.step is not None:
                w.setSingleStep(param.step)
            w.setValue(int(param.default))
            if param.align_multiple:
                def _align_int(val, m=param.align_multiple, sb=w):
                    aligned = (val // m) * m
                    if aligned < sb.minimum():
                        aligned = sb.minimum()
                    if aligned != val:
                        sb.setValue(aligned)
                w.valueChanged.connect(_align_int)
            return w
        elif param.type == ParamType.FLOAT:
            w = QDoubleSpinBox()
            if param.min_val is not None:
                w.setMinimum(param.min_val)
            if param.max_val is not None:
                w.setMaximum(param.max_val)
            if param.step is not None:
                w.setSingleStep(param.step)
            w.setDecimals(param.decimals)
            w.setValue(float(param.default))
            return w
        elif param.type == ParamType.BOOL:
            w = QCheckBox()
            w.setChecked(bool(param.default))
            return w
        elif param.type == ParamType.STR:
            if param.choices:
                w = QComboBox()
                w.addItems([str(c) for c in param.choices])
                idx = w.findText(str(param.default))
                if idx >= 0:
                    w.setCurrentIndex(idx)
                return w
            else:
                w = QLineEdit(str(param.default))
                return w
        return QLineEdit(str(param.default))

    def get_values(self) -> dict:
        result = {}
        for key, widget in self._param_widgets.items():
            pdef = self._param_defs[key]
            if pdef.type == ParamType.INT:
                result[key] = widget.value()
            elif pdef.type == ParamType.FLOAT:
                result[key] = widget.value()
            elif pdef.type == ParamType.BOOL:
                result[key] = widget.isChecked()
            elif pdef.type == ParamType.STR:
                if isinstance(widget, QComboBox):
                    result[key] = widget.currentText()
                else:
                    result[key] = widget.text()
        return result

    def set_values(self, vals: dict) -> None:
        for key, widget in self._param_widgets.items():
            if key not in vals:
                continue
            val = vals[key]
            pdef = self._param_defs[key]
            if pdef.type == ParamType.INT:
                widget.setValue(int(val))
            elif pdef.type == ParamType.FLOAT:
                widget.setValue(float(val))
            elif pdef.type == ParamType.BOOL:
                widget.setChecked(bool(val))
            elif pdef.type == ParamType.STR:
                if isinstance(widget, QComboBox):
                    idx = widget.findText(str(val))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                else:
                    widget.setText(str(val))

    def set_editable(self, editable: bool, keys: Optional[list[str]] = None) -> None:
        for key, widget in self._param_widgets.items():
            if keys is not None and key not in keys:
                continue
            widget.setEnabled(editable)

    def get_widget(self, key: str) -> Optional[QWidget]:
        return self._param_widgets.get(key)


class PresetManager:
    def __init__(self, arch_group: ParamGroupWidget, preset_combo: QComboBox) -> None:
        self._arch_group = arch_group
        self._preset_combo = preset_combo
        self._preset_map = _TFM_PRESETS
        preset_combo.currentTextChanged.connect(self.on_preset_changed)

    def on_preset_changed(self, preset_name: str) -> None:
        if preset_name not in self._preset_map:
            return
        cfg = self._preset_map[preset_name]
        arch_values = {
            "embed_dim": cfg["embed_dim"],
            "depths": str(cfg["depths"]),
            "num_heads": str(cfg["num_heads"]),
            "base_channels": cfg["base_channels"],
            "w_dim": cfg["w_dim"],
        }
        self._arch_group.set_values(arch_values)
        self._arch_group.set_editable(False, keys=["embed_dim", "depths", "num_heads", "base_channels", "w_dim"])

    def apply_saved_preset(self, saved_config: dict) -> None:
        preset = saved_config.get("model_preset", "medium")
        idx = self._preset_combo.findText(preset)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
        self.on_preset_changed(preset)

    def get_current_preset(self) -> str:
        return self._preset_combo.currentText()


class TrainingSignals(QObject):
    iter_signal = pyqtSignal(int, float, float)
    preview_signal = pyqtSignal(object)
    save_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    log_signal = pyqtSignal(str, bool)


class TrainingStatusBar(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)

        self._status_label = QLabel("就绪")
        self._status_label.setStyleSheet("color: #333; font-size: 13px;")
        layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setTextVisible(False)
        layout.addWidget(self._progress_bar)

    def update_status(self, iter_num: int, loss: float, ms: float, lr: float = 0.0) -> None:
        time_str = self._format_time(ms)
        lr_str = f"{lr:.1e}" if lr > 0 else ""
        text = f"迭代: #{iter_num} | 损失: {loss:.4f} | 耗时: {time_str}"
        if lr_str:
            text += f" | 学习率: {lr_str}"
        self._status_label.setText(text)

    def _format_time(self, ms: float) -> str:
        total_sec = int(ms / 1000)
        minutes = total_sec // 60
        seconds = total_sec % 60
        return f"{minutes}:{seconds:02d}"

    def start_pulse(self) -> None:
        self._progress_bar.setRange(0, 0)

    def stop_pulse(self) -> None:
        self._progress_bar.setRange(0, 1)
        self._progress_bar.reset()

    def reset(self) -> None:
        self._status_label.setText("就绪")
        self.stop_pulse()


class PreviewThumbnailBar(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 4, 0, 0)
        self._layout.setSpacing(4)
        self._thumbnail_labels: list[QLabel] = []
        self.setFixedHeight(120)

    def update_preview(self, preview_bgr: np.ndarray) -> None:
        if preview_bgr is None or not isinstance(preview_bgr, np.ndarray):
            return
        thumbnails = self._split_preview(preview_bgr)
        if not thumbnails:
            return
        while len(self._thumbnail_labels) < len(thumbnails):
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._layout.addWidget(lbl, 1)
            self._thumbnail_labels.append(lbl)
        for i, thumb_bgr in enumerate(thumbnails):
            if i >= len(self._thumbnail_labels):
                break
            h, w = thumb_bgr.shape[:2]
            scale = 1.1 if i == len(thumbnails) // 2 else 1.0
            th = int(110 * scale)
            tw = int(th * w / max(h, 1))
            thumb_rgb = thumb_bgr[:, :, ::-1].copy()
            qimg = QImage(thumb_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self._thumbnail_labels[i].setPixmap(pixmap)
            self._thumbnail_labels[i].setFixedHeight(th + 4)

    def _split_preview(self, bgr: np.ndarray) -> list[np.ndarray]:
        h, w = bgr.shape[:2]
        cols = max(1, w // h) if h > 0 else 1
        col_w = w // cols
        result = []
        for i in range(cols):
            x1 = i * col_w
            x2 = (i + 1) * col_w if i < cols - 1 else w
            result.append(bgr[:, x1:x2])
        return result

    def clear(self) -> None:
        for lbl in self._thumbnail_labels:
            lbl.clear()


class AdvancedModelSection(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._collapsed = True
        self._is_training = False
        self._old_trainer = None
        self._old_thread = None
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 8, 0, 0)
        outer.setSpacing(0)

        self._toggle_btn = QPushButton("▶ 高级模型（SAEHD / Quick96 / AMP）")
        self._toggle_btn.setProperty("neutral", True)
        self._toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 8px 12px; font-weight: 600; color: #7A7574; background-color: #F0F0F0; border: 1px solid #D6D6D6; border-radius: 4px; }"
            "QPushButton:hover { background-color: #E4E4E4; }"
        )
        self._toggle_btn.clicked.connect(self._toggle_collapsed)
        outer.addWidget(self._toggle_btn)

        self._content = QWidget()
        self._content.setVisible(False)
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(12, 8, 12, 8)
        content_layout.setSpacing(6)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("模型类型:"))
        self._model_type_combo = QComboBox()
        self._model_type_combo.addItems(["SAEHD", "Quick96", "AMP"])
        self._model_type_combo.currentTextChanged.connect(self._on_old_model_type_changed)
        type_row.addWidget(self._model_type_combo, 1)
        content_layout.addLayout(type_row)

        self._old_params_widget = QWidget()
        self._old_params_layout = QVBoxLayout(self._old_params_widget)
        self._old_params_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self._old_params_widget)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._old_run_btn = QPushButton("开始")
        self._old_run_btn.setFixedWidth(120)
        self._old_run_btn.clicked.connect(self._on_old_run)
        btn_row.addWidget(self._old_run_btn)
        self._old_stop_btn = QPushButton("停止")
        self._old_stop_btn.setProperty("danger", True)
        self._old_stop_btn.setFixedWidth(80)
        self._old_stop_btn.setEnabled(False)
        self._old_stop_btn.clicked.connect(self._on_old_stop)
        btn_row.addWidget(self._old_stop_btn)
        content_layout.addLayout(btn_row)

        outer.addWidget(self._content)
        self._on_old_model_type_changed("SAEHD")

    def _toggle_collapsed(self) -> None:
        self._collapsed = not self._collapsed
        self._content.setVisible(not self._collapsed)
        arrow = "▶" if self._collapsed else "▼"
        self._toggle_btn.setText(f"{arrow} 高级模型（SAEHD / Quick96 / AMP）")

    def _on_old_model_type_changed(self, model_type: str) -> None:
        while self._old_params_layout.count():
            item = self._old_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()
        self._old_param_widgets: dict[str, QWidget] = {}
        self._old_param_defs: dict[str, ParamDef] = {}
        params = OLD_MODEL_PARAMS.get(model_type, [])
        for row_start in range(0, len(params), 3):
            row_params = params[row_start:row_start + 3]
            row = QHBoxLayout()
            for p in row_params:
                unit = QHBoxLayout()
                lbl = QLabel(p.label)
                lbl.setFixedWidth(100)
                lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                w = self._create_widget(p)
                unit.addWidget(lbl)
                unit.addWidget(w, 1)
                self._old_param_widgets[p.key] = w
                self._old_param_defs[p.key] = p
                row.addLayout(unit, 1)
            remaining = 3 - len(row_params)
            for _ in range(remaining):
                row.addStretch(1)
            self._old_params_layout.addLayout(row)

    def _create_widget(self, param: ParamDef) -> QWidget:
        if param.type == ParamType.INT:
            w = QSpinBox()
            if param.min_val is not None:
                w.setMinimum(param.min_val)
            if param.max_val is not None:
                w.setMaximum(param.max_val)
            if param.step is not None:
                w.setSingleStep(param.step)
            w.setValue(int(param.default))
            return w
        elif param.type == ParamType.FLOAT:
            w = QDoubleSpinBox()
            if param.min_val is not None:
                w.setMinimum(param.min_val)
            if param.max_val is not None:
                w.setMaximum(param.max_val)
            if param.step is not None:
                w.setSingleStep(param.step)
            w.setDecimals(param.decimals)
            w.setValue(float(param.default))
            return w
        elif param.type == ParamType.BOOL:
            w = QCheckBox()
            w.setChecked(bool(param.default))
            return w
        elif param.type == ParamType.STR:
            if param.choices:
                w = QComboBox()
                w.addItems([str(c) for c in param.choices])
                idx = w.findText(str(param.default))
                if idx >= 0:
                    w.setCurrentIndex(idx)
                return w
            else:
                w = QLineEdit(str(param.default))
                return w
        return QLineEdit(str(param.default))

    def _on_old_run(self) -> None:
        from DeepFaceLab.business.model_trainer import ModelTrainer, ModelType
        import threading

        mt_map = {"SAEHD": ModelType.SAEHD, "Quick96": ModelType.QUICK96, "AMP": ModelType.AMP}
        mt = mt_map.get(self._model_type_combo.currentText(), ModelType.SAEHD)

        params = {}
        for key, widget in getattr(self, '_old_param_widgets', {}).items():
            pdef = self._old_param_defs.get(key)
            if pdef is None:
                continue
            if pdef.type == ParamType.INT:
                params[key] = widget.value()
            elif pdef.type == ParamType.FLOAT:
                params[key] = widget.value()
            elif pdef.type == ParamType.BOOL:
                params[key] = widget.isChecked()
            elif pdef.type == ParamType.STR:
                if isinstance(widget, QComboBox):
                    params[key] = widget.currentText()
                else:
                    params[key] = widget.text()

        trainer = ModelTrainer()
        self._old_trainer = trainer
        self._is_training = True
        self._old_run_btn.setEnabled(False)
        self._old_stop_btn.setEnabled(True)

        def _task():
            try:
                from DeepFaceLab.setting import DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR, MODEL_DIR
                trainer.train(
                    mt,
                    DATA_SRC_ALIGNED_DIR,
                    DATA_DST_ALIGNED_DIR,
                    MODEL_DIR,
                    **params,
                )
            except Exception as e:
                _logger.error(f"Old model training error: {e}")
            finally:
                self._is_training = False

        self._old_thread = threading.Thread(target=_task, daemon=True)
        self._old_thread.start()

    def _on_old_stop(self) -> None:
        self._is_training = False
        self._old_run_btn.setEnabled(True)
        self._old_stop_btn.setEnabled(False)

    def set_enabled(self, enabled: bool) -> None:
        self._old_run_btn.setEnabled(enabled and not self._is_training)

    def is_training(self) -> bool:
        return self._is_training
