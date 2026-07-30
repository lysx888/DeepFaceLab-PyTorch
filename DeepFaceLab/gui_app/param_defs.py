import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QProgressBar,
    QSpinBox, QVBoxLayout, QWidget,
)

from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("param_defs")


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
    ParamGroup.LOSS_SAMPLING: "损失函数与采样",
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


class ConfigManager:
    def __init__(self, model_dir: Path, config_name: str = "training_config.json") -> None:
        self._model_dir = Path(model_dir)
        self._config_path = self._model_dir / config_name

    def _ensure_dir(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, values: dict) -> None:
        try:
            self._ensure_dir()
            self._config_path.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            _logger.warning(f"Failed to save config: {e}")

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
        self._params = params if params is not None else []
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
            qimg = QImage(thumb_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
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
