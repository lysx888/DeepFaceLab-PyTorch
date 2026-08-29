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

from faceswap.shared.logger import get_logger

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
    OPTIMIZATION = "optimization"


GROUP_COLORS = {
    ParamGroup.BASIC: "#0078D4",
    ParamGroup.ARCHITECTURE: "#CA5010",
    ParamGroup.FACE_DETAIL: "#0F8B4D",
    ParamGroup.LOSS_SAMPLING: "#7A7574",
    ParamGroup.OPTIMIZATION: "#6B5B95",
}

GROUP_NAMES = {
    ParamGroup.BASIC: "基础训练",
    ParamGroup.ARCHITECTURE: "模型架构",
    ParamGroup.FACE_DETAIL: "人脸细节",
    ParamGroup.LOSS_SAMPLING: "损失函数与采样",
    ParamGroup.OPTIMIZATION: "优化选项",
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
    render_hint: Optional[str] = None
    tooltip: Optional[str] = None
    archi_filter: Optional[list[str]] = None


class ConfigManager:
    def __init__(self, model_dir: Path, config_name: str = "SAEHD_training_config.json") -> None:
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

    def set_model_dir(self, model_dir: Path) -> None:
        self._model_dir = Path(model_dir)
        self._config_path = self._model_dir / self._config_path.name

    @property
    def config_path(self) -> Path:
        return self._config_path


class ArchiSelector(QWidget):
    """架构选择器: df/liae 下拉框 + c/d/t/u 多选按钮。"""

    archi_changed = pyqtSignal(str)

    def __init__(self, default: str = "df", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._combo = QComboBox()
        self._combo.addItems(["df", "liae"])
        self._combo.setFixedWidth(70)
        layout.addWidget(self._combo)

        self._opts: dict[str, QCheckBox] = {}
        for opt, desc in [("c", "cos激活"), ("d", "分辨率倍增"),
                          ("t", "深层"), ("u", "像素归一化")]:
            cb = QCheckBox(opt)
            cb.setToolTip(f"{opt} {desc}")
            self._opts[opt] = cb
            layout.addWidget(cb)

        layout.addStretch(1)
        self._combo.currentTextChanged.connect(self._on_base_changed)
        self.setValue(default)

    def _on_base_changed(self, text: str) -> None:
        self.archi_changed.emit(text)

    def value(self) -> str:
        base = self._combo.currentText()
        suffix = "".join(opt for opt in ("c", "d", "t", "u")
                         if self._opts[opt].isChecked())
        return f"{base}-{suffix}" if suffix else base

    def setValue(self, val: str) -> None:
        parts = val.split("-", 1)
        base = parts[0]
        opts = parts[1] if len(parts) == 2 else ""
        idx = self._combo.findText(base)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
        for opt in ("d", "u", "t", "c"):
            self._opts[opt].setChecked(opt in opts)

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self._combo.setEnabled(enabled)
        for cb in self._opts.values():
            cb.setEnabled(enabled)


class ParamGroupWidget(QGroupBox):
    def __init__(self, group: ParamGroup, params: Optional[list[ParamDef]] = None,
                 title: Optional[str] = None, color: Optional[str] = None) -> None:
        group_color = color if color is not None else GROUP_COLORS.get(group, "#0078D4")
        group_name = title if title is not None else GROUP_NAMES.get(group, "")
        super().__init__(group_name)
        self._group = group
        self._params = params if params is not None else []
        self._param_widgets: dict[str, QWidget] = {}
        self._param_defs: dict[str, ParamDef] = {}
        self._param_labels: dict[str, Optional[QLabel]] = {}
        self.setStyleSheet(
            f"QGroupBox::title {{ color: {group_color}; }}"
        )
        self._build_rows()

    def _build_rows(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(12, 20, 12, 8)

        rows: list[list[ParamDef]] = []
        cur_row: list[ParamDef] = []
        cur_type: Optional[str] = None

        def _flush():
            nonlocal cur_row, cur_type
            if cur_row:
                rows.append(cur_row)
            cur_row = []
            cur_type = None

        for p in self._params:
            if p.render_hint in ("archi", "full_row"):
                _flush()
                rows.append([p])
                continue

            p_type = "check" if p.type == ParamType.BOOL else "long"

            if p.render_hint == "new_row":
                _flush()
            elif cur_type is not None and cur_type != p_type:
                _flush()
            elif len(cur_row) >= 3:
                _flush()

            if cur_type is None:
                cur_type = p_type
            cur_row.append(p)

        _flush()

        for row_params in rows:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(12)

            for p in row_params:
                unit = QHBoxLayout()
                unit.setSpacing(4)
                widget = self._create_widget(p)
                if p.type == ParamType.BOOL:
                    widget.setText(p.label)
                    lbl = None
                else:
                    lbl = QLabel(p.label)
                    lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                if p.tooltip:
                    tip = p.tooltip
                    widget.setToolTip(tip)
                    from faceswap.gui_app.gui_utils import install_persistent_tooltip
                    if lbl is not None:
                        lbl.setToolTip(tip)
                        install_persistent_tooltip(lbl)
                    install_persistent_tooltip(widget)
                if lbl is not None:
                    unit.addWidget(lbl)
                if p.type == ParamType.BOOL or p.render_hint == "archi":
                    unit.addWidget(widget)
                else:
                    unit.addWidget(widget, 1)
                self._param_widgets[p.key] = widget
                self._param_defs[p.key] = p
                self._param_labels[p.key] = lbl
                row_layout.addLayout(unit, 1)

            layout.addLayout(row_layout)

    def _create_widget(self, param: ParamDef) -> QWidget:
        if param.render_hint == "archi":
            w = ArchiSelector(str(param.default))
            self._install_no_wheel(w)
            return w
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
            self._install_no_wheel(w)
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
            self._install_no_wheel(w)
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
                self._install_no_wheel(w)
                return w
            else:
                w = QLineEdit(str(param.default))
                return w
        return QLineEdit(str(param.default))

    def _install_no_wheel(self, widget):
        from faceswap.gui_app.gui_utils import install_no_wheel
        install_no_wheel(widget)

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
                if isinstance(widget, ArchiSelector):
                    result[key] = widget.value()
                elif isinstance(widget, QComboBox):
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
                if isinstance(widget, ArchiSelector):
                    widget.setValue(str(val))
                elif isinstance(widget, QComboBox):
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

    def set_param_visible(self, key: str, visible: bool) -> None:
        widget = self._param_widgets.get(key)
        label = self._param_labels.get(key)
        if widget is not None:
            widget.setVisible(visible)
        if label is not None:
            label.setVisible(visible)

    def get_widget(self, key: str) -> Optional[QWidget]:
        return self._param_widgets.get(key)


class CompositeParamGroup:
    def __init__(self, group: ParamGroup, sub_widgets: Optional[list[ParamGroupWidget]] = None) -> None:
        self._group = group
        self._sub_widgets: list[ParamGroupWidget] = sub_widgets if sub_widgets is not None else []
        self._param_widgets: dict[str, QWidget] = {}
        self._param_defs: dict[str, ParamDef] = {}
        self._params: list[ParamDef] = []
        for sw in self._sub_widgets:
            self._param_widgets.update(sw._param_widgets)
            self._param_defs.update(sw._param_defs)
            self._params.extend(sw._params)

    def add_sub_widget(self, widget: ParamGroupWidget) -> None:
        self._sub_widgets.append(widget)
        self._param_widgets.update(widget._param_widgets)
        self._param_defs.update(widget._param_defs)
        self._params.extend(widget._params)

    def get_values(self) -> dict:
        result = {}
        for sw in self._sub_widgets:
            result.update(sw.get_values())
        return result

    def set_values(self, vals: dict) -> None:
        for sw in self._sub_widgets:
            sw.set_values(vals)

    def set_editable(self, editable: bool, keys: Optional[list[str]] = None) -> None:
        for sw in self._sub_widgets:
            sw.set_editable(editable, keys)

    def set_param_visible(self, key: str, visible: bool) -> None:
        for sw in self._sub_widgets:
            if key in sw._param_widgets:
                sw.set_param_visible(key, visible)

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
        self.setFixedHeight(180)

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
        bar_h = self.height() - 8
        for i, thumb_bgr in enumerate(thumbnails):
            if i >= len(self._thumbnail_labels):
                break
            h, w = thumb_bgr.shape[:2]
            th = bar_h
            tw = int(th * w / max(h, 1))
            thumb_rgb = thumb_bgr[:, :, ::-1].copy()
            qimg = QImage(thumb_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimg).scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self._thumbnail_labels[i].setPixmap(pixmap)

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
