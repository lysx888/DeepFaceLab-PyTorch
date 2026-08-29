import json
import shutil
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QMessageBox, QDialog, QLineEdit, QVBoxLayout, QDialogButtonBox,
)


class ModelNameSelector(QWidget):
    model_changed = pyqtSignal(str)

    def __init__(self, base_dir: Path, config_filename: str = "SAEHD_training_config.json",
                 state_filename: str = "SAEHD_training_state.json",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._base_dir = Path(base_dir)
        self._config_filename = config_filename
        self._state_filename = state_filename
        self._is_new_action = False
        self._build_ui()
        self.refresh_models()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        lbl = QLabel("模型名称:")
        lbl.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(lbl)

        self._combo = QComboBox()
        self._combo.setMinimumWidth(220)
        self._combo.setStyleSheet("font-size: 13px; padding: 2px 4px;")
        self._combo.currentTextChanged.connect(self._on_combo_changed)
        layout.addWidget(self._combo, 1)

        _btn_style = (
            "QPushButton { font-size: 13px; font-weight: bold; "
            "padding: 5px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #E0E0E0; }"
        )
        self._new_btn = QPushButton("新建")
        self._new_btn.setMinimumWidth(90)
        self._new_btn.setStyleSheet(_btn_style)
        self._new_btn.clicked.connect(self._on_new)
        layout.addWidget(self._new_btn)

        self._rename_btn = QPushButton("重命名")
        self._rename_btn.setMinimumWidth(100)
        self._rename_btn.setStyleSheet(_btn_style)
        self._rename_btn.clicked.connect(self._on_rename)
        layout.addWidget(self._rename_btn)

        self._delete_btn = QPushButton("删除")
        self._delete_btn.setMinimumWidth(90)
        self._delete_btn.setStyleSheet(
            "QPushButton { font-size: 13px; font-weight: bold; "
            "padding: 5px 16px; border-radius: 4px; color: #C42B1C; }"
            "QPushButton:hover { background-color: #FCE4E0; }"
        )
        self._delete_btn.clicked.connect(self._on_delete)
        layout.addWidget(self._delete_btn)

        layout.addSpacing(20)

        self._train_btn = QPushButton("开始训练")
        self._train_btn.setMinimumWidth(120)
        self._train_btn.setStyleSheet(
            "QPushButton { font-size: 13px; font-weight: bold; "
            "padding: 5px 16px; border-radius: 4px; "
            "background-color: #D45500; color: white; }"
            "QPushButton:hover { background-color: #E06010; }"
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
        layout.addWidget(self._train_btn)

        self._stop_btn = QPushButton("停止")
        self._stop_btn.setMinimumWidth(80)
        self._stop_btn.setStyleSheet(
            "QPushButton { font-size: 13px; font-weight: bold; "
            "padding: 5px 16px; border-radius: 4px; "
            "background-color: #C42B1C; color: white; }"
            "QPushButton:hover { background-color: #D43B2C; }"
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
        self._stop_btn.setEnabled(False)
        layout.addWidget(self._stop_btn)

    @property
    def train_btn(self) -> QPushButton:
        return self._train_btn

    @property
    def stop_btn(self) -> QPushButton:
        return self._stop_btn

    def refresh_models(self) -> None:
        self._combo.blockSignals(True)
        self._combo.clear()

        model_names = self._scan_models()

        for name in model_names:
            self._combo.addItem(name)

        if not model_names:
            self._combo.addItem("new")

        self._combo.setCurrentIndex(0)
        self._combo.blockSignals(False)
        self.model_changed.emit(self.current_name())

    def _scan_models(self) -> list[str]:
        if not self._base_dir.exists():
            return []

        _EXCLUDED = {"autobackups"}
        names = []
        for sub in sorted(self._base_dir.iterdir()):
            if not sub.is_dir() or sub.name in _EXCLUDED:
                continue
            names.append(sub.name)

        return names

    def current_name(self) -> str:
        return self._combo.currentText()

    def resolve_dir(self) -> Path:
        return self._base_dir / self.current_name()

    def resolve_dir_name(self) -> str:
        return self.current_name()

    def current_dir(self) -> Path:
        return self.resolve_dir()

    def has_training_state(self) -> bool:
        return (self.current_dir() / self._state_filename).exists()

    def has_config(self) -> bool:
        return (self.current_dir() / self._config_filename).exists()

    @property
    def is_new_action(self) -> bool:
        return self._is_new_action

    def update_current_name(self, name: str) -> None:
        idx = self._combo.findText(name)
        if idx < 0:
            self._combo.blockSignals(True)
            self._combo.addItem(name)
            idx = self._combo.count() - 1
            self._combo.blockSignals(False)
        self._combo.setCurrentIndex(idx)

    def _on_combo_changed(self, name: str) -> None:
        self._is_new_action = False
        self.model_changed.emit(name)

    def _on_new(self) -> None:
        name = self._input_dialog("新建模型", "请输入模型名称:", "")
        if not name:
            return
        self.update_current_name(name)
        self._is_new_action = True

    def _on_rename(self) -> None:
        old_name = self.current_name()
        new_name = self._input_dialog("重命名模型", f"将 '{old_name}' 重命名为:", old_name)
        if not new_name or new_name == old_name:
            return

        old_dir = self._base_dir / old_name
        new_dir = self._base_dir / new_name
        if new_dir.exists():
            QMessageBox.warning(self, "名称已存在", f"名称 '{new_name}' 已存在。")
            return

        if old_dir.exists():
            old_dir.rename(new_dir)

        self.refresh_models()
        idx = self._combo.findText(new_name)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)

    def _on_delete(self) -> None:
        name = self.current_name()
        reply = QMessageBox.question(
            self, "删除模型",
            f"确定要删除模型 '{name}' 及其所有文件吗？\n此操作不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        model_dir = self._base_dir / name
        if model_dir.exists():
            shutil.rmtree(str(model_dir))
        self.refresh_models()

    def _input_dialog(self, title: str, label: str, default: str) -> Optional[str]:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setFixedWidth(300)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(label))
        edit = QLineEdit(default)
        layout.addWidget(edit)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return edit.text().strip()
        return None

    def set_enabled(self, enabled: bool) -> None:
        self._combo.setEnabled(enabled)
        self._new_btn.setEnabled(enabled)
        self._rename_btn.setEnabled(enabled)
        self._delete_btn.setEnabled(enabled)
