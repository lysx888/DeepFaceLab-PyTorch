import numpy as np
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QGridLayout, QWidget, QCheckBox, QComboBox,
    QProgressBar, QMessageBox, QFrame, QApplication,
)

from faceswap.shared.logger import get_logger
from faceswap.shared.file_manager import FileManager

_logger = get_logger("face_delete_dialog")


class _ThumbWidget(QFrame):
    selection_changed = pyqtSignal()
    clicked = pyqtSignal(object, bool)

    def __init__(self, image_path: Path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self._selected = False
        self._highlighted = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedSize(112, 140)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(1)

        self._cb = QCheckBox()
        self._cb.setFixedSize(16, 16)
        self._cb.stateChanged.connect(self._on_cb)
        cb_row = QHBoxLayout()
        cb_row.addStretch()
        cb_row.addWidget(self._cb)
        cb_row.addStretch()
        lay.addLayout(cb_row)

        self._lbl = QLabel()
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl.setFixedSize(100, 100)
        lay.addWidget(self._lbl, 1)

        name = image_path.name
        if len(name) > 16:
            name = name[:7] + ".." + name[-7:]
        self._name_lbl = QLabel(name)
        self._name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_lbl.setStyleSheet("font-size: 10px; color: #555;")
        lay.addWidget(self._name_lbl)

    def set_pixmap(self, pixmap: QPixmap):
        self._lbl.setPixmap(pixmap)

    def _on_cb(self, state):
        self._selected = bool(state)
        self._update_style()
        self.selection_changed.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.clicked.emit(self, shift)
            if not shift:
                self._selected = not self._selected
                self._cb.setChecked(self._selected)
        super().mousePressEvent(event)

    @property
    def selected(self) -> bool:
        return self._selected

    @selected.setter
    def selected(self, val: bool):
        self._selected = val
        self._cb.setChecked(val)
        self._update_style()

    def set_highlighted(self, val: bool):
        self._highlighted = val
        self._update_style()

    def _update_style(self):
        if self._selected:
            bg = "#D4E6F1"
            border = "#0078D4"
        elif self._highlighted:
            bg = "#FFF8DC"
            border = "#DAA520"
        else:
            bg = "#FFFFFF"
            border = "#CCC"
        self.setStyleSheet(f"QFrame {{ background: {bg}; border: 1px solid {border}; border-radius: 3px; }}")


class FaceDeleteDialog(QDialog):
    _BATCH_SIZE = 50

    def __init__(self, aligned_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"删除头像 - {aligned_dir.name}")
        self.setMinimumSize(900, 650)
        self._aligned_dir = aligned_dir
        self._all_paths: list[Path] = []
        self._all_pixmaps: dict[Path, QPixmap] = {}
        self._all_embeddings: dict[Path, np.ndarray] = {}
        self._current_order: list[Path] = []
        self._thumb_widgets: list[_ThumbWidget] = []
        self._embeddings_loaded = False
        self._load_index = 0
        self._last_clicked_idx = -1

        self._build_ui()

    def showEvent(self, event):
        super().showEvent(event)
        self.showMaximized()
        if not self._all_paths:
            QTimer.singleShot(50, self._start_loading)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        toolbar.addWidget(QLabel("排序:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["文件名", "人脸相似度"])
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        toolbar.addWidget(self._sort_combo)

        self._sort_info = QLabel("")
        self._sort_info.setStyleSheet("color: #555; font-size: 12px;")
        toolbar.addWidget(self._sort_info, 1)

        self._sel_all_btn = QPushButton("全选")
        self._sel_all_btn.clicked.connect(self._select_all)
        toolbar.addWidget(self._sel_all_btn)

        self._desel_btn = QPushButton("取消全选")
        self._desel_btn.clicked.connect(self._deselect_all)
        toolbar.addWidget(self._desel_btn)

        self._del_btn = QPushButton("删除选中")
        self._del_btn.setStyleSheet(
            "QPushButton { background-color: #D45500; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
        self._del_btn.clicked.connect(self._delete_selected)
        toolbar.addWidget(self._del_btn)

        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet("color: #0078D4; font-weight: bold;")
        toolbar.addWidget(self._count_lbl)

        root.addLayout(toolbar)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(4)
        self._grid_layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(self._grid_container)
        root.addWidget(scroll, 1)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #555; font-size: 12px;")
        root.addWidget(self._status_lbl)

    def _start_loading(self):
        self._all_paths = sorted(FileManager.find_images(self._aligned_dir), key=lambda p: p.name)
        if not self._all_paths:
            self._status_lbl.setText("目录中没有图片")
            return
        self._progress.setVisible(True)
        self._progress.setRange(0, len(self._all_paths))
        self._progress.setValue(0)
        self._load_index = 0
        self._current_order = list(self._all_paths)
        self._load_timer = QTimer(self)
        self._load_timer.timeout.connect(self._load_batch)
        self._load_timer.start(0)

    def _load_batch(self):
        import cv2
        batch_end = min(self._load_index + self._BATCH_SIZE, len(self._all_paths))
        for i in range(self._load_index, batch_end):
            p = self._all_paths[i]
            img = cv2.imread(str(p))
            if img is None:
                img = np.zeros((100, 100, 3), dtype=np.uint8)
            else:
                h, w = img.shape[:2]
                scale = 100 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            rgb = img[:, :, ::-1].copy()
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimg).scaled(
                100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self._all_pixmaps[p] = pixmap
        self._load_index = batch_end
        self._progress.setValue(batch_end)
        if batch_end >= len(self._all_paths):
            self._load_timer.stop()
            self._progress.setVisible(False)
            self._populate_grid()
            self._update_status()
        QApplication.processEvents()

    def _populate_grid(self):
        for w in self._thumb_widgets:
            self._grid_layout.removeWidget(w)
            w.deleteLater()
        self._thumb_widgets.clear()

        cols = max(1, self.width() // 116)
        show_highlight = self._sort_combo.currentText() == "文件名"
        prefix_groups = self._compute_prefix_groups() if show_highlight else set()

        for i, p in enumerate(self._current_order):
            pixmap = self._all_pixmaps.get(p)
            if pixmap is None:
                continue
            tw = _ThumbWidget(p)
            tw.set_pixmap(pixmap)
            tw.selection_changed.connect(self._update_status)
            tw.clicked.connect(self._on_thumb_clicked)
            if show_highlight and p in prefix_groups:
                tw.set_highlighted(True)
            row, col = divmod(i, cols)
            self._grid_layout.addWidget(tw, row, col)
            self._thumb_widgets.append(tw)

    def _compute_prefix_groups(self) -> set:
        from collections import defaultdict
        prefix_map = defaultdict(list)
        for p in self._all_paths:
            prefix = p.stem.split("_")[0]
            prefix_map[prefix].append(p)
        multi = set()
        for paths in prefix_map.values():
            if len(paths) > 1:
                multi.update(paths)
        return multi

    def _on_sort_changed(self, idx):
        if idx == 0:
            self._current_order = list(self._all_paths)
            self._sort_info.setText("")
        elif idx == 1:
            self._sort_by_similarity()
        self._populate_grid()
        self._update_status()

    def _sort_by_similarity(self):
        if not self._embeddings_loaded:
            self._load_embeddings()
        if not self._all_embeddings:
            self._sort_info.setText("(无法计算embedding，使用文件名排序)")
            self._current_order = list(self._all_paths)
            return

        paths_with_emb = [(p, self._all_embeddings[p]) for p in self._all_paths if p in self._all_embeddings]
        paths_without = [p for p in self._all_paths if p not in self._all_embeddings]

        if not paths_with_emb:
            self._current_order = list(self._all_paths)
            return

        embeddings = np.stack([e for _, e in paths_with_emb])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        embeddings_norm = embeddings / norms

        n = len(embeddings_norm)
        visited = np.zeros(n, dtype=bool)
        order = []

        current = 0
        visited[current] = True
        order.append(paths_with_emb[current][0])

        for _ in range(n - 1):
            sims = embeddings_norm @ embeddings_norm[current]
            sims[visited] = -2.0
            next_idx = int(np.argmax(sims))
            visited[next_idx] = True
            order.append(paths_with_emb[next_idx][0])
            current = next_idx

        order.extend(paths_without)
        self._current_order = order
        self._sort_info.setText(f"(已按相似度排序，{len(paths_with_emb)}张有embedding)")

    def _load_embeddings(self):
        self._progress.setVisible(True)
        self._progress.setRange(0, len(self._all_paths))
        self._sort_info.setText("计算人脸embedding...")
        QApplication.processEvents()

        try:
            from faceswap.core.insightface_adapter import InsightFaceAdapter
            adapter = InsightFaceAdapter()
            adapter.warmup()
        except Exception as e:
            _logger.warning(f"Failed to init InsightFace: {e}")
            self._progress.setVisible(False)
            self._sort_info.setText(f"InsightFace初始化失败: {e}")
            return

        import cv2
        for i, p in enumerate(self._all_paths):
            img = cv2.imread(str(p))
            if img is not None:
                emb = adapter.extract_embedding_aligned(img)
                if emb is not None:
                    self._all_embeddings[p] = emb
            self._progress.setValue(i + 1)
            if i % 50 == 0:
                QApplication.processEvents()

        self._embeddings_loaded = True
        self._progress.setVisible(False)
        self._sort_info.setText(f"embedding计算完成: {len(self._all_embeddings)}/{len(self._all_paths)}")

    def _on_thumb_clicked(self, widget: _ThumbWidget, shift: bool):
        idx = -1
        for i, w in enumerate(self._thumb_widgets):
            if w is widget:
                idx = i
                break
        if idx < 0:
            return
        if shift and self._last_clicked_idx >= 0:
            start = min(self._last_clicked_idx, idx)
            end = max(self._last_clicked_idx, idx)
            for i in range(start, end + 1):
                self._thumb_widgets[i].selected = True
        else:
            self._last_clicked_idx = idx
        self._update_status()

    def _select_all(self):
        for w in self._thumb_widgets:
            w.selected = True
        self._update_status()

    def _deselect_all(self):
        for w in self._thumb_widgets:
            w.selected = False
        self._update_status()

    def _update_status(self):
        total = len(self._thumb_widgets)
        sel = sum(1 for w in self._thumb_widgets if w.selected)
        self._count_lbl.setText(f"选中: {sel}/{total}")
        self._status_lbl.setText(f"共 {total} 张图片")

    def _delete_selected(self):
        to_delete = [w for w in self._thumb_widgets if w.selected]
        if not to_delete:
            return

        reply = QMessageBox.warning(
            self, "确认删除",
            f"确定要删除 {len(to_delete)} 张图片及其元数据文件吗？\n此操作不可撤销！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        deleted = 0
        for w in to_delete:
            img_path = w.image_path
            json_path = img_path.with_suffix(".json")
            try:
                if img_path.exists():
                    img_path.unlink()
                if json_path.exists():
                    json_path.unlink()
                deleted += 1
            except OSError as e:
                _logger.warning(f"Failed to delete {img_path}: {e}")

        delete_set = {w.image_path for w in to_delete}
        self._all_paths = [p for p in self._all_paths if p not in delete_set]
        self._current_order = [p for p in self._current_order if p not in delete_set]
        for p in delete_set:
            self._all_pixmaps.pop(p, None)
            self._all_embeddings.pop(p, None)

        for w in to_delete:
            self._grid_layout.removeWidget(w)
            w.deleteLater()
        self._thumb_widgets = [w for w in self._thumb_widgets if w.image_path not in delete_set]

        self._reposition_grid()
        self._update_status()
        QMessageBox.information(self, "删除完成", f"已删除 {deleted} 张图片及对应元数据")

    def _reposition_grid(self):
        cols = max(1, self.width() // 116)
        for i, w in enumerate(self._thumb_widgets):
            row, col = divmod(i, cols)
            self._grid_layout.addWidget(w, row, col)
