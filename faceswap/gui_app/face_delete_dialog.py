import numpy as np
from pathlib import Path
from typing import Optional
import threading
import cv2

from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer, QThread
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QProgressBar, QMessageBox, QApplication,
    QListWidget, QListWidgetItem,
)

from faceswap.shared.logger import get_logger
from faceswap.shared.file_manager import FileManager
from faceswap.shared.image_utils import bgr_to_rgb
from faceswap.gui_app.annotator_common import load_thumbnail

_logger = get_logger("face_delete_dialog")

_THUMB_SIZE = 100


class _ThumbLoader(QThread):
    thumb_loaded = pyqtSignal(int, QIcon)
    loading_finished = pyqtSignal()

    def __init__(self, paths: list[Path]):
        super().__init__()
        self._paths = paths
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        for i, p in enumerate(self._paths):
            if self._stop:
                break
            qimg = load_thumbnail(p, _THUMB_SIZE, _THUMB_SIZE)
            if qimg is not None:
                self.thumb_loaded.emit(i, QIcon(QPixmap.fromImage(qimg)))
        self.loading_finished.emit()


class FaceDeleteDialog(QDialog):
    _emb_progress_signal = pyqtSignal(int)
    _emb_done_signal = pyqtSignal(int)
    _emb_error_signal = pyqtSignal(str)

    def __init__(self, aligned_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"删除头像 - {aligned_dir.name}")
        self.setMinimumSize(900, 650)
        self._aligned_dir = aligned_dir
        self._all_paths: list[Path] = []
        self._all_embeddings: dict[Path, np.ndarray] = {}
        self._current_order: list[Path] = []
        self._embeddings_loaded = False
        self._last_clicked_idx = -1
        self._load_started = False
        self._emb_thread = None
        self._emb_stop = False
        self._emb_pending_sort = False
        self._thumb_loader: Optional[_ThumbLoader] = None
        self._prefix_groups: set = set()
        self._path_to_row: dict = {}
        self._thumb_cache: dict[Path, QIcon] = {}

        self._emb_progress_signal.connect(self._on_emb_progress)
        self._emb_done_signal.connect(self._on_emb_done)
        self._emb_error_signal.connect(self._on_emb_error)

        self._build_ui()

    def showEvent(self, event):
        super().showEvent(event)
        self.showMaximized()
        if not self._load_started:
            self._load_started = True
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

        hint_lbl = QLabel("支持Shift多选")
        hint_lbl.setStyleSheet("color: #888; font-size: 11px;")
        toolbar.addWidget(hint_lbl)

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

        progress_row = QHBoxLayout()
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #0078D4; font-weight: bold; font-size: 12px;")
        self._progress_label.setVisible(False)
        progress_row.addWidget(self._progress_label)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        progress_row.addWidget(self._progress, 1)
        root.addLayout(progress_row)

        self._list = QListWidget()
        self._list.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setSpacing(6)
        self._list.setDragDropMode(QListWidget.DragDropMode.NoDragDrop)
        self._list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._list.setItemAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_font = self._list.font()
        thumb_font.setPixelSize(10)
        self._list.setFont(thumb_font)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_list_context_menu)
        root.addWidget(self._list, 1)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #555; font-size: 12px;")
        root.addWidget(self._status_lbl)

    def _start_loading(self):
        screen = self.screen()
        if screen is not None:
            avail_w = screen.availableGeometry().width()
            if self.width() < avail_w * 0.85:
                QTimer.singleShot(30, self._start_loading)
                return
        self._all_paths = sorted(FileManager.find_images(self._aligned_dir), key=lambda p: p.name)
        if not self._all_paths:
            self._status_lbl.setText("目录中没有图片")
            return
        self._progress.setVisible(True)
        self._progress.setRange(0, len(self._all_paths))
        self._progress.setValue(0)
        self._progress_label.setText("正在加载缩略图...")
        self._progress_label.setVisible(True)
        self._current_order = list(self._all_paths)
        self._update_prefix_info()
        self._populate_list()

        self._thumb_loader = _ThumbLoader(self._all_paths)
        self._thumb_loader.thumb_loaded.connect(self._on_thumb_loaded)
        self._thumb_loader.loading_finished.connect(self._on_thumbs_finished)
        self._thumb_loader.start()

        self._sort_combo.setEnabled(False)

    def _on_thumb_loaded(self, idx: int, icon: QIcon):
        if idx < 0 or idx >= len(self._all_paths):
            return
        p = self._all_paths[idx]
        self._thumb_cache[p] = icon
        row = self._path_to_row.get(p, -1)
        if row >= 0:
            item = self._list.item(row)
            if item is not None:
                item.setIcon(icon)
        self._progress.setValue(idx + 1)

    def _on_thumbs_finished(self):
        self._progress.setValue(len(self._all_paths))
        self._update_status()
        self._progress_label.setText("正在计算人脸相似度，请稍候...")
        self._progress.setRange(0, len(self._all_paths))
        self._progress.setValue(0)
        self._load_embeddings()

    def _populate_list(self):
        self._list.clear()
        self._path_to_row = {}
        for row, p in enumerate(self._current_order):
            name = p.name
            if len(name) > 16:
                name = name[:7] + ".." + name[-7:]
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setSizeHint(QSize(_THUMB_SIZE + 16, _THUMB_SIZE + 36))
            if p in self._prefix_groups:
                item.setForeground(Qt.GlobalColor.darkYellow)
            icon = self._thumb_cache.get(p)
            if icon is not None:
                item.setIcon(icon)
            self._list.addItem(item)
            self._path_to_row[p] = row

    def _update_prefix_info(self):
        if self._sort_combo.currentText() != "文件名":
            self._prefix_groups = set()
            self._sort_info.setText("")
            return
        from collections import defaultdict
        prefix_map = defaultdict(list)
        for p in self._all_paths:
            prefix = p.stem.rsplit("_", 1)[0]
            prefix_map[prefix].append(p)
        multi = set()
        for paths in prefix_map.values():
            if len(paths) > 1:
                multi.update(paths)
        self._prefix_groups = multi
        dup_groups = sum(1 for paths in prefix_map.values() if len(paths) > 1)
        dup_files = sum(len(paths) for paths in prefix_map.values() if len(paths) > 1)
        if dup_groups > 0:
            self._sort_info.setText(f"同名文件: {dup_groups}组/{dup_files}个")
            self._sort_info.setStyleSheet("color: #D45500; font-size: 12px; font-weight: bold;")
        else:
            self._sort_info.setText("无同名文件")
            self._sort_info.setStyleSheet("color: #555; font-size: 12px;")

    def _on_sort_changed(self, idx):
        if idx == 0:
            self._current_order = list(self._all_paths)
            self._update_prefix_info()
        elif idx == 1:
            self._sort_by_similarity()
        self._populate_list()
        self._update_status()

    def _sort_by_similarity(self):
        if not self._embeddings_loaded:
            self._emb_pending_sort = True
            return
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
        if self._emb_thread is not None and self._emb_thread.is_alive():
            return
        self._emb_stop = False

        def _task():
            try:
                from faceswap.core.insightface_adapter import InsightFaceAdapter
                adapter = InsightFaceAdapter.get_instance()
                adapter.warmup()
            except Exception as e:
                self._emb_error_signal.emit(str(e))
                return
            count = 0
            for i, p in enumerate(self._all_paths):
                if self._emb_stop:
                    break
                if p in self._all_embeddings:
                    count += 1
                    self._emb_progress_signal.emit(i + 1)
                    continue
                img = cv2.imread(str(p))
                if img is not None:
                    emb = adapter.extract_embedding_aligned(img)
                    if emb is not None:
                        self._all_embeddings[p] = emb
                        count += 1
                self._emb_progress_signal.emit(i + 1)
            self._emb_done_signal.emit(count)

        self._emb_thread = threading.Thread(target=_task, daemon=True)
        self._emb_thread.start()

    def _on_emb_progress(self, val):
        self._progress.setValue(val)

    def _on_emb_done(self, count):
        if self._emb_stop:
            return
        self._embeddings_loaded = True
        self._progress.setVisible(False)
        self._progress_label.setVisible(False)
        self._sort_combo.setEnabled(True)
        if self._sort_combo.currentIndex() == 1:
            self._sort_info.setText(f"embedding计算完成: {count}/{len(self._all_paths)}")
        else:
            self._update_prefix_info()
        if self._emb_pending_sort and self._sort_combo.currentIndex() == 1:
            self._emb_pending_sort = False
            self._sort_by_similarity()
            self._populate_list()
            self._update_status()

    def _on_emb_error(self, err):
        self._embeddings_loaded = True
        self._progress.setVisible(False)
        self._progress_label.setVisible(False)
        self._sort_combo.setEnabled(True)
        if self._sort_combo.currentIndex() == 1:
            self._sort_info.setText(f"InsightFace初始化失败: {err}")
        else:
            self._update_prefix_info()

    def closeEvent(self, event):
        self._emb_stop = True
        if self._thumb_loader is not None:
            self._thumb_loader.stop()
            self._thumb_loader.wait(5000)
        if self._emb_thread is not None and self._emb_thread.is_alive():
            self._emb_thread.join(timeout=3)
        super().closeEvent(event)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        p = item.data(Qt.ItemDataRole.UserRole)
        if p is None:
            return
        img = cv2.imread(str(p))
        if img is None:
            return
        h, w = img.shape[:2]
        rgb = img[:, :, ::-1].copy()
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        dlg = QDialog()
        dlg.setWindowTitle(f"原图 - {p.name} ({w}x{h})")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        screen = QApplication.primaryScreen()
        if screen:
            max_w = screen.availableGeometry().width() - 40
            max_h = screen.availableGeometry().height() - 80
            if pixmap.width() > max_w or pixmap.height() > max_h:
                pixmap = pixmap.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
        lbl.setPixmap(pixmap)
        lay.addWidget(lbl)
        dlg.exec()

    def _select_all(self):
        for i in range(self._list.count()):
            self._list.item(i).setSelected(True)
        self._update_status()

    def _deselect_all(self):
        self._list.clearSelection()
        self._update_status()

    def _update_status(self):
        total = self._list.count()
        sel = len(self._list.selectedItems())
        self._count_lbl.setText(f"选中: {sel}/{total}")
        self._status_lbl.setText(f"共 {total} 张图片")

    def _on_list_context_menu(self, pos):
        item = self._list.itemAt(pos)
        if item is not None and not item.isSelected():
            self._list.clearSelection()
            item.setSelected(True)
        selected = self._list.selectedItems()
        if not selected:
            return
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        act = menu.addAction(f"删除选中 ({len(selected)} 张)")
        act.triggered.connect(self._delete_selected)
        menu.exec(self._list.viewport().mapToGlobal(pos))

    def _delete_selected(self):
        selected_items = self._list.selectedItems()
        if not selected_items:
            return

        reply = QMessageBox.warning(
            self, "确认删除",
            f"确定要删除 {len(selected_items)} 张图片及其元数据文件吗？\n此操作不可撤销！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        delete_set = set()
        for item in selected_items:
            p = item.data(Qt.ItemDataRole.UserRole)
            if p is not None:
                delete_set.add(p)

        deleted = 0
        for p in delete_set:
            json_path = p.with_suffix(".json")
            try:
                if p.exists():
                    p.unlink()
                if json_path.exists():
                    json_path.unlink()
                deleted += 1
            except OSError as e:
                _logger.warning(f"Failed to delete {p}: {e}")

        self._all_paths = [p for p in self._all_paths if p not in delete_set]
        self._current_order = [p for p in self._current_order if p not in delete_set]
        for p in delete_set:
            self._all_embeddings.pop(p, None)
            self._thumb_cache.pop(p, None)

        self._populate_list()
        self._update_prefix_info()
        self._update_status()
        QMessageBox.information(self, "删除完成", f"已删除 {deleted} 张图片及对应元数据")
