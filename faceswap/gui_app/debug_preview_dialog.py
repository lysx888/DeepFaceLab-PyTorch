from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QPointF, QRectF, QSize, QItemSelectionModel, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QSplitter, QListWidget,
    QMessageBox, QListWidgetItem,
)

from faceswap.setting import FaceType, DATA_SRC_DIR, DATA_DST_DIR, DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR
from faceswap.shared.file_manager import FileManager
from faceswap.shared.image_utils import bgr_to_rgb
from faceswap.shared.logger import get_logger
from faceswap.core.metadata_manager import MetadataManager
from faceswap.gui_app.gui_utils import save_face_annotation

from faceswap.gui_app.annotator_common import (
    ImageCanvas, _CompactThumbDelegate, _ThumbLoader, _resize_pad, load_thumbnail,
    _toggle_canvas_visibility, _generate_default_face_annotation,
    _canvas_kps5_to_insightface,
)

_logger = get_logger("debug_preview")

class _PriorityThumbLoader(QThread):
    thumb_loaded = pyqtSignal(int, object)

    def __init__(self, image_paths: list[Path], cached_icons: list, priority_idx: int = 0):
        super().__init__()
        self._paths = image_paths
        self._cached_icons = cached_icons
        self._priority = priority_idx
        self._stop = False

    def stop(self):
        self._stop = True

    def set_priority(self, idx: int):
        self._priority = idx

    def run(self):
        n = len(self._paths)
        loaded = set()
        while not self._stop:
            priority = self._priority
            remaining = [(abs(i - priority), i) for i in range(n)
                         if i not in loaded and self._cached_icons[i] is None]
            if not remaining:
                break
            remaining.sort()
            for _, i in remaining[:8]:
                if self._stop:
                    return
                qimg = load_thumbnail(self._paths[i], 120, 80)
                if qimg is not None:
                    self.thumb_loaded.emit(i, qimg)
                loaded.add(i)


class _HorizontalThumbDialog(QDialog):
    thumb_selected = pyqtSignal(int)

    def __init__(self, image_list: list[Path], current_idx: int, cached_icons: list[Optional[QIcon]], parent=None):
        super().__init__(parent)
        self._image_list = image_list
        self._current_idx = current_idx
        self._cached_icons = cached_icons
        self._loader: Optional[_PriorityThumbLoader] = None
        self.setWindowTitle("横排缩略图浏览")
        self.setMinimumSize(800, 600)
        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowType.WindowMinMaxButtonsHint |
                            Qt.WindowType.Window)
        self._build_ui()
        self._start_loader()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self._thumb_list = QListWidget()
        self._thumb_list.setIconSize(QSize(120, 80))
        self._thumb_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._thumb_list.setViewMode(QListWidget.ViewMode.IconMode)
        self._thumb_list.setSpacing(4)
        self._thumb_list.setDragDropMode(QListWidget.DragDropMode.NoDragDrop)
        self._thumb_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._thumb_list.setItemDelegate(_CompactThumbDelegate(self._thumb_list))
        thumb_font = self._thumb_list.font()
        thumb_font.setPixelSize(10)
        self._thumb_list.setFont(thumb_font)
        self._thumb_list.itemClicked.connect(self._on_item_clicked)
        self._thumb_list.currentItemChanged.connect(self._on_current_changed)
        layout.addWidget(self._thumb_list)

        for i, img_path in enumerate(self._image_list):
            item = QListWidgetItem(img_path.stem)
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setSizeHint(QSize(126, 96))
            if i < len(self._cached_icons) and self._cached_icons[i] is not None:
                item.setIcon(self._cached_icons[i])
            self._thumb_list.addItem(item)

        if 0 <= self._current_idx < self._thumb_list.count():
            item = self._thumb_list.item(self._current_idx)
            self._thumb_list.setCurrentItem(item)
            self._thumb_list.scrollToItem(item, QListWidget.ScrollHint.PositionAtCenter)

    def _start_loader(self):
        self._loader = _PriorityThumbLoader(self._image_list, self._cached_icons, self._current_idx)
        self._loader.thumb_loaded.connect(self._on_thumb_loaded)
        self._loader.start()

    def _on_thumb_loaded(self, idx: int, qimg: QImage):
        icon = QIcon(QPixmap.fromImage(qimg))
        if idx < len(self._cached_icons):
            self._cached_icons[idx] = icon
        if idx < self._thumb_list.count():
            self._thumb_list.item(idx).setIcon(icon)

    def _on_current_changed(self, current, previous):
        if current is not None and self._loader is not None:
            self._loader.set_priority(current.data(Qt.ItemDataRole.UserRole))

    def _on_item_clicked(self, item):
        idx = item.data(Qt.ItemDataRole.UserRole)
        self.thumb_selected.emit(idx)
        self.accept()

    def done(self, result):
        if self._loader is not None:
            self._loader.stop()
            self._loader.wait(5000)
        super().done(result)



class DebugPreviewDialog(QDialog):
    def __init__(self, is_src: bool, parent=None):
        super().__init__(parent)
        self._is_src = is_src
        self._current_img_path = None
        self._current_aligned_path = None
        self._current_img = None
        self._current_source_img = None
        self._current_meta = None
        self._image_list: list[Path] = []
        self._cached_icons: list[Optional[QIcon]] = []
        self._current_image_idx = -1
        self._saving = False
        self._thumb_loader = None
        self._thumb_mode = "all"
        self.setWindowTitle("调试图预览 - " + ("源 (SRC)" if is_src else "目标 (DST)"))
        self.setMinimumSize(1300, 800)
        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowType.WindowMinMaxButtonsHint |
                            Qt.WindowType.Window)
        self._build_ui()
        self._build_aligned_index()
        self._start_loading_thumbnails()
        self.showMaximized()

    def _build_aligned_index(self):
        self._aligned_index: dict[str, list[Path]] = {}
        aligned_dir = DATA_SRC_ALIGNED_DIR if self._is_src else DATA_DST_ALIGNED_DIR
        if not aligned_dir.exists():
            return
        from faceswap.core.metadata_manager import MetadataManager
        index = MetadataManager.get_source_index(aligned_dir)
        for source_stem, face_names in index.items():
            self._aligned_index[source_stem] = [aligned_dir / n for n in face_names]

    def done(self, result):
        if self._thumb_loader is not None:
            self._thumb_loader.stop()
            self._thumb_loader.wait(5000)
        super().done(result)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_edit_canvas') and self._current_img is not None:
            self._edit_canvas._fit_to_view()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        left_widget.setFixedWidth(200)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(2)
        self._all_btn = QPushButton("所有图")
        self._all_btn.setCheckable(True)
        self._all_btn.setChecked(True)
        self._all_btn.setStyleSheet(
            "QPushButton { padding: 4px 8px; font-weight: bold; }"
            "QPushButton:checked { background-color: #0078D4; color: white; }")
        self._all_btn.clicked.connect(lambda: self._switch_thumb_mode("all"))
        mode_row.addWidget(self._all_btn)

        self._no_debug_btn = QPushButton("无调试图")
        self._no_debug_btn.setCheckable(True)
        self._no_debug_btn.setStyleSheet(
            "QPushButton { padding: 4px 8px; font-weight: bold; }"
            "QPushButton:checked { background-color: #0078D4; color: white; }")
        self._no_debug_btn.clicked.connect(lambda: self._switch_thumb_mode("no_debug"))
        mode_row.addWidget(self._no_debug_btn)
        left_layout.addLayout(mode_row)

        self._thumb_list = QListWidget()
        self._thumb_list.setIconSize(QSize(120, 80))
        self._thumb_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._thumb_list.setViewMode(QListWidget.ViewMode.IconMode)
        self._thumb_list.setSpacing(0)
        self._thumb_list.setDragDropMode(QListWidget.DragDropMode.NoDragDrop)
        self._thumb_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._thumb_list.setItemDelegate(_CompactThumbDelegate(self._thumb_list, full_width=True))
        thumb_font = self._thumb_list.font()
        thumb_font.setPixelSize(10)
        self._thumb_list.setFont(thumb_font)
        self._thumb_list.currentItemChanged.connect(self._on_thumb_selected)
        left_layout.addWidget(self._thumb_list)

        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        name_row = QHBoxLayout()
        self._image_name_label = QLabel("")
        self._image_name_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px; color: #0078D4;")
        name_row.addWidget(self._image_name_label)
        name_row.addStretch()
        self._prev_btn = QPushButton("◀ 上一帧")
        self._prev_btn.clicked.connect(self._prev_thumb)
        name_row.addWidget(self._prev_btn)
        self._next_btn = QPushButton("下一帧 ▶")
        self._next_btn.clicked.connect(self._next_thumb)
        name_row.addWidget(self._next_btn)
        name_row.addStretch()
        self._hthumb_btn = QPushButton("横排缩略图")
        self._hthumb_btn.setStyleSheet(
            "QPushButton { background-color: #555; color: white; font-weight: bold; padding: 8px 16px; }")
        self._hthumb_btn.clicked.connect(self._on_horizontal_thumbs)
        name_row.addWidget(self._hthumb_btn)
        right_layout.addLayout(name_row)

        self._edit_canvas = ImageCanvas(show_point_numbers=False, right_click_pan=True)
        right_layout.addWidget(self._edit_canvas, 1)

        nav_row = QHBoxLayout()
        self._reset_view_btn = QPushButton("重置位置")
        self._reset_view_btn.setStyleSheet(
            "QPushButton { background-color: #555; color: white; font-weight: bold; padding: 8px 16px; }")
        self._reset_view_btn.clicked.connect(self._on_reset_view)
        nav_row.addWidget(self._reset_view_btn)

        self._manual_btn = QPushButton("手工标注")
        self._manual_btn.setStyleSheet(
            "QPushButton { background-color: #5B2D8E; color: white; font-weight: bold; padding: 8px 24px; }")
        self._manual_btn.setEnabled(False)
        self._manual_btn.clicked.connect(self._on_manual_annotate)
        nav_row.addWidget(self._manual_btn)

        self._add_lm_btn = QPushButton("添加特征点")
        self._add_lm_btn.setStyleSheet(
            "QPushButton { background-color: #2E7D32; color: white; font-weight: bold; padding: 8px 24px; }")
        self._add_lm_btn.setEnabled(False)
        self._add_lm_btn.clicked.connect(self._on_add_landmarks)
        nav_row.addWidget(self._add_lm_btn)

        nav_row.addStretch()

        self._toggle_kps5_btn = QPushButton("隐藏5点")
        self._toggle_kps5_btn.setStyleSheet(
            "QPushButton { background-color: #555; color: white; font-weight: bold; padding: 6px 12px; }")
        self._toggle_kps5_btn.setCheckable(True)
        self._toggle_kps5_btn.setChecked(False)
        self._toggle_kps5_btn.clicked.connect(self._on_toggle_kps5)
        nav_row.addWidget(self._toggle_kps5_btn)

        self._toggle_106_btn = QPushButton("隐藏106点")
        self._toggle_106_btn.setStyleSheet(
            "QPushButton { background-color: #555; color: white; font-weight: bold; padding: 6px 12px; }")
        self._toggle_106_btn.setCheckable(True)
        self._toggle_106_btn.setChecked(False)
        self._toggle_106_btn.clicked.connect(self._on_toggle_106)
        nav_row.addWidget(self._toggle_106_btn)

        nav_row.addStretch()

        self._save_btn = QPushButton("保存")
        self._save_btn.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; font-weight: bold; padding: 8px 24px; }")
        self._save_btn.clicked.connect(self._on_save_edit)
        nav_row.addWidget(self._save_btn)

        self._delete_btn = QPushButton("删除")
        self._delete_btn.setStyleSheet(
            "QPushButton { background-color: #C42B1C; color: white; font-weight: bold; padding: 8px 24px; }")
        self._delete_btn.clicked.connect(self._on_delete_edit)
        nav_row.addWidget(self._delete_btn)

        right_layout.addLayout(nav_row)

        self._help_label = QLabel(
            "操作说明：滚轮缩放图片，右键按住移动图片，左键拖动特征点调整位置，"
            "按V切换可见性（锁定组=五官级，未锁定=单点级），修改后点击「保存」更新元数据。"
        )
        self._help_label.setStyleSheet("color: #888; font-size: 11px; padding: 4px; border-top: 1px solid #444;")
        self._help_label.setWordWrap(True)
        right_layout.addWidget(self._help_label)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([200, 800])

        layout.addWidget(splitter)

    def _get_filtered_images(self) -> list[Path]:
        frames_dir = DATA_SRC_DIR if self._is_src else DATA_DST_DIR
        images = sorted(FileManager.find_images(frames_dir), key=lambda p: p.name)
        if self._thumb_mode == "no_debug":
            aligned_dir = DATA_SRC_ALIGNED_DIR if self._is_src else DATA_DST_ALIGNED_DIR
            debug_dir = aligned_dir.parent / (aligned_dir.name + "_debug")
            debug_stems = {f.stem for f in debug_dir.iterdir() if f.is_file()} if debug_dir.exists() else set()
            images = [img for img in images if img.stem not in debug_stems]
        return images

    def _switch_thumb_mode(self, mode: str):
        if mode == self._thumb_mode:
            return
        self._thumb_mode = mode
        if mode == "all":
            self._no_debug_btn.setChecked(False)
        else:
            self._all_btn.setChecked(False)

        if self._thumb_loader is not None:
            self._thumb_loader.stop()
            self._thumb_loader.wait(5000)
            self._thumb_loader = None

        self._thumb_list.clear()
        self._image_list = []
        self._cached_icons = []
        self._current_image_idx = -1
        self._current_img_path = None
        self._current_aligned_path = None
        self._current_img = None
        self._current_source_img = None
        self._current_meta = None
        self._edit_canvas.clear_landmarks()
        self._edit_canvas.set_image(None)
        self._image_name_label.setText("")
        self._manual_btn.setEnabled(False)
        self._add_lm_btn.setEnabled(False)

        self._start_loading_thumbnails()

    def _start_loading_thumbnails(self):
        frames_dir = DATA_SRC_DIR if self._is_src else DATA_DST_DIR

        if not frames_dir.exists():
            QMessageBox.information(self, "提示", f"原帧目录不存在: {frames_dir}")
            return

        images = self._get_filtered_images()
        if not images:
            QMessageBox.information(self, "提示", "没有匹配的图片")
            return

        self._image_list = images
        self._cached_icons = [None] * len(images)
        for img_path in images:
            item = QListWidgetItem(img_path.stem)
            item.setData(Qt.ItemDataRole.UserRole, str(img_path))
            item.setSizeHint(QSize(126, 96))
            self._thumb_list.addItem(item)

        self._thumb_list.setCurrentRow(0)

        self._thumb_loader = _ThumbLoader(images, preload_count=0)
        self._thumb_loader.thumb_loaded.connect(self._on_thumb_loaded)
        self._thumb_loader.loading_finished.connect(self._on_thumbs_loaded)
        self._thumb_loader.start()

        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self._show_first_preview)

    def _show_first_preview(self):
        if self._thumb_list.count() > 0:
            self._on_thumb_selected(self._thumb_list.currentItem(), None)

    def _on_thumb_loaded(self, idx: int, qimg: QImage, stem: str, path_str: str):
        icon = QIcon(QPixmap.fromImage(qimg))
        if idx < self._thumb_list.count():
            item = self._thumb_list.item(idx)
            item.setIcon(icon)
            item.setSizeHint(QSize(126, 96))
        if idx < len(self._cached_icons):
            self._cached_icons[idx] = icon

    def _on_thumbs_loaded(self):
        pass

    def _on_thumb_selected(self, current, previous):
        if current is None:
            return
        if self._saving:
            if previous is not None and self._current_image_idx >= 0:
                self._thumb_list.blockSignals(True)
                self._thumb_list.setCurrentRow(self._current_image_idx)
                self._thumb_list.blockSignals(False)
            return

        img_path = Path(current.data(Qt.ItemDataRole.UserRole))
        img = cv2.imread(str(img_path))
        if img is None:
            return

        self._current_img_path = img_path
        self._current_img = img
        self._current_source_img = img
        self._current_image_idx = self._thumb_list.row(current)
        self._image_name_label.setText(img_path.name)

        aligned_paths = [p for p in self._aligned_index.get(img_path.stem, []) if p.exists()]
        self._current_aligned_path = aligned_paths[0] if aligned_paths else None

        from faceswap.core.metadata_manager import MetadataManager
        self._current_meta = MetadataManager.load(self._current_aligned_path) if self._current_aligned_path else None

        self._edit_canvas.clear_landmarks()
        self._edit_canvas.set_show_106(True)
        self._edit_canvas.set_visible_group(None)
        self._edit_canvas.set_image(img)

        if self._current_meta is not None:
            src_lm = self._current_meta.source_landmarks_106
            if src_lm is not None and src_lm.shape[0] == 106:
                landmarks_106 = {}
                for i in range(106):
                    landmarks_106[i] = QPointF(float(src_lm[i, 0]), float(src_lm[i, 1]))
                self._edit_canvas.set_landmarks_106(landmarks_106)

            if self._current_meta.source_rect is not None:
                sr = self._current_meta.source_rect
                self._edit_canvas.set_bbox_rect(QRectF(sr[0], sr[1], sr[2] - sr[0], sr[3] - sr[1]))

            if self._current_meta.landmarks_106_visibility is not None:
                vis = self._current_meta.landmarks_106_visibility
                self._edit_canvas.set_lm106_visibility({i: bool(vis[i]) for i in range(min(len(vis), 106))})

            src_kps5 = self._current_meta.source_kps_5
            if src_kps5 is not None and src_kps5.shape[0] == 5:
                kps5_dict = {}
                kps5_dict[0] = QPointF(float(src_kps5[1, 0]), float(src_kps5[1, 1]))
                kps5_dict[1] = QPointF(float(src_kps5[0, 0]), float(src_kps5[0, 1]))
                kps5_dict[2] = QPointF(float(src_kps5[2, 0]), float(src_kps5[2, 1]))
                kps5_dict[3] = QPointF(float(src_kps5[4, 0]), float(src_kps5[4, 1]))
                kps5_dict[4] = QPointF(float(src_kps5[3, 0]), float(src_kps5[3, 1]))
                self._edit_canvas.set_kps_5(kps5_dict)

        self._edit_canvas.set_show_kps5(True)
        self._edit_canvas.set_show_106(True)
        self._edit_canvas._fit_to_view()
        self._manual_btn.setEnabled(self._current_meta is not None)
        self._add_lm_btn.setEnabled(True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_V:
            msg = _toggle_canvas_visibility(self._edit_canvas)
            if not msg:
                QMessageBox.information(self, "提示", "请先双击锁定一个五官部位，或点击选中一个点")
            return
        super().keyPressEvent(event)

    def _prev_thumb(self):
        if self._current_image_idx > 0:
            idx = self._current_image_idx - 1
            model = self._thumb_list.model()
            sel = self._thumb_list.selectionModel()
            sel.clearSelection()
            sel.select(model.index(idx, 0), QItemSelectionModel.SelectionFlag.Select)
            self._thumb_list.setCurrentIndex(model.index(idx, 0))

    def _next_thumb(self):
        if self._current_image_idx < self._thumb_list.count() - 1:
            idx = self._current_image_idx + 1
            model = self._thumb_list.model()
            sel = self._thumb_list.selectionModel()
            sel.clearSelection()
            sel.select(model.index(idx, 0), QItemSelectionModel.SelectionFlag.Select)
            self._thumb_list.setCurrentIndex(model.index(idx, 0))

    def _on_save_edit(self):
        lm106 = self._edit_canvas.get_landmarks_106()
        if len(lm106) < 5:
            return

        self._saving = True
        try:
            aligned_dir = DATA_SRC_ALIGNED_DIR if self._is_src else DATA_DST_ALIGNED_DIR
            debug_stem = self._current_aligned_path.stem if self._current_aligned_path else self._current_img_path.stem

            lm_106 = np.zeros((106, 2), dtype=np.float32)
            for idx, pt in lm106.items():
                lm_106[idx] = [pt.x(), pt.y()]

            kps_5 = self._current_meta.source_kps_5.astype(np.float32) if self._current_meta and hasattr(self._current_meta, 'source_kps_5') and self._current_meta.source_kps_5 is not None else None
            if kps_5 is None:
                QMessageBox.warning(self, "缺少5点标注", "请先标注5个关键点后再保存")
                self._saving = False
                return

            canvas_kps5 = self._edit_canvas.get_kps_5()
            if len(canvas_kps5) == 5:
                kps_5 = _canvas_kps5_to_insightface(canvas_kps5)

            face_type = self._current_meta.face_type if self._current_meta else FaceType.WHOLE_FACE
            output_size = self._current_meta.output_size if self._current_meta and hasattr(self._current_meta, 'output_size') and self._current_meta.output_size else 512

            canvas_vis = self._edit_canvas.get_lm106_visibility()
            lm106_vis = []
            for i in range(106):
                if i in lm106:
                    lm106_vis.append(canvas_vis.get(i, True))
                else:
                    lm106_vis.append(True)
            kps5_vis = [True] * 5

            bbox_rect = self._edit_canvas.get_bbox_rect()
            if bbox_rect is not None:
                source_rect = [bbox_rect.x(), bbox_rect.y(), bbox_rect.x() + bbox_rect.width(), bbox_rect.y() + bbox_rect.height()]
            else:
                source_rect = [0, 0, self._current_source_img.shape[1], self._current_source_img.shape[0]]

            existing_face_path = self._current_aligned_path

            result = save_face_annotation(
                source_img=self._current_source_img,
                lm_106=lm_106,
                kps_5=kps_5,
                face_type=face_type,
                output_size=output_size,
                is_src=self._is_src,
                source_filename=self._current_meta.source_filename if self._current_meta else self._current_img_path.name,
                source_rect=source_rect,
                landmarks_106_visibility=lm106_vis,
                kps_5_visibility=kps5_vis,
                existing_face_path=existing_face_path,
            )

            self._current_meta = result.metadata
            if result.face_path is not None and existing_face_path is None:
                frame_stem = self._current_img_path.stem
                self._aligned_index.setdefault(frame_stem, []).append(result.face_path)
                self._current_aligned_path = result.face_path
            self._thumb_list.blockSignals(True)
            self._refresh_current_thumbnail()
            self._thumb_list.blockSignals(False)
            QMessageBox.information(self, "保存成功", f"编辑保存完毕\n{result.face_path.name if result.face_path else debug_stem}")
        finally:
            self._saving = False

    def _on_delete_edit(self):
        lm106 = self._edit_canvas.get_landmarks_106()
        if len(lm106) > 0:
            reply = QMessageBox.question(self, "确认删除",
                                         "删除当前帧的所有特征点？将删除对应的头像图和元数据，并用低质量图覆盖调试图。",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        aligned_dir = DATA_SRC_ALIGNED_DIR if self._is_src else DATA_DST_ALIGNED_DIR
        debug_dir = aligned_dir.parent / (aligned_dir.name + "_debug")
        deleted = []

        frame_stem = self._current_img_path.stem
        for aligned_file in self._aligned_index.get(frame_stem, []):
            if aligned_file.exists():
                aligned_file.unlink()
                deleted.append(aligned_file.name)
                json_path = aligned_file.with_suffix(".json")
                if json_path.exists():
                    json_path.unlink()
                    deleted.append(json_path.name)
                MetadataManager.remove_from_source_index(aligned_dir, frame_stem, aligned_file.name)

        self._aligned_index.pop(frame_stem, None)
        self._current_aligned_path = None

        source_img = self._current_source_img
        if source_img is not None:
            low_q_path = debug_dir / (frame_stem + ".jpg")
            from faceswap.shared.file_manager import imwrite_auto
            imwrite_auto(low_q_path, source_img, jpg_quality=30)
            self._current_img = source_img

        self._current_meta = None
        self._refresh_current_thumbnail()

    def _on_reset_view(self):
        self._edit_canvas.reset_view()

    def _on_add_landmarks(self):
        if self._current_img is None:
            return
        if len(self._edit_canvas.get_landmarks_106()) > 0:
            QMessageBox.information(self, "提示", "已经有特征点，无法重复添加！")
            return
        bbox = None
        if self._current_meta is not None and self._current_meta.source_rect is not None:
            bbox = self._current_meta.source_rect
        landmarks_106, kps_5 = _generate_default_face_annotation(self._current_img, bbox=bbox)
        self._edit_canvas.set_landmarks_106(landmarks_106)
        self._edit_canvas.set_kps_5(kps_5)
        self._edit_canvas.set_show_kps5(True)
        self._edit_canvas.set_show_106(True)
        self._edit_canvas.set_visible_group(None)
        self._edit_canvas._fit_to_view()

    def _on_toggle_kps5(self):
        self._edit_canvas.set_visible_group(None)
        if self._toggle_kps5_btn.isChecked():
            self._edit_canvas.set_show_kps5(False)
            self._toggle_kps5_btn.setText("显示5点")
        else:
            self._edit_canvas.set_show_kps5(True)
            self._toggle_kps5_btn.setText("隐藏5点")

    def _on_toggle_106(self):
        self._edit_canvas.set_visible_group(None)
        if self._toggle_106_btn.isChecked():
            self._edit_canvas.set_show_106(False)
            self._toggle_106_btn.setText("显示106点")
        else:
            self._edit_canvas.set_show_106(True)
            self._toggle_106_btn.setText("隐藏106点")

    def _refresh_current_thumbnail(self):
        if self._current_image_idx < 0 or self._current_image_idx >= len(self._image_list):
            return
        img_path = self._image_list[self._current_image_idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_REDUCED_COLOR_4)
        if img is None:
            return
        thumb = _resize_pad(img, 120, 80)
        rgb = bgr_to_rgb(thumb)
        qimg = QImage(rgb.data, 120, 80, 3 * 120, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        item = self._thumb_list.item(self._current_image_idx)
        if item is not None:
            item.setIcon(QIcon(pixmap))
            item.setSizeHint(QSize(126, 96))

    def _on_horizontal_thumbs(self):
        dlg = _HorizontalThumbDialog(self._image_list, self._current_image_idx, self._cached_icons, self)
        dlg.thumb_selected.connect(self._on_hthumb_selected)
        dlg.exec()

    def _on_hthumb_selected(self, idx: int):
        if 0 <= idx < self._thumb_list.count():
            self._thumb_list.setCurrentRow(idx)

    def _on_manual_annotate(self):
        if self._current_img_path is None:
            return
        from faceswap.gui_app.manual_annotator_dialog import ManualAnnotatorDialog
        dlg = ManualAnnotatorDialog(self, is_src=self._is_src, img_path=self._current_img_path)
        dlg.exec()
        self._build_aligned_index()
        if self._current_image_idx >= 0:
            item = self._thumb_list.item(self._current_image_idx)
            if item is not None:
                self._on_thumb_selected(item, None)
