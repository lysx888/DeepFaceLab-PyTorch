import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from faceswap.core.landmarks106 import (
    LANDMARK_GROUPS_106, LINE_CONNECTIONS_106,
)
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QSize, QThread, QItemSelectionModel
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont, QFontMetrics, QWheelEvent, QMouseEvent, QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QSplitter, QListWidget,
    QMessageBox, QListWidgetItem,
    QSizePolicy, QStyledItemDelegate, QStyleOptionViewItem, QStyle,
)

from faceswap.setting import DATA_SRC_DIR, DATA_DST_DIR, DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR
from faceswap.shared.file_manager import FileManager
from faceswap.shared.image_utils import bgr_to_rgb
from faceswap.shared.logger import get_logger
from faceswap.gui_app.gui_utils import RectDragHelper

_logger = get_logger("annotator_common")

_KPS5_NAMES = ["左眼球", "右眼球", "鼻尖", "左嘴角", "右嘴角"]
_KPS5_BUTTON_ORDER = [(0, 1), (2,), (3, 4)]
_KPS5_COLORS = [QColor(0, 255, 255), QColor(0, 255, 255), QColor(0, 255, 255), QColor(0, 255, 255), QColor(0, 255, 255)]

_CN_NAMES = {
    "left_eyebrow": "左眉毛", "right_eyebrow": "右眼眉",
    "left_eye": "左眼睛", "left_eyeball": "左眼球",
    "right_eye": "右眼睛", "right_eyeball": "右眼球",
    "nose_bridge": "鼻梁", "nose": "鼻子",
    "inner_lip": "内嘴唇", "outer_lip": "外嘴唇", "jaw_cheek": "脸颊",
}
_CN_COLORS = {
    "left_eyebrow": QColor(80, 255, 80), "right_eyebrow": QColor(255, 80, 80),
    "left_eye": QColor(255, 255, 80), "left_eyeball": QColor(255, 200, 80),
    "right_eye": QColor(80, 80, 255), "right_eyeball": QColor(150, 150, 255),
    "nose_bridge": QColor(80, 255, 255), "nose": QColor(255, 80, 255),
    "inner_lip": QColor(200, 80, 255), "outer_lip": QColor(255, 160, 80),
    "jaw_cheek": QColor(255, 255, 255),
}

_LANDMARK_GROUPS_106 = [
    (_CN_NAMES.get(gname, gname), idxs, _CN_COLORS.get(gname, QColor(200, 200, 200)))
    for gname, idxs in LANDMARK_GROUPS_106
]

_ALL_106_INDICES = []
_IDX_TO_COLOR_106 = {}
_IDX_TO_GROUP_106 = {}
for _gname, _idxs, _color in _LANDMARK_GROUPS_106:
    for _i in _idxs:
        _ALL_106_INDICES.append(_i)
        _IDX_TO_COLOR_106[_i] = _color
        _IDX_TO_GROUP_106[_i] = _gname

_106_LINE_CONNECTIONS = {
    _CN_NAMES.get(gname, gname): lines
    for gname, lines in LINE_CONNECTIONS_106.items()
}


_DEFAULT_GROUP_SHAPES = {
    "右眼眉": [(43, 0.15, 0.23), (48, 0.20, 0.19), (49, 0.25, 0.17), (51, 0.30, 0.16),
              (50, 0.38, 0.18), (46, 0.38, 0.23), (47, 0.30, 0.23), (45, 0.25, 0.24), (44, 0.20, 0.24)],
    "左眉毛": [(101, 0.85, 0.23), (105, 0.81, 0.19), (104, 0.77, 0.17), (103, 0.72, 0.16),
              (102, 0.65, 0.18), (97, 0.65, 0.23), (98, 0.72, 0.23), (99, 0.77, 0.24), (100, 0.81, 0.24)],
    "右眼睛": [(35, 0.22, 0.36), (41, 0.26, 0.33), (40, 0.31, 0.32), (42, 0.36, 0.33),
              (39, 0.40, 0.36), (37, 0.36, 0.39), (33, 0.31, 0.40), (36, 0.26, 0.39)],
    "右眼球": [(34, 0.29, 0.35), (38, 0.33, 0.36)],
    "左眼睛": [(93, 0.78, 0.36), (96, 0.74, 0.33), (94, 0.69, 0.32), (95, 0.64, 0.33),
              (89, 0.60, 0.36), (90, 0.64, 0.39), (87, 0.69, 0.40), (91, 0.74, 0.39)],
    "左眼球": [(88, 0.67, 0.35), (92, 0.71, 0.36)],
    "鼻梁": [(72, 0.48, 0.32), (73, 0.49, 0.40), (74, 0.50, 0.47), (86, 0.50, 0.55)],
    "鼻子": [(75, 0.42, 0.50), (76, 0.43, 0.56), (77, 0.44, 0.60), (78, 0.47, 0.60),
             (79, 0.49, 0.60), (80, 0.50, 0.60), (85, 0.51, 0.60), (84, 0.53, 0.60),
             (83, 0.56, 0.60), (82, 0.57, 0.56), (81, 0.58, 0.50)],
    "内嘴唇": [(65, 0.43, 0.74), (66, 0.45, 0.72), (62, 0.50, 0.70), (70, 0.55, 0.72),
              (69, 0.57, 0.74), (57, 0.55, 0.76), (60, 0.50, 0.78), (54, 0.45, 0.76)],
    "外嘴唇": [(52, 0.36, 0.74), (64, 0.38, 0.70), (63, 0.43, 0.68), (71, 0.50, 0.67),
              (67, 0.57, 0.68), (68, 0.62, 0.70), (61, 0.64, 0.74), (58, 0.62, 0.78),
              (59, 0.57, 0.80), (53, 0.50, 0.81), (56, 0.43, 0.80), (55, 0.38, 0.78)],
    "脸颊": [(1, 0.15, 0.30), (9, 0.14, 0.35), (10, 0.13, 0.40), (11, 0.12, 0.45),
             (12, 0.12, 0.50), (13, 0.12, 0.55), (14, 0.13, 0.60), (15, 0.14, 0.65),
             (16, 0.16, 0.70), (2, 0.19, 0.75), (3, 0.23, 0.79), (4, 0.28, 0.83),
             (5, 0.33, 0.86), (6, 0.38, 0.89), (7, 0.44, 0.91), (8, 0.50, 0.93),
             (0, 0.50, 0.96), (24, 0.56, 0.91), (23, 0.62, 0.89), (22, 0.67, 0.86),
             (21, 0.72, 0.83), (20, 0.77, 0.79), (19, 0.81, 0.75), (18, 0.84, 0.70),
             (32, 0.86, 0.65), (31, 0.87, 0.60), (30, 0.88, 0.55), (29, 0.88, 0.50),
             (28, 0.87, 0.45), (27, 0.86, 0.40), (26, 0.86, 0.35), (25, 0.86, 0.32),
             (17, 0.84, 0.28)],
}


def _generate_default_group_points(group_name: str, anchor_rect: QRectF) -> dict[int, QPointF]:
    shapes = _DEFAULT_GROUP_SHAPES.get(group_name, [])
    if not shapes:
        center = QPointF(anchor_rect.center().x(), anchor_rect.center().y())
        for gname, gidxs, _ in _LANDMARK_GROUPS_106:
            if gname == group_name:
                return {idx: QPointF(center.x(), center.y()) for idx in gidxs}
        return {}
    x = anchor_rect.x()
    y = anchor_rect.y()
    w = anchor_rect.width()
    h = anchor_rect.height()
    result = {}
    for idx, fx, fy in shapes:
        result[idx] = QPointF(x + fx * w, y + fy * h)
    return result


_KPS5_DEFAULT_POS = {
    0: (0.69, 0.36),
    1: (0.31, 0.36),
    2: (0.50, 0.55),
    3: (0.61, 0.74),
    4: (0.39, 0.74),
}


def _generate_default_face_annotation(img: np.ndarray, bbox=None) -> tuple[dict[int, QPointF], dict[int, QPointF]]:
    h, w = img.shape[:2]
    if bbox is not None:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw, bh = x2 - x1, y2 - y1
        margin = max(bw, bh) * 0.3
        anchor_rect = QRectF(x1 - margin, y1 - margin, bw + 2 * margin, bh + 2 * margin)
    else:
        margin = 0.1
        anchor_rect = QRectF(w * margin, h * margin, w * (1 - 2 * margin), h * (1 - 2 * margin))

    landmarks_106: dict[int, QPointF] = {}
    for gname, _idxs, _color in _LANDMARK_GROUPS_106:
        landmarks_106.update(_generate_default_group_points(gname, anchor_rect))

    kps_5: dict[int, QPointF] = {}
    for kps_idx, (fx, fy) in _KPS5_DEFAULT_POS.items():
        kps_5[kps_idx] = QPointF(anchor_rect.x() + fx * anchor_rect.width(),
                                 anchor_rect.y() + fy * anchor_rect.height())

    return landmarks_106, kps_5


def _canvas_kps5_to_insightface(canvas_kps5: dict[int, QPointF]) -> np.ndarray:
    kps = np.zeros((5, 2), dtype=np.float32)
    kps[0] = [canvas_kps5[1].x(), canvas_kps5[1].y()]
    kps[1] = [canvas_kps5[0].x(), canvas_kps5[0].y()]
    kps[2] = [canvas_kps5[2].x(), canvas_kps5[2].y()]
    kps[3] = [canvas_kps5[4].x(), canvas_kps5[4].y()]
    kps[4] = [canvas_kps5[3].x(), canvas_kps5[3].y()]
    return kps


from enum import IntEnum



class ImageCanvas(QWidget):
    landmark_changed = pyqtSignal()

    def __init__(self, parent=None, show_point_numbers: bool = True, right_click_pan: bool = False):
        super().__init__(parent)
        self._image = None
        self._image_is_empty = True
        self._pixmap = None
        self._landmarks_106: dict[int, QPointF] = {}
        self._lm106_visibility: dict[int, bool] = {}
        self._bbox_rect: Optional[QRectF] = None
        self._show_bbox = True
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._dragging_idx = -1
        self._panning = False
        self._pan_start = QPointF()
        self._active_idx = -1
        self._visible_group: Optional[str] = None
        self._show_106 = True
        self._show_point_numbers = show_point_numbers
        self._dragging_bbox_handle = RectDragHelper.HANDLE_NONE
        self._right_click_pan = right_click_pan
        self._kps_5: dict[int, QPointF] = {}
        self._dragging_kps_idx = -1
        self._show_kps5 = True
        self._dragging_group: Optional[str] = None
        self._group_drag_start: Optional[QPointF] = None
        self._group_drag_initial: Optional[dict[int, QPointF]] = None
        self._empty_hint: Optional[str] = None
        self._scaled_pixmap: Optional[QPixmap] = None
        self._scaled_pixmap_key: Optional[tuple] = None
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, img: Optional[np.ndarray]):
        if img is None:
            self._image = None
            self._image_is_empty = True
            self._pixmap = None
            self._scaled_pixmap = None
            self._scaled_pixmap_key = None
            self.update()
            return
        self._image = img.copy()
        self._image_is_empty = (img.size == 0)
        h, w = img.shape[:2]
        rgb = bgr_to_rgb(img) if img.ndim == 3 else img
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._scaled_pixmap = None
        self._scaled_pixmap_key = None
        self._fit_to_view()
        self.update()

    def set_landmarks_106(self, landmarks: dict[int, QPointF]):
        self._landmarks_106 = dict(landmarks)
        self.update()
        self.landmark_changed.emit()

    def get_landmarks_106(self) -> dict[int, QPointF]:
        return dict(self._landmarks_106)

    def get_lm106_visibility(self) -> dict[int, bool]:
        return dict(self._lm106_visibility)

    def set_lm106_visibility(self, vis: dict[int, bool]):
        self._lm106_visibility = dict(vis)
        self.update()

    def toggle_group_visibility(self, group_name: Optional[str]):
        if group_name is None or not self._landmarks_106:
            return
        for gname, idxs, _ in _LANDMARK_GROUPS_106:
            if gname == group_name:
                present = [i for i in idxs if i in self._landmarks_106]
                if not present:
                    return
                all_invisible = all(not self._lm106_visibility.get(i, True) for i in present)
                for i in present:
                    self._lm106_visibility[i] = all_invisible
                self.update()
                return

    def toggle_point_visibility(self, idx: int):
        if idx < 0 or idx not in self._landmarks_106:
            return
        self._lm106_visibility[idx] = not self._lm106_visibility.get(idx, True)
        self.update()

    def clear_landmarks(self):
        self._landmarks_106.clear()
        self._lm106_visibility.clear()
        self._active_idx = -1
        self._visible_group = None
        self._kps_5.clear()
        self.update()
        self.landmark_changed.emit()

    def set_kps_5(self, kps: dict[int, QPointF]):
        self._kps_5 = dict(kps)
        self.update()

    def get_kps_5(self) -> dict[int, QPointF]:
        return dict(self._kps_5)

    def set_show_kps5(self, show: bool):
        self._show_kps5 = show
        self.update()

    def set_bbox_rect(self, rect: Optional[QRectF]):
        self._bbox_rect = rect
        self.update()

    def get_bbox_rect(self) -> Optional[QRectF]:
        return self._bbox_rect

    def set_show_bbox(self, show: bool):
        self._show_bbox = show
        self.update()

    def set_visible_group(self, group_name: Optional[str]):
        self._visible_group = group_name
        self.update()

    def set_show_106(self, show: bool):
        self._show_106 = show
        self.update()

    def set_empty_hint(self, text: Optional[str]):
        self._empty_hint = text
        self.update()

    def _is_in_group_area(self, img_pos: QPointF) -> bool:
        if self._visible_group is None:
            return False
        for gname, idxs, _ in _LANDMARK_GROUPS_106:
            if gname == self._visible_group:
                pts = [self._landmarks_106[i] for i in idxs if i in self._landmarks_106]
                if len(pts) < 2:
                    return False
                xs = [p.x() for p in pts]
                ys = [p.y() for p in pts]
                margin = 15.0 / self._scale
                return min(xs) - margin <= img_pos.x() <= max(xs) + margin and min(ys) - margin <= img_pos.y() <= max(ys) + margin
        return False

    def set_right_click_pan(self, enabled: bool):
        self._right_click_pan = enabled

    def _fit_to_view(self):
        if self._pixmap is None:
            return
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        vw = self.width()
        vh = self.height()
        if vw <= 0 or vh <= 0:
            return
        self._scale = min(vw / pw, vh / ph) * 0.9
        self._offset = QPointF((vw - pw * self._scale) / 2, (vh - ph * self._scale) / 2)
        self.update()

    def reset_view(self):
        self._fit_to_view()

    def _img_to_screen(self, pt: QPointF) -> QPointF:
        return QPointF(pt.x() * self._scale + self._offset.x(), pt.y() * self._scale + self._offset.y())

    def _screen_to_img(self, pt: QPointF) -> QPointF:
        return QPointF((pt.x() - self._offset.x()) / self._scale, (pt.y() - self._offset.y()) / self._scale)

    def _find_nearest(self, pos: QPointF, threshold: float = 12.0) -> int:
        best_idx = -1
        best_dist = threshold
        if self._show_106:
            visible_indices = self._get_visible_106_indices()
            for idx, pt in self._landmarks_106.items():
                if visible_indices is not None and idx not in visible_indices:
                    continue
                sp = self._img_to_screen(pt)
                d = ((sp.x() - pos.x()) ** 2 + (sp.y() - pos.y()) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
        return best_idx

    def _find_nearest_kps5(self, pos: QPointF, threshold: float = 14.0) -> int:
        best_idx = -1
        best_dist = threshold
        if self._show_kps5:
            for idx, pt in self._kps_5.items():
                sp = self._img_to_screen(pt)
                d = ((sp.x() - pos.x()) ** 2 + (sp.y() - pos.y()) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
        return best_idx

    def _get_visible_106_indices(self) -> Optional[set[int]]:
        if self._visible_group is None:
            return None
        for gname, idxs, _ in _LANDMARK_GROUPS_106:
            if gname == self._visible_group:
                return set(idxs)
        return None

    def _get_visible_106_indices_set(self) -> set[int]:
        result = self._get_visible_106_indices()
        if result is not None:
            return result
        return set(self._landmarks_106.keys())

    def resizeEvent(self, event):
        self._fit_to_view()
        super().resizeEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        mouse_pos = QPointF(event.position())
        img_before = self._screen_to_img(mouse_pos)
        self._scale *= factor
        self._scale = max(0.05, min(self._scale, 30.0))
        img_after = self._screen_to_img(mouse_pos)
        self._offset += QPointF((img_after.x() - img_before.x()) * self._scale,
                                (img_after.y() - img_before.y()) * self._scale)
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Up:
            factor = 1.15
        elif event.key() == Qt.Key.Key_Down:
            factor = 1.0 / 1.15
        else:
            super().keyPressEvent(event)
            return
        if self._pixmap is None:
            return
        center = QPointF(self.width() / 2, self.height() / 2)
        img_before = self._screen_to_img(center)
        self._scale *= factor
        self._scale = max(0.05, min(self._scale, 30.0))
        img_after = self._screen_to_img(center)
        self._offset += QPointF((img_after.x() - img_before.x()) * self._scale,
                                (img_after.y() - img_before.y()) * self._scale)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        pos = QPointF(event.position())
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.RightButton and self._right_click_pan:
            self._panning = True
            self._pan_start = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            locked = self._visible_group is not None
            if not locked:
                if self._show_bbox and self._bbox_rect is not None:
                    handle = RectDragHelper.hit_test(pos, self._bbox_rect, self._img_to_screen)
                    if handle != RectDragHelper.HANDLE_NONE:
                        self._dragging_bbox_handle = handle
                        return
                kps_idx = self._find_nearest_kps5(pos)
                if kps_idx >= 0:
                    self._dragging_kps_idx = kps_idx
                    self.update()
                    return
            nearest = self._find_nearest(pos)
            if nearest >= 0:
                self._dragging_idx = nearest
                self._active_idx = nearest
                self.update()
                return
            if locked:
                img_pos = self._screen_to_img(pos)
                if self._is_in_group_area(img_pos):
                    self._dragging_group = self._visible_group
                    self._group_drag_start = img_pos
                    self._group_drag_initial = {idx: QPointF(self._landmarks_106[idx]) for idx in self._landmarks_106
                                                if idx in self._get_visible_106_indices_set()}
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                    return

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = QPointF(event.position())
        if self._panning:
            delta = pos - self._pan_start
            self._offset += delta
            self._pan_start = pos
            self.update()
            return
        if self._dragging_bbox_handle != RectDragHelper.HANDLE_NONE and self._bbox_rect is not None:
            img_pt = self._screen_to_img(pos)
            self._bbox_rect = RectDragHelper.apply_drag(self._dragging_bbox_handle, img_pt, self._bbox_rect).normalized()
            self.update()
            return
        if self._dragging_kps_idx >= 0:
            img_pt = self._screen_to_img(pos)
            self._kps_5[self._dragging_kps_idx] = img_pt
            self.update()
            return
        if self._dragging_idx >= 0:
            img_pt = self._screen_to_img(pos)
            self._landmarks_106[self._dragging_idx] = img_pt
            self.update()
            self.landmark_changed.emit()
            return
        if self._dragging_group is not None and self._group_drag_start is not None and self._group_drag_initial is not None:
            img_pt = self._screen_to_img(pos)
            dx = img_pt.x() - self._group_drag_start.x()
            dy = img_pt.y() - self._group_drag_start.y()
            for idx, orig_pt in self._group_drag_initial.items():
                self._landmarks_106[idx] = QPointF(orig_pt.x() + dx, orig_pt.y() + dy)
            self.update()
            self.landmark_changed.emit()
            return

        if not self._panning and self._dragging_idx < 0 and self._dragging_kps_idx < 0:
            cursor = Qt.CursorShape.ArrowCursor
            locked = self._visible_group is not None
            if not locked:
                if self._show_bbox and self._bbox_rect is not None:
                    handle = RectDragHelper.hit_test(pos, self._bbox_rect, self._img_to_screen)
                    if handle != RectDragHelper.HANDLE_NONE:
                        cursor = RectDragHelper.cursor_shape(handle)
            if cursor == Qt.CursorShape.ArrowCursor and locked:
                if self._find_nearest(pos) < 0:
                    img_pos = self._screen_to_img(pos)
                    if self._is_in_group_area(img_pos):
                        cursor = Qt.CursorShape.SizeAllCursor
            self.setCursor(cursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = QPointF(event.position())
        nearest = self._find_nearest(pos)
        if nearest >= 0:
            group = _IDX_TO_GROUP_106.get(nearest)
            if group is not None:
                self._visible_group = group
                self._active_idx = -1
                self.update()
        else:
            self._visible_group = None
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        if event.button() == Qt.MouseButton.RightButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_idx = -1
            self._dragging_bbox_handle = RectDragHelper.HANDLE_NONE
            self._dragging_kps_idx = -1
            self._dragging_group = None
            self._group_drag_start = None
            self._group_drag_initial = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._empty_hint and (self._pixmap is None or self._image is None or self._image_is_empty):
            painter.setPen(QColor(180, 180, 180))
            font = painter.font()
            font.setPixelSize(18)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty_hint)
            painter.end()
            return

        if self._pixmap is not None:
            pw = self._pixmap.width()
            ph = self._pixmap.height()
            scaled_w = max(1, int(round(pw * self._scale)))
            scaled_h = max(1, int(round(ph * self._scale)))
            if scaled_w == pw and scaled_h == ph:
                target_pixmap = self._pixmap
            else:
                cache_key = (scaled_w, scaled_h)
                if self._scaled_pixmap is None or self._scaled_pixmap_key != cache_key:
                    self._scaled_pixmap = self._pixmap.scaled(scaled_w, scaled_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self._scaled_pixmap_key = cache_key
                target_pixmap = self._scaled_pixmap
            painter.drawPixmap(int(round(self._offset.x())), int(round(self._offset.y())), target_pixmap)

            painter.save()
            painter.translate(self._offset)
            painter.scale(self._scale, self._scale)

            if self._show_bbox and self._bbox_rect is not None and self._visible_group is None:
                pen = QPen(QColor(0, 0, 255), 2.0 / self._scale)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(self._bbox_rect)
                corner_r = 4.0 / self._scale
                corner_brush = QBrush(QColor(0, 0, 255))
                painter.setBrush(corner_brush)
                painter.setPen(Qt.PenStyle.NoPen)
                bbox_corners = [
                    QPointF(self._bbox_rect.left(), self._bbox_rect.top()),
                    QPointF(self._bbox_rect.right(), self._bbox_rect.top()),
                    QPointF(self._bbox_rect.right(), self._bbox_rect.bottom()),
                    QPointF(self._bbox_rect.left(), self._bbox_rect.bottom()),
                ]
                for c in bbox_corners:
                    painter.drawRect(QRectF(c.x() - corner_r, c.y() - corner_r, corner_r * 2, corner_r * 2))

            if self._show_106:
                for gname, idxs, color in _LANDMARK_GROUPS_106:
                    is_active_group = (self._visible_group is None or gname == self._visible_group)
                    lines = _106_LINE_CONNECTIONS.get(gname, [])
                    base_alpha = 150 if is_active_group else 50
                    pen = QPen(QColor(color.red(), color.green(), color.blue(), base_alpha), 1.5 / self._scale)
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    for i1, i2 in lines:
                        if i1 in self._landmarks_106 and i2 in self._landmarks_106:
                            p1 = self._landmarks_106[i1]
                            p2 = self._landmarks_106[i2]
                            line_pen = QPen(QColor(color.red(), color.green(), color.blue(), base_alpha), 1.5 / self._scale)
                            painter.setPen(line_pen)
                            painter.drawLine(QPointF(p1.x(), p1.y()), QPointF(p2.x(), p2.y()))
                    for idx in idxs:
                        if idx not in self._landmarks_106:
                            continue
                        pt = self._landmarks_106[idx]
                        r = 3.0 / self._scale
                        pen_w = 1.5 / self._scale
                        is_active = (idx == self._active_idx)
                        if is_active:
                            r = 5.0 / self._scale
                            pen_w = 2.5 / self._scale
                        is_visible = self._lm106_visibility.get(idx, True)
                        if is_visible:
                            if is_active_group:
                                draw_color = color
                                brush_color = color
                            else:
                                draw_color = QColor(color.red(), color.green(), color.blue(), 80)
                                brush_color = QColor(color.red(), color.green(), color.blue(), 40)
                        else:
                            draw_color = QColor(color.red(), color.green(), color.blue(), 120)
                            brush_color = QColor(0, 0, 0, 0)
                        pen = QPen(draw_color, pen_w)
                        painter.setPen(pen)
                        painter.setBrush(QBrush(brush_color))
                        painter.drawEllipse(QPointF(pt.x(), pt.y()), r, r)
                        if self._show_point_numbers and is_active_group:
                            font = painter.font()
                            font.setPixelSize(max(8, int(10 / self._scale)))
                            painter.setFont(font)
                            painter.setPen(QPen(QColor(255, 255, 0), 1.0 / self._scale))
                            painter.drawText(QPointF(pt.x() + r + 2 / self._scale, pt.y() - r), str(idx))

            if self._show_kps5 and self._kps_5 and self._visible_group is None:
                for idx in range(5):
                    if idx not in self._kps_5:
                        continue
                    pt = self._kps_5[idx]
                    color = _KPS5_COLORS[idx]
                    r = 6.0 / self._scale
                    pen = QPen(QColor(255, 255, 255), 2.0 / self._scale)
                    painter.setPen(pen)
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(QPointF(pt.x(), pt.y()), r, r)
                    if self._show_point_numbers:
                        font = painter.font()
                        font.setPixelSize(max(9, int(11 / self._scale)))
                        painter.setFont(font)
                        painter.setPen(QPen(QColor(255, 255, 0), 1.0 / self._scale))
                        painter.drawText(QPointF(pt.x() + r + 2 / self._scale, pt.y() - r), _KPS5_NAMES[idx])
                if len(self._kps_5) >= 2:
                    pen = QPen(QColor(0, 255, 255, 120), 1.5 / self._scale)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    for i in range(min(4, len(self._kps_5) - 1)):
                        if i in self._kps_5 and (i + 1) in self._kps_5:
                            p1 = self._kps_5[i]
                            p2 = self._kps_5[i + 1]
                            painter.drawLine(QPointF(p1.x(), p1.y()), QPointF(p2.x(), p2.y()))

            painter.restore()


class _CompactThumbDelegate(QStyledItemDelegate):
    _THUMB_W = 120
    _THUMB_H = 80

    def __init__(self, parent=None, full_width: bool = False):
        super().__init__(parent)
        self._text_gap = 1
        self._full_width = full_width

    def _item_width(self) -> int:
        if self._full_width:
            lv = self.parent()
            if lv is not None and hasattr(lv, 'viewport'):
                return lv.viewport().width()
        return self._THUMB_W + 6

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        painter.save()
        self.initStyleOption(option, index)
        icon = option.icon
        text = option.text
        rect = option.rect

        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(rect, QColor(0, 120, 212, 60))

        tw, th = self._THUMB_W, self._THUMB_H
        icon_x = rect.x() + (rect.width() - tw) // 2
        icon_y = rect.y() + 2
        if not icon.isNull():
            icon.paint(painter, icon_x, icon_y, tw, th)
            text_y = icon_y + th + self._text_gap
        else:
            text_y = rect.y() + 2

        font = option.font
        font.setPixelSize(10)
        painter.setFont(font)
        painter.setPen(QColor(0, 0, 0))
        text_rect = QRectF(rect.x(), text_y, rect.width(), rect.bottom() - text_y)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, text)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        font = option.font
        font.setPixelSize(10)
        fm = QFontMetrics(font)
        text_h = fm.height()
        h = self._THUMB_H + self._text_gap + text_h + 4
        w = self._item_width()
        return QSize(w, h)


def _resize_pad(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def load_thumbnail(path: Path, target_w: int, target_h: int, pad: bool = True) -> Optional[QImage]:
    """读取图片生成缩略图QImage。pad=True时等比缩放+居中padding到固定尺寸，False时仅等比缩放不padding。"""
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        img = cv2.imread(str(path))
        if img is None:
            return None
    if img.shape[1] < target_w or img.shape[0] < target_h:
        full = cv2.imread(str(path))
        if full is not None:
            img = full
    if pad:
        thumb = _resize_pad(img, target_w, target_h)
        w, h = target_w, target_h
    else:
        h0, w0 = img.shape[:2]
        scale = min(target_w / w0, target_h / h0)
        if scale < 1:
            thumb = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
        else:
            thumb = img
        h, w = thumb.shape[:2]
    rgb = bgr_to_rgb(thumb)
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class _ThumbLoader(QThread):
    thumb_loaded = pyqtSignal(int, object, str, str)
    loading_finished = pyqtSignal()

    def __init__(self, image_paths: list[Path], preload_count: int = 5):
        super().__init__()
        self._paths = image_paths
        self._preload_count = preload_count
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        for i, img_path in enumerate(self._paths):
            if self._stop:
                break
            if i < self._preload_count:
                continue
            qimg = load_thumbnail(img_path, 120, 80)
            if qimg is not None:
                self.thumb_loaded.emit(i, qimg, img_path.stem, str(img_path))
        if not self._stop:
            self.loading_finished.emit()



def _toggle_canvas_visibility(canvas: ImageCanvas) -> str:
    """切换可见性，返回状态描述文本"""
    group = canvas._visible_group
    if group is not None:
        canvas.toggle_group_visibility(group)
        invisible_count = sum(1 for v in canvas.get_lm106_visibility().values() if not v)
        return f"已切换 {group} 可见性，当前不可见点: {invisible_count}"
    active = canvas._active_idx
    if active >= 0 and active in canvas._landmarks_106:
        canvas.toggle_point_visibility(active)
        vis = canvas.get_lm106_visibility().get(active, True)
        return f"点 {active} 已标记为{'可见' if vis else '不可见'}"
    return ""

