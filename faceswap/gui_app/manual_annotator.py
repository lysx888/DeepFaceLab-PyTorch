import json

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
    QComboBox, QWidget, QSplitter, QListWidget,
    QMessageBox, QGroupBox, QListWidgetItem, QScrollArea,
    QSizePolicy, QStackedWidget, QStyledItemDelegate, QStyleOptionViewItem, QStyle,
)

from faceswap.setting import FaceType, DATA_SRC_DIR, DATA_DST_DIR, DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR, INSIGHTFACE_TRAIN_DIR, WORKSPACE_DIR
from faceswap.shared.file_manager import FileManager
from faceswap.shared.image_utils import bgr_to_rgb
from faceswap.shared.logger import get_logger
from faceswap.core.debug_image_generator import save_debug_image
from faceswap.gui_app.gui_utils import RectDragHelper, save_face_annotation
from faceswap.business.insightface_training_data_generator import InsightFaceTrainingDataGenerator

_logger = get_logger("manual_annotator")

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


def _generate_default_group_points(group_name: str, face_rect: QRectF) -> dict[int, QPointF]:
    shapes = _DEFAULT_GROUP_SHAPES.get(group_name, [])
    if not shapes:
        center = QPointF(face_rect.center().x(), face_rect.center().y())
        for gname, gidxs, _ in _LANDMARK_GROUPS_106:
            if gname == group_name:
                return {idx: QPointF(center.x(), center.y()) for idx in gidxs}
        return {}
    x = face_rect.x()
    y = face_rect.y()
    w = face_rect.width()
    h = face_rect.height()
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


def _canvas_kps5_to_insightface(canvas_kps5: dict[int, QPointF]) -> np.ndarray:
    kps = np.zeros((5, 2), dtype=np.float32)
    kps[0] = [canvas_kps5[1].x(), canvas_kps5[1].y()]
    kps[1] = [canvas_kps5[0].x(), canvas_kps5[0].y()]
    kps[2] = [canvas_kps5[2].x(), canvas_kps5[2].y()]
    kps[3] = [canvas_kps5[4].x(), canvas_kps5[4].y()]
    kps[4] = [canvas_kps5[3].x(), canvas_kps5[3].y()]
    return kps


def _compute_visibility_for_point(point: QPointF, face_rect: Optional[QRectF]) -> bool:
    if face_rect is None:
        return True
    return face_rect.contains(point)


def _compute_landmarks_106_visibility(landmarks_106: dict[int, QPointF], face_rect: Optional[QRectF]) -> list[bool]:
    vis = [True] * 106
    if face_rect is None:
        return vis
    for idx, pt in landmarks_106.items():
        if 0 <= idx < 106:
            vis[idx] = face_rect.contains(pt)
    return vis


def _compute_kps_5_visibility(kps_5: dict[int, QPointF], face_rect: Optional[QRectF]) -> list[bool]:
    vis = [True] * 5
    if face_rect is None:
        return vis
    for idx, pt in kps_5.items():
        if 0 <= idx < 5:
            vis[idx] = face_rect.contains(pt)
    return vis


from enum import IntEnum


class AnnotationStep(IntEnum):
    STEP_0_IDLE = 0
    STEP_1_FACE_RECT = 1
    STEP_2_KPS5 = 2
    STEP_3_LANDMARKS_106 = 3
    STEP_COMPLETE = 4


class ImageCanvas(QWidget):
    landmark_changed = pyqtSignal()
    face_rect_changed = pyqtSignal()

    def __init__(self, parent=None, show_point_numbers: bool = True, right_click_pan: bool = False):
        super().__init__(parent)
        self._image = None
        self._pixmap = None
        self._landmarks_106: dict[int, QPointF] = {}
        self._lm106_visibility: dict[int, bool] = {}
        self._face_rect: Optional[QRectF] = None
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
        self._show_face_rect = True
        self._show_point_numbers = show_point_numbers
        self._dragging_face_rect_handle = RectDragHelper.HANDLE_NONE
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

    def set_image(self, img: np.ndarray):
        self._image = img.copy()
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

    def set_face_rect(self, rect: Optional[QRectF]):
        self._face_rect = rect
        self.update()
        self.face_rect_changed.emit()

    def get_face_rect(self) -> Optional[QRectF]:
        return self._face_rect

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

    def set_show_face_rect(self, show: bool):
        self._show_face_rect = show
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
        if event.button() == Qt.MouseButton.RightButton:
            self._panning = True
            self._pan_start = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._show_face_rect and self._face_rect is not None:
                handle = RectDragHelper.hit_test(pos, self._face_rect, self._img_to_screen)
                if handle != RectDragHelper.HANDLE_NONE:
                    self._dragging_face_rect_handle = handle
                    return
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
        if self._dragging_face_rect_handle != RectDragHelper.HANDLE_NONE and self._face_rect is not None:
            img_pt = self._screen_to_img(pos)
            self._face_rect = RectDragHelper.apply_drag(self._dragging_face_rect_handle, img_pt, self._face_rect).normalized()
            self.update()
            self.face_rect_changed.emit()
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
            if self._show_face_rect and self._face_rect is not None:
                handle = RectDragHelper.hit_test(pos, self._face_rect, self._img_to_screen)
                if handle != RectDragHelper.HANDLE_NONE:
                    cursor = RectDragHelper.cursor_shape(handle)
            if cursor == Qt.CursorShape.ArrowCursor and self._show_bbox and self._bbox_rect is not None:
                handle = RectDragHelper.hit_test(pos, self._bbox_rect, self._img_to_screen)
                if handle != RectDragHelper.HANDLE_NONE:
                    cursor = RectDragHelper.cursor_shape(handle)
            if cursor == Qt.CursorShape.ArrowCursor:
                nearest_106 = self._find_nearest(pos)
                nearest_kps = self._find_nearest_kps5(pos)
                if nearest_106 < 0 and nearest_kps < 0:
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
            self._dragging_face_rect_handle = RectDragHelper.HANDLE_NONE
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

        if self._empty_hint and (self._pixmap is None or self._image is None or self._image.max() == 0):
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

            if self._show_face_rect and self._face_rect is not None:
                pen = QPen(QColor(0, 120, 212), 2.5 / self._scale)
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(self._face_rect)
                corner_r = 5.0 / self._scale
                corner_brush = QBrush(QColor(0, 120, 212))
                painter.setBrush(corner_brush)
                painter.setPen(Qt.PenStyle.NoPen)
                corners = [
                    QPointF(self._face_rect.left(), self._face_rect.top()),
                    QPointF(self._face_rect.right(), self._face_rect.top()),
                    QPointF(self._face_rect.right(), self._face_rect.bottom()),
                    QPointF(self._face_rect.left(), self._face_rect.bottom()),
                ]
                for c in corners:
                    painter.drawRect(QRectF(c.x() - corner_r, c.y() - corner_r, corner_r * 2, corner_r * 2))

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
                visible_indices = self._get_visible_106_indices()
                for gname, idxs, color in _LANDMARK_GROUPS_106:
                    if visible_indices is not None and gname != self._visible_group:
                        continue
                    lines = _106_LINE_CONNECTIONS.get(gname, [])
                    pen = QPen(QColor(color.red(), color.green(), color.blue(), 150), 1.5 / self._scale)
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    for i1, i2 in lines:
                        if i1 in self._landmarks_106 and i2 in self._landmarks_106:
                            p1 = self._landmarks_106[i1]
                            p2 = self._landmarks_106[i2]
                            p1_out = (self._show_face_rect and self._face_rect is not None
                                      and not self._face_rect.contains(p1))
                            p2_out = (self._show_face_rect and self._face_rect is not None
                                      and not self._face_rect.contains(p2))
                            if p1_out or p2_out:
                                line_pen = QPen(QColor(color.red(), color.green(), color.blue(), 40), 1.5 / self._scale)
                            else:
                                line_pen = QPen(QColor(color.red(), color.green(), color.blue(), 150), 1.5 / self._scale)
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
                        if self._show_face_rect and self._face_rect is not None:
                            is_visible = is_visible and self._face_rect.contains(pt)
                        if is_visible:
                            draw_color = color
                            brush_color = color
                        else:
                            draw_color = QColor(color.red(), color.green(), color.blue(), 120)
                            brush_color = QColor(0, 0, 0, 0)
                        pen = QPen(draw_color, pen_w)
                        painter.setPen(pen)
                        painter.setBrush(QBrush(brush_color))
                        painter.drawEllipse(QPointF(pt.x(), pt.y()), r, r)
                        if self._show_point_numbers:
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


class _ThumbLoader(QThread):
    thumb_loaded = pyqtSignal(int, object, str, str)
    loading_finished = pyqtSignal()

    def __init__(self, image_paths: list[Path], preload_count: int = 5):
        super().__init__()
        self._paths = image_paths
        self._preload_count = preload_count

    def run(self):
        for i, img_path in enumerate(self._paths):
            if i < self._preload_count:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            thumb = cv2.resize(img, (120, 80))
            rgb = bgr_to_rgb(thumb)
            qimg = QImage(rgb.data, 120, 80, 3 * 120, QImage.Format.Format_RGB888).copy()
            self.thumb_loaded.emit(i, qimg, img_path.stem, str(img_path))
        self.loading_finished.emit()


class ManualAnnotatorDialog(QDialog):
    def __init__(self, parent=None, is_src: bool = True, img_path: Optional[Path] = None):
        super().__init__(parent)
        self.setWindowTitle("手动人脸标注")
        self.setMinimumSize(1300, 800)
        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowType.WindowMinMaxButtonsHint |
                            Qt.WindowType.Window)
        self.showMaximized()
        self._is_src = is_src
        self._current_img_path = img_path
        self._current_img = None
        self._face_type = FaceType.WHOLE_FACE
        self._output_size = 512
        self._face_rect_generated = False
        self._kps5_saved = False
        self._annotation_step = AnnotationStep.STEP_0_IDLE
        self._to_annotate_images = []
        self._current_to_annotate_idx = -1
        self._build_ui()
        if img_path is not None:
            self._init_navigation_list(img_path)
            self._load_single_image(img_path)
        else:
            self._to_annotate_images = InsightFaceTrainingDataGenerator(WORKSPACE_DIR).get_to_annotate_images()
            if self._to_annotate_images:
                self._load_to_annotate_image(0)
            else:
                self._canvas.set_empty_hint("请先把图片复制到to_annotate目录再打开此页面")
                self._status.setText("请先把图片复制到to_annotate目录再打开此页面")

    def _build_ui(self):
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()

        src_btn = QPushButton("源 (SRC)")
        src_btn.setCheckable(True)
        src_btn.setChecked(self._is_src)
        src_btn.clicked.connect(lambda: self._set_target(True))
        dst_btn = QPushButton("目标 (DST)")
        dst_btn.setCheckable(True)
        dst_btn.setChecked(not self._is_src)
        dst_btn.clicked.connect(lambda: self._set_target(False))
        self._src_btn = src_btn
        self._dst_btn = dst_btn
        toolbar.addWidget(src_btn)
        toolbar.addWidget(dst_btn)

        to_annotate_btn = QPushButton("标注训练 (to_annotate)")
        to_annotate_btn.setStyleSheet("QPushButton { background-color: #0078D4; color: white; font-weight: bold; padding: 4px 8px; }")
        to_annotate_btn.clicked.connect(self._set_to_annotate)
        toolbar.addWidget(to_annotate_btn)

        self._img_label = QLabel("未加载")
        self._img_label.setMinimumWidth(150)
        self._img_label.setStyleSheet("font-weight: bold; padding: 2px 8px;")
        toolbar.addWidget(self._img_label)

        prev_btn = QPushButton("◀ 上一帧")
        prev_btn.clicked.connect(self._prev_image)
        toolbar.addWidget(prev_btn)
        next_btn = QPushButton("下一帧 ▶")
        next_btn.clicked.connect(self._next_image)
        toolbar.addWidget(next_btn)

        toolbar.addSpacing(10)

        ft_label = QLabel("人脸类型:")
        toolbar.addWidget(ft_label)
        self._ft_combo = QComboBox()
        self._ft_combo.addItems(["whole_face", "head"])
        self._ft_combo.setFixedWidth(120)
        self._ft_combo.currentTextChanged.connect(self._on_face_type_changed)
        toolbar.addWidget(self._ft_combo)

        sz_label = QLabel("尺寸:")
        toolbar.addWidget(sz_label)
        self._size_combo = QComboBox()
        self._size_combo.addItems(["512", "768", "384", "256", "640", "1024"])
        self._size_combo.setCurrentText("512")
        self._size_combo.setFixedWidth(80)
        toolbar.addWidget(self._size_combo)

        toolbar.addSpacing(10)

        auto_btn = QPushButton("自动标注")
        auto_btn.setStyleSheet("QPushButton { background-color: #5B2D8E; color: white; font-weight: bold; padding: 5px 14px; }")
        auto_btn.clicked.connect(self._auto_annotate)
        toolbar.addWidget(auto_btn)

        apply_next_btn = QPushButton("应用到下一帧")
        apply_next_btn.setStyleSheet("QPushButton { background-color: #2D6A8E; color: white; font-weight: bold; padding: 5px 14px; }")
        apply_next_btn.clicked.connect(self._apply_to_next)
        toolbar.addWidget(apply_next_btn)

        toolbar.addStretch()

        layout.addLayout(toolbar)

        self._canvas = ImageCanvas(show_point_numbers=False, right_click_pan=True)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        splitter.addWidget(self._canvas)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(8)

        help_label = QLabel(
            "操作说明:\n"
            "1、左键拖动调整点位置\n"
            "2、按V切换当前组可见/不可见（不可见点用空心显示）\n"
        )
        help_label.setStyleSheet("font-size: 11px; color: #888888; padding: 4px;")
        scroll_layout.addWidget(help_label)

        self._point_num_toggle_btn = QPushButton("显示/隐藏 序号")
        self._point_num_toggle_btn.setCheckable(True)
        self._point_num_toggle_btn.setChecked(False)
        self._point_num_toggle_btn.setStyleSheet("QPushButton { color: #CCCCCC; padding: 4px 8px; }")
        self._point_num_toggle_btn.clicked.connect(self._on_point_num_toggle)
        scroll_layout.addWidget(self._point_num_toggle_btn)

        self._vis_toggle_btn = QPushButton("标记不可见 (V)")
        self._vis_toggle_btn.setStyleSheet("QPushButton { color: #FF6600; padding: 4px 8px; }")
        self._vis_toggle_btn.clicked.connect(self._toggle_visibility)
        scroll_layout.addWidget(self._vis_toggle_btn)

        face_rect_grp = QGroupBox("人脸区域框")
        face_rect_grp.setStyleSheet("QGroupBox { padding-top: 14px; padding-right: 4px; padding-bottom: 2px; padding-left: 4px; margin-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 2px; }")
        face_rect_layout = QVBoxLayout(face_rect_grp)
        self._gen_rect_btn = QPushButton("生成人脸区域框")
        self._gen_rect_btn.setStyleSheet("QPushButton { background-color: #0078D4; color: white; font-weight: bold; padding: 4px 8px; }")
        self._gen_rect_btn.clicked.connect(self._generate_face_rect)
        face_rect_layout.addWidget(self._gen_rect_btn)
        scroll_layout.addWidget(face_rect_grp)

        kps5_grp = QGroupBox("5点检测器及检测框")
        kps5_grp.setStyleSheet("QGroupBox { padding-top: 14px; padding-right: 4px; padding-bottom: 2px; padding-left: 4px; margin-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 2px; }")
        kps5_layout = QVBoxLayout(kps5_grp)
        self._gen_bbox_btn = QPushButton("生成检测框")
        self._gen_bbox_btn.setStyleSheet("QPushButton { background-color: #0078D4; color: white; font-weight: bold; padding: 4px 8px; }")
        self._gen_bbox_btn.clicked.connect(self._generate_bbox)
        kps5_layout.addWidget(self._gen_bbox_btn)
        kps5_btn_style = "QPushButton { background-color: #0078D4; color: white; font-weight: bold; padding: 3px 6px; border-radius: 3px; }"
        for row_indices in _KPS5_BUTTON_ORDER:
            kps5_row = QHBoxLayout()
            for kps_idx in row_indices:
                btn = QPushButton(_KPS5_NAMES[kps_idx])
                btn.setStyleSheet(kps5_btn_style)
                btn.clicked.connect(lambda checked, ki=kps_idx: self._on_kps5_point_btn(ki))
                kps5_row.addWidget(btn)
            kps5_layout.addLayout(kps5_row)
        self._kps5_toggle_btn = QPushButton("显示/隐藏 5点和检测框")
        self._kps5_toggle_btn.setCheckable(True)
        self._kps5_toggle_btn.setChecked(True)
        self._kps5_toggle_btn.setStyleSheet("QPushButton { color: #00FFFF; padding: 4px 8px; }")
        self._kps5_toggle_btn.clicked.connect(self._on_kps5_toggle)
        kps5_layout.addWidget(self._kps5_toggle_btn)
        self._kps5_info_label = QLabel("点击按钮在人脸区域框中心添加点\n拖动调整位置，拖到框外=不可见")
        self._kps5_info_label.setStyleSheet("font-size: 10px; color: #888888;")
        kps5_layout.addWidget(self._kps5_info_label)
        scroll_layout.addWidget(kps5_grp)

        align106_grp = QGroupBox("106点对齐器")
        align106_grp.setStyleSheet("QGroupBox { padding-top: 14px; padding-right: 4px; padding-bottom: 2px; padding-left: 4px; margin-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 2px; }")
        align106_layout = QVBoxLayout(align106_grp)
        self._lm106_toggle_btn = QPushButton("显示/隐藏 106点")
        self._lm106_toggle_btn.setCheckable(True)
        self._lm106_toggle_btn.setChecked(True)
        self._lm106_toggle_btn.setStyleSheet("QPushButton { color: #CCCCCC; padding: 4px 8px; }")
        self._lm106_toggle_btn.clicked.connect(self._on_lm106_toggle)
        align106_layout.addWidget(self._lm106_toggle_btn)
        btn_row_106 = None
        for gidx, (gname, gidxs, gcolor) in enumerate(_LANDMARK_GROUPS_106):
            btn = QPushButton(gname)
            btn.setStyleSheet(f"QPushButton {{ color: {gcolor.name()}; padding: 3px 4px; }}")
            btn.clicked.connect(lambda checked, gn=gname: self._on_106_group_btn(gn))
            if gidx % 2 == 0:
                btn_row_106 = QHBoxLayout()
            btn_row_106.addWidget(btn)
            if gidx % 2 == 1 or gidx == len(_LANDMARK_GROUPS_106) - 1:
                align106_layout.addLayout(btn_row_106)

        scroll_layout.addWidget(align106_grp)

        action_grp = QGroupBox("操作")
        action_grp.setStyleSheet("QGroupBox { padding-top: 14px; padding-right: 4px; padding-bottom: 2px; padding-left: 4px; margin-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 2px; }")
        action_layout = QVBoxLayout(action_grp)
        btn_row = QHBoxLayout()
        save106_btn = QPushButton("保存")
        save106_btn.setStyleSheet("QPushButton { background-color: #0078D4; color: white; font-weight: bold; padding: 3px 8px; }")
        save106_btn.clicked.connect(self._save_106)
        btn_row.addWidget(save106_btn)
        delete_btn = QPushButton("删除")
        delete_btn.setStyleSheet("QPushButton { background-color: #C42B1C; color: white; font-weight: bold; padding: 3px 8px; }")
        delete_btn.clicked.connect(self._delete_current_group)
        btn_row.addWidget(delete_btn)
        action_layout.addLayout(btn_row)
        scroll_layout.addWidget(action_grp)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        right_layout.addWidget(scroll)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 8)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([800, 200])

        layout.addWidget(splitter, 1)

        self._status = QLabel("请加载图片开始标注")
        self._status.setStyleSheet("padding: 4px; color: #CCCCCC;")
        layout.addWidget(self._status)

        self._canvas.landmark_changed.connect(self._update_progress)
        self._canvas.face_rect_changed.connect(self._on_face_rect_changed)

    def _set_target(self, is_src: bool):
        self._is_src = is_src
        self._src_btn.setChecked(is_src)
        self._dst_btn.setChecked(not is_src)
        frames_dir = DATA_SRC_DIR if is_src else DATA_DST_DIR
        from faceswap.shared.file_manager import FileManager
        frame_images = sorted(FileManager.find_images(frames_dir))
        if not frame_images:
            tag = "源" if is_src else "目标"
            QMessageBox.information(self, "提示", f"{tag}帧目录为空")
            return
        dlg = _FrameSelectDialog(frame_images, self)
        dlg.frame_selected.connect(self._on_frame_selected)
        dlg.exec()

    def _set_to_annotate(self):
        to_annotate_dir = InsightFaceTrainingDataGenerator(WORKSPACE_DIR).to_annotate_dir
        if not to_annotate_dir.exists() or not any(to_annotate_dir.iterdir()):
            QMessageBox.information(self, "提示", f"to_annotate目录为空\n{to_annotate_dir}")
            return
        to_annotate_images = InsightFaceTrainingDataGenerator(WORKSPACE_DIR).get_to_annotate_images()
        dlg = _FrameSelectDialog(to_annotate_images, self)
        dlg.frame_selected.connect(self._on_frame_selected)
        dlg.exec()

    def _on_frame_selected(self, img_path: str):
        p = Path(img_path)
        self._init_navigation_list(p)
        self._load_single_image(p)

    def _on_face_type_changed(self, text):
        size_map = {"whole_face": "512", "head": "768", "full": "384", "mid_full": "256", "half": "128"}
        if text in size_map:
            self._size_combo.setCurrentText(size_map[text])

    def _on_face_rect_changed(self):
        pass

    def _on_kps5_toggle(self, checked):
        self._canvas.set_show_kps5(checked)
        self._canvas.set_show_bbox(checked)

    def _on_lm106_toggle(self, checked):
        self._canvas.set_show_106(checked)

    def _on_point_num_toggle(self, checked):
        self._canvas._show_point_numbers = checked
        self._canvas.update()

    def _toggle_visibility(self):
        canvas = self._edit_canvas if self._editing else self._canvas
        active = canvas._active_idx
        if active >= 0 and active in canvas._landmarks_106:
            canvas.toggle_point_visibility(active)
            vis = canvas.get_lm106_visibility().get(active, True)
            self._status.setText(f"点 {active} 已标记为{'可见' if vis else '不可见'}")
            return
        group = canvas._visible_group
        if group is None:
            QMessageBox.information(self, "提示", "请先双击锁定一个五官部位，或点击选中一个点")
            return
        canvas.toggle_group_visibility(group)
        invisible_count = sum(1 for v in canvas.get_lm106_visibility().values() if not v)
        self._status.setText(f"已切换 {group} 可见性，当前不可见点: {invisible_count}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_V:
            self._toggle_visibility()
            return
        super().keyPressEvent(event)

    def _get_current_step(self) -> AnnotationStep:
        if self._current_img is None:
            return AnnotationStep.STEP_0_IDLE
        if not self._face_rect_generated or self._canvas.get_face_rect() is None:
            return AnnotationStep.STEP_1_FACE_RECT
        kps5 = self._canvas.get_kps_5()
        if not self._kps5_saved or len(kps5) < 5 or self._canvas.get_bbox_rect() is None:
            return AnnotationStep.STEP_2_KPS5
        lm106 = self._canvas.get_landmarks_106()
        if len(lm106) < 106:
            return AnnotationStep.STEP_3_LANDMARKS_106
        return AnnotationStep.STEP_COMPLETE

    def _check_step_prerequisite(self, required_step: AnnotationStep) -> bool:
        current = self._get_current_step()
        if current >= required_step:
            return True
        if required_step == AnnotationStep.STEP_2_KPS5:
            QMessageBox.warning(self, "提示", "请先点击\"生成人脸区域框\"")
        elif required_step == AnnotationStep.STEP_3_LANDMARKS_106:
            if not self._face_rect_generated or self._canvas.get_face_rect() is None:
                QMessageBox.warning(self, "提示", "请先点击\"生成人脸区域框\"")
            elif len(self._canvas.get_kps_5()) < 5:
                QMessageBox.warning(self, "提示", "请先添加5点检测器")
            elif self._canvas.get_bbox_rect() is None:
                QMessageBox.warning(self, "提示", "请先点击\"生成检测框\"")
            else:
                QMessageBox.warning(self, "提示", "请先添加5点检测器和检测框")
        return False

    def _on_kps5_point_btn(self, kps_idx: int):
        if not self._check_step_prerequisite(AnnotationStep.STEP_2_KPS5):
            return
        kps5 = self._canvas.get_kps_5()
        if kps_idx in kps5:
            QMessageBox.warning(self, "提示", f"{_KPS5_NAMES[kps_idx]}已添加，请拖动调整位置")
            return
        face_rect = self._canvas.get_face_rect()
        if kps_idx in _KPS5_DEFAULT_POS and face_rect is not None:
            fx, fy = _KPS5_DEFAULT_POS[kps_idx]
            pos = QPointF(face_rect.x() + fx * face_rect.width(), face_rect.y() + fy * face_rect.height())
        else:
            pos = QPointF(face_rect.center().x(), face_rect.center().y())
        kps5[kps_idx] = pos
        self._canvas.set_kps_5(kps5)
        self._canvas.set_show_kps5(self._kps5_toggle_btn.isChecked())
        if len(kps5) == 5:
            self._kps5_saved = True
        self._status.setText(f"已添加 {_KPS5_NAMES[kps_idx]}，请拖动调整位置")

    def _load_single_image(self, img_path: Path):
        self._current_img_path = img_path
        self._current_img = cv2.imread(str(img_path))
        if self._current_img is None:
            self._status.setText(f"无法读取: {img_path.name}")
            return
        self._canvas.set_empty_hint(None)
        self._canvas.set_image(self._current_img)
        self._canvas.clear_landmarks()
        self._canvas.set_face_rect(None)
        self._canvas.set_bbox_rect(None)
        self._face_rect_generated = False
        self._kps5_saved = False
        self._annotation_step = AnnotationStep.STEP_1_FACE_RECT
        self._img_label.setText(img_path.name)
        self._status.setText(f"已加载: {img_path.name} ({self._current_img.shape[1]}x{self._current_img.shape[0]})")
        self._try_load_existing_annotation(img_path)

    def _try_load_existing_annotation(self, img_path: Path):
        generator = InsightFaceTrainingDataGenerator(WORKSPACE_DIR)
        json_path = generator.find_manual_annotation(img_path)
        if json_path is None:
            return
        try:
            with open(str(json_path), "r", encoding="utf-8") as f:
                ann = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        bbox = ann.get("bbox")
        face_rect = ann.get("source_face_rect")
        kps_5 = ann.get("kps_5")
        landmarks_106 = ann.get("landmarks_106")

        if bbox is not None and len(bbox) == 4:
            self._canvas.set_bbox_rect(QRectF(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))

        if kps_5 is not None and len(kps_5) == 5:
            canvas_kps = {
                0: QPointF(kps_5[1][0], kps_5[1][1]),
                1: QPointF(kps_5[0][0], kps_5[0][1]),
                2: QPointF(kps_5[2][0], kps_5[2][1]),
                3: QPointF(kps_5[4][0], kps_5[4][1]),
                4: QPointF(kps_5[3][0], kps_5[3][1]),
            }
            self._canvas.set_kps_5(canvas_kps)

        if landmarks_106 is not None and len(landmarks_106) == 106:
            canvas_lm = {i: QPointF(pt[0], pt[1]) for i, pt in enumerate(landmarks_106)}
            self._canvas.set_landmarks_106(canvas_lm)
            lm106_vis = ann.get("landmarks_106_visibility")
            if lm106_vis is not None and len(lm106_vis) == 106:
                self._canvas.set_lm106_visibility({i: bool(lm106_vis[i]) for i in range(106)})

        if face_rect is not None and len(face_rect) == 4:
            self._canvas.set_face_rect(QRectF(face_rect[0], face_rect[1], face_rect[2], face_rect[3]))
            self._face_rect_generated = True

        self._kps5_saved = True
        self._annotation_step = AnnotationStep.STEP_COMPLETE
        self._status.setText(f"已加载: {img_path.name} (已恢复标注)")

    def _init_navigation_list(self, img_path: Path):
        img_path = Path(img_path)
        aligned_dir = DATA_SRC_ALIGNED_DIR if self._is_src else DATA_DST_ALIGNED_DIR
        debug_dir = aligned_dir.parent / (aligned_dir.name + "_debug")
        to_annotate_dir = InsightFaceTrainingDataGenerator(WORKSPACE_DIR).to_annotate_dir
        try:
            img_parent = img_path.parent.resolve()
        except Exception:
            img_parent = img_path.parent
        if debug_dir.exists() and str(img_parent).startswith(str(debug_dir.resolve())):
            self._to_annotate_images = sorted(FileManager.find_images(debug_dir), key=lambda p: p.name)
        elif to_annotate_dir.exists() and str(img_parent).startswith(str(to_annotate_dir.resolve())):
            self._to_annotate_images = InsightFaceTrainingDataGenerator(WORKSPACE_DIR).get_to_annotate_images()
        else:
            frames_dir = DATA_SRC_DIR if self._is_src else DATA_DST_DIR
            self._to_annotate_images = sorted(FileManager.find_images(frames_dir), key=lambda p: p.name)
        for i, p in enumerate(self._to_annotate_images):
            if str(p) == str(img_path):
                self._current_to_annotate_idx = i
                break

    def _load_to_annotate_image(self, idx: int):
        if idx < 0 or idx >= len(self._to_annotate_images):
            return
        self._current_to_annotate_idx = idx
        img_path = self._to_annotate_images[idx]
        self._load_single_image(img_path)
        total = len(self._to_annotate_images)
        self._img_label.setText(f"[{idx + 1}/{total}] {img_path.name}")
        self._status.setText(
            f"[{idx + 1}/{total}] {img_path.name} "
            f"({self._current_img.shape[1]}x{self._current_img.shape[0]})"
        )

    def _prev_image(self):
        if self._current_to_annotate_idx > 0:
            self._load_to_annotate_image(self._current_to_annotate_idx - 1)

    def _next_image(self):
        if self._current_to_annotate_idx < len(self._to_annotate_images) - 1:
            self._load_to_annotate_image(self._current_to_annotate_idx + 1)

    def _generate_face_rect(self):
        if self._current_img is None:
            QMessageBox.warning(self, "提示", "请先加载图片")
            return
        h, w = self._current_img.shape[:2]
        self._canvas.set_face_rect(QRectF(w * 0.15, -h * 0.05, w * 0.7, h * 0.95))
        self._face_rect_generated = True
        self._annotation_step = AnnotationStep.STEP_2_KPS5
        self._status.setText("人脸区域框已生成，请添加5点检测器")

    def _generate_bbox(self):
        if self._current_img is None:
            QMessageBox.warning(self, "提示", "请先加载图片")
            return
        h, w = self._current_img.shape[:2]
        self._canvas.set_bbox_rect(QRectF(w * 0.25, h * 0.15, w * 0.5, h * 0.7))
        self._status.setText("检测框已生成，请拖动调整位置")

    def _check_face_rect(self) -> bool:
        if not self._face_rect_generated or self._canvas.get_face_rect() is None:
            QMessageBox.warning(self, "提示", "请先点击\"生成人脸区域框\"按钮")
            return False
        return True

    def _on_106_group_btn(self, group_name: str):
        if not self._check_step_prerequisite(AnnotationStep.STEP_3_LANDMARKS_106):
            return

        missing = []
        kps5 = self._canvas.get_kps_5()
        if len(kps5) < 5:
            missing.append("5点检测器")
        if self._canvas.get_bbox_rect() is None:
            missing.append("检测框")
        if missing:
            QMessageBox.warning(self, "提示", f"缺少: {', '.join(missing)}\n请先添加5点检测器和检测框后再添加106点。")
            return

        face_rect = self._canvas.get_face_rect()
        default_pts = _generate_default_group_points(group_name, face_rect) if face_rect else {}

        lm106 = self._canvas.get_landmarks_106()
        for gname, gidxs, _ in _LANDMARK_GROUPS_106:
            if gname == group_name:
                for idx in gidxs:
                    if idx not in lm106:
                        if idx in default_pts:
                            lm106[idx] = default_pts[idx]
                        elif face_rect:
                            center = QPointF(face_rect.center().x(), face_rect.center().y())
                            lm106[idx] = center
                break

        self._canvas.set_landmarks_106(lm106)
        self._canvas.set_show_106(self._lm106_toggle_btn.isChecked())
        self._canvas.set_visible_group(group_name)
        self._canvas.set_show_kps5(self._kps5_toggle_btn.isChecked())

        for gname, gidxs, _ in _LANDMARK_GROUPS_106:
            if gname == group_name:
                for idx in gidxs:
                    if idx in lm106:
                        self._canvas._active_idx = idx
                        self._status.setText(f"编辑: {group_name} - 左键拖动调整")
                        return
                if gidxs:
                    self._status.setText(f"查看: {group_name} - 无特征点")
                return

    def _auto_annotate_silent(self):
        if self._current_img is None:
            return
        from faceswap.core.insightface_adapter import InsightFaceAdapter
        adapter = InsightFaceAdapter()
        faces = adapter.detect_faces(self._current_img, max_num=1)
        if not faces:
            return
        face = faces[0]
        if face.landmarks_106 is not None and face.landmarks_106.shape[0] == 106:
            lm = face.landmarks_106
            landmarks_106 = {}
            for i in range(106):
                landmarks_106[i] = QPointF(float(lm[i, 0]), float(lm[i, 1]))
            self._canvas.set_landmarks_106(landmarks_106)
        if face.kps_5 is not None and face.kps_5.shape[0] == 5:
            kps5_dict = {}
            kps5_dict[0] = QPointF(float(face.kps_5[1, 0]), float(face.kps_5[1, 1]))
            kps5_dict[1] = QPointF(float(face.kps_5[0, 0]), float(face.kps_5[0, 1]))
            kps5_dict[2] = QPointF(float(face.kps_5[2, 0]), float(face.kps_5[2, 1]))
            kps5_dict[3] = QPointF(float(face.kps_5[4, 0]), float(face.kps_5[4, 1]))
            kps5_dict[4] = QPointF(float(face.kps_5[3, 0]), float(face.kps_5[3, 1]))
            self._canvas.set_kps_5(kps5_dict)
            self._canvas.set_show_kps5(self._kps5_toggle_btn.isChecked())
        if not self._face_rect_generated and face.bbox is not None:
            bx, by, bx2, by2 = face.bbox
            margin = max(bx2 - bx, by2 - by) * 0.3
            self._canvas.set_face_rect(QRectF(bx - margin, by - margin, bx2 - bx + 2 * margin, by2 - by + 2 * margin))
            self._face_rect_generated = True
        if face.bbox is not None:
            bx, by, bx2, by2 = face.bbox
            self._canvas.set_bbox_rect(QRectF(bx, by, bx2 - bx, by2 - by))

    def _auto_annotate(self):
        if self._current_img is None:
            QMessageBox.warning(self, "提示", "请先加载图片")
            return
        try:
            from faceswap.core.insightface_adapter import InsightFaceAdapter
            adapter = InsightFaceAdapter()
            faces = adapter.detect_faces(self._current_img, max_num=1)
            if not faces:
                QMessageBox.information(self, "自动标注失败", "未检测到人脸，请手动标注。")
                return
            self._auto_annotate_silent()
            if not self._validate_auto_annotate_result():
                return
            self._canvas.set_show_106(True)
            self._canvas.set_visible_group(None)
            self._face_rect_generated = True
            self._kps5_saved = True
            self._annotation_step = AnnotationStep.STEP_COMPLETE
            self._status.setText("自动标注完成，可拖动调整各点")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"自动标注失败: {e}")

    def _validate_auto_annotate_result(self) -> bool:
        missing = []
        if not self._face_rect_generated or self._canvas.get_face_rect() is None:
            missing.append("人脸区域框")
        kps5 = self._canvas.get_kps_5()
        if len(kps5) < 5:
            missing.append("5点检测器")
        lm106 = self._canvas.get_landmarks_106()
        if len(lm106) < 106:
            missing.append(f"106点 (当前: {len(lm106)})")
        if missing:
            QMessageBox.warning(self, "自动标注失败",
                                f"自动标注结果缺少: {', '.join(missing)}\n\n"
                                "请采用正常标注流程：\n"
                                "1. 点击\"生成人脸区域框\"\n"
                                "2. 添加5点检测器并保存\n"
                                "3. 添加106点并保存")
            return False
        return True

    def _apply_to_next(self):
        if self._current_to_annotate_idx < 0 or self._current_to_annotate_idx >= len(self._to_annotate_images) - 1:
            QMessageBox.warning(self, "提示", "没有下一帧")
            return
        lm106 = self._canvas.get_landmarks_106()
        face_rect = self._canvas.get_face_rect()
        if len(lm106) < 5:
            QMessageBox.warning(self, "提示", "请先标注特征点")
            return
        saved_106 = dict(lm106)
        saved_rect = QRectF(face_rect) if face_rect else None
        saved_rect_generated = self._face_rect_generated
        saved_kps5 = dict(self._canvas.get_kps_5())
        saved_bbox = QRectF(self._canvas.get_bbox_rect()) if self._canvas.get_bbox_rect() is not None else None
        saved_kps5_saved = self._kps5_saved
        self._next_image()
        if saved_106:
            self._canvas.set_landmarks_106(saved_106)
        if saved_rect:
            self._canvas.set_face_rect(saved_rect)
            self._face_rect_generated = saved_rect_generated
        if saved_kps5:
            self._canvas.set_kps_5(saved_kps5)
            self._kps5_saved = saved_kps5_saved
        if saved_bbox:
            self._canvas.set_bbox_rect(saved_bbox)

    def _delete_current_group(self):
        reply = QMessageBox.question(self, "确认删除", "是否要删除？删除的话则106点、5点、人脸区域框和检测框将被全部删除。",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._canvas.clear_landmarks()
            self._canvas._kps_5.clear()
            self._canvas.set_show_kps5(False)
            self._canvas.set_bbox_rect(None)
            self._canvas.set_face_rect(None)
            self._kps5_saved = False
            self._face_rect_generated = False
            self._status.setText("106点、5点、人脸区域框和检测框已全部删除，请重新添加")

    def _save_106(self):
        lm106 = self._canvas.get_landmarks_106()

        if len(lm106) == 0:
            if self._current_img_path is not None:
                self._delete_face_files()
            return

        missing = []
        if not self._face_rect_generated or self._canvas.get_face_rect() is None:
            missing.append("人脸区域框")
        if not self._kps5_saved or len(self._canvas.get_kps_5()) < 5:
            missing.append("5点检测器")
        if len(lm106) < 106:
            missing.append(f"106点不完整 (当前: {len(lm106)})")
        if missing:
            QMessageBox.warning(self, "提示", f"缺少: {', '.join(missing)}")
            return

        self._save_full_annotation()

    def _delete_face_files(self):
        from faceswap.core.metadata_manager import MetadataManager
        aligned_dir = DATA_SRC_ALIGNED_DIR if self._is_src else DATA_DST_ALIGNED_DIR
        stem = self._current_img_path.stem
        deleted = []
        for ext in [".jpg", ".jpeg", ".png"]:
            face_path = aligned_dir / (stem + ext)
            if face_path.exists():
                face_path.unlink()
                deleted.append(face_path.name)
            json_path = face_path.with_suffix(".json")
            if json_path.exists():
                json_path.unlink()
                deleted.append(json_path.name)
        debug_dir = aligned_dir.parent / (aligned_dir.name + "_debug")
        for ext in [".jpg", ".jpeg", ".png"]:
            debug_path = debug_dir / (stem + ext)
            if debug_path.exists():
                debug_path.unlink()
                deleted.append(debug_path.name)
        if deleted:
            self._status.setText(f"已删除: {', '.join(deleted)}")
            _logger.info(f"Deleted face files for {stem}: {deleted}")
        else:
            self._status.setText("无对应文件可删除")

    def _save_full_annotation(self):
        lm106 = self._canvas.get_landmarks_106()
        if len(lm106) < 106:
            return

        ft_map = {"whole_face": FaceType.WHOLE_FACE, "head": FaceType.HEAD}
        face_type = ft_map.get(self._ft_combo.currentText(), FaceType.WHOLE_FACE)
        output_size = int(self._size_combo.currentText())

        lm_106 = np.zeros((106, 2), dtype=np.float32)
        for idx, pt in lm106.items():
            lm_106[idx] = [pt.x(), pt.y()]

        canvas_kps5 = self._canvas.get_kps_5()
        if canvas_kps5 and len(canvas_kps5) == 5:
            kps_5 = _canvas_kps5_to_insightface(canvas_kps5)
        else:
            QMessageBox.warning(self, "缺少5点标注", "请先标注5个关键点后再保存")
            return

        face_rect = self._canvas.get_face_rect()
        canvas_vis = self._canvas.get_lm106_visibility()
        lm106_vis = []
        for i in range(106):
            if i in lm106:
                pt = lm106[i]
                in_face = face_rect is None or face_rect.contains(pt)
                manual_vis = canvas_vis.get(i, True)
                lm106_vis.append(in_face and manual_vis)
            else:
                lm106_vis.append(True)
        kps5_vis = _compute_kps_5_visibility(canvas_kps5 if canvas_kps5 and len(canvas_kps5) == 5 else {}, face_rect)

        source_face_rect = None
        if face_rect is not None:
            source_face_rect = [face_rect.x(), face_rect.y(), face_rect.width(), face_rect.height()]

        source_rect = [0, 0, self._current_img.shape[1], self._current_img.shape[0]]
        bbox_rect = self._canvas.get_bbox_rect()
        if bbox_rect is not None:
            source_rect = [bbox_rect.x(), bbox_rect.y(), bbox_rect.x() + bbox_rect.width(), bbox_rect.y() + bbox_rect.height()]

        aligned_dir = DATA_SRC_ALIGNED_DIR if self._is_src else DATA_DST_ALIGNED_DIR
        to_annotate_dir = InsightFaceTrainingDataGenerator(WORKSPACE_DIR).to_annotate_dir
        skip_aligned = False
        try:
            if to_annotate_dir.exists() and str(Path(self._current_img_path).parent.resolve()).startswith(str(to_annotate_dir.resolve())):
                skip_aligned = True
        except Exception:
            pass
        existing_face_path = None
        debug_stem = self._current_img_path.stem
        if not skip_aligned and aligned_dir.exists():
            for aligned_file in sorted(aligned_dir.iterdir()):
                if aligned_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                if aligned_file.stem.startswith(debug_stem):
                    existing_face_path = aligned_file
                    break

        result = save_face_annotation(
            source_img=self._current_img,
            lm_106=lm_106,
            kps_5=kps_5,
            face_type=face_type,
            output_size=output_size,
            is_src=self._is_src,
            source_filename=self._current_img_path.name,
            source_rect=source_rect,
            source_face_rect=source_face_rect,
            landmarks_106_visibility=lm106_vis,
            kps_5_visibility=kps5_vis,
            existing_face_path=existing_face_path,
            skip_aligned=skip_aligned,
        )

        from faceswap.gui_app.gui_utils import _save_insightface_training_data
        _save_insightface_training_data(result.metadata, self._current_img_path, self._current_img_path.name)

        self._annotation_step = AnnotationStep.STEP_COMPLETE
        saved_name = result.face_path.name if result.face_path else self._current_img_path.name
        self._status.setText(f"手动标注完毕: {saved_name}")
        QMessageBox.information(self, "保存成功", f"手动标注完毕\n{saved_name}")
        _logger.info(f"Manual annotation complete: {saved_name}")

    def _update_progress(self):
        pass

    def load_image(self, img_path: Path):
        self._load_single_image(img_path)


class _FrameSelectDialog(QDialog):
    frame_selected = pyqtSignal(str)

    def __init__(self, image_list: list[Path], parent=None):
        super().__init__(parent)
        self._image_list = image_list
        self._cached_icons: list[Optional[QIcon]] = [None] * len(image_list)
        self.setWindowTitle("选择帧图片")
        self.setMinimumSize(900, 600)
        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowType.WindowMinMaxButtonsHint |
                            Qt.WindowType.Window)
        self._build_ui()
        self._start_loading()

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
        layout.addWidget(self._thumb_list)

        for i, img_path in enumerate(self._image_list):
            item = QListWidgetItem(img_path.stem)
            item.setData(Qt.ItemDataRole.UserRole, str(img_path))
            item.setSizeHint(QSize(126, 96))
            self._thumb_list.addItem(item)

    def _start_loading(self):
        self._thumb_loader = _ThumbLoader(self._image_list, preload_count=0)
        self._thumb_loader.thumb_loaded.connect(self._on_thumb_loaded)
        self._thumb_loader.start()

    def _on_thumb_loaded(self, idx: int, qimg: QImage, stem: str, path_str: str):
        icon = QIcon(QPixmap.fromImage(qimg))
        if idx < self._thumb_list.count():
            item = self._thumb_list.item(idx)
            item.setIcon(icon)
            item.setSizeHint(QSize(126, 96))
        if idx < len(self._cached_icons):
            self._cached_icons[idx] = icon

    def _on_item_clicked(self, item):
        path_str = item.data(Qt.ItemDataRole.UserRole)
        self.frame_selected.emit(path_str)
        self.accept()


class _HorizontalThumbDialog(QDialog):
    thumb_selected = pyqtSignal(int)

    def __init__(self, image_list: list[Path], current_idx: int, cached_icons: list[Optional[QIcon]], parent=None):
        super().__init__(parent)
        self._image_list = image_list
        self._current_idx = current_idx
        self._cached_icons = cached_icons
        self.setWindowTitle("横排缩略图浏览")
        self.setMinimumSize(800, 600)
        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowType.WindowMinMaxButtonsHint |
                            Qt.WindowType.Window)
        self._build_ui()

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
        layout.addWidget(self._thumb_list)

        for i, img_path in enumerate(self._image_list):
            item = QListWidgetItem(img_path.stem)
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setSizeHint(QSize(126, 96))
            if i < len(self._cached_icons) and self._cached_icons[i] is not None:
                item.setIcon(self._cached_icons[i])
            self._thumb_list.addItem(item)

        if 0 <= self._current_idx < self._thumb_list.count():
            self._thumb_list.setCurrentRow(self._current_idx)

    def _on_item_clicked(self, item):
        idx = item.data(Qt.ItemDataRole.UserRole)
        self.thumb_selected.emit(idx)
        self.accept()


class DebugPreviewDialog(QDialog):
    def __init__(self, is_src: bool, parent=None):
        super().__init__(parent)
        self._is_src = is_src
        self._current_img_path = None
        self._current_img = None
        self._current_source_img = None
        self._current_meta = None
        self._image_list: list[Path] = []
        self._cached_icons: list[Optional[QIcon]] = []
        self._current_image_idx = -1
        self._editing = False
        self._saving = False
        self.setWindowTitle("调试图预览 - " + ("源 (SRC)" if is_src else "目标 (DST)"))
        self.setMinimumSize(1300, 800)
        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowType.WindowMinMaxButtonsHint |
                            Qt.WindowType.Window)
        self._build_ui()
        self._start_loading_thumbnails()
        self.showMaximized()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_preview_label') and self._current_img is not None:
            self._update_preview()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        left_widget.setFixedWidth(200)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)

        left_label = QLabel("缩略图列表")
        left_label.setStyleSheet("font-weight: bold; padding: 4px;")
        left_layout.addWidget(left_label)

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

        self._image_name_label = QLabel("")
        self._image_name_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px; color: #0078D4;")
        right_layout.addWidget(self._image_name_label)

        self._stack = QStackedWidget()

        self._preview_label = QLabel("选择左侧缩略图查看大图")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(500, 400)
        self._stack.addWidget(self._preview_label)

        self._edit_canvas = ImageCanvas(show_point_numbers=False, right_click_pan=True)
        self._stack.addWidget(self._edit_canvas)

        right_layout.addWidget(self._stack, 1)

        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("◀ 上一帧")
        self._prev_btn.clicked.connect(self._prev_thumb)
        nav_row.addWidget(self._prev_btn)
        self._next_btn = QPushButton("下一帧 ▶")
        self._next_btn.clicked.connect(self._next_thumb)
        nav_row.addWidget(self._next_btn)
        nav_row.addStretch()

        self._edit_btn = QPushButton("编辑")
        self._edit_btn.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; font-weight: bold; padding: 8px 24px; }")
        self._edit_btn.setEnabled(False)
        self._edit_btn.clicked.connect(self._on_edit)
        nav_row.addWidget(self._edit_btn)

        self._save_btn = QPushButton("保存")
        self._save_btn.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; font-weight: bold; padding: 8px 24px; }")
        self._save_btn.setVisible(False)
        self._save_btn.clicked.connect(self._on_save_edit)
        nav_row.addWidget(self._save_btn)

        self._delete_btn = QPushButton("删除")
        self._delete_btn.setStyleSheet(
            "QPushButton { background-color: #C42B1C; color: white; font-weight: bold; padding: 8px 24px; }")
        self._delete_btn.setVisible(False)
        self._delete_btn.clicked.connect(self._on_delete_edit)
        nav_row.addWidget(self._delete_btn)

        self._cancel_btn = QPushButton("取消编辑")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel_edit)
        nav_row.addWidget(self._cancel_btn)

        self._reset_view_btn = QPushButton("重置位置")
        self._reset_view_btn.setStyleSheet(
            "QPushButton { background-color: #555; color: white; font-weight: bold; padding: 8px 16px; }")
        self._reset_view_btn.setVisible(False)
        self._reset_view_btn.clicked.connect(self._on_reset_view)
        nav_row.addWidget(self._reset_view_btn)

        self._manual_btn = QPushButton("手工标注")
        self._manual_btn.setStyleSheet(
            "QPushButton { background-color: #5B2D8E; color: white; font-weight: bold; padding: 8px 24px; }")
        self._manual_btn.setEnabled(False)
        self._manual_btn.clicked.connect(self._on_manual_annotate)
        nav_row.addWidget(self._manual_btn)

        nav_row.addSpacing(10)

        self._toggle_kps5_btn = QPushButton("隐藏5点")
        self._toggle_kps5_btn.setStyleSheet(
            "QPushButton { background-color: #555; color: white; font-weight: bold; padding: 6px 12px; }")
        self._toggle_kps5_btn.setVisible(False)
        self._toggle_kps5_btn.setCheckable(True)
        self._toggle_kps5_btn.setChecked(False)
        self._toggle_kps5_btn.clicked.connect(self._on_toggle_kps5)
        nav_row.addWidget(self._toggle_kps5_btn)

        self._toggle_106_btn = QPushButton("隐藏106点")
        self._toggle_106_btn.setStyleSheet(
            "QPushButton { background-color: #555; color: white; font-weight: bold; padding: 6px 12px; }")
        self._toggle_106_btn.setVisible(False)
        self._toggle_106_btn.setCheckable(True)
        self._toggle_106_btn.setChecked(False)
        self._toggle_106_btn.clicked.connect(self._on_toggle_106)
        nav_row.addWidget(self._toggle_106_btn)

        self._hthumb_btn = QPushButton("横排缩略图")
        self._hthumb_btn.setStyleSheet(
            "QPushButton { background-color: #555; color: white; font-weight: bold; padding: 8px 16px; }")
        self._hthumb_btn.clicked.connect(self._on_horizontal_thumbs)
        nav_row.addWidget(self._hthumb_btn)

        right_layout.addLayout(nav_row)

        self._help_label = QLabel(
            "操作说明：点击「编辑」进入编辑模式，滚轮缩放图片，右键按住移动图片，左键拖动特征点调整位置，"
            "修改后点击「保存」更新元数据及调试图，点击「删除」删除头像及元数据。"
        )
        self._help_label.setStyleSheet("color: #888; font-size: 11px; padding: 4px; border-top: 1px solid #444;")
        self._help_label.setWordWrap(True)
        right_layout.addWidget(self._help_label)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([200, 800])

        layout.addWidget(splitter)

    def _start_loading_thumbnails(self):
        aligned_dir = DATA_SRC_ALIGNED_DIR if self._is_src else DATA_DST_ALIGNED_DIR
        debug_dir = aligned_dir.parent / (aligned_dir.name + "_debug")

        if not debug_dir.exists():
            QMessageBox.information(self, "提示", f"调试图目录不存在: {debug_dir}")
            return

        images = sorted(FileManager.find_images(debug_dir), key=lambda p: p.name)
        if not images:
            QMessageBox.information(self, "提示", "调试图目录为空")
            return

        self._image_list = images
        self._cached_icons = [None] * len(images)
        for img_path in images:
            item = QListWidgetItem(img_path.stem)
            item.setData(Qt.ItemDataRole.UserRole, str(img_path))
            item.setSizeHint(QSize(126, 96))
            self._thumb_list.addItem(item)

        preload = min(5, len(images))
        for i in range(preload):
            img = cv2.imread(str(images[i]))
            if img is not None:
                thumb = cv2.resize(img, (120, 80))
                rgb = bgr_to_rgb(thumb)
                qimg = QImage(rgb.data, 120, 80, 3 * 120, QImage.Format.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(qimg)
                icon = QIcon(pixmap)
                self._thumb_list.item(i).setIcon(icon)
                self._cached_icons[i] = icon

        if preload > 0:
            self._thumb_list.setCurrentRow(0)

        if len(images) > preload:
            self._thumb_loader = _ThumbLoader(images, preload_count=preload)
            self._thumb_loader.thumb_loaded.connect(self._on_thumb_loaded)
            self._thumb_loader.loading_finished.connect(self._on_thumbs_loaded)
            self._thumb_loader.start()

        if preload > 0:
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
        if self._editing or self._saving:
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
        self._current_source_img = None
        self._current_image_idx = self._thumb_list.row(current)
        self._image_name_label.setText(img_path.name)

        from faceswap.core.metadata_manager import MetadataManager
        aligned_dir = DATA_SRC_ALIGNED_DIR if self._is_src else DATA_DST_ALIGNED_DIR
        self._current_meta = None
        if aligned_dir.exists():
            debug_stem = img_path.stem
            for aligned_file in sorted(aligned_dir.iterdir()):
                if aligned_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                if aligned_file.stem.startswith(debug_stem):
                    self._current_meta = MetadataManager.load(aligned_file)
                    if self._current_meta is not None:
                        break

        self._update_preview()

    def _update_preview(self):
        if self._current_img is None:
            return
        img = self._current_img
        h, w = img.shape[:2]
        max_w = self._preview_label.width() - 20
        max_h = self._preview_label.height() - 20
        if max_w <= 0:
            max_w = 600
        if max_h <= 0:
            max_h = 500
        scale = min(max_w / w, max_h / h, 1.0)
        display_w = max(1, int(w * scale))
        display_h = max(1, int(h * scale))
        display = cv2.resize(img, (display_w, display_h), interpolation=cv2.INTER_LANCZOS4)
        rgb = bgr_to_rgb(display)
        qimg = QImage(rgb.data, display_w, display_h, 3 * display_w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        self._preview_label.setPixmap(pixmap)
        self._edit_btn.setEnabled(True)
        self._manual_btn.setEnabled(True)

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

    def _on_edit(self):
        if self._current_img_path is None:
            return
        if self._current_meta is None:
            QMessageBox.information(self, "提示", "无元数据，请使用\"手工标注\"按钮")
            return

        frames_dir = DATA_SRC_DIR if self._is_src else DATA_DST_DIR
        source_name = self._current_meta.source_filename
        source_path = frames_dir / source_name

        if not source_path.exists():
            for ext in [".png", ".jpg", ".jpeg"]:
                alt = frames_dir / (source_path.stem + ext)
                if alt.exists():
                    source_path = alt
                    break

        if not source_path.exists():
            QMessageBox.warning(self, "提示", f"找不到源图片: {source_name}")
            return

        source_img = cv2.imread(str(source_path))
        if source_img is None:
            QMessageBox.warning(self, "提示", f"无法读取源图片: {source_name}")
            return

        self._current_source_img = source_img
        self._edit_canvas.set_image(source_img)
        self._edit_canvas.clear_landmarks()
        self._edit_canvas.set_face_rect(None)
        self._edit_canvas.set_show_106(True)
        self._edit_canvas.set_visible_group(None)

        src_lm = self._current_meta.source_landmarks_106
        if src_lm is not None and src_lm.shape[0] == 106:
            landmarks_106 = {}
            for i in range(106):
                landmarks_106[i] = QPointF(float(src_lm[i, 0]), float(src_lm[i, 1]))
            self._edit_canvas.set_landmarks_106(landmarks_106)

        if self._current_meta.source_face_rect is not None:
            fr = self._current_meta.source_face_rect
            self._edit_canvas.set_face_rect(QRectF(fr[0], fr[1], fr[2], fr[3]))

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

        self._stack.setCurrentIndex(1)
        self._editing = True
        self._thumb_list.setEnabled(False)
        self._prev_btn.setVisible(False)
        self._next_btn.setVisible(False)
        self._hthumb_btn.setVisible(False)
        self._edit_btn.setVisible(False)
        self._manual_btn.setVisible(False)
        self._save_btn.setVisible(True)
        self._delete_btn.setVisible(True)
        self._cancel_btn.setVisible(True)
        self._reset_view_btn.setVisible(True)
        self._toggle_kps5_btn.setVisible(True)
        self._toggle_kps5_btn.setChecked(False)
        self._toggle_kps5_btn.setText("隐藏5点")
        self._toggle_106_btn.setVisible(True)
        self._toggle_106_btn.setChecked(False)
        self._toggle_106_btn.setText("隐藏106点")

    def _on_save_edit(self):
        lm106 = self._edit_canvas.get_landmarks_106()
        if len(lm106) < 5:
            return

        self._saving = True
        try:
            aligned_dir = DATA_SRC_ALIGNED_DIR if self._is_src else DATA_DST_ALIGNED_DIR
            debug_stem = self._current_img_path.stem

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

            face_rect = self._edit_canvas.get_face_rect()
            canvas_vis = self._edit_canvas.get_lm106_visibility()
            lm106_vis = []
            for i in range(106):
                if i in lm106:
                    pt = lm106[i]
                    in_face = face_rect is None or face_rect.contains(pt)
                    manual_vis = canvas_vis.get(i, True)
                    lm106_vis.append(in_face and manual_vis)
                else:
                    lm106_vis.append(True)
            kps5_vis = _compute_kps_5_visibility(canvas_kps5 if len(canvas_kps5) == 5 else {}, face_rect)

            source_rect = self._current_meta.source_rect if self._current_meta and self._current_meta.source_rect else [0, 0, self._current_source_img.shape[1], self._current_source_img.shape[0]]

            source_face_rect = None
            if face_rect is not None:
                source_face_rect = [face_rect.x(), face_rect.y(), face_rect.width(), face_rect.height()]

            existing_face_path = None
            if aligned_dir.exists():
                for aligned_file in sorted(aligned_dir.iterdir()):
                    if aligned_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                        continue
                    if aligned_file.stem.startswith(debug_stem):
                        existing_face_path = aligned_file
                        break

            result = save_face_annotation(
                source_img=self._current_source_img,
                lm_106=lm_106,
                kps_5=kps_5,
                face_type=face_type,
                output_size=output_size,
                is_src=self._is_src,
                source_filename=self._current_meta.source_filename if self._current_meta else "",
                source_rect=source_rect,
                source_face_rect=source_face_rect,
                landmarks_106_visibility=lm106_vis,
                kps_5_visibility=kps5_vis,
                existing_face_path=existing_face_path,
            )

            self._current_meta = result.metadata
            self._thumb_list.blockSignals(True)
            self._refresh_current_thumbnail()
            self._thumb_list.blockSignals(False)
            _logger.info(f"Edit saved: {result.face_path.name if result.face_path else debug_stem}")
            QMessageBox.information(self, "保存成功", f"编辑保存完毕\n{result.face_path.name if result.face_path else debug_stem}")
            self._exit_edit_mode()
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
        debug_stem = self._current_img_path.stem
        deleted = []

        for aligned_file in list(aligned_dir.iterdir()):
            if aligned_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if aligned_file.stem.startswith(debug_stem):
                aligned_file.unlink()
                deleted.append(aligned_file.name)
                json_path = aligned_file.with_suffix(".json")
                if json_path.exists():
                    json_path.unlink()
                    deleted.append(json_path.name)

        source_img = self._current_source_img
        if source_img is not None:
            low_q_path = debug_dir / (debug_stem + ".jpg")
            from faceswap.shared.file_manager import imwrite_auto
            imwrite_auto(low_q_path, source_img, jpg_quality=30)
            self._current_img = source_img

        self._current_meta = None
        self._exit_edit_mode()
        self._refresh_current_thumbnail()

        if deleted:
            _logger.info(f"Deleted face files for {debug_stem}: {deleted}")

    def _on_cancel_edit(self):
        self._exit_edit_mode()

    def _on_reset_view(self):
        self._edit_canvas.reset_view()

    def _on_toggle_kps5(self):
        if self._toggle_kps5_btn.isChecked():
            self._edit_canvas.set_show_kps5(False)
            self._toggle_kps5_btn.setText("显示5点")
        else:
            self._edit_canvas.set_show_kps5(True)
            self._toggle_kps5_btn.setText("隐藏5点")

    def _on_toggle_106(self):
        if self._toggle_106_btn.isChecked():
            self._edit_canvas.set_show_106(False)
            self._toggle_106_btn.setText("显示106点")
        else:
            self._edit_canvas.set_show_106(True)
            self._toggle_106_btn.setText("隐藏106点")

    def _exit_edit_mode(self):
        self._stack.setCurrentIndex(0)
        self._editing = False
        self._thumb_list.setEnabled(True)
        self._prev_btn.setVisible(True)
        self._next_btn.setVisible(True)
        self._hthumb_btn.setVisible(True)
        self._edit_btn.setVisible(True)
        self._manual_btn.setVisible(True)
        self._save_btn.setVisible(False)
        self._delete_btn.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._reset_view_btn.setVisible(False)
        self._toggle_kps5_btn.setVisible(False)
        self._toggle_106_btn.setVisible(False)
        self._update_preview()

    def _refresh_current_thumbnail(self):
        if self._current_image_idx < 0 or self._current_image_idx >= len(self._image_list):
            return
        img_path = self._image_list[self._current_image_idx]
        img = cv2.imread(str(img_path))
        if img is None:
            return
        thumb = cv2.resize(img, (120, 80))
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
        frames_dir = DATA_SRC_DIR if self._is_src else DATA_DST_DIR

        source_path = None
        if self._current_meta is not None and self._current_meta.source_filename:
            source_name = self._current_meta.source_filename
            candidate = frames_dir / source_name
            if candidate.exists():
                source_path = candidate
            else:
                for ext in [".png", ".jpg", ".jpeg"]:
                    alt = frames_dir / (candidate.stem + ext)
                    if alt.exists():
                        source_path = alt
                        break

        if source_path is None:
            stem_parts = self._current_img_path.stem.rsplit("_", 1)
            base_stem = stem_parts[0] if len(stem_parts) == 2 else self._current_img_path.stem
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = frames_dir / (base_stem + ext)
                if candidate.exists():
                    source_path = candidate
                    break

        if source_path is None:
            QMessageBox.warning(self, "提示", "找不到源图片")
            return

        dlg = ManualAnnotatorDialog(self, is_src=self._is_src, img_path=source_path)
        dlg.exec()
