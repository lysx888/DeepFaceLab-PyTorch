import enum
import math
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF, QSize, pyqtSignal, QThread, QObject
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QImage, QPixmap, QIcon,
    QTransform, QMouseEvent, QWheelEvent, QKeyEvent, QPolygonF,
    QFont, QCursor, QFontMetrics,
)
from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QSizePolicy, QApplication, QFrame,
)

from DeepFaceLab.core.metadata_manager import MetadataManager
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("xseg_editor_dialog")

_ICONS_DIR = Path(__file__).resolve().parent.parent / "icons"


class _OpMode(enum.IntEnum):
    NONE = 0
    DRAW_PTS = 1
    EDIT_PTS = 2
    VIEW_BAKED = 3
    VIEW_XSEG_MASK = 4


class _PolyType(enum.IntEnum):
    INCLUDE = 1
    EXCLUDE = 0


class _SegIEPoly:
    def __init__(self, poly_type: _PolyType = _PolyType.INCLUDE):
        self.type = poly_type
        self.pts: list[QPointF] = []
        self._history: list[list[QPointF]] = []
        self._redo_stack: list[list[QPointF]] = []

    def add_pt(self, pt: QPointF):
        self._history.append(list(self.pts))
        self._redo_stack.clear()
        self.pts.append(QPointF(pt))

    def remove_pt(self, idx: int):
        if 0 <= idx < len(self.pts):
            self._history.append(list(self.pts))
            self._redo_stack.clear()
            self.pts.pop(idx)

    def insert_pt(self, idx: int, pt: QPointF):
        self._history.append(list(self.pts))
        self._redo_stack.clear()
        self.pts.insert(idx, QPointF(pt))

    def undo(self):
        if self._history:
            self._redo_stack.append(list(self.pts))
            self.pts = self._history.pop()

    def redo(self):
        if self._redo_stack:
            self._history.append(list(self.pts))
            self.pts = self._redo_stack.pop()

    def dump(self) -> dict:
        return {"type": int(self.type), "pts": [[p.x(), p.y()] for p in self.pts]}

    @staticmethod
    def load(data: dict) -> "_SegIEPoly":
        poly = _SegIEPoly(_PolyType(data.get("type", 1)))
        for pt in data.get("pts", []):
            poly.pts.append(QPointF(pt[0], pt[1]))
        return poly


class _XSegCanvas(QWidget):
    image_changed = pyqtSignal()
    mode_changed = pyqtSignal()

    POLY_COLORS = {
        _PolyType.INCLUDE: QColor(0, 192, 0),
        _PolyType.EXCLUDE: QColor(192, 0, 0),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Optional[np.ndarray] = None
        self._qimage: Optional[QImage] = None
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._polys: list[_SegIEPoly] = []
        self._cur_poly_idx: int = -1
        self._op_mode: _OpMode = _OpMode.NONE
        self._poly_type: _PolyType = _PolyType.INCLUDE
        self._dragging = False
        self._drag_pt_idx = -1
        self._panning = False
        self._pan_start = QPointF()
        self._pan_offset_start = QPointF()
        self._hover_poly_idx = -1
        self._mouse_pos = QPointF()
        self._img_path: Optional[Path] = None
        self._baked_mask: Optional[np.ndarray] = None
        self._need_fit = False
        self._user_zoomed = False
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def load_image(self, img_path: Path):
        self._save_current_polys()
        self._img_path = img_path
        img = cv2.imread(str(img_path))
        if img is None:
            self._image = None
            self._qimage = None
            self._polys = []
            self._cur_poly_idx = -1
            self._op_mode = _OpMode.NONE
            self.update()
            return
        self._image = img
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self._qimage = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self._load_polys(img_path)
        self._baked_mask = None
        self._zoom = min(self.width() / w, self.height() / h) * 0.9
        self._offset = QPointF(0, 0)
        self._op_mode = _OpMode.NONE
        self._cur_poly_idx = -1
        self._need_fit = True
        self._user_zoomed = False
        self._notify_mode_changed()
        self.update()
        self.image_changed.emit()

    def fit_image(self):
        if self._qimage is None:
            return
        self._zoom = min(self.width() / self._qimage.width(), self.height() / self._qimage.height()) * 0.9
        self._offset = QPointF(0, 0)
        self.update()

    def _load_polys(self, img_path: Path):
        self._polys = []
        self._cur_poly_idx = -1
        meta = MetadataManager.load(img_path)
        if meta is None or meta.seg_ie_polys is None:
            return
        for poly_data in meta.seg_ie_polys:
            poly = _SegIEPoly.load(poly_data)
            self._polys.append(poly)

    def _save_current_polys(self):
        if self._img_path is None:
            return
        meta = MetadataManager.load(self._img_path)
        if meta is None:
            return
        if self._polys:
            meta.seg_ie_polys = [p.dump() for p in self._polys]
        else:
            meta.seg_ie_polys = None
        MetadataManager.save(self._img_path, meta)

    def get_annotated_count_text(self) -> str:
        if self._img_path is None:
            return ""
        d = self._img_path.parent
        count = 0
        for p in FileManager.find_images(d):
            m = MetadataManager.load(p)
            if m is not None and m.seg_ie_polys is not None:
                count += 1
        return f"已标记: {count}"

    def get_current_has_polys(self) -> bool:
        return len(self._polys) > 0

    def set_poly_type(self, poly_type: _PolyType):
        self._poly_type = poly_type

    def undo_pt(self):
        if 0 <= self._cur_poly_idx < len(self._polys):
            self._polys[self._cur_poly_idx].undo()

    def redo_pt(self):
        if 0 <= self._cur_poly_idx < len(self._polys):
            self._polys[self._cur_poly_idx].redo()

    def delete_poly(self):
        if 0 <= self._cur_poly_idx < len(self._polys):
            self._polys.pop(self._cur_poly_idx)
            self._cur_poly_idx = -1
            self._op_mode = _OpMode.NONE
            self._notify_mode_changed()

    def clear_all_polys(self):
        self._polys.clear()
        self._cur_poly_idx = -1
        self._op_mode = _OpMode.NONE
        self._notify_mode_changed()

    def toggle_view_baked(self):
        if self._op_mode == _OpMode.VIEW_BAKED:
            self._op_mode = _OpMode.NONE
            self._baked_mask = None
        else:
            self._op_mode = _OpMode.VIEW_BAKED
            self._bake_mask()
        self.update()

    def toggle_view_xseg(self):
        if self._op_mode == _OpMode.VIEW_XSEG_MASK:
            self._op_mode = _OpMode.NONE
        else:
            self._op_mode = _OpMode.VIEW_XSEG_MASK
        self.update()

    def auto_mask_from_landmarks(self, landmarks_106: np.ndarray):
        if self._image is None:
            return
        if landmarks_106 is None or len(landmarks_106) < 3:
            return
        from DeepFaceLab.core.auto_mask_generator import AutoMaskGenerator
        generator = AutoMaskGenerator.get_instance()
        result = generator.generate_sam2(self._image, landmarks_106)
        self._apply_auto_mask_result(result)

    def auto_mask_face_parsing(self):
        if self._image is None:
            return
        from DeepFaceLab.core.auto_mask_generator import AutoMaskGenerator
        generator = AutoMaskGenerator.get_instance()
        result = generator.generate_face_parsing(self._image)
        self._apply_auto_mask_result(result)

    def auto_mask_simple(self, landmarks_106: np.ndarray):
        if self._image is None:
            return
        if landmarks_106 is None or len(landmarks_106) < 3:
            return
        from DeepFaceLab.core.auto_mask_generator import AutoMaskGenerator
        generator = AutoMaskGenerator.get_instance()
        result = generator.generate_simple(landmarks_106)
        self._apply_auto_mask_result(result)

    def _apply_auto_mask_result(self, result):
        from DeepFaceLab.core.auto_mask_generator import AutoMaskResult
        for poly_pts in result.include_polys:
            poly = _SegIEPoly(_PolyType.INCLUDE)
            for x, y in poly_pts:
                poly.pts.append(QPointF(x, y))
            self._polys.append(poly)
        for poly_pts in result.exclude_polys:
            poly = _SegIEPoly(_PolyType.EXCLUDE)
            for x, y in poly_pts:
                poly.pts.append(QPointF(x, y))
            self._polys.append(poly)
        if self._polys:
            self._cur_poly_idx = len(self._polys) - 1
            self._op_mode = _OpMode.EDIT_PTS
        self._notify_mode_changed()
        self.update()

    def _bake_mask(self):
        if self._image is None:
            return
        h, w = self._image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in self._polys:
            if len(poly.pts) < 3:
                continue
            pts = np.array([[int(p.x()), int(p.y())] for p in poly.pts], dtype=np.int32)
            if poly.type == _PolyType.INCLUDE:
                cv2.fillPoly(mask, [pts], 255)
            else:
                cv2.fillPoly(mask, [pts], 0)
        self._baked_mask = mask

    def _render_saved_polys_mask(self) -> Optional[np.ndarray]:
        if self._img_path is None or self._image is None:
            return None
        meta = MetadataManager.load(self._img_path)
        if meta is None or meta.seg_ie_polys is None:
            return None
        h, w = self._image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly_data in meta.seg_ie_polys:
            pts_data = poly_data.get("pts", [])
            poly_type = poly_data.get("type", 1)
            if len(pts_data) < 3:
                continue
            pts = np.array(pts_data, dtype=np.int32)
            if poly_type == int(_PolyType.INCLUDE):
                cv2.fillPoly(mask, [pts], 255)
            else:
                cv2.fillPoly(mask, [pts], 0)
        return mask

    def _notify_mode_changed(self):
        self.mode_changed.emit()

    def _img_to_screen(self, pt: QPointF) -> QPointF:
        cx = self.width() / 2 + self._offset.x()
        cy = self.height() / 2 + self._offset.y()
        return QPointF(pt.x() * self._zoom + cx - self._qimage.width() * self._zoom / 2,
                       pt.y() * self._zoom + cy - self._qimage.height() * self._zoom / 2)

    def _screen_to_img(self, pt: QPointF) -> QPointF:
        cx = self.width() / 2 + self._offset.x()
        cy = self.height() / 2 + self._offset.y()
        ox = cx - self._qimage.width() * self._zoom / 2
        oy = cy - self._qimage.height() * self._zoom / 2
        return QPointF((pt.x() - ox) / self._zoom, (pt.y() - oy) / self._zoom)

    def _find_nearest_pt(self, img_pt: QPointF, threshold: float = 8.0) -> tuple[int, int]:
        best_dist = threshold / self._zoom
        best_poly = -1
        best_pt = -1
        for pi, poly in enumerate(self._polys):
            for vi, pt in enumerate(poly.pts):
                d = math.hypot(pt.x() - img_pt.x(), pt.y() - img_pt.y())
                if d < best_dist:
                    best_dist = d
                    best_poly = pi
                    best_pt = vi
        return best_poly, best_pt

    def _find_nearest_edge(self, img_pt: QPointF, threshold: float = 8.0) -> tuple[int, int]:
        best_dist = threshold / self._zoom
        best_poly = -1
        best_edge = -1
        for pi, poly in enumerate(self._polys):
            n = len(poly.pts)
            if n < 2:
                continue
            for ei in range(n):
                p1 = poly.pts[ei]
                p2 = poly.pts[(ei + 1) % n]
                d = self._point_to_segment_dist(img_pt, p1, p2)
                if d < best_dist:
                    best_dist = d
                    best_poly = pi
                    best_edge = ei
        return best_poly, best_edge

    @staticmethod
    def _point_to_segment_dist(p: QPointF, a: QPointF, b: QPointF) -> float:
        dx, dy = b.x() - a.x(), b.y() - a.y()
        if dx == 0 and dy == 0:
            return math.hypot(p.x() - a.x(), p.y() - a.y())
        t = max(0, min(1, ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / (dx * dx + dy * dy)))
        proj = QPointF(a.x() + t * dx, a.y() + t * dy)
        return math.hypot(p.x() - proj.x(), p.y() - proj.y())

    def _update_hover(self, img_pt: QPointF):
        self._hover_poly_idx = -1
        pi, vi = self._find_nearest_pt(img_pt)
        if pi >= 0:
            self._hover_poly_idx = pi
            return
        pi, _ = self._find_nearest_edge(img_pt)
        if pi >= 0:
            self._hover_poly_idx = pi

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(243, 243, 243))

        if self._qimage is None:
            painter.end()
            return

        base_image = self._qimage

        if self._op_mode == _OpMode.VIEW_BAKED and self._baked_mask is not None:
            h, w = self._baked_mask.shape
            rgb = cv2.cvtColor(self._baked_mask, cv2.COLOR_GRAY2RGB)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            base_image = qimg
        elif self._op_mode == _OpMode.VIEW_XSEG_MASK:
            saved_mask = self._render_saved_polys_mask()
            if saved_mask is not None:
                h, w = saved_mask.shape
                rgb = cv2.cvtColor(saved_mask, cv2.COLOR_GRAY2RGB)
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
                base_image = qimg

        origin = self._img_to_screen(QPointF(0, 0))
        painter.drawImage(QRectF(origin.x(), origin.y(),
                                  base_image.width() * self._zoom,
                                  base_image.height() * self._zoom), base_image)

        for pi, poly in enumerate(self._polys):
            if len(poly.pts) < 1:
                continue
            color = self.POLY_COLORS.get(poly.type, QColor(0, 192, 0))
            is_current = (pi == self._cur_poly_idx)
            is_hover = (pi == self._hover_poly_idx and not is_current)

            pen = QPen(color, 2)
            if poly.type == _PolyType.EXCLUDE:
                pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)

            if len(poly.pts) >= 2:
                screen_pts = [self._img_to_screen(p) for p in poly.pts]
                polygon = QPolygonF(screen_pts)
                if is_current and self._op_mode == _OpMode.DRAW_PTS:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawPolyline(polygon)
                else:
                    if is_hover:
                        fill_color = QColor(color)
                        fill_color.setAlpha(72)
                        painter.setBrush(QBrush(fill_color))
                    else:
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawPolygon(polygon)
            elif len(poly.pts) == 1:
                sp = self._img_to_screen(poly.pts[0])
                painter.drawEllipse(sp, 4, 4)

            if is_current and self._op_mode != _OpMode.NONE:
                for vi, pt in enumerate(poly.pts):
                    sp = self._img_to_screen(pt)
                    painter.setPen(QPen(QColor(255, 255, 255), 1))
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(sp, 4, 4)
                    painter.setBrush(Qt.BrushStyle.NoBrush)

            if is_current and self._op_mode == _OpMode.DRAW_PTS and len(poly.pts) > 0:
                last = self._img_to_screen(poly.pts[-1])
                mouse = self._img_to_screen(self._mouse_pos)
                painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
                painter.drawLine(last, mouse)
                if len(poly.pts) >= 3:
                    first = self._img_to_screen(poly.pts[0])
                    dist = math.hypot(mouse.x() - first.x(), mouse.y() - first.y())
                    if dist < 15:
                        painter.setPen(QPen(color, 2))
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.drawEllipse(first, 8, 8)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        if self._qimage is None:
            return
        pos = event.position()
        img_pt = self._screen_to_img(pos)

        if event.button() == Qt.MouseButton.RightButton:
            self._panning = True
            self._pan_start = pos
            self._pan_offset_start = QPointF(self._offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self._op_mode == _OpMode.NONE:
            pi, vi = self._find_nearest_pt(img_pt)
            if pi >= 0:
                self._cur_poly_idx = pi
                self._op_mode = _OpMode.EDIT_PTS
                self._drag_pt_idx = vi
                self._dragging = True
            else:
                ei_pi, _ = self._find_nearest_edge(img_pt)
                if ei_pi >= 0:
                    self._cur_poly_idx = ei_pi
                    self._op_mode = _OpMode.EDIT_PTS
                else:
                    poly = _SegIEPoly(self._poly_type)
                    poly.add_pt(img_pt)
                    self._polys.append(poly)
                    self._cur_poly_idx = len(self._polys) - 1
                    self._op_mode = _OpMode.DRAW_PTS

        elif self._op_mode == _OpMode.DRAW_PTS:
            poly = self._polys[self._cur_poly_idx]
            if len(poly.pts) >= 3:
                first = poly.pts[0]
                dist = math.hypot(img_pt.x() - first.x(), img_pt.y() - first.y())
                if dist < 8.0 / self._zoom:
                    self._op_mode = _OpMode.EDIT_PTS
                    self.update()
                    self._notify_mode_changed()
                    return
            poly.add_pt(img_pt)

        elif self._op_mode == _OpMode.EDIT_PTS:
            modifiers = event.modifiers()
            pi, vi = self._find_nearest_pt(img_pt)
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                if pi >= 0 and pi == self._cur_poly_idx:
                    self._polys[pi].remove_pt(vi)
                    if len(self._polys[pi].pts) < 3:
                        self._polys.pop(pi)
                        self._cur_poly_idx = -1
                        self._op_mode = _OpMode.NONE
                else:
                    _, ei = self._find_nearest_edge(img_pt)
                    if ei >= 0 and 0 <= self._cur_poly_idx < len(self._polys):
                        poly = self._polys[self._cur_poly_idx]
                        insert_idx = ei + 1
                        poly.insert_pt(insert_idx, img_pt)
            else:
                if pi >= 0 and pi == self._cur_poly_idx:
                    self._drag_pt_idx = vi
                    self._dragging = True
                else:
                    ei_pi, _ = self._find_nearest_edge(img_pt)
                    if ei_pi >= 0:
                        self._cur_poly_idx = ei_pi
                    else:
                        self._op_mode = _OpMode.NONE
                        self._cur_poly_idx = -1

        self.update()
        self._notify_mode_changed()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        self._mouse_pos = self._screen_to_img(pos)

        if self._panning:
            delta = pos - self._pan_start
            self._offset = QPointF(self._pan_offset_start.x() + delta.x(),
                                    self._pan_offset_start.y() + delta.y())
            self.update()
            return

        if self._dragging and self._drag_pt_idx >= 0:
            img_pt = self._screen_to_img(pos)
            if 0 <= self._cur_poly_idx < len(self._polys):
                poly = self._polys[self._cur_poly_idx]
                if 0 <= self._drag_pt_idx < len(poly.pts):
                    poly.pts[self._drag_pt_idx] = QPointF(img_pt)
            self.update()
            return

        self._update_hover(self._mouse_pos)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._drag_pt_idx = -1

    def wheelEvent(self, event: QWheelEvent):
        if self._qimage is None:
            return
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        new_zoom = self._zoom * factor
        new_zoom = max(0.1, min(20.0, new_zoom))
        pos = event.position()
        img_before = self._screen_to_img(pos)
        self._zoom = new_zoom
        self._user_zoomed = True
        img_after = self._screen_to_img(pos)
        self._offset += QPointF((img_after.x() - img_before.x()) * self._zoom,
                                 (img_after.y() - img_before.y()) * self._zoom)
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_Q:
            self.set_poly_type(_PolyType.INCLUDE)
        elif key == Qt.Key.Key_W:
            self.set_poly_type(_PolyType.EXCLUDE)
        elif key == Qt.Key.Key_4:
            self.toggle_view_baked()
        elif key == Qt.Key.Key_5:
            self.toggle_view_xseg()
        elif key == Qt.Key.Key_Delete:
            self.delete_poly()
        elif key == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.redo_pt()
            else:
                self.undo_pt()
        else:
            super().keyPressEvent(event)
        self.update()


_THUMB_W = 72
_THUMB_H = 48
_CENTER_THUMB_W = 90
_CENTER_THUMB_H = 60
_VISIBLE_COUNT = 9
_CENTER_POS = 4


def _load_icon(name: str) -> QIcon:
    p = _ICONS_DIR / name
    if p.exists():
        return QIcon(str(p))
    return QIcon()


_SLOT_W = _CENTER_THUMB_W + 4


class _ThumbLoadWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, paths):
        super().__init__()
        self._paths = paths
        self.images: dict[str, QImage] = {}

    def run(self):
        for p in self._paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            thumb = cv2.resize(img, (_CENTER_THUMB_W, _CENTER_THUMB_H))
            rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, _CENTER_THUMB_W, _CENTER_THUMB_H,
                          3 * _CENTER_THUMB_W, QImage.Format.Format_RGB888).copy()
            self.images[p.name] = qimg
        self.finished.emit()


class _BottomThumbBar(QWidget):
    image_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: list[Path] = []
        self._pixmaps: dict[str, QPixmap] = {}
        self._current_idx = -1
        self._load_thread: Optional[QThread] = None
        self._load_worker: Optional[_ThumbLoadWorker] = None
        self.setFixedHeight(80)
        self.setMinimumWidth(400)
        self.setMouseTracking(True)

    def set_paths(self, paths: list[Path]):
        self._paths = list(paths)
        self._pixmaps.clear()
        self._current_idx = -1
        self.update()
        self._start_load_thumbnails()

    def _start_load_thumbnails(self):
        if self._load_thread is not None and self._load_thread.isRunning():
            self._load_thread.quit()
            self._load_thread.wait()
        self._load_thread = QThread()
        self._load_worker = _ThumbLoadWorker(self._paths)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_thread.start()

    def _on_load_finished(self):
        if self._load_worker is not None:
            for name, qimg in self._load_worker.images.items():
                self._pixmaps[name] = QPixmap.fromImage(qimg)
            self._load_worker = None
        self.update()

    def remove_path(self, idx: int):
        if idx < 0 or idx >= len(self._paths):
            return
        removed_name = self._paths[idx].name
        self._paths.pop(idx)
        self._pixmaps.pop(removed_name, None)
        if self._current_idx >= len(self._paths):
            self._current_idx = len(self._paths) - 1
        self.update()

    def set_current(self, idx: int):
        self._current_idx = idx
        self.update()

    def _get_pixmap(self, idx: int) -> Optional[QPixmap]:
        if 0 <= idx < len(self._paths):
            return self._pixmaps.get(self._paths[idx].name)
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(225, 225, 225))

        if not self._paths:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "无图片")
            painter.end()
            return

        y_center = self.height() // 2
        bar_w = _VISIBLE_COUNT * _SLOT_W
        x_start = max(8, (self.width() - bar_w) // 2)

        for slot in range(_VISIBLE_COUNT):
            i = self._current_idx - _CENTER_POS + slot
            is_center_slot = (slot == _CENTER_POS)
            x = x_start + slot * _SLOT_W

            if is_center_slot:
                tw, th = _CENTER_THUMB_W, _CENTER_THUMB_H
            else:
                tw, th = _THUMB_W, _THUMB_H

            x_offset = (_SLOT_W - tw) // 2
            y = y_center - th // 2

            if 0 <= i < len(self._paths):
                pm = self._get_pixmap(i)
                is_current = (i == self._current_idx)

                if pm and not pm.isNull():
                    draw_pm = pm.scaled(tw, th, Qt.AspectRatioMode.IgnoreAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
                    if is_current:
                        painter.setPen(QPen(QColor(0, 120, 212), 2))
                    else:
                        painter.setPen(QPen(QColor(180, 180, 180), 1))
                    painter.drawRect(x + x_offset - 1, y - 1, tw + 2, th + 2)
                    painter.drawPixmap(x + x_offset, y, draw_pm)
                else:
                    painter.setPen(QColor(200, 200, 200))
                    painter.setBrush(QColor(240, 240, 240))
                    painter.drawRect(x + x_offset, y, tw, th)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(215, 215, 215))
                painter.drawRect(x + x_offset, y, tw, th)
                painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        idx = self._hit_test(event.position().toPoint())
        if idx >= 0:
            self.image_selected.emit(idx)

    def _hit_test(self, pos: QPoint) -> int:
        if not self._paths:
            return -1
        y_center = self.height() // 2
        bar_w = _VISIBLE_COUNT * _SLOT_W
        x_start = max(8, (self.width() - bar_w) // 2)

        for slot in range(_VISIBLE_COUNT):
            i = self._current_idx - _CENTER_POS + slot
            is_center_slot = (slot == _CENTER_POS)
            x = x_start + slot * _SLOT_W
            tw = _CENTER_THUMB_W if is_center_slot else _THUMB_W
            th = _CENTER_THUMB_H if is_center_slot else _THUMB_H
            x_offset = (_SLOT_W - tw) // 2
            y = y_center - th // 2

            if 0 <= i < len(self._paths):
                rect = QRectF(x + x_offset, y, tw, th)
                if rect.contains(QPointF(pos)):
                    return i
        return -1


_TOOL_BTN_STYLE = (
    "QPushButton {{ background-color: {bg}; color: {fg}; border: none; border-radius: 4px; "
    "padding: 6px 4px; font-weight: bold; min-width: 40px; }}"
    "QPushButton:hover {{ background-color: {hover}; }}"
    "QPushButton:disabled {{ background-color: #D6D6D6; color: #A0A0A0; }}"
    "QPushButton:checked {{ background-color: {checked}; }}"
)

_ICON_SIZE = QSize(36, 36)
_NAV_ICON_SIZE = QSize(48, 48)

_TOOLBAR_ICON_SIZE = QSize(48, 48)

_FRAME_STYLE = "QFrame { border: 1px solid #B0B0B0; border-radius: 3px; }"

_ICON_BTN_STYLE = (
    "QPushButton { background-color: #E8E8E8; border: none; border-radius: 4px; padding: 4px; }"
    "QPushButton:hover { background-color: #D0D0D0; }"
    "QPushButton:disabled { background-color: #D6D6D6; }"
    "QPushButton:checked { background-color: #C8C8C8; }"
)

_NAV_BTN_STYLE = (
    "QPushButton { background: #FFF; border: 1px solid #CCC; border-radius: 4px; padding: 4px; }"
    "QPushButton:hover { background: #E0E0E0; }"
)


class XSegEditorDialog(QDialog):
    def __init__(self, aligned_dir: Path, parent=None):
        super().__init__(parent)
        self._aligned_dir = Path(aligned_dir)
        self._trash_dir = self._aligned_dir.parent / (self._aligned_dir.name + "_trash")
        self._images: list[Path] = []
        self._current_idx = -1
        self._annotated_count = 0

        self.setWindowTitle(f"XSeg 遮罩编辑器 — {self._aligned_dir.name}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint)
        self._build_ui()
        self.showMaximized()
        self._load_images()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        main_split = QSplitter(Qt.Orientation.Horizontal)

        # ========== Left toolbar ==========
        left_toolbar = QWidget()
        left_toolbar.setStyleSheet("background: #F0F0F0;")
        lt_outer = QVBoxLayout(left_toolbar)
        lt_outer.setContentsMargins(4, 4, 4, 4)
        lt_outer.setSpacing(72)
        lt_outer.addSpacing(32)

        # Frame: Include / Exclude
        frame1 = QFrame()
        frame1.setStyleSheet(_FRAME_STYLE)
        f1_lay = QVBoxLayout(frame1)
        f1_lay.setContentsMargins(2, 2, 2, 2)
        f1_lay.setSpacing(2)

        self._btn_include = QPushButton(_load_icon("poly_type_include.png"), "")
        self._btn_include.setCheckable(True)
        self._btn_include.setChecked(True)
        self._btn_include.setIconSize(_TOOLBAR_ICON_SIZE)
        self._btn_include.setToolTip("包含选区 (Q)\n绘制包含区域的遮罩")
        self._btn_include.setStyleSheet(_ICON_BTN_STYLE)
        self._btn_include.clicked.connect(lambda: self._set_poly_type(_PolyType.INCLUDE))
        f1_lay.addWidget(self._btn_include)

        self._btn_exclude = QPushButton(_load_icon("poly_type_exclude.png"), "")
        self._btn_exclude.setCheckable(True)
        self._btn_exclude.setIconSize(_TOOLBAR_ICON_SIZE)
        self._btn_exclude.setToolTip("排除选区 (W)\n绘制排除区域的遮罩")
        self._btn_exclude.setStyleSheet(_ICON_BTN_STYLE)
        self._btn_exclude.clicked.connect(lambda: self._set_poly_type(_PolyType.EXCLUDE))
        f1_lay.addWidget(self._btn_exclude)

        lt_outer.addWidget(frame1)

        # Frame: Undo / Redo / Delete
        frame2 = QFrame()
        frame2.setStyleSheet(_FRAME_STYLE)
        f2_lay = QVBoxLayout(frame2)
        f2_lay.setContentsMargins(2, 2, 2, 2)
        f2_lay.setSpacing(2)

        undo_btn = QPushButton(_load_icon("undo_pt.png"), "")
        undo_btn.setIconSize(_TOOLBAR_ICON_SIZE)
        undo_btn.setToolTip("撤销 (Ctrl+Z)")
        undo_btn.setStyleSheet(_ICON_BTN_STYLE)
        undo_btn.clicked.connect(self._canvas_undo)
        f2_lay.addWidget(undo_btn)

        redo_btn = QPushButton(_load_icon("redo_pt.png"), "")
        redo_btn.setIconSize(_TOOLBAR_ICON_SIZE)
        redo_btn.setToolTip("重做 (Ctrl+Shift+Z)")
        redo_btn.setStyleSheet(_ICON_BTN_STYLE)
        redo_btn.clicked.connect(self._canvas_redo)
        f2_lay.addWidget(redo_btn)

        del_btn = QPushButton(_load_icon("delete_poly.png"), "")
        del_btn.setIconSize(_TOOLBAR_ICON_SIZE)
        del_btn.setToolTip("删除多边形 (Delete)")
        del_btn.setStyleSheet(_ICON_BTN_STYLE)
        del_btn.clicked.connect(self._canvas_delete)
        f2_lay.addWidget(del_btn)

        lt_outer.addWidget(frame2)

        # Frame: Clear
        frame3 = QFrame()
        frame3.setStyleSheet(_FRAME_STYLE)
        f3_lay = QVBoxLayout(frame3)
        f3_lay.setContentsMargins(2, 2, 2, 2)
        f3_lay.setSpacing(2)

        clear_btn = QPushButton(_load_icon("qingkong.png"), "")
        clear_btn.setIconSize(_TOOLBAR_ICON_SIZE)
        clear_btn.setToolTip("清空所有多边形")
        clear_btn.setStyleSheet(_ICON_BTN_STYLE)
        clear_btn.clicked.connect(self._canvas_clear_all)
        f3_lay.addWidget(clear_btn)

        lt_outer.addWidget(frame3)

        lt_outer.addStretch()
        main_split.addWidget(left_toolbar)

        # ========== Center ==========
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self._canvas = _XSegCanvas()
        self._canvas.mode_changed.connect(self._update_poly_type_buttons)
        center_layout.addWidget(self._canvas, 1)

        # Status row: filename + annotated count
        status_row = QWidget()
        status_layout = QHBoxLayout(status_row)
        status_layout.setContentsMargins(8, 2, 8, 2)
        status_layout.addStretch()
        self._filename_label = QLabel("")
        self._filename_label.setStyleSheet("color: #1B1B1B; font-weight: bold;")
        status_layout.addWidget(self._filename_label)
        self._annotated_label = QLabel("")
        self._annotated_label.setStyleSheet("color: #616161;")
        status_layout.addWidget(self._annotated_label)
        status_layout.addStretch()
        center_layout.addWidget(status_row)

        # Preview bar row: [pad] [prev] [thumbs] [next] | [delete]
        preview_bar = QWidget()
        preview_bar.setStyleSheet("background: #E8E8E8; border-top: 1px solid #D6D6D6;")
        pb_layout = QHBoxLayout(preview_bar)
        pb_layout.setContentsMargins(4, 4, 4, 4)
        pb_layout.setSpacing(4)

        # Left part: prev + thumbs + next (centered)
        nav_frame = QFrame()
        nav_frame.setStyleSheet(_FRAME_STYLE)
        nav_lay = QHBoxLayout(nav_frame)
        nav_lay.setContentsMargins(4, 4, 4, 4)
        nav_lay.setSpacing(4)

        prev_btn = QPushButton(_load_icon("left.png"), "")
        prev_btn.setIconSize(_NAV_ICON_SIZE)
        prev_btn.setToolTip("上一张 (A)")
        prev_btn.setStyleSheet(_NAV_BTN_STYLE)
        prev_btn.clicked.connect(self._prev_image)
        nav_lay.addWidget(prev_btn)

        self._thumb_bar = _BottomThumbBar()
        self._thumb_bar.image_selected.connect(self._on_thumb_selected)
        nav_lay.addWidget(self._thumb_bar, 1)

        next_btn = QPushButton(_load_icon("right.png"), "")
        next_btn.setIconSize(_NAV_ICON_SIZE)
        next_btn.setToolTip("下一张 (D)")
        next_btn.setStyleSheet(_NAV_BTN_STYLE)
        next_btn.clicked.connect(self._next_image)
        nav_lay.addWidget(next_btn)

        pb_layout.addWidget(nav_frame, 1)

        # Right part: delete
        del_frame = QFrame()
        del_frame.setStyleSheet(_FRAME_STYLE)
        del_lay = QVBoxLayout(del_frame)
        del_lay.setContentsMargins(4, 4, 4, 4)

        trash_btn = QPushButton(_load_icon("trashcan.png"), "")
        trash_btn.setIconSize(_NAV_ICON_SIZE)
        trash_btn.setToolTip("删除当前图片 (X)\n将不需要的图片移至回收目录")
        trash_btn.setStyleSheet(_NAV_BTN_STYLE)
        trash_btn.clicked.connect(self._delete_current_image)
        del_lay.addWidget(trash_btn)

        pb_layout.addWidget(del_frame)

        center_layout.addWidget(preview_bar)
        main_split.addWidget(center)

        # ========== Right toolbar ==========
        right_toolbar = QWidget()
        right_toolbar.setStyleSheet("background: #F0F0F0;")
        rt_outer = QVBoxLayout(right_toolbar)
        rt_outer.setContentsMargins(4, 4, 4, 4)
        rt_outer.setSpacing(72)
        rt_outer.addSpacing(32)

        # Frame: Baked / XSeg mask
        rframe1 = QFrame()
        rframe1.setStyleSheet(_FRAME_STYLE)
        rf1_lay = QVBoxLayout(rframe1)
        rf1_lay.setContentsMargins(2, 2, 2, 2)
        rf1_lay.setSpacing(2)

        baked_btn = QPushButton(_load_icon("view_baked.png"), "")
        baked_btn.setIconSize(_TOOLBAR_ICON_SIZE)
        baked_btn.setToolTip("烘焙遮罩 (4)\n预览多边形渲染的遮罩效果")
        baked_btn.setStyleSheet(_ICON_BTN_STYLE)
        baked_btn.setCheckable(True)
        baked_btn.clicked.connect(self._toggle_baked)
        rf1_lay.addWidget(baked_btn)
        self._baked_btn = baked_btn

        xseg_btn = QPushButton(_load_icon("view_xseg.png"), "")
        xseg_btn.setIconSize(_TOOLBAR_ICON_SIZE)
        xseg_btn.setToolTip("XSeg遮罩 (5)\n查看训练好的XSeg遮罩")
        xseg_btn.setStyleSheet(_ICON_BTN_STYLE)
        xseg_btn.setCheckable(True)
        xseg_btn.clicked.connect(self._toggle_xseg)
        rf1_lay.addWidget(xseg_btn)
        self._xseg_btn = xseg_btn

        rt_outer.addWidget(rframe1)

        # Frame: Auto mask (Face Parsing)
        rframe3 = QFrame()
        rframe3.setStyleSheet(_FRAME_STYLE)
        rf3_lay = QVBoxLayout(rframe3)
        rf3_lay.setContentsMargins(2, 2, 2, 2)
        rf3_lay.setSpacing(2)

        fp_btn = QPushButton(_load_icon("zhezhao.png"), "")
        fp_btn.setIconSize(_TOOLBAR_ICON_SIZE)
        fp_btn.setToolTip("语义自动遮罩 (N)\n使用Face Parsing生成语义遮罩")
        fp_btn.setStyleSheet(_ICON_BTN_STYLE)
        fp_btn.clicked.connect(self._auto_mask_face_parsing)
        rf3_lay.addWidget(fp_btn)

        rt_outer.addWidget(rframe3)

        rt_outer.addStretch()
        main_split.addWidget(right_toolbar)

        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setStretchFactor(2, 0)
        root.addWidget(main_split, 1)

    def _load_images(self):
        self._images = FileManager.find_images(self._aligned_dir)
        if not self._images:
            self._filename_label.setText("目录为空")
            return
        self._thumb_bar.set_paths(self._images)
        self._current_idx = 0
        self._thumb_bar.set_current(0)
        self._load_image_at(0)
        self._refresh_annotated_count()

    def _refresh_annotated_count(self):
        count = 0
        for p in self._images:
            m = MetadataManager.load(p)
            if m is not None and m.seg_ie_polys is not None:
                count += 1
        self._annotated_count = count
        self._annotated_label.setText(f"已标记: {self._annotated_count}")

    def _on_thumb_selected(self, idx: int):
        if 0 <= idx < len(self._images):
            self._current_idx = idx
            self._thumb_bar.set_current(idx)
            self._load_image_at(idx)

    def _load_image_at(self, idx: int):
        if idx < 0 or idx >= len(self._images):
            return
        self._current_idx = idx
        self._canvas.load_image(self._images[idx])
        self._filename_label.setText(self._images[idx].name)
        self._annotated_label.setText(f"已标记: {self._annotated_count}")

    def _prev_image(self):
        if self._current_idx > 0:
            self._on_thumb_selected(self._current_idx - 1)

    def _next_image(self):
        if self._current_idx < len(self._images) - 1:
            self._on_thumb_selected(self._current_idx + 1)

    def _delete_current_image(self):
        if self._current_idx < 0 or self._current_idx >= len(self._images):
            return
        img_path = self._images[self._current_idx]
        self._trash_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(img_path), str(self._trash_dir / img_path.name))
            json_path = img_path.with_suffix(".json")
            if json_path.exists():
                shutil.move(str(json_path), str(self._trash_dir / json_path.name))
        except Exception as e:
            _logger.error(f"移动图片失败: {e}")
            return

        self._thumb_bar.remove_path(self._current_idx)
        self._images.pop(self._current_idx)

        if not self._images:
            self._current_idx = -1
            self._filename_label.setText("目录为空")
            self._canvas._image = None
            self._canvas._qimage = None
            self._canvas.update()
            return

        if self._current_idx >= len(self._images):
            self._current_idx = len(self._images) - 1
        self._thumb_bar.set_current(self._current_idx)
        self._load_image_at(self._current_idx)

    def _set_poly_type(self, poly_type: _PolyType):
        self._canvas.set_poly_type(poly_type)
        self._btn_include.setChecked(poly_type == _PolyType.INCLUDE)
        self._btn_exclude.setChecked(poly_type == _PolyType.EXCLUDE)

    def _update_poly_type_buttons(self):
        if self._canvas._op_mode == _OpMode.DRAW_PTS:
            if self._canvas._poly_type == _PolyType.INCLUDE:
                self._btn_exclude.setEnabled(False)
                self._btn_include.setEnabled(True)
            else:
                self._btn_include.setEnabled(False)
                self._btn_exclude.setEnabled(True)
        else:
            self._btn_include.setEnabled(True)
            self._btn_exclude.setEnabled(True)

    def _canvas_undo(self):
        self._canvas.undo_pt()
        self._canvas.update()
        self._update_poly_type_buttons()

    def _canvas_redo(self):
        self._canvas.redo_pt()
        self._canvas.update()

    def _canvas_clear_all(self):
        self._canvas.clear_all_polys()
        self._canvas.update()
        self._update_poly_type_buttons()

    def _canvas_delete(self):
        self._canvas.delete_poly()
        self._canvas.update()
        self._update_poly_type_buttons()

    def _toggle_baked(self):
        self._canvas.toggle_view_baked()
        self._xseg_btn.setChecked(False)

    def _toggle_xseg(self):
        self._canvas.toggle_view_xseg()
        self._baked_btn.setChecked(False)

    def _auto_mask_face_parsing(self):
        if self._current_idx < 0 or self._current_idx >= len(self._images):
            return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._canvas.auto_mask_face_parsing()
            self._refresh_annotated_count()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            import logging
            logging.getLogger(__name__).error(f"Face Parsing自动遮罩失败: {e}", exc_info=True)
        finally:
            QApplication.restoreOverrideCursor()

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_A:
            self._prev_image()
        elif key == Qt.Key.Key_D:
            self._next_image()
        elif key == Qt.Key.Key_X:
            self._delete_current_image()
        elif key == Qt.Key.Key_M:
            self._auto_mask_face_parsing()
        elif key == Qt.Key.Key_Escape:
            self._canvas._save_current_polys()
            self.accept()
        else:
            self._canvas.keyPressEvent(event)

    def closeEvent(self, event):
        self._canvas._save_current_polys()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._canvas._need_fit or not self._canvas._user_zoomed:
            self._canvas.fit_image()
            self._canvas._need_fit = False
