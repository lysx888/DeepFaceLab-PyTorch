import enum
import math
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF, QSize, pyqtSignal, QThread, QObject, QTimer
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QImage, QPixmap, QIcon,
    QTransform, QMouseEvent, QWheelEvent, QKeyEvent, QPolygonF,
    QFont, QCursor, QFontMetrics,
)
from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QSizePolicy, QApplication, QFrame, QMessageBox, QProgressBar,
)

from faceswap.core.metadata_manager import MetadataManager, FaceMetadata
from faceswap.shared.file_manager import FileManager
from faceswap.shared.image_utils import bgr_to_rgb
from faceswap.shared.logger import get_logger

_logger = get_logger("xseg_editor_dialog")

_ICONS_DIR = Path(__file__).resolve().parent.parent / "icons"


class _OpMode(enum.IntEnum):
    NONE = 0
    DRAW_PTS = 1
    EDIT_PTS = 2
    VIEW_BAKED = 3
    VIEW_XSEG_MASK = 4
    SAM2_BOX = 5


class _PolyType(enum.IntEnum):
    INCLUDE = 1
    EXCLUDE = 0


class _PTEditMode(enum.IntEnum):
    MOVE = 0
    ADD_DEL = 1


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
        self._pt_edit_mode: _PTEditMode = _PTEditMode.MOVE
        self._ctrl_held = False
        self._img_path: Optional[Path] = None
        self._baked_mask: Optional[np.ndarray] = None
        self._xseg_mask_original: Optional[list[dict]] = None
        self._from_xseg_mask: bool = False
        self._need_fit = False
        self._user_zoomed = False
        self._sam2_box_start: Optional[QPointF] = None
        self._sam2_box_end: Optional[QPointF] = None
        self._sam2_box_dragging = False
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
        rgb = bgr_to_rgb(img)
        h, w, ch = rgb.shape
        self._qimage = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self._load_polys(img_path)
        self._baked_mask = None
        self._xseg_mask_original = None
        self._from_xseg_mask = False
        self._zoom = min(self.width() / w, self.height() / h) * 0.9
        self._offset = QPointF(0, 0)
        self._op_mode = _OpMode.NONE
        self._cur_poly_idx = -1
        self._need_fit = True
        self._user_zoomed = False
        self._sam2_box_start = None
        self._sam2_box_end = None
        self._sam2_box_dragging = False
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
        if self._from_xseg_mask:
            current_dump = [p.dump() for p in self._polys] if self._polys else []
            if self._xseg_mask_original is not None and current_dump == self._xseg_mask_original:
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

    def set_pt_edit_mode(self, mode: _PTEditMode):
        if self._pt_edit_mode != mode:
            self._pt_edit_mode = mode
            self._update_cursor()
            self.update()
        self.mode_changed.emit()

    def _effective_pt_edit_mode(self) -> _PTEditMode:
        if self._ctrl_held:
            return _PTEditMode.ADD_DEL
        return self._pt_edit_mode

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
            self.update()
            return
        if self._img_path is None or self._image is None:
            return
        meta = MetadataManager.load(self._img_path)
        if meta is None or meta.xseg_mask is None:
            return
        h, w = self._image.shape[:2]
        xseg_arr = meta.get_xseg_mask_array(h, w)
        if xseg_arr is None:
            return
        self._load_xseg_mask_as_polys(xseg_arr)
        self._op_mode = _OpMode.VIEW_XSEG_MASK
        self._xseg_mask_original = [p.dump() for p in self._polys]
        self._from_xseg_mask = True
        self.update()

    def _load_xseg_mask_as_polys(self, mask_uint8: np.ndarray):
        self._polys = []
        self._cur_poly_idx = -1
        h, w = mask_uint8.shape[:2]
        binary = ((mask_uint8 > 64) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue
            approx = cv2.approxPolyDP(contour, 2.0, True)
            if len(approx) < 3:
                continue
            poly = _SegIEPoly(_PolyType.INCLUDE)
            for pt in approx:
                poly.pts.append(QPointF(float(pt[0][0]), float(pt[0][1])))
            self._polys.append(poly)
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        for poly in self._polys:
            if len(poly.pts) >= 3:
                pts = np.array([[int(p.x()), int(p.y())] for p in poly.pts], dtype=np.int32)
                cv2.fillPoly(hull_mask, [pts], 255)
        exclude_mask = cv2.bitwise_and(hull_mask, cv2.bitwise_not(binary))
        ex_contours, _ = cv2.findContours(exclude_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in ex_contours:
            if cv2.contourArea(contour) < 100:
                continue
            approx = cv2.approxPolyDP(contour, 2.0, True)
            if len(approx) < 3:
                continue
            poly = _SegIEPoly(_PolyType.EXCLUDE)
            for pt in approx:
                poly.pts.append(QPointF(float(pt[0][0]), float(pt[0][1])))
            self._polys.append(poly)
        if self._polys:
            self._cur_poly_idx = 0

    def auto_mask_face_parsing(self):
        if self._image is None:
            return
        from faceswap.core.face_masker import FaceMasker
        masker = FaceMasker.get_instance()
        result = masker.auto_draw_mask(self._image, use_occlusion=True)
        self._apply_auto_mask_result(result)

    def auto_mask_dfl(self):
        if self._image is None:
            return
        from faceswap.core.face_masker import FaceMasker
        masker = FaceMasker.get_instance()
        result = masker.auto_draw_mask_dfl(self._image)
        self._apply_auto_mask_result(result)

    def auto_mask_simple(self, landmarks_106: np.ndarray):
        if self._image is None:
            return
        if landmarks_106 is None or len(landmarks_106) < 3:
            return
        from faceswap.core.auto_mask_generator import AutoMaskGenerator
        generator = AutoMaskGenerator.get_instance()
        result = generator.generate_simple(landmarks_106)
        self._apply_auto_mask_result(result)

    def _apply_auto_mask_result(self, result):
        from faceswap.core.auto_mask_generator import AutoMaskResult
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

    def enter_sam2_box_mode(self):
        if self._image is None:
            return
        if self._op_mode == _OpMode.SAM2_BOX:
            self._op_mode = _OpMode.NONE
        else:
            self._op_mode = _OpMode.SAM2_BOX
        self._sam2_box_start = None
        self._sam2_box_end = None
        self._sam2_box_dragging = False
        self.setCursor(Qt.CursorShape.CrossCursor if self._op_mode == _OpMode.SAM2_BOX else Qt.CursorShape.ArrowCursor)
        self._notify_mode_changed()
        self.update()

    def sam2_segment_box(self, box: tuple[float, float, float, float], poly_type: _PolyType):
        if self._image is None:
            return
        from faceswap.core.sam2_segmenter import SAM2Segmenter
        segmenter = SAM2Segmenter.get_instance()
        polys = segmenter.segment_box(self._image, box)
        for poly_pts in polys:
            poly = _SegIEPoly(poly_type)
            for x, y in poly_pts:
                poly.pts.append(QPointF(x, y))
            self._polys.append(poly)
        if self._polys:
            self._cur_poly_idx = len(self._polys) - 1
            self._op_mode = _OpMode.EDIT_PTS
        self._notify_mode_changed()
        self.update()

    def auto_mask_sam3(self):
        if self._image is None:
            return
        from faceswap.core.sam3_segmenter import SAM3Segmenter
        segmenter = SAM3Segmenter.get_instance()
        include_polys, exclude_polys = segmenter.segment_auto(self._image)
        for poly_pts in include_polys:
            poly = _SegIEPoly(_PolyType.INCLUDE)
            for x, y in poly_pts:
                poly.pts.append(QPointF(x, y))
            self._polys.append(poly)
        for poly_pts in exclude_polys:
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

    @staticmethod
    def _project_on_segment(p: QPointF, a: QPointF, b: QPointF) -> QPointF:
        dx, dy = b.x() - a.x(), b.y() - a.y()
        if dx == 0 and dy == 0:
            return QPointF(a)
        t = max(0, min(1, ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / (dx * dx + dy * dy)))
        return QPointF(a.x() + t * dx, a.y() + t * dy)

    def _update_hover(self, img_pt: QPointF):
        self._hover_poly_idx = -1
        pi, vi = self._find_nearest_pt(img_pt)
        if pi >= 0:
            self._hover_poly_idx = pi
            return
        pi, _ = self._find_nearest_edge(img_pt)
        if pi >= 0:
            self._hover_poly_idx = pi

    def _update_cursor(self):
        if self._op_mode == _OpMode.EDIT_PTS:
            img_pt = self._mouse_pos
            pi, vi = self._find_nearest_pt(img_pt)
            if pi >= 0 and pi == self._cur_poly_idx:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                return
            if self._effective_pt_edit_mode() == _PTEditMode.ADD_DEL:
                ei_pi, ei = self._find_nearest_edge(img_pt)
                if ei_pi >= 0 and ei_pi == self._cur_poly_idx:
                    self.setCursor(Qt.CursorShape.CrossCursor)
                    return
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _find_nearest_edge_with_proj(self, img_pt: QPointF, threshold: float = 8.0) -> tuple[int, int, QPointF]:
        best_dist = threshold / self._zoom
        best_poly = -1
        best_edge = -1
        best_proj = QPointF()
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
                    best_proj = self._project_on_segment(img_pt, p1, p2)
        return best_poly, best_edge, best_proj

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
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
            pass

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

        if self._op_mode == _OpMode.SAM2_BOX and self._sam2_box_start is not None and self._sam2_box_end is not None:
            sp1 = self._img_to_screen(self._sam2_box_start)
            sp2 = self._img_to_screen(self._sam2_box_end)
            rect = QRectF(min(sp1.x(), sp2.x()), min(sp1.y(), sp2.y()),
                          abs(sp2.x() - sp1.x()), abs(sp2.y() - sp1.y()))
            painter.setPen(QPen(QColor(0, 120, 215), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(0, 120, 215, 40)))
            painter.drawRect(rect)

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

        if self._op_mode == _OpMode.SAM2_BOX:
            self._sam2_box_start = QPointF(img_pt)
            self._sam2_box_end = QPointF(img_pt)
            self._sam2_box_dragging = True
            self.update()
            return

        if self._op_mode == _OpMode.NONE or self._op_mode == _OpMode.VIEW_XSEG_MASK:
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
            pi, vi = self._find_nearest_pt(img_pt)
            eff_mode = self._effective_pt_edit_mode()
            if pi >= 0 and pi == self._cur_poly_idx:
                if eff_mode == _PTEditMode.ADD_DEL:
                    self._polys[pi].remove_pt(vi)
                    if len(self._polys[pi].pts) < 3:
                        self._polys.pop(pi)
                        self._cur_poly_idx = -1
                        self._op_mode = _OpMode.NONE
                else:
                    self._drag_pt_idx = vi
                    self._dragging = True
            else:
                ei_pi, ei, proj = self._find_nearest_edge_with_proj(img_pt)
                if eff_mode == _PTEditMode.ADD_DEL and ei_pi >= 0 and ei_pi == self._cur_poly_idx:
                    poly = self._polys[ei_pi]
                    poly.insert_pt(ei + 1, proj)
                elif ei_pi >= 0:
                    self._cur_poly_idx = ei_pi
                else:
                    self._op_mode = _OpMode.NONE
                    self._cur_poly_idx = -1
                    self._pt_edit_mode = _PTEditMode.MOVE

        self.update()
        self._notify_mode_changed()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._qimage is None:
            return
        pos = event.position()
        self._mouse_pos = self._screen_to_img(pos)

        if self._panning:
            delta = pos - self._pan_start
            self._offset = QPointF(self._pan_offset_start.x() + delta.x(),
                                    self._pan_offset_start.y() + delta.y())
            self.update()
            return

        if self._sam2_box_dragging and self._op_mode == _OpMode.SAM2_BOX:
            self._sam2_box_end = self._screen_to_img(pos)
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
        if self._op_mode == _OpMode.EDIT_PTS:
            self._update_cursor()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._drag_pt_idx = -1
            if self._sam2_box_dragging and self._op_mode == _OpMode.SAM2_BOX:
                self._sam2_box_dragging = False
                if self._sam2_box_start is not None and self._sam2_box_end is not None:
                    x1 = min(self._sam2_box_start.x(), self._sam2_box_end.x())
                    y1 = min(self._sam2_box_start.y(), self._sam2_box_end.y())
                    x2 = max(self._sam2_box_start.x(), self._sam2_box_end.x())
                    y2 = max(self._sam2_box_start.y(), self._sam2_box_end.y())
                    if (x2 - x1) > 3 and (y2 - y1) > 3:
                        self.sam2_segment_box((x1, y1, x2, y2), self._poly_type)
                self._sam2_box_start = None
                self._sam2_box_end = None
                self.update()

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
        if key == Qt.Key.Key_Escape and self._op_mode == _OpMode.SAM2_BOX:
            self.enter_sam2_box_mode()
            return
        if key == Qt.Key.Key_Q:
            if self._op_mode != _OpMode.DRAW_PTS or self._poly_type == _PolyType.INCLUDE:
                self.set_poly_type(_PolyType.INCLUDE)
                self._notify_mode_changed()
        elif key == Qt.Key.Key_W:
            if self._op_mode != _OpMode.DRAW_PTS or self._poly_type == _PolyType.EXCLUDE:
                self.set_poly_type(_PolyType.EXCLUDE)
                self._notify_mode_changed()
        elif key == Qt.Key.Key_E:
            if self._op_mode == _OpMode.EDIT_PTS:
                new_mode = _PTEditMode.MOVE if self._pt_edit_mode == _PTEditMode.ADD_DEL else _PTEditMode.ADD_DEL
                self.set_pt_edit_mode(new_mode)
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
        elif key == Qt.Key.Key_Control:
            if self._op_mode == _OpMode.EDIT_PTS and not self._ctrl_held:
                self._ctrl_held = True
                self._update_cursor()
                self.update()
        else:
            super().keyPressEvent(event)
            return
        self.update()

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Control:
            if self._ctrl_held:
                self._ctrl_held = False
                if self._op_mode == _OpMode.EDIT_PTS:
                    self._update_cursor()
                    self.update()
        super().keyReleaseEvent(event)


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


class _BottomThumbBar(QWidget):
    image_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: list[Path] = []
        self._pixmaps: dict[str, QPixmap] = {}
        self._current_idx = -1
        self.setFixedHeight(80)
        self.setMinimumWidth(400)
        self.setMouseTracking(True)

    def set_paths(self, paths: list[Path]):
        self._paths = list(paths)
        self._pixmaps.clear()
        self._current_idx = -1
        self.update()

    def _ensure_pixmap(self, idx: int) -> Optional[QPixmap]:
        if idx < 0 or idx >= len(self._paths):
            return None
        name = self._paths[idx].name
        pm = self._pixmaps.get(name)
        if pm is not None:
            return pm
        img = cv2.imread(str(self._paths[idx]))
        if img is None:
            return None
        thumb = cv2.resize(img, (_CENTER_THUMB_W, _CENTER_THUMB_H))
        rgb = bgr_to_rgb(thumb)
        qimg = QImage(rgb.data, _CENTER_THUMB_W, _CENTER_THUMB_H,
                      3 * _CENTER_THUMB_W, QImage.Format.Format_RGB888).copy()
        pm = QPixmap.fromImage(qimg)
        self._pixmaps[name] = pm
        return pm

    def _trim_cache(self):
        keep = set()
        for slot in range(_VISIBLE_COUNT):
            i = self._current_idx - _CENTER_POS + slot
            if 0 <= i < len(self._paths):
                keep.add(self._paths[i].name)
        self._pixmaps = {k: v for k, v in self._pixmaps.items() if k in keep}

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
        for slot in range(_VISIBLE_COUNT):
            self._ensure_pixmap(idx - _CENTER_POS + slot)
        self._trim_cache()
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

_ICON_BTN_STYLE_NP = (
    "QPushButton { background-color: #E8E8E8; border: none; border-radius: 4px; padding: 4px; }"
    "QPushButton:hover { background-color: #D0D0D0; }"
    "QPushButton:disabled { background-color: #D6D6D6; }"
    "QPushButton:checked { background-color: #C8C8C8; }"
)

_ICON_BTN_STYLE_NP_MT = (
    "QPushButton { background-color: #E8E8E8; border: none; border-radius: 4px; padding: 4px; margin-top: 10px; }"
    "QPushButton:hover { background-color: #D0D0D0; }"
    "QPushButton:disabled { background-color: #D6D6D6; }"
    "QPushButton:checked { background-color: #C8C8C8; }"
)

_NAV_BTN_STYLE = (
    "QPushButton { background: #FFF; border: 1px solid #CCC; border-radius: 4px; padding: 4px; }"
    "QPushButton:hover { background: #E0E0E0; }"
)


class _ImageLoaderThread(QThread):
    progress = pyqtSignal(int, int, str)
    finished_loading = pyqtSignal(list, int)

    def __init__(self, aligned_dir: Path):
        super().__init__()
        self._aligned_dir = aligned_dir

    def run(self):
        images = FileManager.find_images(self._aligned_dir)
        total = len(images)
        annotated_count = 0
        for i, img_path in enumerate(images):
            meta = MetadataManager.load(img_path, lightweight=True)
            if meta is not None and meta.seg_ie_polys is not None:
                annotated_count += 1
            self.progress.emit(i + 1, total, img_path.name)
        self.finished_loading.emit(images, annotated_count)


class XSegEditorDialog(QDialog):
    def __init__(self, aligned_dir: Path, parent=None):
        super().__init__(parent)
        self._aligned_dir = Path(aligned_dir)
        self._trash_dir = self._aligned_dir.parent / (self._aligned_dir.name + "_trash")
        self._images: list[Path] = []
        self._current_idx = -1
        self._annotated_count = 0
        self._xseg_view_active = False
        self._meta_cache = None

        self.setWindowTitle(f"XSeg 遮罩编辑器 — {self._aligned_dir.name}")
        self.setMinimumSize(1300, 800)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint)
        self._build_ui()
        self.showMaximized()
        QApplication.processEvents()

        self._loading_frame = QFrame(self)
        self._loading_frame.setAutoFillBackground(True)
        self._loading_frame.setStyleSheet("background: #1E1E1E;")
        loading_layout = QVBoxLayout(self._loading_frame)
        loading_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label = QLabel("正在加载图片...")
        self._loading_label.setStyleSheet("color: #CCCCCC; font-size: 16px;")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_progress = QProgressBar()
        self._loading_progress.setFixedWidth(400)
        self._loading_progress.setStyleSheet("QProgressBar { color: white; }")
        loading_layout.addWidget(self._loading_label)
        loading_layout.addWidget(self._loading_progress, alignment=Qt.AlignmentFlag.AlignCenter)
        self._loading_frame.setGeometry(self.rect())
        self._loading_frame.show()
        self._loading_frame.raise_()

        self._loader_thread = _ImageLoaderThread(self._aligned_dir)
        self._loader_thread.progress.connect(self._on_load_progress)
        self._loader_thread.finished_loading.connect(self._on_load_finished)
        self._loader_thread.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_loading_frame') and self._loading_frame is not None:
            self._loading_frame.setGeometry(self.rect())

    def _on_load_progress(self, done, total, name):
        self._loading_progress.setMaximum(total)
        self._loading_progress.setValue(done)
        self._loading_label.setText(f"正在加载... {done}/{total}  {name}")

    def _on_load_finished(self, images, annotated_count):
        self._loader_thread = None
        self._loading_frame.hide()
        self._loading_frame = None

        self._images = images
        self._annotated_count = annotated_count
        if not self._images:
            self._filename_label.setText("目录为空")
            return
        self._current_idx = 0
        self._load_image_at(0)
        self._thumb_bar.set_paths(self._images)
        self._thumb_bar.set_current(0)
        self._annotated_label.setText(f"已标记: {self._annotated_count}")

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 0, 16, 0)
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

        # Frame: Add/Del point mode
        frame_pt_edit = QFrame()
        frame_pt_edit.setStyleSheet(_FRAME_STYLE)
        fpe_lay = QVBoxLayout(frame_pt_edit)
        fpe_lay.setContentsMargins(2, 2, 2, 2)
        fpe_lay.setSpacing(2)

        self._btn_pt_edit_mode = QPushButton(_load_icon("pt_edit_mode.png"), "")
        self._btn_pt_edit_mode.setCheckable(True)
        self._btn_pt_edit_mode.setIconSize(_TOOLBAR_ICON_SIZE)
        self._btn_pt_edit_mode.setToolTip("添加/删除点 (E)\n点击边插入点，点击点删除点\n也可按住Ctrl临时激活")
        self._btn_pt_edit_mode.setStyleSheet(_ICON_BTN_STYLE)
        self._btn_pt_edit_mode.clicked.connect(self._toggle_pt_edit_mode)
        fpe_lay.addWidget(self._btn_pt_edit_mode)

        lt_outer.addWidget(frame_pt_edit)

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
        left_toolbar.setMinimumWidth(72)
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
        prev_btn.setToolTip("上一张 (A)\nShift+A 或 Shift+点击: 后退5张")
        prev_btn.setStyleSheet(_NAV_BTN_STYLE)
        prev_btn.clicked.connect(lambda: self._prev_image(5 if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier else 1))
        nav_lay.addWidget(prev_btn)

        self._thumb_bar = _BottomThumbBar()
        self._thumb_bar.image_selected.connect(self._on_thumb_selected)
        nav_lay.addWidget(self._thumb_bar, 1)

        next_btn = QPushButton(_load_icon("right.png"), "")
        next_btn.setIconSize(_NAV_ICON_SIZE)
        next_btn.setToolTip("下一张 (D)\nShift+D 或 Shift+点击: 前进5张")
        next_btn.setStyleSheet(_NAV_BTN_STYLE)
        next_btn.clicked.connect(lambda: self._next_image(5 if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier else 1))
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
        fp_btn.setToolTip("自动遮罩 (N)\nFace Parsing语义分割 + XSeg遮挡检测\n自动生成include/exclude多边形")
        fp_btn.setStyleSheet(_ICON_BTN_STYLE_NP)
        fp_btn.clicked.connect(self._auto_mask_face_parsing)
        rf3_lay.addWidget(fp_btn)
        self._fp_btn = fp_btn

        dfl_btn = QPushButton(_load_icon("zhezhao0.png"), "")
        dfl_btn.setIconSize(_TOOLBAR_ICON_SIZE)
        dfl_btn.setToolTip("DFL遮罩自动绘制\n基于DFL XSeg权重(PyTorch)直接输出人脸遮罩\n生成include多边形")
        dfl_btn.setStyleSheet(_ICON_BTN_STYLE_NP)
        dfl_btn.clicked.connect(self._auto_mask_dfl)
        rf3_lay.addWidget(dfl_btn)
        self._dfl_btn = dfl_btn
        from faceswap.core.face_masker import FaceOccluderPyTorch
        dfl_btn.setEnabled(FaceOccluderPyTorch.get_instance().is_available())

        sam3_btn = QPushButton(_load_icon("sam3xseg.png"), "")
        sam3_btn.setIconSize(_TOOLBAR_ICON_SIZE)
        sam3_btn.setToolTip("SAM3自动分割\n用预设提示词自动分割人脸/头发/眼镜等\n自动生成include/exclude多边形")
        sam3_btn.setStyleSheet(_ICON_BTN_STYLE_NP_MT)
        sam3_btn.clicked.connect(self._auto_mask_sam3)
        rf3_lay.addWidget(sam3_btn)
        self._sam3_btn = sam3_btn

        sam2_btn = QPushButton(_load_icon("sam2xseg.png"), "")
        sam2_btn.setIconSize(_TOOLBAR_ICON_SIZE)
        sam2_btn.setToolTip("SAM2框选分割\n拖拽矩形框选目标区域，SAM2精确分割\n用Q/W切换Include/Exclude类型，Esc退出")
        sam2_btn.setStyleSheet(_ICON_BTN_STYLE_NP_MT)
        sam2_btn.setCheckable(True)
        sam2_btn.clicked.connect(self._toggle_sam2_mode)
        rf3_lay.addWidget(sam2_btn)
        self._sam2_btn = sam2_btn

        rt_outer.addWidget(rframe3)

        rt_outer.addStretch()
        right_toolbar.setMinimumWidth(72)
        main_split.addWidget(right_toolbar)

        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setStretchFactor(2, 0)
        root.addWidget(main_split, 1)

    def _on_thumb_selected(self, idx: int):
        if 0 <= idx < len(self._images):
            self._current_idx = idx
            self._thumb_bar.set_current(idx)
            self._load_image_at(idx)

    def _refresh_annotated_count(self):
        count = 0
        for p in self._images:
            meta = MetadataManager.load(p, lightweight=True)
            if meta is not None and meta.seg_ie_polys is not None:
                count += 1
        self._annotated_count = count
        self._annotated_label.setText(f"已标记: {self._annotated_count}")

    def _load_image_at(self, idx: int):
        if idx < 0 or idx >= len(self._images):
            return
        self._current_idx = idx
        self._canvas.load_image(self._images[idx])
        if self._xseg_view_active:
            self._canvas.toggle_view_xseg()
        self._filename_label.setText(self._images[idx].name)
        self._annotated_label.setText(f"已标记: {self._annotated_count}")

    def _prev_image(self, step: int = 1):
        new_idx = self._current_idx - step
        if new_idx < 0:
            new_idx = 0
        if new_idx != self._current_idx:
            self._on_thumb_selected(new_idx)

    def _next_image(self, step: int = 1):
        new_idx = self._current_idx + step
        if new_idx >= len(self._images):
            new_idx = len(self._images) - 1
        if new_idx != self._current_idx:
            self._on_thumb_selected(new_idx)

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
        self._btn_include.setChecked(self._canvas._poly_type == _PolyType.INCLUDE)
        self._btn_exclude.setChecked(self._canvas._poly_type == _PolyType.EXCLUDE)
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
        self._btn_pt_edit_mode.setChecked(self._canvas._pt_edit_mode == _PTEditMode.ADD_DEL)
        is_sam2_active = self._canvas._op_mode == _OpMode.SAM2_BOX
        self._sam2_btn.setChecked(is_sam2_active)
        if not is_sam2_active and not self._fp_btn.isEnabled():
            self._fp_btn.setEnabled(True)
            self._sam3_btn.setEnabled(True)
            from faceswap.core.face_masker import FaceOccluderPyTorch
            self._dfl_btn.setEnabled(FaceOccluderPyTorch.get_instance().is_available())

    def _toggle_pt_edit_mode(self):
        if self._canvas._op_mode == _OpMode.EDIT_PTS:
            new_mode = _PTEditMode.MOVE if self._canvas._pt_edit_mode == _PTEditMode.ADD_DEL else _PTEditMode.ADD_DEL
            self._canvas.set_pt_edit_mode(new_mode)
        self._btn_pt_edit_mode.setChecked(self._canvas._pt_edit_mode == _PTEditMode.ADD_DEL)

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
        self._xseg_view_active = False
        self._xseg_btn.setChecked(False)
        self._canvas.toggle_view_baked()

    def _toggle_xseg(self):
        self._xseg_view_active = not self._xseg_view_active
        if self._xseg_view_active:
            self._canvas.toggle_view_xseg()
        else:
            if self._canvas._from_xseg_mask:
                self._canvas._save_current_polys()
                self._canvas._from_xseg_mask = False
                self._canvas._xseg_mask_original = None
            self._canvas._op_mode = _OpMode.NONE
            self._canvas._load_polys(self._canvas._img_path) if self._canvas._img_path else None
            self._canvas.update()
        self._baked_btn.setChecked(False)

    def _confirm_clear_polys(self) -> bool:
        if self._canvas._polys:
            ret = QMessageBox.warning(
                self, "确认清除",
                "画布上已有绘制内容，继续将清除所有已绘制的多边形。\n确定要继续吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return False
            self._canvas._polys.clear()
            self._canvas._cur_poly_idx = -1
            self._canvas.update()
        return True

    def _auto_mask_face_parsing(self):
        if self._current_idx < 0 or self._current_idx >= len(self._images):
            return
        if not self._confirm_clear_polys():
            return
        self._sam2_btn.setEnabled(False)
        self._sam3_btn.setEnabled(False)
        self._dfl_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._canvas.auto_mask_face_parsing()
            self._refresh_annotated_count()
        except Exception as e:
            _logger.error(f"Face Parsing自动遮罩失败: {e}", exc_info=True)
        finally:
            QApplication.restoreOverrideCursor()
            from faceswap.core.face_masker import FaceMasker
            FaceMasker.get_instance().release()
            self._sam2_btn.setEnabled(True)
            self._sam3_btn.setEnabled(True)
            from faceswap.core.face_masker import FaceOccluderPyTorch
            self._dfl_btn.setEnabled(FaceOccluderPyTorch.get_instance().is_available())

    def _auto_mask_dfl(self):
        if self._current_idx < 0 or self._current_idx >= len(self._images):
            return
        if not self._confirm_clear_polys():
            return
        self._fp_btn.setEnabled(False)
        self._sam3_btn.setEnabled(False)
        self._sam2_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._canvas.auto_mask_dfl()
            self._refresh_annotated_count()
        except Exception as e:
            _logger.error(f"DFL遮罩自动绘制失败: {e}", exc_info=True)
        finally:
            QApplication.restoreOverrideCursor()
            from faceswap.core.face_masker import FaceOccluderPyTorch
            FaceOccluderPyTorch.get_instance().release()
            self._fp_btn.setEnabled(True)
            self._sam3_btn.setEnabled(True)
            self._sam2_btn.setEnabled(True)
            self._dfl_btn.setEnabled(FaceOccluderPyTorch.get_instance().is_available())

    def _toggle_sam2_mode(self):
        if self._current_idx < 0 or self._current_idx >= len(self._images):
            return
        from faceswap.core.face_masker import FaceOccluderPyTorch
        was_active = self._canvas._op_mode == _OpMode.SAM2_BOX
        if not was_active and not self._confirm_clear_polys():
            return
        self._canvas.enter_sam2_box_mode()
        is_active = self._canvas._op_mode == _OpMode.SAM2_BOX
        self._sam2_btn.setChecked(is_active)
        self._fp_btn.setEnabled(not is_active)
        self._sam3_btn.setEnabled(not is_active)
        self._dfl_btn.setEnabled(not is_active and FaceOccluderPyTorch.get_instance().is_available())
        if was_active and not is_active:
            from faceswap.core.sam2_segmenter import SAM2Segmenter
            SAM2Segmenter.get_instance().release()

    def _auto_mask_sam3(self):
        if self._current_idx < 0 or self._current_idx >= len(self._images):
            return
        if not self._confirm_clear_polys():
            return
        self._fp_btn.setEnabled(False)
        self._sam2_btn.setEnabled(False)
        self._dfl_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._canvas.auto_mask_sam3()
            self._refresh_annotated_count()
        except Exception as e:
            _logger.error(f"SAM3自动分割失败: {e}", exc_info=True)
        finally:
            QApplication.restoreOverrideCursor()
            from faceswap.core.sam3_segmenter import SAM3Segmenter
            SAM3Segmenter.get_instance().release()
            self._fp_btn.setEnabled(True)
            self._sam2_btn.setEnabled(True)
            from faceswap.core.face_masker import FaceOccluderPyTorch
            self._dfl_btn.setEnabled(FaceOccluderPyTorch.get_instance().is_available())

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if key == Qt.Key.Key_A:
            self._prev_image(5 if shift else 1)
        elif key == Qt.Key.Key_D:
            self._next_image(5 if shift else 1)
        elif key == Qt.Key.Key_X:
            self._delete_current_image()
        elif key == Qt.Key.Key_M:
            self._auto_mask_face_parsing()
        elif key == Qt.Key.Key_Escape:
            if self._canvas._op_mode == _OpMode.SAM2_BOX:
                self._toggle_sam2_mode()
            else:
                self._canvas._save_current_polys()
                self.accept()
        else:
            self._canvas.keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        self._canvas.keyReleaseEvent(event)

    def closeEvent(self, event):
        self._canvas._save_current_polys()
        from faceswap.core.sam2_segmenter import SAM2Segmenter
        from faceswap.core.sam3_segmenter import SAM3Segmenter
        from faceswap.core.face_masker import FaceMasker
        SAM2Segmenter.get_instance().release()
        SAM3Segmenter.get_instance().release()
        FaceMasker.get_instance().release()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._canvas._need_fit or not self._canvas._user_zoomed:
            self._canvas.fit_image()
            self._canvas._need_fit = False
