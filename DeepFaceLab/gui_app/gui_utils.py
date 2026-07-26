from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF

from DeepFaceLab.setting import FaceType, DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR
from DeepFaceLab.core.metadata_manager import FaceMetadata, MetadataManager
from DeepFaceLab.core.debug_image_generator import save_debug_image
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("gui_utils")


class RectDragHelper:
    HANDLE_NONE = -1
    HANDLE_TL = 0
    HANDLE_TR = 1
    HANDLE_BR = 2
    HANDLE_BL = 3
    HANDLE_TOP = 4
    HANDLE_RIGHT = 5
    HANDLE_BOTTOM = 6
    HANDLE_LEFT = 7

    @staticmethod
    def hit_test(screen_pos: QPointF, rect: QRectF, img_to_screen_fn, threshold: float = 10.0) -> int:
        if rect is None:
            return RectDragHelper.HANDLE_NONE
        tl = img_to_screen_fn(QPointF(rect.left(), rect.top()))
        tr = img_to_screen_fn(QPointF(rect.right(), rect.top()))
        br = img_to_screen_fn(QPointF(rect.right(), rect.bottom()))
        bl = img_to_screen_fn(QPointF(rect.left(), rect.bottom()))
        corners = [tl, tr, br, bl]
        for i, c in enumerate(corners):
            d = ((c.x() - screen_pos.x()) ** 2 + (c.y() - screen_pos.y()) ** 2) ** 0.5
            if d < threshold:
                return i
        x, y = screen_pos.x(), screen_pos.y()
        if abs(y - tl.y()) < threshold and min(tl.x(), tr.x()) <= x <= max(tl.x(), tr.x()):
            return RectDragHelper.HANDLE_TOP
        if abs(x - tr.x()) < threshold and min(tr.y(), br.y()) <= y <= max(tr.y(), br.y()):
            return RectDragHelper.HANDLE_RIGHT
        if abs(y - bl.y()) < threshold and min(bl.x(), br.x()) <= x <= max(bl.x(), br.x()):
            return RectDragHelper.HANDLE_BOTTOM
        if abs(x - tl.x()) < threshold and min(tl.y(), bl.y()) <= y <= max(tl.y(), bl.y()):
            return RectDragHelper.HANDLE_LEFT
        return RectDragHelper.HANDLE_NONE

    @staticmethod
    def apply_drag(handle: int, img_pt: QPointF, rect: QRectF) -> QRectF:
        if handle == RectDragHelper.HANDLE_TL:
            return QRectF(img_pt.x(), img_pt.y(), rect.right() - img_pt.x(), rect.bottom() - img_pt.y())
        if handle == RectDragHelper.HANDLE_TR:
            return QRectF(rect.left(), img_pt.y(), img_pt.x() - rect.left(), rect.bottom() - img_pt.y())
        if handle == RectDragHelper.HANDLE_BR:
            return QRectF(rect.left(), rect.top(), img_pt.x() - rect.left(), img_pt.y() - rect.top())
        if handle == RectDragHelper.HANDLE_BL:
            return QRectF(img_pt.x(), rect.top(), rect.right() - img_pt.x(), img_pt.y() - rect.top())
        if handle == RectDragHelper.HANDLE_TOP:
            return QRectF(rect.left(), img_pt.y(), rect.width(), rect.bottom() - img_pt.y())
        if handle == RectDragHelper.HANDLE_RIGHT:
            return QRectF(rect.left(), rect.top(), img_pt.x() - rect.left(), rect.height())
        if handle == RectDragHelper.HANDLE_BOTTOM:
            return QRectF(rect.left(), rect.top(), rect.width(), img_pt.y() - rect.top())
        if handle == RectDragHelper.HANDLE_LEFT:
            return QRectF(img_pt.x(), rect.top(), rect.right() - img_pt.x(), rect.height())
        return rect

    @staticmethod
    def cursor_shape(handle: int):
        from PyQt6.QtCore import Qt
        if handle in (RectDragHelper.HANDLE_TL, RectDragHelper.HANDLE_BR):
            return Qt.CursorShape.SizeFDiagCursor
        if handle in (RectDragHelper.HANDLE_TR, RectDragHelper.HANDLE_BL):
            return Qt.CursorShape.SizeBDiagCursor
        if handle in (RectDragHelper.HANDLE_TOP, RectDragHelper.HANDLE_BOTTOM):
            return Qt.CursorShape.SizeVerCursor
        if handle in (RectDragHelper.HANDLE_LEFT, RectDragHelper.HANDLE_RIGHT):
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.ArrowCursor


@dataclass
class FaceSaveResult:
    face_path: Optional[Path]
    metadata: FaceMetadata
    transform_matrix: np.ndarray


def save_face_annotation(
    source_img: np.ndarray,
    lm_106: np.ndarray,
    kps_5: np.ndarray,
    face_type: FaceType,
    output_size: int,
    is_src: bool,
    source_filename: str,
    source_rect: list[float],
    source_face_rect: Optional[list[float]],
    landmarks_106_visibility: list[bool],
    kps_5_visibility: list[bool],
    existing_face_path: Optional[Path] = None,
    generate_training_data: bool = False,
    source_image_path: Optional[Path] = None,
    output_format: str = "jpg",
) -> FaceSaveResult:
    from DeepFaceLab.core.insightface_adapter import InsightFaceAdapter

    adapter = InsightFaceAdapter()
    aligned = adapter.align_face(source_img, lm_106.astype(np.int64), face_type, output_size, kps_5=kps_5)
    transform_mat = aligned.transform_matrix

    aligned_lm = cv2.transform(
        lm_106.reshape(1, -1, 2).astype(np.float32),
        transform_mat,
    ).reshape(-1, 2).astype(np.int64)

    meta = FaceMetadata(
        landmarks_106=aligned_lm,
        face_type=face_type,
        source_filename=source_filename,
        source_rect=source_rect,
        source_landmarks_106=lm_106.astype(np.int64),
        image_to_face_mat=transform_mat,
        output_size=output_size,
        source_kps_5=kps_5,
        source_face_rect=source_face_rect,
        landmarks_106_visibility=landmarks_106_visibility,
        kps_5_visibility=kps_5_visibility,
    )

    aligned_dir = DATA_SRC_ALIGNED_DIR if is_src else DATA_DST_ALIGNED_DIR
    aligned_dir.mkdir(parents=True, exist_ok=True)

    if existing_face_path is not None:
        face_path = existing_face_path
        existing_meta = MetadataManager.load(face_path)
        if existing_meta is not None:
            if existing_meta.arcface_embedding is not None:
                meta.arcface_embedding = existing_meta.arcface_embedding
            if existing_meta.seg_ie_polys is not None:
                meta.seg_ie_polys = existing_meta.seg_ie_polys
            if existing_meta.yaw is not None:
                meta.yaw = existing_meta.yaw
    else:
        from DeepFaceLab.business.face_extractor import _compute_yaw
        try:
            faces = adapter.detect_faces(aligned.image, max_num=1)
            if faces and faces[0].embedding is not None:
                meta.arcface_embedding = faces[0].embedding.astype(np.float32)
        except Exception:
            pass
        meta.yaw = _compute_yaw(kps_5)
        ext = f".{output_format.lower().lstrip('.')}"
        stem = Path(source_filename).stem
        face_idx = 0
        while (aligned_dir / f"{stem}_{face_idx}{ext}").exists():
            face_idx += 1
        face_path = aligned_dir / f"{stem}_{face_idx}{ext}"

    from DeepFaceLab.shared.file_manager import imwrite_auto
    imwrite_auto(face_path, aligned.image, jpg_quality=100)
    MetadataManager.save(face_path, meta)

    debug_dir = aligned_dir.parent / (aligned_dir.name + "_debug")
    face_rect_tuple = None
    if source_face_rect is not None:
        face_rect_tuple = tuple(source_face_rect)
    debug_stem = Path(source_filename).stem
    try:
        save_debug_image(debug_dir, debug_stem, source_img, lm_106, transform_mat, output_size, face_rect=face_rect_tuple)
    except Exception as e:
        _logger.warning(f"调试图生成失败: {e}")

    if generate_training_data:
        _save_insightface_training_data(meta, source_image_path, source_filename)

    return FaceSaveResult(face_path=face_path, metadata=meta, transform_matrix=transform_mat)


def _save_insightface_training_data(meta: FaceMetadata, source_image_path: Optional[Path], source_filename: str):
    try:
        from DeepFaceLab.business.insightface_training_data_generator import InsightFaceTrainingDataGenerator
        from DeepFaceLab.setting import WORKSPACE_DIR
        generator = InsightFaceTrainingDataGenerator(WORKSPACE_DIR)
        generator.generate_from_annotation(
            metadata=meta,
            source_image_path=source_image_path,
            source_stem=Path(source_filename).stem,
        )
    except Exception as e:
        _logger.warning(f"insightface训练数据生成失败: {e}")
