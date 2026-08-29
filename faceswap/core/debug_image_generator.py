import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from faceswap.core.landmarks106 import (
    LANDMARK_GROUPS_106, LINE_CONNECTIONS_106, fill_hull_mask_106,
)
from faceswap.shared.logger import get_logger

_logger = get_logger("debug_image_generator")

_LINE_COLOR = (0, 255, 0)
_LINE_THICKNESS = 2


def draw_debug_image(
    source_img: np.ndarray,
    landmarks_106: np.ndarray,
    transform_mat: Optional[np.ndarray] = None,
    output_size: int = 512,
    face_rect: Optional[tuple] = None,
    draw_hull: bool = True,
) -> np.ndarray:
    debug_img = source_img.copy()
    lm = landmarks_106.astype(np.float32)

    if face_rect is not None:
        rx, ry, rw, rh = [int(v) for v in face_rect]
        cv2.rectangle(debug_img, (rx, ry), (rx + rw, ry + rh), (212, 120, 0), 2)

    for gname, idxs in LANDMARK_GROUPS_106:
        lines = LINE_CONNECTIONS_106.get(gname, [])
        for i1, i2 in lines:
            if i1 < len(lm) and i2 < len(lm):
                p1 = (int(lm[i1, 0]), int(lm[i1, 1]))
                p2 = (int(lm[i2, 0]), int(lm[i2, 1]))
                cv2.line(debug_img, p1, p2, _LINE_COLOR, _LINE_THICKNESS)
        for idx in idxs:
            if idx < len(lm):
                cv2.circle(debug_img, (int(lm[idx, 0]), int(lm[idx, 1])), 2, _LINE_COLOR, -1)

    if draw_hull:
        hull_mask = np.zeros(debug_img.shape[:2], dtype=np.uint8)
        fill_hull_mask_106(hull_mask, lm)
        debug_img[hull_mask > 0] = (debug_img[hull_mask > 0] * 0.5).astype(np.uint8)

    return debug_img


def save_debug_image(
    debug_dir: Path,
    stem: str,
    source_img: np.ndarray,
    landmarks_106: np.ndarray,
    transform_mat: Optional[np.ndarray] = None,
    output_size: int = 512,
    face_rect: Optional[tuple] = None,
    draw_hull: bool = True,
    quality: int = 95,
    fmt: str = "jpg",
) -> np.ndarray:
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_img = draw_debug_image(source_img, landmarks_106, transform_mat, output_size, face_rect, draw_hull)
    ext = f".{fmt.lower().lstrip('.')}"
    debug_path = debug_dir / (stem + ext)
    from faceswap.shared.file_manager import imwrite_auto
    imwrite_auto(debug_path, debug_img, jpg_quality=quality)
    return debug_img
