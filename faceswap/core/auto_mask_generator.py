from dataclasses import dataclass, field

import cv2
import numpy as np

from faceswap.shared.logger import get_logger
from faceswap.core.face_parsing_adapter import FaceParsingAdapter, INCLUDE_LABELS, CLASS_EYEGLASSES, CLASS_EARRING, CLASS_HAIR

_logger = get_logger(__name__)

_MIN_INCLUDE_AREA = 300
_MIN_EXCLUDE_AREA = 80
_INCLUDE_EPSILON_FACTOR = 0.002
_EXCLUDE_EPSILON_FACTOR = 0.003


@dataclass
class AutoMaskResult:
    include_polys: list[list[tuple[float, float]]] = field(default_factory=list)
    exclude_polys: list[list[tuple[float, float]]] = field(default_factory=list)


def _mask_to_polys(mask: np.ndarray, epsilon_factor: float = 0.005, min_area: int = 100) -> list[list[tuple[float, float]]]:
    blurred = cv2.GaussianBlur(mask, (7, 7), 0)
    _, smoothed = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    polys = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        poly = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
        polys.append(poly)
    return polys


def _morphology_clean(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


class AutoMaskGenerator:
    _instance: "AutoMaskGenerator | None" = None

    @classmethod
    def get_instance(cls) -> "AutoMaskGenerator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._face_parsing = FaceParsingAdapter.get_instance()

    def generate_face_parsing(
        self,
        image_bgr: np.ndarray,
    ) -> AutoMaskResult:
        result = AutoMaskResult()
        h, w = image_bgr.shape[:2]

        try:
            parsing_map = self._face_parsing.predict(image_bgr)
        except Exception as e:
            _logger.warning(f"Face Parsing推理失败: {e}")
            return result

        face_mask = np.zeros((h, w), dtype=np.uint8)
        for cls_id in INCLUDE_LABELS:
            face_mask[parsing_map == cls_id] = 255
        face_mask = _morphology_clean(face_mask)

        exclude_mask = np.zeros((h, w), dtype=np.uint8)
        face_expanded = cv2.dilate(face_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        for cls_id in (CLASS_EYEGLASSES, CLASS_EARRING):
            region = self._face_parsing.get_class_mask(parsing_map, cls_id)
            overlap = cv2.bitwise_and(face_expanded, region)
            if np.any(overlap > 0):
                exclude_mask = cv2.bitwise_or(exclude_mask, region)
        hair_region = self._face_parsing.get_class_mask(parsing_map, CLASS_HAIR)
        hair_on_face = cv2.bitwise_and(face_mask, hair_region)
        if np.any(hair_on_face > 0):
            hair_dilated = cv2.dilate(hair_region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            exclude_mask = cv2.bitwise_or(exclude_mask, hair_dilated)
        if np.any(exclude_mask > 0):
            exclude_mask = _morphology_clean(exclude_mask, 3)

        result.include_polys = _mask_to_polys(face_mask, _INCLUDE_EPSILON_FACTOR, _MIN_INCLUDE_AREA)
        result.exclude_polys = _mask_to_polys(exclude_mask, _EXCLUDE_EPSILON_FACTOR, _MIN_EXCLUDE_AREA)
        return result

    def generate_simple(
        self,
        landmarks_106: np.ndarray,
    ) -> AutoMaskResult:
        from faceswap.core.landmarks106 import fill_hull_mask_106
        result = AutoMaskResult()
        if landmarks_106 is None or len(landmarks_106) < 3:
            return result
        lm = landmarks_106.astype(np.float32)
        h, w = int(lm[:, 1].max()) + 1, int(lm[:, 0].max()) + 1
        mask = np.zeros((h, w), dtype=np.uint8)
        fill_hull_mask_106(mask, lm)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(largest, 2.0, True)
            if len(approx) >= 3:
                poly = [(float(p[0][0]), float(p[0][1])) for p in approx]
                result.include_polys = [poly]
        return result

    def release(self):
        self._face_parsing.release()
