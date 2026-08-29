import threading
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from faceswap.shared.logger import get_logger
from faceswap.shared.config import auto_select_device, is_gpu_device

_logger = get_logger(__name__)
warnings.filterwarnings("ignore", message="cannot import name '_C'")

_MIN_AREA = 200
_EPSILON_FACTOR = 0.004
_MAX_HOLE_AREA = 50
_MAX_SPRINKLE_AREA = 50


def _postprocess_mask(mask_u8: np.ndarray) -> np.ndarray:
    if mask_u8 is None or mask_u8.size == 0:
        return mask_u8
    result = mask_u8.copy()
    inv = cv2.bitwise_not(result)
    n_bg, _, stats_bg, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, n_bg):
        if stats_bg[i, cv2.CC_STAT_AREA] <= _MAX_HOLE_AREA:
            result[inv == i] = 255
    n_fg, _, stats_fg, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    for i in range(1, n_fg):
        if stats_fg[i, cv2.CC_STAT_AREA] <= _MAX_SPRINKLE_AREA:
            result[result == i] = 0
    return result


def _mask_to_polys(mask: np.ndarray, epsilon_factor: float = _EPSILON_FACTOR, min_area: int = _MIN_AREA) -> list[list[tuple[float, float]]]:
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


class SAM2Segmenter:
    _instance: "SAM2Segmenter | None" = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "SAM2Segmenter":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._predictor = None
        self._device = None
        self._image_set: bool = False
        self._current_image_hash: Optional[int] = None

    def _ensure_model(self):
        if self._predictor is not None:
            return
        import sys
        import torch
        _plugin_dir = str(Path(__file__).resolve().parent.parent / "plugin")
        if _plugin_dir not in sys.path:
            sys.path.insert(0, _plugin_dir)
        from faceswap.setting import SAM2_CHECKPOINT_PATH, SAM2_CONFIG_PATH
        from faceswap.plugin.sam2.build_sam import build_sam2
        from faceswap.plugin.sam2.sam2_image_predictor import SAM2ImagePredictor

        self._device = auto_select_device()
        _logger.info(f"加载SAM2模型: {SAM2_CHECKPOINT_PATH.name}, 设备: {self._device}")
        try:
            sam_model = build_sam2(
                config_file=str(SAM2_CONFIG_PATH),
                ckpt_path=str(SAM2_CHECKPOINT_PATH),
                device=str(self._device),
                mode="eval",
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            _logger.warning(f"GPU加载SAM2失败: {e}，降级到CPU")
            self._device = torch.device("cpu")
            sam_model = build_sam2(
                config_file=str(SAM2_CONFIG_PATH),
                ckpt_path=str(SAM2_CHECKPOINT_PATH),
                device="cpu",
                mode="eval",
            )
        self._predictor = SAM2ImagePredictor(sam_model)
        _logger.info(f"SAM2模型加载完成, 设备: {self._device}")

    def _set_image(self, image_bgr: np.ndarray):
        self._ensure_model()
        img_hash = hash(image_bgr.tobytes())
        if self._image_set and self._current_image_hash == img_hash:
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(image_rgb)
        self._image_set = True
        self._current_image_hash = img_hash

    def segment_box(
        self,
        image_bgr: np.ndarray,
        box: tuple[float, float, float, float],
    ) -> list[list[tuple[float, float]]]:
        """
        用SAM2对框选区域进行分割，返回多边形列表。
        box+前景点组合提示 + 迭代细化。

        Args:
            image_bgr: BGR格式图像（HWC, uint8）
            box: (x1, y1, x2, y2) 框选矩形坐标

        Returns:
            多边形点列表，每个多边形是[(x, y), ...]格式
        """
        try:
            self._set_image(image_bgr)
            box_arr = np.array(box, dtype=np.float32)
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            point_coords = np.array([[cx, cy]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
            masks, ious, low_res = self._predictor.predict(
                box=box_arr,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            if masks is None or len(masks) == 0:
                return []
            best_idx = int(np.argmax(ious))
            mask = masks[best_idx]
            low_res_mask = low_res[best_idx:best_idx+1]
            masks2, ious2, _ = self._predictor.predict(
                box=box_arr,
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=low_res_mask,
                multimask_output=False,
            )
            if masks2 is not None and len(masks2) > 0 and ious2[0] >= ious[best_idx]:
                mask = masks2[0]
            mask_u8 = (mask.astype(np.uint8)) * 255
            mask_u8 = _postprocess_mask(mask_u8)
            return _mask_to_polys(mask_u8)
        except Exception as e:
            _logger.error(f"SAM2分割失败: {e}", exc_info=True)
            return []

    def segment_point(
        self,
        image_bgr: np.ndarray,
        point: tuple[float, float],
        label: int = 1,
    ) -> list[list[tuple[float, float]]]:
        """
        用SAM2对点提示进行分割。迭代细化提升边界精度。

        Args:
            image_bgr: BGR格式图像
            point: (x, y) 点坐标
            label: 1=前景, 0=背景

        Returns:
            多边形点列表
        """
        try:
            self._set_image(image_bgr)
            point_coords = np.array([[point[0], point[1]]], dtype=np.float32)
            point_labels = np.array([label], dtype=np.int32)
            masks, ious, low_res = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            if masks is None or len(masks) == 0:
                return []
            best_idx = int(np.argmax(ious))
            mask = masks[best_idx]
            low_res_mask = low_res[best_idx:best_idx+1]
            masks2, ious2, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=low_res_mask,
                multimask_output=False,
            )
            if masks2 is not None and len(masks2) > 0 and ious2[0] >= ious[best_idx]:
                mask = masks2[0]
            mask_u8 = (mask.astype(np.uint8)) * 255
            mask_u8 = _postprocess_mask(mask_u8)
            return _mask_to_polys(mask_u8)
        except Exception as e:
            _logger.error(f"SAM2点分割失败: {e}", exc_info=True)
            return []

    def release(self):
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
        self._image_set = False
        self._current_image_hash = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
