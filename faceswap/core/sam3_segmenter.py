import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from faceswap.shared.config import auto_select_device, is_gpu_device

_logger = logging.getLogger(__name__)

_MIN_AREA = 200
_EPSILON_FACTOR = 0.004

SAM3_PROMPT_MAP = {
    "human face skin": "include",
    "hair": "exclude",
    "eyeglasses": "exclude",
    "hands in front of face": "exclude",
    "objects occluding face": "exclude",
}

def _mask_to_polys(mask: np.ndarray, epsilon_factor: float = _EPSILON_FACTOR, min_area: int = _MIN_AREA, apply_blur: bool = True):
    if apply_blur:
        blurred = cv2.GaussianBlur(mask, (7, 7), 0)
        _, smoothed = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    else:
        smoothed = mask
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


class SAM3Segmenter:
    _instance: "SAM3Segmenter | None" = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "SAM3Segmenter":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._processor = None
        self._device = None
        self._current_image_hash: Optional[int] = None
        self._current_state = None

    def _ensure_model(self):
        if self._processor is not None:
            return
        import sys
        import torch
        _plugin_dir = str(Path(__file__).resolve().parent.parent / "plugin")
        if _plugin_dir not in sys.path:
            sys.path.insert(0, _plugin_dir)
        from faceswap.setting import SAM3_CHECKPOINT_PATH
        from faceswap.plugin.sam3.model_builder import build_sam3_image_model
        from faceswap.plugin.sam3.model.sam3_image_processor import Sam3Processor

        self._device = auto_select_device()
        _logger.info(f"加载SAM3模型: {SAM3_CHECKPOINT_PATH.name}, 设备: {self._device}")
        _device_str = "cuda" if is_gpu_device(self._device) else "cpu"
        try:
            model = build_sam3_image_model(
                checkpoint_path=str(SAM3_CHECKPOINT_PATH),
                device=_device_str,
                eval_mode=True,
                load_from_HF=False,
            )
            model = model.to(self._device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            _logger.warning(f"GPU加载SAM3失败: {e}，降级到CPU")
            self._device = torch.device("cpu")
            model = build_sam3_image_model(
                checkpoint_path=str(SAM3_CHECKPOINT_PATH),
                device="cpu",
                eval_mode=True,
                load_from_HF=False,
            )
        self._processor = Sam3Processor(
            model,
            device=str(self._device),
            confidence_threshold=0.3,
        )
        _logger.info(f"SAM3模型加载完成, 设备: {self._device}")

    def _get_amp_ctx(self):
        import torch
        from contextlib import nullcontext
        if self._device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self._device.type == "xpu":
            return torch.autocast(device_type="xpu", dtype=torch.bfloat16)
        return nullcontext()

    def _set_image(self, image_bgr: np.ndarray):
        self._ensure_model()
        import torch
        import PIL.Image as PILImage

        img_hash = hash(image_bgr.tobytes())
        if self._current_image_hash == img_hash and self._current_state is not None:
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)
        with self._get_amp_ctx():
            self._current_state = self._processor.set_image(pil_image)
        self._current_image_hash = img_hash

    def segment_text(
        self,
        image_bgr: np.ndarray,
        text_prompt: str,
    ) -> list[tuple[list[list[tuple[float, float]]], float]]:
        """
        用SAM3文本提示词进行分割。

        Args:
            image_bgr: BGR格式图像（HWC, uint8）
            text_prompt: 文本提示词（如"human face skin"）

        Returns:
            列表，每个元素为(多边形列表, 置信度)
        """
        try:
            self._set_image(image_bgr)
            import torch

            with self._get_amp_ctx():
                state = self._processor.set_text_prompt(text_prompt, self._current_state)
            self._current_state = state

            merged_mask, best_score = self._extract_mask(state)
            if merged_mask is None:
                return []
            polys = _mask_to_polys(merged_mask)
            if polys:
                return [(polys, best_score)]
            return []
        except Exception as e:
            _logger.error(f"SAM3文本分割失败: {e}", exc_info=True)
            return []

    def _extract_mask(self, state) -> tuple[Optional[np.ndarray], float]:
        import torch
        masks = state.get("masks")
        scores = state.get("scores")
        if masks is None or scores is None or len(scores) == 0:
            return None, 0.0
        masks_np = masks.squeeze(1).cpu().to(torch.float32).numpy().astype(np.uint8) * 255
        scores_np = scores.cpu().to(torch.float32).numpy()
        merged_mask = np.zeros_like(masks_np[0])
        best_score = 0.0
        for mask, score in zip(masks_np, scores_np):
            merged_mask = cv2.bitwise_or(merged_mask, mask)
            best_score = max(best_score, float(score))
        return merged_mask, best_score

    def segment_auto(
        self,
        image_bgr: np.ndarray,
    ) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
        """
        用预设提示词自动分割，返回(include_polys, exclude_polys)。

        Args:
            image_bgr: BGR格式图像

        Returns:
            (include_polys, exclude_polys) 两个多边形列表
        """
        try:
            import torch
            self._set_image(image_bgr)
            amp_ctx = self._get_amp_ctx()

            include_mask = None

            for prompt, role in SAM3_PROMPT_MAP.items():
                with amp_ctx:
                    state = self._processor.set_text_prompt(prompt, self._current_state)
                self._current_state = state
                mask, score = self._extract_mask(state)
                if mask is None or score < 0.3:
                    continue
                if role == "include":
                    include_mask = mask if include_mask is None else cv2.bitwise_or(include_mask, mask)

            if include_mask is None:
                return [], []

            include_polys = _mask_to_polys(include_mask)
            return include_polys, []
        except Exception as e:
            _logger.error(f"SAM3自动分割失败: {e}", exc_info=True)
            return [], []

    def release(self):
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._current_state = None
        self._current_image_hash = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
