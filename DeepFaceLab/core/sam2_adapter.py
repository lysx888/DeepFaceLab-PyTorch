import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from DeepFaceLab.setting import SAM2_MODEL_DIR, SAM2_CONFIG_DIR, SAM2_DEFAULT_MODEL, SAM2_DEFAULT_CONFIG

_logger = logging.getLogger(__name__)


class SAM2Adapter:
    _instance: Optional["SAM2Adapter"] = None

    @classmethod
    def get_instance(cls) -> "SAM2Adapter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._predictor = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self):
        if self._predictor is not None:
            return
        config_path = str(SAM2_CONFIG_DIR / SAM2_DEFAULT_CONFIG)
        checkpoint_path = str(SAM2_MODEL_DIR / SAM2_DEFAULT_MODEL)
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"SAM2权重文件不存在: {checkpoint_path}")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _logger.info(f"加载SAM2模型: {checkpoint_path}")
        model = build_sam2(config_path, checkpoint_path, device=self._device)
        self._predictor = SAM2ImagePredictor(model)
        _logger.info("SAM2模型加载完成")

    def predict_with_points(
        self,
        image_rgb: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> np.ndarray:
        self._ensure_loaded()
        with torch.inference_mode(), torch.autocast(self._device, dtype=torch.bfloat16, enabled=self._device == "cuda"):
            self._predictor.set_image(image_rgb)
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        best_idx = int(np.argmax(scores))
        return masks[best_idx]

    def predict_with_box(
        self,
        image_rgb: np.ndarray,
        box: np.ndarray,
    ) -> np.ndarray:
        self._ensure_loaded()
        with torch.inference_mode(), torch.autocast(self._device, dtype=torch.bfloat16, enabled=self._device == "cuda"):
            self._predictor.set_image(image_rgb)
            masks, scores, _ = self._predictor.predict(
                box=box,
                multimask_output=True,
            )
        best_idx = int(np.argmax(scores))
        return masks[best_idx]

    def predict_face_mask(
        self,
        image_rgb: np.ndarray,
        landmarks_106: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if landmarks_106 is not None and len(landmarks_106) >= 5:
            points = landmarks_106.astype(np.float32)
            labels = np.ones(len(points), dtype=np.int32)
            h, w = image_rgb.shape[:2]
            border_pts = np.array([
                [2, 2], [w - 2, 2], [2, h - 2], [w - 2, h - 2],
                [w // 2, 2], [w // 2, h - 2], [2, h // 2], [w - 2, h // 2],
            ], dtype=np.float32)
            border_labels = np.zeros(len(border_pts), dtype=np.int32)
            all_points = np.concatenate([points, border_pts], axis=0)
            all_labels = np.concatenate([labels, border_labels], axis=0)
            return self.predict_with_points(image_rgb, all_points, all_labels)
        else:
            h, w = image_rgb.shape[:2]
            box = np.array([0, 0, w, h], dtype=np.float32)
            return self.predict_with_box(image_rgb, box)

    def release(self):
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
