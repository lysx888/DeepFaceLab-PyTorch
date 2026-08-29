from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from faceswap.shared.logger import get_logger
from faceswap.setting import YOLO_MODEL_DIR, YOLO_DEFAULT_MODEL

_logger = get_logger(__name__)

COCO_PERSON_CLASS_ID = 0


class YoloAdapter:
    _instance: Optional["YoloAdapter"] = None

    @classmethod
    def get_instance(cls) -> "YoloAdapter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, device: str = "auto"):
        self._model = None
        self._device = device

    def _resolve_device(self) -> str:
        if self._device != "auto":
            return self._device
        from faceswap.shared.config import auto_select_device
        return auto_select_device().type

    def _ensure_loaded(self):
        if self._model is not None:
            return
        model_path = YOLO_MODEL_DIR / YOLO_DEFAULT_MODEL
        if not model_path.exists():
            _logger.info(f"YOLO模型本地不存在，将自动下载: {model_path}")
        _logger.info(f"加载YOLO模型: {model_path}")
        from ultralytics import YOLO
        self._model = YOLO(str(model_path))
        _logger.info("YOLO模型加载完成")

    def predict_occlusion_mask(
        self,
        image_bgr: np.ndarray,
        conf: float = 0.25,
    ) -> np.ndarray:
        self._ensure_loaded()
        results = self._model(
            image_bgr, conf=conf, retina_masks=True, verbose=False,
            device=self._resolve_device(),
        )
        h, w = image_bgr.shape[:2]
        occlusion_mask = np.zeros((h, w), dtype=np.uint8)
        for result in results:
            if result.masks is None:
                _logger.info("YOLO: 未检测到任何实例")
                continue
            n = len(result.masks)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            masks_data = result.masks.data.cpu().numpy()
            for i in range(n):
                cls_id = int(cls_ids[i])
                name = result.names.get(cls_id, str(cls_id))
                conf_i = float(result.boxes.conf[i].cpu().numpy())
                is_person = cls_id == COCO_PERSON_CLASS_ID
                _logger.info(f"  YOLO [{i}] {name} (cls={cls_id}) conf={conf_i:.2f} person={is_person}")
                if is_person:
                    continue
                mask_i = (masks_data[i] * 255).astype(np.uint8)
                occlusion_mask = cv2.bitwise_or(occlusion_mask, mask_i)
        _logger.info(f"YOLO occlusion mask: {np.count_nonzero(occlusion_mask)} px")
        return occlusion_mask

    def release(self):
        if self._model is not None:
            del self._model
            self._model = None
