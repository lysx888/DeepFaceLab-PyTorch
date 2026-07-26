import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from DeepFaceLab.setting import FACE_PARSING_MODEL_PATH

_logger = logging.getLogger(__name__)

_FACE_PARSING_INPUT_SIZE = (512, 512)

CLASS_BACKGROUND = 0
CLASS_SKIN = 1
CLASS_NOSE = 2
CLASS_EYEGLASSES = 3
CLASS_LEFT_EYE = 4
CLASS_RIGHT_EYE = 5
CLASS_LEFT_BROW = 6
CLASS_RIGHT_BROW = 7
CLASS_LEFT_EAR = 8
CLASS_RIGHT_EAR = 9
CLASS_MOUTH = 10
CLASS_UPPER_LIP = 11
CLASS_LOWER_LIP = 12
CLASS_HAIR = 13
CLASS_HAT = 14
CLASS_EARRING = 15
CLASS_NECKLACE = 16
CLASS_NECK = 17
CLASS_CLOTH = 18

INCLUDE_LABELS = frozenset({
    CLASS_SKIN, CLASS_NOSE,
    CLASS_LEFT_EYE, CLASS_RIGHT_EYE,
    CLASS_LEFT_BROW, CLASS_RIGHT_BROW,
    CLASS_MOUTH, CLASS_UPPER_LIP, CLASS_LOWER_LIP,
})

EXCLUDE_LABELS = frozenset({
    CLASS_EYEGLASSES, CLASS_LEFT_EAR, CLASS_RIGHT_EAR,
    CLASS_HAIR, CLASS_HAT, CLASS_EARRING,
    CLASS_NECKLACE, CLASS_NECK, CLASS_CLOTH,
})

HEAD_LABELS = frozenset({
    CLASS_SKIN, CLASS_NOSE, CLASS_EYEGLASSES,
    CLASS_LEFT_EYE, CLASS_RIGHT_EYE,
    CLASS_LEFT_BROW, CLASS_RIGHT_BROW,
    CLASS_LEFT_EAR, CLASS_RIGHT_EAR,
    CLASS_MOUTH, CLASS_UPPER_LIP, CLASS_LOWER_LIP,
    CLASS_HAIR, CLASS_HAT, CLASS_EARRING,
    CLASS_NECKLACE, CLASS_NECK, CLASS_CLOTH,
})

CLASS_NAMES = {
    0: "background", 1: "skin", 2: "nose", 3: "eyeglasses",
    4: "left_eye", 5: "right_eye", 6: "left_brow", 7: "right_brow",
    8: "left_ear", 9: "right_ear", 10: "mouth", 11: "upper_lip",
    12: "lower_lip", 13: "hair", 14: "hat", 15: "earring",
    16: "necklace", 17: "neck", 18: "cloth",
}


class FaceParsingAdapter:
    _instance: Optional["FaceParsingAdapter"] = None

    @classmethod
    def get_instance(cls) -> "FaceParsingAdapter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._session = None

    def _ensure_loaded(self):
        if self._session is not None:
            return
        model_path = str(FACE_PARSING_MODEL_PATH)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Face Parsing权重文件不存在: {model_path}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _logger.info(f"加载Face Parsing模型: {model_path}")
        self._session = ort.InferenceSession(model_path, providers=providers)
        _logger.info("Face Parsing模型加载完成")

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        orig_h, orig_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, _FACE_PARSING_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - 0.5) / 0.5
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: blob})[0]
        parsing_map = np.argmax(output[0], axis=0)
        if parsing_map.shape[0] != orig_h or parsing_map.shape[1] != orig_w:
            parsing_map = cv2.resize(
                parsing_map.astype(np.uint8), (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
        return parsing_map

    def get_include_mask(self, parsing_map: np.ndarray) -> np.ndarray:
        mask = np.zeros(parsing_map.shape, dtype=np.uint8)
        for cls_id in INCLUDE_LABELS:
            mask[parsing_map == cls_id] = 255
        return mask

    def get_exclude_mask(self, parsing_map: np.ndarray) -> np.ndarray:
        mask = np.zeros(parsing_map.shape, dtype=np.uint8)
        for cls_id in EXCLUDE_LABELS:
            mask[parsing_map == cls_id] = 255
        return mask

    def get_head_mask(self, parsing_map: np.ndarray) -> np.ndarray:
        mask = np.zeros(parsing_map.shape, dtype=np.uint8)
        for cls_id in HEAD_LABELS:
            mask[parsing_map == cls_id] = 255
        return mask

    def get_class_mask(self, parsing_map: np.ndarray, class_id: int) -> np.ndarray:
        mask = np.zeros(parsing_map.shape, dtype=np.uint8)
        mask[parsing_map == class_id] = 255
        return mask

    def release(self):
        if self._session is not None:
            del self._session
            self._session = None
