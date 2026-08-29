from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import torch

from faceswap.shared.logger import get_logger
from faceswap.core.face_parsing_adapter import (
    FaceParsingAdapter, INCLUDE_LABELS, EXCLUDE_LABELS, HEAD_LABELS,
    CLASS_EYEGLASSES, CLASS_EARRING, CLASS_HAIR,
)
from faceswap.core.auto_mask_generator import AutoMaskResult, _mask_to_polys, _morphology_clean
from faceswap.core.dfl_weight_loader import load_dfl_xseg_weights
from faceswap.models.xseg_model import XSegNet
from faceswap.shared.config import auto_select_device
from faceswap.setting import FACE_OCCLUDER_MODEL_DIR

_logger = get_logger(__name__)

_OCCLUDER_MODELS = {
    'xseg_1': ('xseg_1.onnx', (256, 256)),
    'xseg_2': ('xseg_2.onnx', (256, 256)),
    'xseg_3': ('xseg_3.onnx', (256, 256)),
}

_MIN_INCLUDE_AREA = 300
_MIN_EXCLUDE_AREA = 80
_INCLUDE_EPSILON_FACTOR = 0.002
_EXCLUDE_EPSILON_FACTOR = 0.003


class FaceOccluder:
    _instance: Optional["FaceOccluder"] = None

    @classmethod
    def get_instance(cls) -> "FaceOccluder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, ctx_id: int = 0):
        self._ctx_id = ctx_id
        self._sessions: dict[str, ort.InferenceSession] = {}

    def _get_providers(self) -> list[str]:
        if self._ctx_id < 0:
            return ["CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def _ensure_loaded(self, model_name: str) -> ort.InferenceSession:
        if model_name in self._sessions:
            return self._sessions[model_name]
        filename, _ = _OCCLUDER_MODELS[model_name]
        model_path = FACE_OCCLUDER_MODEL_DIR / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Face occluder模型不存在: {model_path}")
        _logger.info(f"加载遮挡检测模型: {model_path}")
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess = ort.InferenceSession(str(model_path), sess_options=sess_opts,
                                    providers=["CPUExecutionProvider"])
        self._sessions[model_name] = sess
        return sess

    def is_available(self, model_name: str = 'xseg_1') -> bool:
        try:
            filename, _ = _OCCLUDER_MODELS[model_name]
            return (FACE_OCCLUDER_MODEL_DIR / filename).exists()
        except Exception:
            return False

    def predict(self, image_bgr: np.ndarray, model_names: list[str] = None) -> np.ndarray:
        if model_names is None:
            model_names = ['xseg_1', 'xseg_2', 'xseg_3']
        h, w = image_bgr.shape[:2]
        masks = []
        for model_name in model_names:
            _, input_size = _OCCLUDER_MODELS[model_name]
            sess = self._ensure_loaded(model_name)
            resized = cv2.resize(image_bgr, input_size, interpolation=cv2.INTER_LINEAR)
            blob = resized.astype(np.float32) / 255.0
            blob = blob[np.newaxis, ...]
            input_name = sess.get_inputs()[0].name
            output = sess.run(None, {input_name: blob})[0][0]
            if output.ndim == 3:
                output = output.squeeze(-1)
            output = output.clip(0, 1).astype(np.float32)
            if output.shape != (h, w):
                output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LINEAR)
            masks.append(output)
        result = np.minimum.reduce(masks) if len(masks) > 1 else masks[0]
        result = (cv2.GaussianBlur(result.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return result.clip(0, 1).astype(np.float32)

    def release(self):
        self._sessions.clear()


_DFL_NPY_FILENAME = "XSeg_256.npy"


class FaceOccluderPyTorch:
    _instance: Optional["FaceOccluderPyTorch"] = None

    @classmethod
    def get_instance(cls) -> "FaceOccluderPyTorch":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._model: Optional[XSegNet] = None
        self._device: Optional[torch.device] = None
        self._loaded: bool = False
        self._first_inference: bool = True

    def is_available(self) -> bool:
        try:
            return (FACE_OCCLUDER_MODEL_DIR / _DFL_NPY_FILENAME).exists()
        except Exception:
            return False

    def _ensure_model_loaded(self) -> None:
        if self._loaded:
            return
        npy_path = FACE_OCCLUDER_MODEL_DIR / _DFL_NPY_FILENAME
        model = XSegNet(resolution=256, base_ch=32)
        load_dfl_xseg_weights(npy_path, model)
        device = auto_select_device()
        try:
            model = model.to(device)
        except RuntimeError as e:
            _logger.warning(f"CUDA加载失败，降级到CPU: {e}")
            device = torch.device('cpu')
            model = model.to(device)
        model.eval()
        self._model = model
        self._device = device
        self._loaded = True

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("输入图片为空")
        self._ensure_model_loaded()
        h, w = image_bgr.shape[:2]
        resized = cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(blob).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model.forward(tensor, skip_enabled=True)
            prob = torch.sigmoid(logits)
        mask = prob.cpu().numpy()[0, 0]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        if self._first_inference:
            _logger.debug(f"DFL XSeg首次推理: device={self._device}, 输入={image_bgr.shape[:2]}, 输出={mask.shape}")
            self._first_inference = False
        return mask.astype(np.float32)

    def release(self) -> None:
        self._model = None
        self._device = None
        self._loaded = False
        self._first_inference = True
        FaceOccluderPyTorch._instance = None


class FaceMasker:
    _instance: Optional["FaceMasker"] = None

    @classmethod
    def get_instance(cls) -> "FaceMasker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._face_parsing = FaceParsingAdapter.get_instance()
        self._face_occluder = FaceOccluder.get_instance()
        self._face_occluder_pytorch = FaceOccluderPyTorch.get_instance()

    def auto_draw_mask(
        self,
        image_bgr: np.ndarray,
        use_occlusion: bool = True,
        occlusion_model_names: list[str] = None,
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

        if use_occlusion and self._face_occluder.is_available():
            try:
                occlusion_mask = self._face_occluder.predict(image_bgr, occlusion_model_names)
                occlusion_exclude = (1.0 - occlusion_mask) * 255
                occlusion_exclude = occlusion_exclude.astype(np.uint8)
                occlusion_exclude = cv2.bitwise_and(occlusion_exclude, face_mask)
                exclude_mask = cv2.bitwise_or(exclude_mask, occlusion_exclude)
            except Exception as e:
                _logger.warning(f"遮挡检测失败: {e}")

        if np.any(exclude_mask > 0):
            exclude_mask = _morphology_clean(exclude_mask, 3)

        result.include_polys = _mask_to_polys(face_mask, _INCLUDE_EPSILON_FACTOR, _MIN_INCLUDE_AREA)
        result.exclude_polys = _mask_to_polys(exclude_mask, _EXCLUDE_EPSILON_FACTOR, _MIN_EXCLUDE_AREA)
        return result

    def auto_draw_mask_dfl(
        self,
        image_bgr: np.ndarray,
    ) -> AutoMaskResult:
        result = AutoMaskResult()
        if not self._face_occluder_pytorch.is_available():
            _logger.warning("DFL遮罩不可用: XSeg_256.npy文件不存在")
            return result
        try:
            mask = self._face_occluder_pytorch.predict(image_bgr)
        except Exception as e:
            _logger.warning(f"DFL遮罩推理失败: {e}")
            return result
        include_mask = ((mask > 0.5) * 255).astype(np.uint8)
        include_mask = _morphology_clean(include_mask)
        result.include_polys = _mask_to_polys(include_mask, _INCLUDE_EPSILON_FACTOR, _MIN_INCLUDE_AREA)
        return result

    def create_occlusion_mask(
        self,
        image_bgr: np.ndarray,
        model_names: list[str] = None,
    ) -> np.ndarray:
        if not self._face_occluder.is_available():
            return np.ones(image_bgr.shape[:2], dtype=np.float32)
        return self._face_occluder.predict(image_bgr, model_names)

    def create_region_mask(
        self,
        image_bgr: np.ndarray,
        region_labels: set[int],
    ) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        try:
            parsing_map = self._face_parsing.predict(image_bgr)
        except Exception as e:
            _logger.warning(f"Face Parsing推理失败: {e}")
            return np.ones((h, w), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)
        for cls_id in region_labels:
            mask[parsing_map == cls_id] = 1.0
        mask = (cv2.GaussianBlur(mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return mask.clip(0, 1).astype(np.float32)

    def release(self):
        self._face_parsing.release()
        self._face_occluder.release()
        self._face_occluder_pytorch.release()
