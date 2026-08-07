import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from faceswap.setting import INSIGHTFACE_MODEL_DIR
from faceswap.core.face_parsing_adapter import FaceParsingAdapter

_logger = logging.getLogger(__name__)

_MODEL_INPUT_SIZE = 192


class OcclusionAdapter:
    _instance: Optional["OcclusionAdapter"] = None

    @classmethod
    def get_instance(cls) -> "OcclusionAdapter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, ctx_id: int = 0):
        self._session = None
        self._input_mean = 0.0
        self._input_std = 1.0
        self._face_parsing = None
        self._ctx_id = ctx_id

    @staticmethod
    def _get_providers_for_ctx(ctx_id: int) -> list[str]:
        if ctx_id < 0:
            return ["CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def _get_providers(self) -> list[str]:
        return self._get_providers_for_ctx(self._ctx_id)

    def _ensure_loaded(self):
        if self._session is not None:
            return
        model_path = str(INSIGHTFACE_MODEL_DIR / "antelopev2" / "1k3d68.onnx")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"1k3d68模型文件不存在: {model_path}")
        providers = self._get_providers()
        _logger.info(f"加载1k3d68模型: {model_path}")
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._detect_normalization(model_path)
        _logger.info("1k3d68模型加载完成")

    def _detect_normalization(self, model_path: str):
        try:
            import onnx
            model = onnx.load(model_path)
            graph = model.graph
            find_sub = False
            find_mul = False
            for nid, node in enumerate(graph.node[:8]):
                name = node.name
                if name.startswith('Sub') or name.startswith('_minus'):
                    find_sub = True
                if name.startswith('Mul') or name.startswith('_mul'):
                    find_mul = True
                if nid < 3 and name == 'bn_data':
                    find_sub = True
                    find_mul = True
            if find_sub and find_mul:
                self._input_mean = 0.0
                self._input_std = 1.0
            else:
                self._input_mean = 127.5
                self._input_std = 128.0
        except Exception:
            self._input_mean = 0.0
            self._input_std = 1.0

    def get_3d_landmarks(
        self,
        image_bgr: np.ndarray,
        bbox: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        self._ensure_loaded()
        h, w = image_bgr.shape[:2]
        if bbox is None:
            bbox = np.array([0, 0, w, h], dtype=np.float32)

        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        center = ((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2)
        scale = _MODEL_INPUT_SIZE / (max(bw, bh) * 1.5)

        M = np.float64([
            [scale, 0, _MODEL_INPUT_SIZE / 2 - center[0] * scale],
            [0, scale, _MODEL_INPUT_SIZE / 2 - center[1] * scale],
        ])

        aimg = cv2.warpAffine(
            image_bgr, M, (_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE),
            borderValue=0.0,
        )

        rgb = aimg[:, :, ::-1].copy()
        blob = (rgb.astype(np.float32) - self._input_mean) / self._input_std
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        input_name = self._session.get_inputs()[0].name
        pred = self._session.run(None, {input_name: blob})[0][0]

        pred = pred.reshape((-1, 3))
        pred = pred[-68:, :]

        pred[:, 0:2] += 1
        pred[:, 0:2] *= (_MODEL_INPUT_SIZE // 2)
        pred[:, 2] *= (_MODEL_INPUT_SIZE // 2)

        IM = cv2.invertAffineTransform(M)
        sf = np.sqrt(IM[0, 0] ** 2 + IM[0, 1] ** 2)
        ones = np.ones((pred.shape[0], 1), dtype=np.float32)
        pts_homo = np.hstack([pred[:, :2], ones])
        pred[:, :2] = (IM @ pts_homo.T).T
        pred[:, 2] *= sf

        return pred

    def get_face_region_mask(
        self,
        image_bgr: np.ndarray,
        bbox: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        landmarks_3d = self.get_3d_landmarks(image_bgr, bbox)
        if landmarks_3d is None:
            return np.zeros((h, w), dtype=np.uint8)

        pts_2d = landmarks_3d[:, :2].astype(np.float64)

        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(pts_2d)
            mask = np.zeros((h, w), dtype=np.uint8)
            for simplex in tri.simplices:
                triangle_pts = pts_2d[simplex].astype(np.int32)
                cv2.fillPoly(mask, [triangle_pts], 255)
        except Exception:
            hull = cv2.convexHull(pts_2d.astype(np.float32))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [hull.astype(np.int32)], 255)

        return mask

    def detect_occlusion(
        self,
        image_bgr: np.ndarray,
        bbox: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        h, w = image_bgr.shape[:2]

        face_region = self.get_face_region_mask(image_bgr, bbox)
        if not np.any(face_region):
            return np.zeros((h, w), dtype=np.uint8)

        if self._face_parsing is None:
            self._face_parsing = FaceParsingAdapter.get_instance()
        try:
            parsing_map = self._face_parsing.predict(image_bgr)
        except Exception as e:
            _logger.warning(f"Face Parsing推理失败: {e}")
            return np.zeros((h, w), dtype=np.uint8)

        bg_mask = (parsing_map == 0).astype(np.uint8) * 255
        occlusion_mask = cv2.bitwise_and(face_region, bg_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_CLOSE, kernel)
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_OPEN, kernel)

        return occlusion_mask

    def release(self):
        if self._session is not None:
            del self._session
            self._session = None
