from typing import Optional
from pathlib import Path

import os
import cv2
import numpy as np
import numpy.typing as npt

from faceswap.setting import (
    FaceType,
    FACE_TYPE_SCALE,
    FACE_TYPE_CHIN_OFFSET,
    INSIGHTFACE_MODEL_DIR,
    INSIGHTFACE_MODEL_PACKAGE,
    LANDMARK_POINTS,
)
from faceswap.shared.logger import get_logger

_logger = get_logger("insightface_adapter")


class DetectedFace:
    __slots__ = ("bbox", "det_score", "landmarks_106", "kps_5", "embedding")

    def __init__(
        self,
        bbox: npt.NDArray[np.float32],
        det_score: float,
        landmarks_106: npt.NDArray[np.int64],
        kps_5: Optional[npt.NDArray[np.float32]] = None,
        embedding: Optional[npt.NDArray[np.float32]] = None,
    ):
        self.bbox = bbox
        self.det_score = det_score
        self.landmarks_106 = landmarks_106
        self.kps_5 = kps_5
        self.embedding = embedding


class AlignedFace:
    __slots__ = ("image", "transform_matrix", "landmarks_106")

    def __init__(
        self,
        image: npt.NDArray[np.uint8],
        transform_matrix: npt.NDArray[np.float32],
        landmarks_106: npt.NDArray[np.int64],
    ):
        self.image = image
        self.transform_matrix = transform_matrix
        self.landmarks_106 = landmarks_106


class InsightFaceAdapter:

    def __init__(self, model_dir: Optional[Path] = None, ctx_id: int = 0, det_thresh: float = 0.5) -> None:
        self._model_dir = Path(model_dir) if model_dir else INSIGHTFACE_MODEL_DIR
        self._ctx_id = ctx_id
        self._det_thresh = det_thresh
        self._app = None

    def _ensure_app(self) -> None:
        if self._app is not None:
            return
        try:
            os.environ["ORT_LOGGING_LEVEL"] = "3"
            import warnings
            import onnxruntime as ort
            import io
            import contextlib
            warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
            ort.set_default_logger_severity(3)
            from insightface.app import FaceAnalysis
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self._app = FaceAnalysis(
                    name=INSIGHTFACE_MODEL_PACKAGE,
                    root=str(self._model_dir),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if self._ctx_id >= 0 else ["CPUExecutionProvider"],
                )
                self._app.prepare(ctx_id=self._ctx_id, det_thresh=self._det_thresh, det_size=(640, 640))
            _logger.info("InsightFace FaceAnalysis app initialized.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize InsightFace: {e}") from e

    def warmup(self) -> None:
        self._ensure_app()
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        self._app.get(dummy, max_num=1)

    def detect_faces(
        self,
        img: npt.NDArray[np.uint8],
        max_num: int = 0,
    ) -> list[DetectedFace]:
        self._ensure_app()
        faces = self._app.get(img, max_num=max_num)
        if len(faces) == 0:
            self._app.det_model.prepare(self._ctx_id, input_size=[(128, 128), (640, 640)], det_thresh=self._det_thresh)
            faces = self._app.get(img, max_num=max_num)
            self._app.det_model.prepare(self._ctx_id, input_size=(640, 640), det_thresh=self._det_thresh)
        results = []
        for face in faces:
            lm = np.zeros((LANDMARK_POINTS, 2), dtype=np.int64)
            kps = None
            if hasattr(face, "kps") and face.kps is not None:
                kps = face.kps.astype(np.float32)
            if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
                lm = face.landmark_2d_106.astype(np.int64)
            elif kps is not None:
                if kps.shape[0] == 5:
                    lm[:5] = kps.astype(np.int64)
                elif kps.shape[0] == LANDMARK_POINTS:
                    lm = kps.astype(np.int64)
            bbox = face.bbox.astype(np.float32) if face.bbox is not None else np.zeros(4, dtype=np.float32)
            det_score = float(face.det_score) if face.det_score is not None else 0.0
            embedding = face.embedding.astype(np.float32) if face.embedding is not None else None
            results.append(DetectedFace(bbox=bbox, det_score=det_score, landmarks_106=lm, kps_5=kps, embedding=embedding))
        return results

    def align_face(
        self,
        img: npt.NDArray[np.uint8],
        landmarks_106: npt.NDArray[np.int64],
        face_type: FaceType,
        output_size: int = 256,
        kps_5: Optional[npt.NDArray[np.float32]] = None,
    ) -> AlignedFace:
        from insightface.utils.face_align import estimate_norm
        if kps_5 is None:
            raise ValueError("kps_5 is required for face alignment (5-point keypoints not available)")
        M = estimate_norm(kps_5, output_size, mode=None)
        scale = FACE_TYPE_SCALE.get(face_type, 1.0)
        if scale != 1.0:
            cx, cy = output_size / 2.0, output_size / 2.0
            M[0, 0] /= scale
            M[0, 1] /= scale
            M[1, 0] /= scale
            M[1, 1] /= scale
            M[0, 2] = cx + (M[0, 2] - cx) / scale
            M[1, 2] = cy + (M[1, 2] - cy) / scale
        chin_extend = FACE_TYPE_CHIN_OFFSET.get(face_type, 0.0)
        if chin_extend != 0.0:
            M[1, 2] -= output_size * chin_extend
        aligned = cv2.warpAffine(img, M, (output_size, output_size), flags=cv2.INTER_LANCZOS4, borderValue=0.0)
        return AlignedFace(
            image=aligned,
            transform_matrix=M,
            landmarks_106=landmarks_106,
        )

    def compute_similarity(self, face1: DetectedFace, face2: DetectedFace) -> float:
        if face1.embedding is None or face2.embedding is None:
            return 0.0
        e1 = face1.embedding / (np.linalg.norm(face1.embedding) + 1e-10)
        e2 = face2.embedding / (np.linalg.norm(face2.embedding) + 1e-10)
        return float(np.dot(e1, e2))

    def get_rec_model(self):
        self._ensure_app()
        return self._app.models.get("recognition", None)

    def extract_embedding_aligned(self, aligned_face_bgr: npt.NDArray[np.uint8]) -> Optional[npt.NDArray[np.float32]]:
        self._ensure_app()
        rec_model = self._app.models.get("recognition", None)
        if rec_model is None:
            return None
        embedding = rec_model.get_feat(aligned_face_bgr)
        if embedding is not None:
            return embedding.astype(np.float32).flatten()
        return None


