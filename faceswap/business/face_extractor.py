import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from faceswap.setting import FaceType, DEFAULT_FACE_OUTPUT_SIZE, DEFAULT_DET_THRESH, DEFAULT_JPG_QUALITY, FACE_TYPE_SCALE, FACE_TYPE_FOREHEAD_OFFSET, LANDMARK_POINTS, KPS5_POINTS
from faceswap.core.insightface_adapter import InsightFaceAdapter, DetectedFace, AlignedFace
from faceswap.core.metadata_manager import MetadataManager, FaceMetadata
from faceswap.core.debug_image_generator import draw_debug_image
from faceswap.shared.file_manager import FileManager
from faceswap.shared.logger import get_logger

_logger = get_logger("face_extractor")


def _compute_pose(kps_5: np.ndarray, pose: Optional[np.ndarray] = None,
                  kps_5_visibility: Optional[list[bool]] = None) -> Optional[list[float]]:
    """返回归一化pose [pitch, yaw, roll]，范围约[-1,1]。
    优先使用外部提供的pose（如InsightFace检测），否则用5点几何估算。"""
    if pose is not None and len(pose) >= 3:
        return [float(pose[0]) / 90.0, float(pose[1]) / 90.0, float(pose[2]) / 90.0]
    if kps_5 is None or kps_5.shape[0] < 3:
        return None

    kps = kps_5.astype(np.float64)
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]
    left_mouth = kps[3] if kps_5.shape[0] > 3 else np.zeros(2)
    right_mouth = kps[4] if kps_5.shape[0] > 4 else np.zeros(2)

    vis = [True] * 5
    if kps_5_visibility is not None:
        for i in range(min(len(kps_5_visibility), 5)):
            vis[i] = bool(kps_5_visibility[i])
    for i in range(kps_5.shape[0]):
        if np.all(kps[i] == 0):
            vis[i] = False

    if not vis[2]:
        return None
    if not vis[0] and not vis[1]:
        return None

    le, re = left_eye, right_eye
    le_ok, re_ok = vis[0], vis[1]
    lm_ok = vis[3] if kps_5.shape[0] > 3 else False
    rm_ok = vis[4] if kps_5.shape[0] > 4 else False

    roll_rad = 0.0
    if le_ok and re_ok:
        dx = re[0] - le[0]
        dy = re[1] - le[1]
        if abs(dx) > 1.0:
            roll_rad = math.atan2(dy, dx)

    if le_ok and re_ok:
        eye_mid = (le + re) / 2.0
        ref = math.hypot(re[0] - le[0], re[1] - le[1])
    elif re_ok:
        eye_mid = re
        ref = math.hypot(nose[0] - re[0], nose[1] - re[1]) * 2.0
    else:
        eye_mid = le
        ref = math.hypot(nose[0] - le[0], nose[1] - le[1]) * 2.0

    if ref < 1.0:
        return None

    yaw = (nose[0] - eye_mid[0]) / ref
    if lm_ok and rm_ok:
        mouth_mid = (left_mouth + right_mouth) / 2.0
        yaw = (yaw + (mouth_mid[0] - eye_mid[0]) / ref) / 2.0

    pitch_ratio = (nose[1] - eye_mid[1]) / ref
    pitch = (pitch_ratio - 0.5) / 0.3

    yaw_norm = max(-1.0, min(1.0, yaw / 0.5))
    pitch_norm = max(-1.0, min(1.0, pitch))
    roll_norm = max(-1.0, min(1.0, roll_rad / (math.pi / 2)))

    return [pitch_norm, yaw_norm, roll_norm]


@dataclass
class ExtractConfig:
    face_type: FaceType = FaceType.WHOLE_FACE
    max_faces: int = 0
    det_thresh: float = DEFAULT_DET_THRESH
    output_size: int = DEFAULT_FACE_OUTPUT_SIZE
    jpg_quality: int = DEFAULT_JPG_QUALITY
    output_format: str = "jpg"
    num_workers: int = 0
    debug_output: bool = False
    manual_fix: bool = False

    @property
    def ext(self) -> str:
        return f".{self.output_format.lower().lstrip('.')}"


class FaceExtractor:
    def __init__(self, adapter: InsightFaceAdapter = None, gpu_ids: list[int] = None) -> None:
        self._adapter = adapter
        self._gpu_ids = gpu_ids or []

    def _write_face(self, path: Path, img: np.ndarray, quality: int) -> None:
        from faceswap.shared.file_manager import imwrite_auto
        imwrite_auto(path, img, jpg_quality=quality)

    def _print_progress(self, current: int, total: int, start_time: float,
                         progress_callback=None) -> None:
        import time
        elapsed = time.time() - start_time
        if current > 0:
            it_per_sec = current / elapsed
            remaining = (total - current) / it_per_sec
            speed_str = f"{1.0/it_per_sec:.2f}s/it" if it_per_sec < 1 else f"{it_per_sec:.2f}it/s"
        else:
            remaining = 0
            speed_str = ""
        if progress_callback is not None:
            progress_callback(current, total, elapsed, remaining, speed_str)

    def _extract_faces(
        self,
        frames_dir: Path,
        aligned_dir: Path,
        config: ExtractConfig,
        progress_callback=None,
        index_progress_callback=None,
    ) -> int:
        from faceswap.shared.torch_config import configure_torch
        configure_torch("gpu_infer")
        frames_dir = Path(frames_dir)
        aligned_dir = Path(aligned_dir)
        aligned_dir.mkdir(parents=True, exist_ok=True)

        debug_dir = None
        if config.debug_output:
            debug_dir = aligned_dir.parent / (aligned_dir.name + "_debug")
            debug_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(FileManager.find_images(frames_dir), key=lambda p: p.name)
        if not images:
            raise ValueError(f"No images found in {frames_dir}")

        total = len(images)
        _logger.info(f"Extracting faces from {total} frames (output_size={config.output_size}, quality={config.jpg_quality})...")

        if self._adapter is None:
            # 设备跟随全局设备管理器: cuda -> CUDA EP(device_id=0), cpu -> CPU EP
            if self._gpu_ids:
                ctx_id = self._gpu_ids[0]
            else:
                from faceswap.shared.config import auto_select_device
                _dev = auto_select_device()
                ctx_id = _dev.index if _dev.type == "cuda" else -1
            self._adapter = InsightFaceAdapter(det_thresh=config.det_thresh, ctx_id=ctx_id)

        _logger.info("Warming up face detection model...")
        self._adapter.warmup()
        _logger.info("Model ready.")

        total_extracted = 0
        import time
        start_time = time.time()
        for i, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                self._print_progress(i + 1, total, start_time, progress_callback)
                continue

            faces = self._adapter.detect_faces(img, max_num=config.max_faces)

            if not faces:
                if config.debug_output and debug_dir is not None:
                    from faceswap.shared.file_manager import imwrite_auto
                    imwrite_auto(debug_dir / f"{img_path.stem}.jpg", img, jpg_quality=30)
                self._print_progress(i + 1, total, start_time, progress_callback)
                continue

            aligned_faces = []
            for idx, face in enumerate(faces):
                aligned = self._adapter.align_face(
                    img, face.landmarks_106, config.face_type, config.output_size, kps_5=face.kps_5, pose=face.pose
                )
                aligned_faces.append((face, aligned))

            if config.debug_output and debug_dir is not None:
                debug_img = img.copy()
                for face, aligned in aligned_faces:
                    debug_img = draw_debug_image(debug_img, face.landmarks_106, aligned.transform_matrix, config.output_size, bbox=face.bbox)
                from faceswap.shared.file_manager import imwrite_auto
                imwrite_auto(debug_dir / f"{img_path.stem}.jpg", debug_img, jpg_quality=50)

            for idx, (face, aligned) in enumerate(aligned_faces):
                face_filename = f"{img_path.stem}_{idx}{config.ext}"
                face_path = aligned_dir / face_filename
                self._write_face(face_path, aligned.image, config.jpg_quality)

                aligned_lm = cv2.transform(
                    face.landmarks_106.reshape(1, -1, 2).astype(np.float32),
                    aligned.transform_matrix,
                ).reshape(-1, 2).astype(np.int64)

                pose_val = _compute_pose(face.kps_5, face.pose)

                meta = FaceMetadata(
                    landmarks_106=aligned_lm,
                    face_type=config.face_type,
                    source_filename=img_path.name,
                    source_rect=face.bbox.tolist(),
                    source_landmarks_106=face.landmarks_106,
                    image_to_face_mat=aligned.transform_matrix,
                    output_size=config.output_size,
                    source_kps_5=face.kps_5,
                    pose=pose_val,
                    landmarks_106_visibility=[True] * LANDMARK_POINTS,
                    kps_5_visibility=[True] * KPS5_POINTS,
                )
                MetadataManager.save(face_path, meta)
                total_extracted += 1

            self._print_progress(i + 1, total, start_time, progress_callback)

        MetadataManager.build_source_index(aligned_dir, progress_callback=index_progress_callback)
        return total_extracted

    def extract_src_faces(self, frames_dir, aligned_dir, config, progress_callback=None, index_progress_callback=None):
        return self._extract_faces(frames_dir, aligned_dir, config, progress_callback, index_progress_callback)

    def extract_dst_faces(self, frames_dir, aligned_dir, config, progress_callback=None, index_progress_callback=None):
        return self._extract_faces(frames_dir, aligned_dir, config, progress_callback, index_progress_callback)

    def manual_extract(self, frames_dir, aligned_dir, config):
        return self.extract_src_faces(frames_dir, aligned_dir, config)

    def manual_reextract(self, frames_dir, aligned_dir, debug_dir, config):
        return self.extract_dst_faces(frames_dir, aligned_dir, config)
