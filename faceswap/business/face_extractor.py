import threading
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from queue import Empty

import cv2
import numpy as np

from faceswap.setting import FaceType, DEFAULT_FACE_OUTPUT_SIZE, DEFAULT_DET_THRESH, DEFAULT_JPG_QUALITY, FACE_TYPE_SCALE, FACE_TYPE_FOREHEAD_OFFSET, LANDMARK_POINTS, KPS5_POINTS
from faceswap.core.insightface_adapter import InsightFaceAdapter, DetectedFace, AlignedFace
from faceswap.core.metadata_manager import MetadataManager, FaceMetadata
from faceswap.core.debug_image_generator import draw_debug_image
from faceswap.shared.file_manager import FileManager
from faceswap.shared.logger import get_logger

_logger = get_logger("face_extractor")


def _compute_pose(kps_5: np.ndarray, pose: Optional[np.ndarray] = None) -> Optional[list[float]]:
    """返回归一化pose [pitch, yaw, roll]，各角度/90.0，范围约[-1,1]。"""
    if pose is not None and len(pose) >= 3:
        return [float(pose[0]) / 90.0, float(pose[1]) / 90.0, float(pose[2]) / 90.0]
    if kps_5 is None or kps_5.shape[0] < 3:
        return None
    kps = kps_5.astype(np.float64)
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]
    left_ok = not np.all(left_eye == 0)
    right_ok = not np.all(right_eye == 0)
    if not left_ok and not right_ok:
        return None
    if not left_ok:
        face_width = abs(right_eye[0] - nose[0]) * 2.0
        if face_width < 1.0:
            return None
        yaw = float((nose[0] - right_eye[0]) / face_width)
        return [0.0, yaw, 0.0]
    if not right_ok:
        face_width = abs(nose[0] - left_eye[0]) * 2.0
        if face_width < 1.0:
            return None
        yaw = float((nose[0] - left_eye[0]) / face_width)
        return [0.0, yaw, 0.0]
    eye_mid = (left_eye + right_eye) / 2
    eye_dist = abs(left_eye[0] - right_eye[0])
    if eye_dist < 1.0:
        return None
    yaw = float((nose[0] - eye_mid[0]) / eye_dist)
    return [0.0, yaw, 0.0]


def _bbox_to_face_rect(bbox: np.ndarray, expand: float = 0.3) -> list[float]:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    margin = max(w, h) * expand
    return [float(x1 - margin), float(y1 - margin), float(w + 2 * margin), float(h + 2 * margin)]


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


def _gpu_worker(task_queue, result_queue, gpu_idx, config_dict, stop_event):
    import os
    os.environ["ORT_LOGGING_LEVEL"] = "3"

    adapter = InsightFaceAdapter(det_thresh=config_dict["det_thresh"], ctx_id=gpu_idx)
    adapter.warmup()

    face_type = FaceType(config_dict["face_type"])
    output_size = config_dict["output_size"]
    jpg_quality = config_dict["jpg_quality"]
    max_faces = config_dict["max_faces"]
    debug_output = config_dict["debug_output"]

    while not stop_event.is_set():
        try:
            item = task_queue.get(timeout=0.5)
        except Empty:
            continue
        if item is None:
            break
        img_path, idx, total = item
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                result_queue.put((idx, 0, None))
                continue

            faces = adapter.detect_faces(img, max_num=max_faces)
            if not faces:
                debug_info = None
                if debug_output:
                    debug_info = (img_path.stem, img, None, None, 30)
                result_queue.put((idx, 0, debug_info))
                continue

            aligned_faces = []
            for f_idx, face in enumerate(faces):
                aligned = adapter.align_face(
                    img, face.landmarks_106, face_type, output_size, kps_5=face.kps_5, pose=face.pose
                )
                aligned_faces.append((face, aligned))

            debug_info = None
            if debug_output:
                debug_img = img.copy()
                for face, aligned in aligned_faces:
                    debug_img = draw_debug_image(debug_img, face.landmarks_106, aligned.transform_matrix, output_size, face_rect=_bbox_to_face_rect(face.bbox))
                debug_info = (img_path.stem, debug_img, None, None, 50)

            face_data = []
            for f_idx, (face, aligned) in enumerate(aligned_faces):
                aligned_lm = cv2.transform(
                    face.landmarks_106.reshape(1, -1, 2).astype(np.float32),
                    aligned.transform_matrix,
                ).reshape(-1, 2).astype(np.int64)
                meta = FaceMetadata(
                    landmarks_106=aligned_lm,
                    face_type=face_type,
                    source_filename=img_path.name,
                    source_rect=face.bbox.tolist(),
                    source_landmarks_106=face.landmarks_106,
                    image_to_face_mat=aligned.transform_matrix,
                    output_size=output_size,
                    source_kps_5=face.kps_5,
                    source_face_rect=_bbox_to_face_rect(face.bbox),
                    pose=_compute_pose(face.kps_5, face.pose),
                    landmarks_106_visibility=[True] * LANDMARK_POINTS,
                    kps_5_visibility=[True] * KPS5_POINTS,
                )
                face_data.append((img_path.stem, f_idx, aligned.image, meta))

            result_queue.put((idx, len(aligned_faces), debug_info, face_data))
        except Exception as e:
            result_queue.put((idx, -1, str(e)))

    result_queue.put(None)


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

        use_multi_gpu = len(self._gpu_ids) > 1

        if use_multi_gpu:
            return self._extract_multi_gpu(images, aligned_dir, debug_dir, config, total, progress_callback)

        if self._adapter is None:
            self._adapter = InsightFaceAdapter(det_thresh=config.det_thresh, ctx_id=self._gpu_ids[0] if self._gpu_ids else -1)

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
                    debug_img = draw_debug_image(debug_img, face.landmarks_106, aligned.transform_matrix, config.output_size, face_rect=_bbox_to_face_rect(face.bbox))
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
                    source_face_rect=_bbox_to_face_rect(face.bbox),
                    pose=pose_val,
                    landmarks_106_visibility=[True] * LANDMARK_POINTS,
                    kps_5_visibility=[True] * KPS5_POINTS,
                )
                MetadataManager.save(face_path, meta)
                total_extracted += 1

            self._print_progress(i + 1, total, start_time, progress_callback)

        _logger.info("-------------------------")
        _logger.info(f"Images found:        {total}")
        _logger.info(f"Faces detected:      {total_extracted}")
        _logger.info("-------------------------")
        _logger.info("========== 人脸提取完成 ==========")
        return total_extracted

    def _extract_multi_gpu(self, images, aligned_dir, debug_dir, config, total, progress_callback):
        config_dict = {
            "face_type": int(config.face_type),
            "output_size": config.output_size,
            "jpg_quality": config.jpg_quality,
            "output_format": config.output_format,
            "max_faces": config.max_faces,
            "det_thresh": config.det_thresh,
            "debug_output": config.debug_output,
        }

        task_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()
        stop_event = multiprocessing.Event()

        for i, img_path in enumerate(images):
            task_queue.put((img_path, i, total))

        num_gpus = len(self._gpu_ids)
        for _ in range(num_gpus):
            task_queue.put(None)

        workers = []
        for gpu_idx in self._gpu_ids:
            p = multiprocessing.Process(
                target=_gpu_worker,
                args=(task_queue, result_queue, gpu_idx, config_dict, stop_event),
                daemon=True,
            )
            p.start()
            workers.append(p)

        total_extracted = 0
        completed = 0
        import time
        start_time = time.time()

        while completed < total:
            try:
                result = result_queue.get(timeout=0.5)
            except Empty:
                if not any(p.is_alive() for p in workers):
                    alive_count = sum(1 for p in workers if p.exitcode is None)
                    if alive_count == 0:
                        _logger.error(f"All workers exited prematurely (completed {completed}/{total}). Remaining frames skipped.")
                        break
                continue
            if result is None:
                continue
            idx, count, *rest = result
            completed += 1

            if count < 0:
                _logger.error(f"Error processing frame {idx}: {rest[0] if rest else 'unknown'}")
                self._print_progress(completed, total, start_time, progress_callback)
                continue

            debug_info = rest[0] if rest else None
            face_data = rest[1] if len(rest) > 1 else None

            if debug_info and debug_dir:
                stem, img_or_debug, _, _, quality = debug_info
                if img_or_debug is not None:
                    from faceswap.shared.file_manager import imwrite_auto
                    imwrite_auto(debug_dir / f"{stem}.jpg", img_or_debug, jpg_quality=quality)

            if face_data:
                for stem, f_idx, face_img, meta in face_data:
                    face_path = aligned_dir / f"{stem}_{f_idx}{config.ext}"
                    self._write_face(face_path, face_img, config.jpg_quality)
                    MetadataManager.save(face_path, meta)
                    total_extracted += 1

            self._print_progress(completed, total, start_time, progress_callback)

        stop_event.set()
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        _logger.info("-------------------------")
        _logger.info(f"Images found:        {total}")
        _logger.info(f"Faces detected:      {total_extracted}")
        _logger.info("-------------------------")
        _logger.info("========== 人脸提取完成 ==========")
        return total_extracted

    def extract_src_faces(self, frames_dir, aligned_dir, config, progress_callback=None):
        return self._extract_faces(frames_dir, aligned_dir, config, progress_callback)

    def extract_dst_faces(self, frames_dir, aligned_dir, config, progress_callback=None):
        return self._extract_faces(frames_dir, aligned_dir, config, progress_callback)

    def manual_extract(self, frames_dir, aligned_dir, config):
        _logger.info("Manual extract mode - using auto detection as base")
        return self.extract_src_faces(frames_dir, aligned_dir, config)

    def manual_reextract(self, frames_dir, aligned_dir, debug_dir, config):
        _logger.info("Manual re-extract from aligned_debug")
        return self.extract_dst_faces(frames_dir, aligned_dir, config)
