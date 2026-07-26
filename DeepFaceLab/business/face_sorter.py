import os
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from DeepFaceLab.core.metadata_manager import MetadataManager, FaceMetadata
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("face_sorter")


class SortAlgorithm(Enum):
    BLUR = "blur"
    HIST = "hist"
    YAW = "yaw"
    PITCH = "pitch"
    BRIGHTNESS = "brightness"
    HUE = "hue"
    ONEFACE = "oneface"
    FINAL = "final"


class FaceSorter:
    def sort_aligned(
        self,
        aligned_dir: Path,
        algorithm: SortAlgorithm,
        progress_callback=None,
    ) -> int:
        import time
        aligned_dir = Path(aligned_dir)
        images = FileManager.find_images(aligned_dir)
        if not images:
            raise ValueError(f"No images found in {aligned_dir}")

        metadata_map = MetadataManager.load_all(aligned_dir)
        t0 = time.time()

        def _report(done, total):
            if progress_callback is not None:
                progress_callback(done, total, time.time() - t0)

        if algorithm == SortAlgorithm.ONEFACE:
            result = self._oneface(aligned_dir, images, metadata_map)
            _report(len(images), len(images))
            return result

        if algorithm == SortAlgorithm.HIST:
            scored = self._sort_by_hist(images, metadata_map, _report)
        elif algorithm == SortAlgorithm.FINAL:
            scored = self._sort_by_final(images, metadata_map, _report)
        else:
            compute_fn = {
                SortAlgorithm.BLUR: self._compute_blur,
                SortAlgorithm.YAW: self._compute_yaw,
                SortAlgorithm.PITCH: self._compute_pitch,
                SortAlgorithm.BRIGHTNESS: self._compute_brightness,
                SortAlgorithm.HUE: self._compute_hue,
            }.get(algorithm)
            if compute_fn is None:
                raise ValueError(f"Unknown sort algorithm: {algorithm}")
            scored = []
            for i, img_path in enumerate(images):
                meta = metadata_map.get(img_path.name)
                score = compute_fn(img_path, meta)
                scored.append((score, img_path))
                _report(i + 1, len(images))
            scored.sort(key=lambda x: x[0])

        return self._rename_sorted(aligned_dir, scored, algorithm)

    def _rename_sorted(self, aligned_dir: Path, scored: list[tuple[float, Path]], algorithm: SortAlgorithm) -> int:
        temp_dir = aligned_dir / "_sort_tmp"
        temp_dir.mkdir(exist_ok=True)

        for rank, (score, old_path) in enumerate(scored):
            ext = old_path.suffix
            new_name = f"{rank:05d}_0{ext}"
            old_path.rename(temp_dir / new_name)
            old_json = old_path.with_suffix(".json")
            if old_json.exists():
                old_json.rename(temp_dir / f"{rank:05d}_0.json")

        renamed = 0
        for f in temp_dir.iterdir():
            f.rename(aligned_dir / f.name)
            renamed += 1

        temp_dir.rmdir()
        _logger.info(f"Sorted {len(scored)} faces by {algorithm.value}, renamed {renamed}")
        return renamed

    def _oneface(self, aligned_dir: Path, images: list[Path], metadata_map: dict[str, FaceMetadata]) -> int:
        groups: dict[str, list[tuple[float, Path]]] = {}
        for img_path in images:
            meta = metadata_map.get(img_path.name)
            source = meta.source_filename if meta else img_path.stem.split("_")[0]
            area = self._face_area(img_path, meta)
            groups.setdefault(source, []).append((area, img_path))

        deleted = 0
        trash_dir = aligned_dir.parent / (aligned_dir.name + "_trash")
        trash_dir.mkdir(parents=True, exist_ok=True)
        for source, faces in groups.items():
            if len(faces) <= 1:
                continue
            faces.sort(key=lambda x: x[0], reverse=True)
            for _, img_path in faces[1:]:
                json_path = img_path.with_suffix(".json")
                img_path.rename(trash_dir / img_path.name)
                if json_path.exists():
                    json_path.rename(trash_dir / json_path.name)
                deleted += 1

        _logger.info(f"Oneface: deleted {deleted} extra faces, kept {len(groups)} source frames")
        return len(groups)

    @staticmethod
    def _face_area(img_path: Path, meta: Optional[FaceMetadata]) -> float:
        if meta is not None and meta.landmarks_106 is not None and len(meta.landmarks_106) >= 2:
            lm = meta.landmarks_106.astype(np.float64)
            x_min, y_min = lm.min(axis=0)
            x_max, y_max = lm.max(axis=0)
            return float((x_max - x_min) * (y_max - y_min))
        img = cv2.imread(str(img_path))
        if img is not None:
            return float(img.shape[0] * img.shape[1])
        return 0.0

    @staticmethod
    def _compute_blur(img_path: Path, meta: Optional[FaceMetadata]) -> float:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())

    @staticmethod
    def _compute_yaw(img_path: Path, meta: Optional[FaceMetadata]) -> float:
        if meta is None or meta.landmarks_106 is None or len(meta.landmarks_106) < 70:
            return 0.0
        lm = meta.landmarks_106.astype(np.float64)
        left_eye = lm[36:42].mean(axis=0)
        right_eye = lm[42:48].mean(axis=0)
        nose = lm[57]
        eye_center = (left_eye + right_eye) / 2.0
        dx = nose[0] - eye_center[0]
        eye_dist = right_eye[0] - left_eye[0]
        if eye_dist < 1.0:
            return 0.0
        return float(dx / eye_dist)

    @staticmethod
    def _compute_pitch(img_path: Path, meta: Optional[FaceMetadata]) -> float:
        if meta is None or meta.landmarks_106 is None or len(meta.landmarks_106) < 70:
            return 0.0
        lm = meta.landmarks_106.astype(np.float64)
        left_eye = lm[36:42].mean(axis=0)
        right_eye = lm[42:48].mean(axis=0)
        nose = lm[57]
        eye_center = (left_eye + right_eye) / 2.0
        dy = nose[1] - eye_center[1]
        face_h = lm[8, 1] - eye_center[1] if lm.shape[0] > 8 else 1.0
        if face_h < 1.0:
            return 0.0
        return float(dy / face_h)

    @staticmethod
    def _compute_brightness(img_path: Path, meta: Optional[FaceMetadata]) -> float:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(img.mean())

    @staticmethod
    def _compute_hue(img_path: Path, meta: Optional[FaceMetadata]) -> float:
        img = cv2.imread(str(img_path))
        if img is None:
            return 0.0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 0].mean())

    @staticmethod
    def _compute_3d_hist(img_path: Path) -> Optional[np.ndarray]:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _sort_by_hist(self, images: list[Path], metadata_map: dict[str, FaceMetadata], report=None) -> list[tuple[float, Path]]:
        hists: list[tuple[np.ndarray, Path]] = []
        for i, img_path in enumerate(images):
            h = self._compute_3d_hist(img_path)
            if h is not None:
                hists.append((h, img_path))
            if report:
                report(i + 1, len(images))

        if not hists:
            return [(0.0, p) for p in images]

        mean_hist = np.mean([h for h, _ in hists], axis=0)
        cv2.normalize(mean_hist, mean_hist)

        scored: list[tuple[float, Path]] = []
        for h, img_path in hists:
            corr = cv2.compareHist(h, mean_hist, cv2.HISTCMP_CORREL)
            scored.append((corr, img_path))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _sort_by_final(self, images: list[Path], metadata_map: dict[str, FaceMetadata], report=None) -> list[tuple[float, Path]]:
        raw_scores: dict[str, list[float]] = {}
        for i, img_path in enumerate(images):
            meta = metadata_map.get(img_path.name)
            blur = self._compute_blur(img_path, meta)
            yaw = abs(self._compute_yaw(img_path, meta))
            pitch = abs(self._compute_pitch(img_path, meta))
            brightness = self._compute_brightness(img_path, meta)
            raw_scores[img_path.name] = [blur, yaw, pitch, brightness]
            if report:
                report(i + 1, len(images))

        blur_vals = [v[0] for v in raw_scores.values()]
        yaw_vals = [v[1] for v in raw_scores.values()]
        pitch_vals = [v[2] for v in raw_scores.values()]
        bright_vals = [v[3] for v in raw_scores.values()]

        def _norm(vals: list[float]) -> tuple[float, float]:
            mn, mx = min(vals), max(vals)
            return (mn, mx) if mx > mn else (0.0, 1.0)

        b_min, b_max = _norm(blur_vals)
        y_min, y_max = _norm(yaw_vals)
        p_min, p_max = _norm(pitch_vals)
        br_min, br_max = _norm(bright_vals)

        scored: list[tuple[float, Path]] = []
        for img_path in images:
            s = raw_scores[img_path.name]
            n_blur = (s[0] - b_min) / (b_max - b_min)
            n_yaw = 1.0 - (s[1] - y_min) / (y_max - y_min)
            n_pitch = 1.0 - (s[2] - p_min) / (p_max - p_min)
            n_bright = 1.0 - abs((s[3] - br_min) / (br_max - br_min) - 0.5) * 2.0
            score = n_blur * 0.3 + n_yaw * 0.3 + n_pitch * 0.2 + n_bright * 0.2
            scored.append((score, img_path))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
