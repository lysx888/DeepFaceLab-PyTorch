import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

from faceswap.core.metadata_manager import MetadataManager, FaceMetadata
from faceswap.shared.file_manager import FileManager, imwrite_auto
from faceswap.shared.image_utils import bgr_to_hsv
from faceswap.shared.logger import get_logger

_logger = get_logger("model_merger")


class MaskMode(Enum):
    DST = "dst"
    XSEG = "xseg"
    LEARNED = "learned"


@dataclass
class MergeConfig:
    mask_mode: MaskMode = MaskMode.XSEG
    erode_mask_modifier: float = 0.0
    blur_mask_modifier: float = 0.0
    output_size: int = 256
    seamless_clone: bool = False
    color_transfer: bool = True
    enhancer_model: str = ""
    enhancer_blend: int = 80


class ModelMerger:
    def __init__(self) -> None:
        self._enhancer = None

    def _get_enhancer(self, model_name: str, blend: int):
        if self._enhancer is not None and self._enhancer.model_name == model_name and self._enhancer.blend == blend:
            return self._enhancer
        from faceswap.models.face_enhancer import FaceEnhancer
        self._enhancer = FaceEnhancer(model_name=model_name, blend=blend)
        return self._enhancer

    def composite_to_frames(
        self,
        dst_frames_dir: Path,
        dst_aligned_dir: Path,
        swapped_dir: Path,
        merged_dir: Path,
        config: MergeConfig,
        progress_callback: Optional[Callable] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> int:
        dst_frames_dir = Path(dst_frames_dir)
        dst_aligned_dir = Path(dst_aligned_dir)
        swapped_dir = Path(swapped_dir)
        merged_dir = Path(merged_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)

        if (swapped_dir / ".frame_level").exists():
            return self._composite_frame_level(
                swapped_dir, merged_dir, progress_callback, stop_check,
            )

        mask_dir = merged_dir.parent / "merged_mask"
        mask_dir.mkdir(parents=True, exist_ok=True)

        all_meta = MetadataManager.load_all(dst_aligned_dir)
        if not all_meta:
            raise ValueError(f"No aligned face metadata found in {dst_aligned_dir}")

        frame_to_faces: dict[str, list[tuple[str, FaceMetadata]]] = {}
        for face_name, meta in all_meta.items():
            src_fn = meta.source_filename
            if src_fn not in frame_to_faces:
                frame_to_faces[src_fn] = []
            frame_to_faces[src_fn].append((face_name, meta))

        dst_frames = FileManager.find_images(dst_frames_dir)
        if not dst_frames:
            raise ValueError(f"No destination frames found in {dst_frames_dir}")

        total = len(dst_frames)
        count = 0
        start_time = time.time()

        for i, frame_path in enumerate(dst_frames):
            faces_for_frame = frame_to_faces.get(frame_path.name, [])
            if not faces_for_frame:
                frame_img = cv2.imread(str(frame_path))
                if frame_img is not None:
                    imwrite_auto(merged_dir / frame_path.name, frame_img)
                count += 1
                if progress_callback is not None:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / count) * (total - count) if count > 0 else 0
                    speed = f"{count / elapsed:.1f}it/s" if elapsed > 0 else ""
                    progress_callback(i + 1, total, elapsed, remaining, speed)
                continue

            frame_img = cv2.imread(str(frame_path))
            if frame_img is None:
                continue

            frame_h, frame_w = frame_img.shape[:2]
            result = frame_img.copy()
            full_mask = np.zeros((frame_h, frame_w), dtype=np.float32)

            for face_name, meta in faces_for_frame:
                swapped_path = swapped_dir / face_name
                if not swapped_path.exists():
                    for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
                        alt = swapped_dir / (swapped_path.stem + ext)
                        if alt.exists():
                            swapped_path = alt
                            break
                    if not swapped_path.exists():
                        _logger.warning(f"Swapped face not found: {face_name}")
                        continue

                swapped_img = cv2.imread(str(swapped_path))
                if swapped_img is None:
                    continue

                if meta.image_to_face_mat is None:
                    _logger.warning(f"No transform matrix for {face_name}")
                    continue

                face_to_image_mat = cv2.invertAffineTransform(meta.image_to_face_mat)

                output_size = meta.output_size
                mask = self._create_aligned_mask(meta, config, output_size)

                if config.enhancer_model:
                    enhancer = self._get_enhancer(config.enhancer_model, config.enhancer_blend)
                    swapped_img = enhancer.enhance(swapped_img, (mask * 255).astype(np.uint8))

                if config.color_transfer:
                    dst_aligned_path = dst_aligned_dir / face_name
                    if not dst_aligned_path.exists():
                        for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
                            alt = dst_aligned_dir / (dst_aligned_path.stem + ext)
                            if alt.exists():
                                dst_aligned_path = alt
                                break
                    if dst_aligned_path.exists():
                        dst_face = cv2.imread(str(dst_aligned_path))
                        if dst_face is not None:
                            swapped_img = self._conditional_match_color(dst_face, swapped_img)

                warped_face = cv2.warpAffine(
                    swapped_img, face_to_image_mat, (frame_w, frame_h),
                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT,
                )
                warped_mask = cv2.warpAffine(
                    mask, face_to_image_mat, (frame_w, frame_h),
                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
                )

                warped_mask_3ch = np.stack([warped_mask] * 3, axis=-1)
                result = result * (1.0 - warped_mask_3ch) + warped_face * warped_mask_3ch
                full_mask = np.maximum(full_mask, warped_mask)

            result = np.clip(result, 0, 255).astype(np.uint8)
            imwrite_auto(merged_dir / frame_path.name, result)

            mask_out = (full_mask * 255).astype(np.uint8)
            imwrite_auto(mask_dir / frame_path.name, mask_out)

            count += 1
            if progress_callback is not None:
                elapsed = time.time() - start_time
                remaining = (elapsed / count) * (total - count) if count > 0 else 0
                speed = f"{count / elapsed:.1f}it/s" if elapsed > 0 else ""
                progress_callback(i + 1, total, elapsed, remaining, speed)

            if stop_check is not None and stop_check():
                break

        _logger.info(f"Composited {count} frames to {merged_dir}")
        return count

    def _composite_frame_level(
        self,
        swapped_dir: Path,
        merged_dir: Path,
        progress_callback: Optional[Callable],
        stop_check: Optional[Callable[[], bool]],
    ) -> int:
        swapped_images = FileManager.find_images(swapped_dir)
        total = len(swapped_images)
        count = 0
        start_time = time.time()

        for i, img_path in enumerate(swapped_images):
            img = cv2.imread(str(img_path))
            if img is not None:
                imwrite_auto(merged_dir / img_path.name, img)
            count += 1

            if progress_callback is not None:
                elapsed = time.time() - start_time
                remaining = (elapsed / count) * (total - count) if count > 0 else 0
                speed = f"{count / elapsed:.1f}it/s" if elapsed > 0 else ""
                progress_callback(i + 1, total, elapsed, remaining, speed)

            if stop_check is not None and stop_check():
                break

        _logger.info(f"Copied {count} frame-level results to {merged_dir}")
        return count

    def _create_aligned_mask(
        self,
        meta: FaceMetadata,
        config: MergeConfig,
        output_size: int,
    ) -> np.ndarray:
        h = w = output_size
        mask = np.zeros((h, w), dtype=np.float32)

        if config.mask_mode in (MaskMode.XSEG, MaskMode.LEARNED):
            if meta.xseg_mask is not None:
                xseg_arr = meta.get_xseg_mask_array(h, w)
                if xseg_arr is not None and np.any(xseg_arr):
                    mask = xseg_arr.astype(np.float32) / 255.0
                    mask = self._apply_mask_modifiers(mask, config)
                    return mask
            if meta.seg_ie_polys is not None and len(meta.seg_ie_polys) > 0:
                mask_uint8 = np.zeros((h, w), dtype=np.uint8)
                for poly_data in meta.seg_ie_polys:
                    pts_data = poly_data.get("pts", [])
                    poly_type = poly_data.get("type", 1)
                    if len(pts_data) < 3:
                        continue
                    pts = np.array(pts_data, dtype=np.int32)
                    if poly_type == 1:
                        cv2.fillPoly(mask_uint8, [pts], 255)
                    else:
                        cv2.fillPoly(mask_uint8, [pts], 0)
                mask = mask_uint8.astype(np.float32) / 255.0
                if np.max(mask) > 0:
                    mask = self._apply_mask_modifiers(mask, config)
                    return mask

        cx, cy = w / 2.0, h / 2.0
        rx = w * 0.42
        ry = h * 0.46
        Y, X = np.ogrid[:h, :w]
        ellipse = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        mask = np.clip(1.0 - ellipse, 0.0, 1.0)
        mask = (mask > 0).astype(np.float32)

        mask = self._apply_mask_modifiers(mask, config)
        return mask

    def _apply_mask_modifiers(self, mask: np.ndarray, config: MergeConfig) -> np.ndarray:
        h, w = mask.shape[:2]

        if config.erode_mask_modifier != 0.0:
            erode_px = int(abs(config.erode_mask_modifier) * min(h, w) / 100)
            if erode_px > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
                if config.erode_mask_modifier > 0:
                    mask = cv2.dilate(mask, kernel, iterations=1)
                else:
                    mask = cv2.erode(mask, kernel, iterations=1)

        if config.blur_mask_modifier > 0.0:
            blur_px = max(1, int(config.blur_mask_modifier * min(h, w) / 100))
            if blur_px % 2 == 0:
                blur_px += 1
            mask = cv2.GaussianBlur(mask, (blur_px, blur_px), 0)

        mask = np.clip(mask, 0.0, 1.0)
        return mask

    @staticmethod
    def _get_mask(shape, meta, config, aligned_dir, filename):
        h, w = shape[:2]
        if config.mask_mode in (MaskMode.XSEG, MaskMode.LEARNED) and meta is not None:
            if meta.xseg_mask is not None:
                xseg_arr = meta.get_xseg_mask_array(h, w)
                if xseg_arr is not None:
                    return xseg_arr
            if meta.seg_ie_polys is not None:
                mask = np.zeros((h, w), dtype=np.uint8)
                for poly_data in meta.seg_ie_polys:
                    pts_data = poly_data.get("pts", [])
                    poly_type = poly_data.get("type", 1)
                    if len(pts_data) < 3:
                        continue
                    pts = np.array(pts_data, dtype=np.int32)
                    if poly_type == 1:
                        cv2.fillPoly(mask, [pts], 255)
                    else:
                        cv2.fillPoly(mask, [pts], 0)
                return mask
        return np.full((h, w), 255, dtype=np.uint8)

    @staticmethod
    def _conditional_match_color(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        hist_factor = ModelMerger._calc_histogram_difference(source, target)
        corrected = ModelMerger._match_frame_color(source, target)
        result = cv2.addWeighted(target, hist_factor, corrected, 1.0 - hist_factor, 0)
        return result

    @staticmethod
    def _match_frame_color(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        h = target.shape[0]
        sizes = np.linspace(16, h, 3, endpoint=False).astype(int)
        src = source.copy()
        for sz in sizes:
            sz = max(sz, 2)
            src = ModelMerger._equalize_frame_color(src, target, (sz, sz))
        result = ModelMerger._equalize_frame_color(src, target, (target.shape[1], target.shape[0]))
        return result

    @staticmethod
    def _equalize_frame_color(source: np.ndarray, target: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        src_down = cv2.resize(source, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        tgt_down = cv2.resize(target, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        diff = np.subtract(src_down, tgt_down)
        diff_up = cv2.resize(diff, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)
        result = np.add(target.astype(np.float32), diff_up).clip(0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _calc_histogram_difference(source: np.ndarray, target: np.ndarray) -> float:
        hsv_src = bgr_to_hsv(source)
        hsv_tgt = bgr_to_hsv(target)
        hist_src = cv2.calcHist([hsv_src], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_tgt = cv2.calcHist([hsv_tgt], [0, 1], None, [50, 60], [0, 180, 0, 256])
        correl = cv2.compareHist(hist_src, hist_tgt, cv2.HISTCMP_CORREL)
        return float(np.interp(correl, [-1, 1], [0, 1]))
