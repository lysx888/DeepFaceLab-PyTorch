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
    LEARNED_PRD = "learned-prd"
    LEARNED_DST = "learned-dst"
    LEARNED_PRD_DST = "learned-prd*dst"


@dataclass
class MergeConfig:
    mask_mode: MaskMode = MaskMode.XSEG
    erode_mask_modifier: float = 0.0
    blur_mask_modifier: float = 0.0
    output_size: int = 256
    seamless_clone: bool = False
    color_transfer: bool = True
    color_transfer_mode: str = "none"
    output_face_scale: float = 0.0
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

                learned_prd_mask = None
                learned_dst_mask = None
                if config.mask_mode in (MaskMode.LEARNED, MaskMode.LEARNED_PRD, MaskMode.LEARNED_DST, MaskMode.LEARNED_PRD_DST):
                    prd_mask_path = swapped_path.parent / (swapped_path.stem + "_mask_prd.png")
                    dst_mask_path = swapped_path.parent / (swapped_path.stem + "_mask_dst.png")
                    if prd_mask_path.exists():
                        m = cv2.imread(str(prd_mask_path), cv2.IMREAD_GRAYSCALE)
                        if m is not None:
                            learned_prd_mask = (m.astype(np.float32) / 255.0)
                    if dst_mask_path.exists():
                        m = cv2.imread(str(dst_mask_path), cv2.IMREAD_GRAYSCALE)
                        if m is not None:
                            learned_dst_mask = (m.astype(np.float32) / 255.0)

                if meta.image_to_face_mat is None:
                    _logger.warning(f"No transform matrix for {face_name}")
                    continue

                face_to_image_mat = cv2.invertAffineTransform(meta.image_to_face_mat)

                output_size = meta.output_size
                mask = self._create_aligned_mask(meta, config, output_size, learned_prd_mask, learned_dst_mask)

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

    def composite_single_frame(
        self,
        frame_img: np.ndarray,
        faces: list[tuple[str, FaceMetadata]],
        predictor_func: Callable,
        config: MergeConfig,
        dst_aligned_dir: Optional[Path] = None,
    ) -> np.ndarray:
        frame_h, frame_w = frame_img.shape[:2]
        result = frame_img.copy().astype(np.float32)

        for face_name, meta in faces:
            if meta.image_to_face_mat is None:
                continue

            face_mat = meta.image_to_face_mat
            output_size = meta.output_size or config.output_size

            dst_face = cv2.warpAffine(
                frame_img, face_mat, (output_size, output_size),
                flags=cv2.INTER_CUBIC,
            )

            try:
                pred_bgr, pred_mask_prd, pred_mask_dst = predictor_func(dst_face)
            except Exception as e:
                _logger.warning(f"predictor_func failed for {face_name}: {e}")
                continue

            swapped_bgr = np.clip(pred_bgr * 255.0, 0, 255).astype(np.uint8)
            if swapped_bgr.shape[0] != output_size:
                swapped_bgr = cv2.resize(swapped_bgr, (output_size, output_size),
                                         interpolation=cv2.INTER_LANCZOS4)

            if abs(config.output_face_scale) > 1e-6:
                scale = 1.0 + 0.01 * config.output_face_scale
                new_size = max(32, int(round(output_size * scale)))
                swapped_bgr = cv2.resize(swapped_bgr, (new_size, new_size), interpolation=cv2.INTER_LANCZOS4)
                if new_size > output_size:
                    off = (new_size - output_size) // 2
                    swapped_bgr = swapped_bgr[off:off+output_size, off:off+output_size]
                elif new_size < output_size:
                    pad = (output_size - new_size) // 2
                    padded = dst_face.copy()
                    padded[pad:pad+new_size, pad:pad+new_size] = swapped_bgr
                    swapped_bgr = padded

            mask = self._select_merge_mask(
                meta, config, output_size, pred_mask_prd, pred_mask_dst)
            if mask is None:
                continue

            mask = self._apply_mask_modifiers(mask, config)

            if config.color_transfer and config.color_transfer_mode != "none":
                swapped_bgr = self._apply_color_transfer(
                    swapped_bgr, dst_face, mask, config.color_transfer_mode)
            elif config.color_transfer:
                swapped_bgr = self._conditional_match_color(dst_face, swapped_bgr)

            if config.enhancer_model:
                enhancer = self._get_enhancer(config.enhancer_model, config.enhancer_blend)
                swapped_bgr = enhancer.enhance(swapped_bgr, (mask * 255).astype(np.uint8))

            face_to_image_mat = cv2.invertAffineTransform(face_mat)

            crop_h, crop_w = swapped_bgr.shape[:2]
            crop_pts = np.array([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]], dtype=np.float32)
            paste_pts = cv2.transform(crop_pts.reshape(1, -1, 2), face_to_image_mat).reshape(-1, 2)
            x1, y1 = int(np.floor(paste_pts[:, 0].min())), int(np.floor(paste_pts[:, 1].min()))
            x2, y2 = int(np.ceil(paste_pts[:, 0].max())), int(np.ceil(paste_pts[:, 1].max()))
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame_w), min(y2, frame_h)

            if x2 > x1 and y2 > y1:
                paste_mat = face_to_image_mat.copy()
                paste_mat[0, 2] -= x1
                paste_mat[1, 2] -= y1
                pw, ph = x2 - x1, y2 - y1

                warped_mask = cv2.warpAffine(mask, paste_mat, (pw, ph), flags=cv2.INTER_CUBIC)
                warped_mask = np.clip(warped_mask, 0.0, 1.0)
                warped_face = cv2.warpAffine(
                    swapped_bgr.astype(np.float32), paste_mat, (pw, ph),
                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
                )

                wm3 = np.stack([warped_mask] * 3, axis=-1)
                paste_region = result[y1:y2, x1:x2]
                paste_region = paste_region * (1.0 - wm3) + warped_face * wm3
                result[y1:y2, x1:x2] = paste_region

        return np.clip(result, 0, 255).astype(np.uint8)

    def _select_merge_mask(
        self,
        meta: FaceMetadata,
        config: MergeConfig,
        output_size: int,
        pred_mask_prd: Optional[np.ndarray] = None,
        pred_mask_dst: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        h = w = output_size
        mode = config.mask_mode

        def _resize_mask(m):
            if m is None:
                return None
            m = np.clip(m, 0.0, 1.0)
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            return m.astype(np.float32)

        if mode in (MaskMode.LEARNED, MaskMode.LEARNED_PRD, MaskMode.LEARNED_PRD_DST):
            m = _resize_mask(pred_mask_prd)
            if m is not None and np.any(m):
                if mode == MaskMode.LEARNED_PRD_DST:
                    m_dst = _resize_mask(pred_mask_dst)
                    if m_dst is not None:
                        m *= m_dst
                return m

        if mode == MaskMode.LEARNED_DST:
            m = _resize_mask(pred_mask_dst)
            if m is not None and np.any(m):
                return m

        if mode == MaskMode.DST:
            m = _resize_mask(pred_mask_dst)
            if m is not None and np.any(m):
                return m

        if mode in (MaskMode.XSEG, MaskMode.LEARNED):
            if meta.xseg_mask is not None:
                xseg_arr = meta.get_xseg_mask_array(h, w)
                if xseg_arr is not None and np.any(xseg_arr):
                    return xseg_arr.astype(np.float32) / 255.0
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
                if np.any(mask_uint8):
                    return mask_uint8.astype(np.float32) / 255.0

        cx, cy = w / 2.0, h / 2.0
        rx, ry = w * 0.42, h * 0.46
        Y, X = np.ogrid[:h, :w]
        ellipse = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        return np.clip(1.0 - ellipse, 0.0, 1.0)

    @staticmethod
    def _apply_color_transfer(
        swapped: np.ndarray,
        dst_face: np.ndarray,
        mask: np.ndarray,
        ct_mode: str,
    ) -> np.ndarray:
        from faceswap.core.color_transfer import color_transfer
        src_f = dst_face.astype(np.float32) / 255.0
        trg_f = swapped.astype(np.float32) / 255.0
        mask_u8 = (mask * 255).astype(np.uint8)
        try:
            out = color_transfer(ct_mode, trg_f, src_f, src_mask=mask_u8, trg_mask=mask_u8)
        except Exception:
            return swapped
        if np.isnan(out).any() or np.isinf(out).any():
            return swapped
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

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
        learned_prd_mask: Optional[np.ndarray] = None,
        learned_dst_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        h = w = output_size
        mask = np.zeros((h, w), dtype=np.float32)

        if config.mask_mode in (MaskMode.LEARNED, MaskMode.LEARNED_PRD, MaskMode.LEARNED_PRD_DST):
            if learned_prd_mask is not None and np.any(learned_prd_mask):
                mask = learned_prd_mask.astype(np.float32)
                if config.mask_mode == MaskMode.LEARNED_PRD_DST and learned_dst_mask is not None:
                    mask *= learned_dst_mask.astype(np.float32)
                mask = self._apply_mask_modifiers(mask, config)
                return mask

        if config.mask_mode == MaskMode.LEARNED_DST:
            if learned_dst_mask is not None and np.any(learned_dst_mask):
                mask = learned_dst_mask.astype(np.float32)
                mask = self._apply_mask_modifiers(mask, config)
                return mask

        if config.mask_mode in (MaskMode.XSEG, MaskMode.LEARNED, MaskMode.LEARNED_PRD, MaskMode.LEARNED_PRD_DST, MaskMode.LEARNED_DST):
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
        ero = int(config.erode_mask_modifier)
        blur = int(config.blur_mask_modifier)

        if ero == 0 and blur == 0:
            return np.clip(mask, 0.0, 1.0).astype(np.float32)

        pad = max(abs(ero), blur) + 1
        mask_p = np.pad(mask, pad)

        if ero > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero, ero))
            mask_p = cv2.erode(mask_p, k, iterations=1)
        elif ero < 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-ero, -ero))
            mask_p = cv2.dilate(mask_p, k, iterations=1)

        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            mask_p = cv2.GaussianBlur(mask_p, (blur, blur), 0)

        mask = mask_p[pad:pad+h, pad:pad+w]
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _get_mask(shape, meta, config, aligned_dir, filename,
                  learned_prd_mask=None, learned_dst_mask=None):
        h, w = shape[:2]
        if config.mask_mode in (MaskMode.LEARNED, MaskMode.LEARNED_PRD, MaskMode.LEARNED_PRD_DST) and learned_prd_mask is not None:
            mask = learned_prd_mask.astype(np.float32)
            if config.mask_mode == MaskMode.LEARNED_PRD_DST and learned_dst_mask is not None:
                mask *= learned_dst_mask.astype(np.float32)
            return (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        if config.mask_mode == MaskMode.LEARNED_DST and learned_dst_mask is not None:
            return (np.clip(learned_dst_mask, 0, 1) * 255).astype(np.uint8)
        if config.mask_mode in (MaskMode.XSEG, MaskMode.LEARNED, MaskMode.LEARNED_PRD, MaskMode.LEARNED_PRD_DST, MaskMode.LEARNED_DST) and meta is not None:
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
