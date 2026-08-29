import collections
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from faceswap.core.metadata_manager import MetadataManager, FaceMetadata
from faceswap.core.color_transfer import color_transfer
from faceswap.core.landmarks106 import fill_hull_mask_106, LANDMARK_GROUPS_106
from faceswap.setting import FaceType
from faceswap.shared.file_manager import FileManager
from faceswap.shared.image_utils import bgr_to_hsv, hsv_to_bgr
from faceswap.shared.logger import get_logger

_logger = get_logger("saehd_dataset")

MIN_MASK_AREA_RATIO = 0.01
ELLIPSE_FALLBACK_W_RATIO = 0.38
ELLIPSE_FALLBACK_H_RATIO = 0.48
WARP_ROTATION_DEGREES = 10
WARP_SCALE_FACTOR = 1.15
WARP_TRANSLATION_FRAC = 0.05
WARP_MAX_OFFSET_FACTOR = 0.72
WARP_OFFSET_STD_FACTOR = 0.24
HSV_HUE_SHIFT_DEGREES = 90
MAX_IMAGE_CACHE = 1024
MIN_IMAGE_CACHE = 128
MAX_META_CACHE = 4096
MIN_META_CACHE = 256
DEGENERATE_MEAN_MIN = 5.0
DEGENERATE_MEAN_MAX = 250.0
DEGENERATE_STD_MIN = 5.0


class SAEHDDataset(Dataset):
    def __init__(
        self,
        aligned_dir: Path,
        resolution: int = 128,
        face_type: str = 'wf',
        random_warp: bool = True,
        random_flip: bool = True,
        random_hsv_power: float = 0.0,
        ct_mode: str = 'none',
        uniform_yaw: bool = False,
        is_src: bool = True,
        augment: bool = True,
        scale_range: tuple[float, float] = (-0.15, 0.15),
        need_em_mask: bool = False,
        need_vis_mask: bool = False,
    ) -> None:
        self._aligned_dir = Path(aligned_dir)
        self._resolution = resolution
        self._face_type = face_type
        self._random_warp = random_warp
        self._random_flip = random_flip
        self._random_hsv_power = random_hsv_power
        self._ct_mode = ct_mode if ct_mode != 'none' else None
        self._uniform_yaw = uniform_yaw
        self._is_src = is_src
        self._augment = augment
        self._scale_range = scale_range
        self._need_em_mask = need_em_mask
        self._need_vis_mask = need_vis_mask

        self._image_paths: list[Path] = []
        self._image_cache: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()
        self._metadata_cache: collections.OrderedDict[str, FaceMetadata] = collections.OrderedDict()

        all_paths = list(FileManager.find_images(self._aligned_dir))
        if not all_paths:
            raise ValueError(f"No images found in {self._aligned_dir}")

        self._image_paths = self._filter_degenerate(all_paths)

        if not self._image_paths:
            _logger.warning(f"All images in {self._aligned_dir} are degenerate, keeping all")
            self._image_paths = all_paths

        n_images = len(self._image_paths)
        self._image_cache_max = max(MIN_IMAGE_CACHE, min(n_images, MAX_IMAGE_CACHE))
        self._meta_cache_max = max(MIN_META_CACHE, min(n_images, MAX_META_CACHE))

        self._yaw_values: dict[str, float] = MetadataManager.load_yaw_only(self._aligned_dir)
        self._yaw_weights: list[float] = []
        if uniform_yaw:
            self._compute_yaw_weights()

        self._dst_sample_for_ct: np.ndarray | None = None
        self._dst_mask_for_ct: np.ndarray | None = None
        _logger.info(f"SAEHDDataset: {len(self._image_paths)} images from {self._aligned_dir}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_metadata_cache'] = collections.OrderedDict()
        state['_image_cache'] = collections.OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_meta(self, img_path: Path) -> FaceMetadata | None:
        key = img_path.name
        meta = self._metadata_cache.get(key)
        if meta is not None:
            self._metadata_cache.move_to_end(key)
            return meta
        meta = MetadataManager.load(img_path, lightweight=False)
        if meta is not None:
            if len(self._metadata_cache) >= self._meta_cache_max:
                self._metadata_cache.popitem(last=False)
            self._metadata_cache[key] = meta
        return meta

    def _compute_yaw_weights(self) -> None:
        yaws = []
        for p in self._image_paths:
            yaw = self._yaw_values.get(p.name)
            if yaw is not None:
                yaws.append(abs(yaw))
            else:
                yaws.append(0.0)
        if not yaws:
            return
        yaws = np.array(yaws)
        raw_max = yaws.max() if len(yaws) > 0 else 0.0
        yaws = np.clip(yaws, 0.0, 70.0)
        n = len(yaws)
        if n > 5000:
            n_bins = 128
        elif n < 500:
            n_bins = 5
        elif n < 2000:
            n_bins = 10
        else:
            n_bins = 15
        bin_max = 70.0
        bin_edges = np.linspace(0.0, bin_max, n_bins + 1)
        bin_edges[-1] += 1e-6
        img_bins = np.digitize(yaws, bin_edges[1:-1])
        bin_counts = np.bincount(img_bins, minlength=n_bins)
        with np.errstate(divide='ignore'):
            bin_weights = np.where(bin_counts > 0, 1.0 / bin_counts, 0.0)
        self._yaw_weights = bin_weights[img_bins].tolist()
        nonzero = bin_counts[bin_counts > 0]
        _logger.info(
            f"uniform_yaw: {n} samples, {n_bins} equal-width bins over [0,{bin_max:.1f}], "
            f"{len(nonzero)} non-empty bins, "
            f"yaw range [{yaws.min():.3f}, {raw_max:.3f}] (clipped to {bin_max:.1f}), "
            f"bin sizes min={nonzero.min()} max={nonzero.max()}"
        )

    def set_dst_sample_for_ct(self, dst_image: np.ndarray, dst_mask: np.ndarray = None) -> None:
        self._dst_sample_for_ct = dst_image
        self._dst_mask_for_ct = dst_mask

    def set_ct_shared(self, ct_img_shared, ct_mask_shared, ct_valid_shared) -> None:
        self._ct_img_shared = ct_img_shared
        self._ct_mask_shared = ct_mask_shared
        self._ct_valid_shared = ct_valid_shared

    def __len__(self) -> int:
        return len(self._image_paths)

    @property
    def image_paths(self) -> list[Path]:
        return self._image_paths

    @property
    def yaw_weights(self) -> list[float] | None:
        return self._yaw_weights

    def _filter_degenerate(self, paths: list[Path]) -> list[Path]:
        valid = []
        skipped = 0
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped += 1
                continue
            mean = img.mean()
            std = img.std()
            if mean < DEGENERATE_MEAN_MIN or mean > DEGENERATE_MEAN_MAX or std < DEGENERATE_STD_MIN:
                skipped += 1
                _logger.warning(f"Skipping degenerate image: {p.name} (mean={mean:.1f}, std={std:.1f})")
                continue
            valid.append(p)
        if skipped > 0:
            _logger.info(f"Filtered {skipped} degenerate images from {self._aligned_dir} "
                         f"({len(valid)}/{len(paths)} kept)")
        return valid

    def _read_image(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self._image_cache:
            self._image_cache.move_to_end(key)
            return self._image_cache[key]
        img = cv2.imread(str(path))
        if img is not None:
            if len(self._image_cache) >= self._image_cache_max:
                self._image_cache.popitem(last=False)
            self._image_cache[key] = img
        return img

    def get_preview_mask(self, idx: int, target_size: tuple[int, int]) -> np.ndarray:
        real_idx = idx % len(self._image_paths)
        img_path = self._image_paths[real_idx]
        img = self._read_image(img_path)
        if img is None:
            return np.zeros(target_size, dtype=np.float32)
        meta = self._get_meta(img_path)
        mask = self._render_full_mask(img.shape[:2], meta)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)
        return mask.astype(np.float32) / 255.0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        real_idx = idx % len(self._image_paths)
        img_path = self._image_paths[real_idx]

        img = self._read_image(img_path)
        if img is None:
            img = np.zeros((self._resolution, self._resolution, 3), dtype=np.uint8)

        meta = self._get_meta(img_path)

        full_mask = self._render_full_mask(img.shape[:2], meta)
        em_mask = self._render_em_mask(img.shape[:2], meta) if self._need_em_mask else None
        vis_mask = self._render_visibility_mask(img.shape[:2], meta) if self._need_vis_mask else None

        res = self._resolution
        img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
        full_mask = cv2.resize(full_mask, (res, res), interpolation=cv2.INTER_LINEAR)
        if em_mask is not None:
            em_mask = cv2.resize(em_mask, (res, res), interpolation=cv2.INTER_LINEAR)
        if vis_mask is not None:
            vis_mask = cv2.resize(vis_mask, (res, res), interpolation=cv2.INTER_LINEAR)

        if self._random_hsv_power > 0 and self._is_src:
            img = self._apply_random_hsv(img, self._random_hsv_power)

        if self._ct_mode is not None and self._is_src:
            ct_img = None
            ct_mask = None
            if hasattr(self, '_ct_valid_shared') and self._ct_valid_shared.item() == 1:
                ct_img = self._ct_img_shared.cpu().numpy()
                ct_mask = self._ct_mask_shared.cpu().numpy()
            elif self._dst_sample_for_ct is not None:
                ct_img = self._dst_sample_for_ct
                ct_mask = self._dst_mask_for_ct
            if ct_img is not None:
                img_f = img.astype(np.float32) / 255.0
                dst_f = ct_img.astype(np.float32) / 255.0
                src_mask_f = full_mask.astype(np.float32) / 255.0
                dst_mask_f = (ct_mask.astype(np.float32) / 255.0
                             if ct_mask is not None else None)
                try:
                    img_ct = color_transfer(self._ct_mode, img_f, dst_f,
                                            src_mask=src_mask_f, trg_mask=dst_mask_f)
                    img = np.clip(img_ct * 255.0, 0, 255).astype(np.uint8)
                except Exception as e:
                    _logger.warning(f"color_transfer '{self._ct_mode}' failed: {e}, using original image")

        target_image = img.astype(np.float32) / 255.0
        full_mask_f = full_mask.astype(np.float32) / 255.0
        em_mask_f = em_mask.astype(np.float32) / 255.0 if em_mask is not None else np.zeros((res, res), dtype=np.float32)

        if self._augment:
            flip = self._random_flip and random.randint(0, 9) < 4
            rotation = random.uniform(-WARP_ROTATION_DEGREES, WARP_ROTATION_DEGREES)
            scale = random.uniform(1.0 / WARP_SCALE_FACTOR, WARP_SCALE_FACTOR)
            tx = random.uniform(-WARP_TRANSLATION_FRAC, WARP_TRANSLATION_FRAC)
            ty = random.uniform(-WARP_TRANSLATION_FRAC, WARP_TRANSLATION_FRAC)
        else:
            flip = False
            rotation = 0.0
            scale = 1.0
            tx = ty = 0.0

        if flip:
            target_image = target_image[:, ::-1].copy()
            full_mask_f = full_mask_f[:, ::-1].copy()
            em_mask_f = em_mask_f[:, ::-1].copy()
            if vis_mask is not None:
                vis_mask = vis_mask[:, ::-1].copy()

        rmat = cv2.getRotationMatrix2D((res / 2.0, res / 2.0), rotation, scale)
        rmat[:, 2] += (tx * res, ty * res)

        target_image = cv2.warpAffine(target_image, rmat, (res, res), borderMode=cv2.BORDER_REPLICATE)
        full_mask_f = cv2.warpAffine(full_mask_f, rmat, (res, res), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        em_mask_f = cv2.warpAffine(em_mask_f, rmat, (res, res), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        if vis_mask is not None:
            vis_mask = cv2.warpAffine(vis_mask, rmat, (res, res), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        if self._random_warp:
            warped_image = self._apply_grid_warp(target_image)
        else:
            warped_image = target_image.copy()

        target_image_t = torch.from_numpy(target_image.transpose(2, 0, 1)).float()
        warped_image_t = torch.from_numpy(warped_image.transpose(2, 0, 1)).float()
        full_mask_t = torch.from_numpy(full_mask_f).unsqueeze(0)
        em_mask_t = torch.from_numpy(em_mask_f).unsqueeze(0)
        vis_mask_t = torch.from_numpy(vis_mask.astype(np.float32)).unsqueeze(0) if vis_mask is not None else torch.zeros(1, res, res)

        return {
            'warped_image': warped_image_t,
            'target_image': target_image_t,
            'target_mask': full_mask_t,
            'target_em_mask': em_mask_t,
            'target_vis_mask': vis_mask_t,
        }

    def _render_visibility_mask(self, shape: tuple, meta: FaceMetadata | None) -> np.ndarray:
        h, w = shape[:2]
        if meta is None or meta.landmarks_106 is None or len(meta.landmarks_106) < 3:
            return np.ones((h, w), dtype=np.float32)

        vis = meta.landmarks_106_visibility
        if all(v for v in vis):
            return np.ones((h, w), dtype=np.float32)

        lm = meta.landmarks_106
        scale_x = w / meta.output_size if meta.output_size > 0 and meta.output_size != w else 1.0
        scale_y = h / meta.output_size if meta.output_size > 0 and meta.output_size != h else 1.0

        visible_pts = []
        for i, v in enumerate(vis):
            if v and i < len(lm):
                pt = (int(lm[i][0] * scale_x), int(lm[i][1] * scale_y))
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    visible_pts.append(pt)

        mask = np.zeros((h, w), dtype=np.uint8)
        if len(visible_pts) >= 3:
            hull = cv2.convexHull(np.array(visible_pts, dtype=np.float32))
            cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        elif len(visible_pts) >= 1:
            for pt in visible_pts:
                cv2.circle(mask, pt, max(w, h) // 8, 255, -1)

        blur_k = max(3, h // 16)
        blur_k += 1 - blur_k % 2
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
        return mask.astype(np.float32) / 255.0

    def _render_full_mask(self, shape: tuple, meta: FaceMetadata | None) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if meta is not None and meta.xseg_mask is not None:
            xseg_arr = meta.get_xseg_mask_array(h, w)
            if xseg_arr is not None and np.any(xseg_arr) and np.sum(xseg_arr > 0) / (h * w) > MIN_MASK_AREA_RATIO:
                return xseg_arr
            _logger.warning(f"xseg_mask decode failed or empty for {getattr(meta, 'source_filename', '?')}, "
                            f"falling back to seg_ie_polys")

        if meta is not None and meta.seg_ie_polys is not None:
            for poly_data in meta.seg_ie_polys:
                pts_data = poly_data.get("pts", [])
                poly_type = poly_data.get("type", 1)
                if len(pts_data) < 3:
                    continue
                scale_x = w / meta.output_size if meta.output_size > 0 and meta.output_size != w else 1.0
                scale_y = h / meta.output_size if meta.output_size > 0 and meta.output_size != h else 1.0
                pts = np.array(pts_data, dtype=np.float32)
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                pts = pts.astype(np.int64)
                if poly_type == 1:
                    cv2.fillPoly(mask, [pts], 255)
                else:
                    cv2.fillPoly(mask, [pts], 0)
            if np.any(mask) and np.sum(mask > 0) / (h * w) > MIN_MASK_AREA_RATIO:
                return mask
            _logger.warning(f"seg_ie_polys rendered empty for {getattr(meta, 'source_filename', '?')}, "
                            f"falling back to landmarks")

        if meta is not None and meta.landmarks_106 is not None and len(meta.landmarks_106) >= 68:
            lm = meta.landmarks_106.astype(np.float32).copy()
            scale_x = w / meta.output_size if meta.output_size > 0 else 1.0
            scale_y = h / meta.output_size if meta.output_size > 0 else 1.0
            lm[:, 0] *= scale_x
            lm[:, 1] *= scale_y
            base_mod = 2.0 if meta.face_type == FaceType.HEAD else 1.5
            yaw = meta.pose[1] if meta.pose is not None else 0.0
            yaw_factor = 0.3 + 0.7 * math.cos(yaw * math.pi / 2)
            fill_hull_mask_106(mask, lm, eyebrows_expand_mod=base_mod * yaw_factor)
            if np.any(mask):
                return mask

        cv2.ellipse(mask, (w // 2, h // 2), (int(w * ELLIPSE_FALLBACK_W_RATIO), int(h * ELLIPSE_FALLBACK_H_RATIO)), 0, 0, 360, 255, -1)
        return mask

    def _render_em_mask(self, shape: tuple, meta: FaceMetadata | None) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if meta is None or meta.landmarks_106 is None:
            return mask
        lm = meta.landmarks_106
        if len(lm) < 106:
            return mask
        scale_x = w / meta.output_size if meta.output_size > 0 else 1.0
        scale_y = h / meta.output_size if meta.output_size > 0 else 1.0
        lm_scaled = lm.astype(np.float32).copy()
        lm_scaled[:, 0] *= scale_x
        lm_scaled[:, 1] *= scale_y

        _lm_groups = dict(LANDMARK_GROUPS_106)
        eye_l = lm_scaled[_lm_groups["left_eye"]].astype(np.int32)
        eye_r = lm_scaled[_lm_groups["right_eye"]].astype(np.int32)
        mouth = lm_scaled[_lm_groups["outer_lip"] + _lm_groups["inner_lip"]].astype(np.int32)

        cv2.fillPoly(mask, [eye_l], 255)
        cv2.fillPoly(mask, [eye_r], 255)
        cv2.fillPoly(mask, [mouth], 255)
        dilate_k = max(1, h // 32)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)), iterations=1)
        blur_k = max(1, h // 16)
        blur_k += (1 - blur_k % 2)
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
        return mask

    def _apply_grid_warp(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        w_work = self._resolution

        cell_size_choices = [w_work // (2 ** i) for i in range(1, 4) if w_work // (2 ** i) >= 2]
        if not cell_size_choices:
            cell_size_choices = [2]
        cell_size = random.choice(cell_size_choices)
        cell_count = w_work // cell_size + 1
        grid_points = np.linspace(0, w_work, cell_count)

        mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
        mapy = mapx.T.copy()

        inner_size = cell_count - 2
        if inner_size > 0:
            max_offset = cell_size * WARP_MAX_OFFSET_FACTOR
            dx = np.clip(np.random.normal(0, 1, (inner_size, inner_size)) * (cell_size * WARP_OFFSET_STD_FACTOR),
                         -max_offset, max_offset)
            dy = np.clip(np.random.normal(0, 1, (inner_size, inner_size)) * (cell_size * WARP_OFFSET_STD_FACTOR),
                         -max_offset, max_offset)
            mapx[1:-1, 1:-1] += dx
            mapy[1:-1, 1:-1] += dy

        half_cell = cell_size // 2
        mapx = cv2.resize(mapx, (w_work + cell_size,) * 2)[half_cell:-half_cell, half_cell:-half_cell].astype(np.float32)
        mapy = cv2.resize(mapy, (w_work + cell_size,) * 2)[half_cell:-half_cell, half_cell:-half_cell].astype(np.float32)

        img = cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return np.clip(img, 0.0, 1.0)

    def _apply_random_hsv(self, img: np.ndarray, power: float) -> np.ndarray:
        h_amount = max(1, int(360 * power * 0.5))
        hsv = bgr_to_hsv(img).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-h_amount, h_amount)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.uniform(-power * 255, power * 255), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.uniform(-power * 255, power * 255), 0, 255)
        return hsv_to_bgr(hsv.astype(np.uint8))
