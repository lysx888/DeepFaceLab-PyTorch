import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from DeepFaceLab.core.metadata_manager import MetadataManager, FaceMetadata
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("saehd_dataset")


def _random_warp_nonlinear(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nonlinear grid warp only (affine transform is applied separately with shared params)."""
    h, w = img.shape[:2]
    cell_size = random.choice([w // 2, w // 4, max(w // 8, 2)])
    cell_count = w // cell_size + 1

    grid_x = np.zeros((cell_count + 1, cell_count + 1), dtype=np.float32)
    grid_y = np.zeros((cell_count + 1, cell_count + 1), dtype=np.float32)
    for i in range(cell_count + 1):
        for j in range(cell_count + 1):
            grid_x[i, j] = j * cell_size
            grid_y[i, j] = i * cell_size

    noise_scale = cell_size * 0.24
    grid_x += np.random.uniform(-noise_scale, noise_scale, grid_x.shape).astype(np.float32)
    grid_y += np.random.uniform(-noise_scale, noise_scale, grid_y.shape).astype(np.float32)

    mapx = cv2.resize(grid_x, (w, h), interpolation=cv2.INTER_CUBIC)
    mapy = cv2.resize(grid_y, (w, h), interpolation=cv2.INTER_CUBIC)

    img = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.remap(mask, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    return img, mask


def _random_hsv_dfl(img: np.ndarray, power: float) -> np.ndarray:
    if power <= 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_amount = max(1, int(179 * power * 0.5))
    hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-h_amount, h_amount + 1)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + (np.random.random() - 0.5) * power * 255, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + (np.random.random() - 0.5) * power * 255, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _color_transfer_rct(src: np.ndarray, dst_sample: np.ndarray) -> np.ndarray:
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float64)
    dst_lab = cv2.cvtColor(dst_sample, cv2.COLOR_BGR2LAB).astype(np.float64)
    result = np.empty_like(src_lab)
    for i in range(3):
        s_ch, d_ch = src_lab[:, :, i], dst_lab[:, :, i]
        s_mean, s_std = s_ch.mean(), s_ch.std() + 1e-6
        d_mean, d_std = d_ch.mean(), d_ch.std() + 1e-6
        result[:, :, i] = np.clip((s_ch - s_mean) / s_std * d_std + d_mean, 0, 255)
    return cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _color_transfer_mkl(src: np.ndarray, dst_sample: np.ndarray) -> np.ndarray:
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float64)
    dst_lab = cv2.cvtColor(dst_sample, cv2.COLOR_BGR2LAB).astype(np.float64)
    src_flat = src_lab.reshape(-1, 3)
    dst_flat = dst_lab.reshape(-1, 3)
    src_mean = src_flat.mean(axis=0)
    dst_mean = dst_flat.mean(axis=0)
    src_centered = src_flat - src_mean
    dst_centered = dst_flat - dst_mean
    src_cov = np.cov(src_centered, rowvar=False)
    dst_cov = np.cov(dst_centered, rowvar=False)
    src_std = np.linalg.cholesky(src_cov + 1e-6 * np.eye(3))
    dst_std = np.linalg.cholesky(dst_cov + 1e-6 * np.eye(3))
    transform = dst_std @ np.linalg.inv(src_std)
    result_flat = (src_centered @ transform.T) + dst_mean
    result = result_flat.reshape(src_lab.shape)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def _color_transfer_lct(src: np.ndarray, dst_sample: np.ndarray) -> np.ndarray:
    """Linear Color Transfer: full covariance matching (DFL lct mode)."""
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float64)
    dst_lab = cv2.cvtColor(dst_sample, cv2.COLOR_BGR2LAB).astype(np.float64)
    src_flat = src_lab.reshape(-1, 3)
    dst_flat = dst_lab.reshape(-1, 3)
    src_mean, dst_mean = src_flat.mean(0), dst_flat.mean(0)
    src_cov = np.cov(src_flat, rowvar=False)
    dst_cov = np.cov(dst_flat, rowvar=False)
    src_std = np.linalg.cholesky(src_cov + 1e-6 * np.eye(3))
    dst_std = np.linalg.cholesky(dst_cov + 1e-6 * np.eye(3))
    T = dst_std @ np.linalg.inv(src_std)
    result = ((src_flat - src_mean) @ T.T) + dst_mean
    return cv2.cvtColor(np.clip(result.reshape(src_lab.shape), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _color_transfer_idt(src: np.ndarray, dst_sample: np.ndarray) -> np.ndarray:
    """Iterative Distribution Transfer (DFL idt mode): histogram matching per channel."""
    result = np.empty_like(src)
    for c in range(3):
        src_ch = src[:, :, c]
        dst_ch = dst_sample[:, :, c]
        # Histogram matching
        src_hist, _ = np.histogram(src_ch.flatten(), 256, [0, 256])
        dst_hist, _ = np.histogram(dst_ch.flatten(), 256, [0, 256])
        src_cdf = src_hist.cumsum().astype(np.float64)
        dst_cdf = dst_hist.cumsum().astype(np.float64)
        src_cdf = src_cdf / src_cdf[-1]
        dst_cdf = dst_cdf / dst_cdf[-1]
        # Map src values to dst distribution
        lookup = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            j = np.searchsorted(dst_cdf, src_cdf[i])
            lookup[i] = min(j, 255)
        result[:, :, c] = lookup[src_ch]
    return result


def _color_transfer_sot(src: np.ndarray, dst_sample: np.ndarray) -> np.ndarray:
    """Sliced Optimal Transfer (DFL sot-m mode): 1D optimal transport per random projection."""
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float64)
    dst_lab = cv2.cvtColor(dst_sample, cv2.COLOR_BGR2LAB).astype(np.float64)
    src_flat = src_lab.reshape(-1, 3)
    dst_flat = dst_lab.reshape(-1, 3)
    n_proj = 10
    result_flat = src_flat.copy()
    for _ in range(n_proj):
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        src_proj = src_flat @ direction
        dst_proj = dst_flat @ direction
        src_order = np.argsort(src_proj)
        dst_order = np.argsort(dst_proj)
        # Map sorted src to sorted dst
        n_src, n_dst = len(src_proj), len(dst_proj)
        indices = np.interp(np.arange(n_src), np.linspace(0, n_dst - 1, n_dst), np.arange(n_dst)).astype(int)
        shift = dst_proj[dst_order[indices[src_order]]] - src_proj
        result_flat += shift[:, None] * direction[None, :]
    return cv2.cvtColor(np.clip(result_flat.reshape(src_lab.shape), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _render_face_mask(res: int, meta: FaceMetadata) -> np.ndarray:
    """Render full-face mask at training resolution from landmarks (DFL: on-the-fly)."""
    mask = np.zeros((res, res), dtype=np.uint8)
    if meta.landmarks_106 is None:
        return np.ones((res, res), dtype=np.uint8) * 255
    lm = meta.landmarks_106.astype(np.float64)
    scale_x = res / meta.output_size if meta.output_size != res else 1.0
    scale_y = res / meta.output_size if meta.output_size != res else 1.0
    lm[:, 0] *= scale_x
    lm[:, 1] *= scale_y
    hull = cv2.convexHull(lm.astype(np.float32))
    cv2.fillPoly(mask, [hull.astype(np.int64)], 255)
    return mask


def _render_mask_from_polys(res: int, meta: FaceMetadata) -> np.ndarray:
    """Render XSeg-style mask from polygons at training resolution."""
    mask = np.ones((res, res), dtype=np.uint8) * 255
    if meta.seg_ie_polys is None:
        return mask
    for poly_data in meta.seg_ie_polys:
        pts_data = poly_data.get("pts", [])
        poly_type = poly_data.get("type", 1)
        if len(pts_data) < 3:
            continue
        scale_x = res / meta.output_size if meta.output_size != res else 1.0
        scale_y = res / meta.output_size if meta.output_size != res else 1.0
        pts = np.array(pts_data, dtype=np.float32)
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        pts = pts.astype(np.int64)
        if poly_type == 1:
            cv2.fillPoly(mask, [pts], 255)
        else:
            cv2.fillPoly(mask, [pts], 0)
    return mask


def _render_eyes_mouth_mask(res: int, meta: FaceMetadata) -> np.ndarray:
    """Render eyes+mouth priority mask at training resolution (DFL: on-the-fly)."""
    mask = np.zeros((res, res), dtype=np.uint8)
    if meta.landmarks_106 is None:
        return mask
    lm = meta.landmarks_106.astype(np.float64)
    scale_x = res / meta.output_size if meta.output_size != res else 1.0
    scale_y = res / meta.output_size if meta.output_size != res else 1.0
    lm[:, 0] *= scale_x
    lm[:, 1] *= scale_y
    left_eye = lm[63:73].astype(np.int64)
    right_eye = lm[73:83].astype(np.int64)
    upper_lip = lm[83:93].astype(np.int64)
    lower_lip = lm[93:103].astype(np.int64)
    inner_lip = lm[103:106].astype(np.int64)
    mouth_pts = np.vstack([upper_lip, lower_lip, inner_lip])
    for pts in [left_eye, right_eye]:
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull], 255)
    mouth_hull = cv2.convexHull(mouth_pts)
    cv2.fillPoly(mask, [mouth_hull], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _generate_masks(res: int, meta: Optional[FaceMetadata]) -> tuple[np.ndarray, np.ndarray]:
    """Generate (face_mask, em_mask) at training resolution on-the-fly (DFL style)."""
    if meta is None:
        return np.ones((res, res), dtype=np.uint8) * 255, np.zeros((res, res), dtype=np.uint8)
    if meta.seg_ie_polys is not None:
        face_mask = _render_mask_from_polys(res, meta)
    elif meta.landmarks_106 is not None:
        face_mask = _render_face_mask(res, meta)
    else:
        face_mask = np.ones((res, res), dtype=np.uint8) * 255
    em_mask = _render_eyes_mouth_mask(res, meta)
    return face_mask, em_mask


class SAEHDDataset(Dataset):
    """
    DFL-style dataset: zero pixel caching, all masks generated on-the-fly.
    
    Memory footprint: only file paths + lightweight metadata (no image/mask arrays).
    Matches DFL's SubprocessGenerator approach where nothing is pre-cached.
    """

    def __init__(
        self,
        aligned_dir: Path,
        resolution: int = 128,
        is_src: bool = True,
        augment: bool = True,
        random_warp: bool = True,
        random_flip: bool = True,
        random_hsv_power: float = 0.0,
        random_ct: bool = False,
        ct_mode: str = "none",
        ct_sample_pool: Optional[list] = None,
        uniform_yaw: bool = False,
        src_face_scale: int = 0,
    ) -> None:
        self._aligned_dir = Path(aligned_dir)
        self._resolution = resolution
        self._is_src = is_src
        self._augment = augment
        self._random_warp = random_warp
        self._random_flip = random_flip
        self._random_hsv_power = random_hsv_power
        self._random_ct = random_ct
        self._ct_mode = ct_mode
        self._ct_sample_pool = ct_sample_pool or []
        self._uniform_yaw = uniform_yaw
        self._src_face_scale = src_face_scale

        # Only store paths + lightweight metadata — NO pixel data cached
        self._image_paths: list[Path] = FileManager.find_images(self._aligned_dir)
        self._metadata_cache: dict[str, FaceMetadata] = MetadataManager.load_all(
            self._aligned_dir, lightweight=True)

        if self._uniform_yaw:
            self._build_yaw_bins()

        if not self._image_paths:
            raise ValueError(f"No face images found in {self._aligned_dir}")

    def _build_yaw_bins(self) -> None:
        yaws = []
        for p in self._image_paths:
            meta = self._metadata_cache.get(p.name)
            yaw = meta.yaw if meta and meta.yaw is not None else 0.0
            yaws.append(yaw)
        n_bins = 20
        self._yaw_bins = [[] for _ in range(n_bins)]
        for idx, yaw in enumerate(yaws):
            bin_idx = min(int((yaw + 90) / 180 * n_bins), n_bins - 1)
            self._yaw_bins[bin_idx].append(idx)
        self._yaw_bins = [b for b in self._yaw_bins if b]

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._uniform_yaw and hasattr(self, '_yaw_bins') and self._yaw_bins:
            chosen_bin = random.choice(self._yaw_bins)
            idx = random.choice(chosen_bin)

        img_path = self._image_paths[idx]
        meta = self._metadata_cache.get(img_path.name)

        # Load image on-demand (DFL: load_bgr per sample, no caching)
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self._resolution, self._resolution, 3), dtype=np.uint8)

        # Resize to training resolution FIRST (DFL: warp at target res)
        res = self._resolution
        img = cv2.resize(img, (res, res))

        # Generate masks on-the-fly at training resolution (DFL: no pre-caching)
        mask, em_mask = _generate_masks(res, meta)

        target_img = img.copy()
        target_mask = mask.copy()
        target_em_mask = em_mask.copy()

        warped_img = img.copy()
        warped_mask = mask.copy()

        if self._augment:
            # Shared affine params (DFL: warped and target use same transform)
            h, w = res, res
            flip = self._random_flip and np.random.random() < 0.5
            angle = np.random.uniform(-10, 10)
            scale = np.random.uniform(0.85, 1.15)
            tx = np.random.uniform(-0.05, 0.05) * w
            ty = np.random.uniform(-0.05, 0.05) * h
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            M[:, 2] += [tx, ty]

            if flip:
                target_img = np.fliplr(target_img).copy()
                target_mask = np.fliplr(target_mask).copy()
                target_em_mask = np.fliplr(target_em_mask).copy()
                warped_img = np.fliplr(warped_img).copy()
                warped_mask = np.fliplr(warped_mask).copy()

            target_img = cv2.warpAffine(target_img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            target_mask = cv2.warpAffine(target_mask, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            target_em_mask = cv2.warpAffine(target_em_mask, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            warped_img = cv2.warpAffine(warped_img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            warped_mask = cv2.warpAffine(warped_mask, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

            # Nonlinear warp ONLY on warped (DFL: both src and dst)
            if self._random_warp:
                warped_img, warped_mask = _random_warp_nonlinear(warped_img, warped_mask)

            # HSV only on warped
            if self._random_hsv_power > 0:
                warped_img = _random_hsv_dfl(warped_img, self._random_hsv_power)

            # src_face_scale: scale src face region (DFL: -30..30% modifier)
            if self._src_face_scale != 0 and self._is_src:
                scale_factor = 1.0 + self._src_face_scale / 100.0
                cx, cy = w // 2, h // 2
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                scaled = cv2.resize(warped_img, (new_w, new_h))
                # Center crop/pad back to original size
                if new_w > w:
                    sx = (new_w - w) // 2
                    sy = (new_h - h) // 2
                    warped_img = scaled[sy:sy+h, sx:sx+w]
                else:
                    sx = (w - new_w) // 2
                    sy = (h - new_h) // 2
                    warped_img = np.zeros_like(warped_img)
                    warped_img[sy:sy+new_h, sx:sx+new_w] = scaled

            # Color transfer on both (DFL: ct applies to warped and target)
            if self._random_ct and self._ct_mode != "none" and self._is_src and self._ct_sample_pool:
                ct_sample = random.choice(self._ct_sample_pool)
                ct_resized = cv2.resize(ct_sample, (res, res))
                ct_func = {
                    "rct": _color_transfer_rct, "mkl": _color_transfer_mkl,
                    "lct": _color_transfer_lct, "idt": _color_transfer_idt,
                    "sot-m": _color_transfer_sot, "sot": _color_transfer_sot,
                }.get(self._ct_mode)
                if ct_func is not None:
                    target_img = ct_func(target_img, ct_resized)
                    warped_img = ct_func(warped_img, ct_resized)

        warped_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
        target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        warped_tensor = torch.from_numpy(warped_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        target_tensor = torch.from_numpy(target_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(target_mask.astype(np.float32) / 255.0).unsqueeze(0)
        em_mask_tensor = torch.from_numpy(target_em_mask.astype(np.float32) / 255.0).unsqueeze(0)

        return {
            "warped": warped_tensor,
            "target": target_tensor,
            "mask": mask_tensor,
            "em_mask": em_mask_tensor,
        }

    def build_ct_sample_pool(self, n: int = 100) -> list:
        pool = []
        paths = random.sample(self._image_paths, min(n, len(self._image_paths)))
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                img = cv2.resize(img, (self._resolution, self._resolution))
                pool.append(img)
        return pool
