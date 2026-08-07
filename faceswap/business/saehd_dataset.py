import collections
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from faceswap.core.metadata_manager import MetadataManager, FaceMetadata
from faceswap.core.color_transfer import color_transfer
from faceswap.core.landmarks106 import fill_hull_mask_106, LANDMARK_GROUPS_106
from faceswap.shared.file_manager import FileManager
from faceswap.shared.image_utils import bgr_to_hsv, hsv_to_bgr
from faceswap.shared.logger import get_logger

_logger = get_logger("saehd_dataset")


class SAEHDDataset(Dataset):
    def __init__(
        self,
        aligned_dir: Path,
        resolution: int = 128,
        face_type: str = 'f',
        random_warp: bool = True,
        random_flip: bool = True,
        random_hsv_power: float = 0.0,
        ct_mode: str = 'none',
        uniform_yaw: bool = False,
        is_src: bool = True,
        augment: bool = True,
        scale_range: tuple[float, float] = (-0.15, 0.15),
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

        self._image_paths: list[Path] = []
        _t_meta = time.time()
        self._metadata_cache: dict[str, FaceMetadata] = MetadataManager.load_all(self._aligned_dir, lightweight=True)
        _logger.info(f"[DIAG] MetadataManager.load_all({self._aligned_dir.name}): "
                     f"{time.time()-_t_meta:.2f}s  entries={len(self._metadata_cache)}")
        self._image_cache_max = 512
        self._image_cache: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()

        for p in FileManager.find_images(self._aligned_dir):
            self._image_paths.append(p)

        if not self._image_paths:
            raise ValueError(f"No images found in {self._aligned_dir}")

        self._yaw_weights: list[float] = []
        if uniform_yaw:
            self._compute_yaw_weights()

        self._dst_sample_for_ct: np.ndarray | None = None
        self._dst_mask_for_ct: np.ndarray | None = None
        _logger.info(f"SAEHDDataset: {len(self._image_paths)} images from {self._aligned_dir}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_metadata_cache'] = {}
        state['_image_cache'] = collections.OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_meta(self, img_path: Path) -> FaceMetadata | None:
        key = img_path.name
        meta = self._metadata_cache.get(key)
        if meta is not None:
            return meta
        meta = MetadataManager.load(img_path, lightweight=True)
        if meta is not None:
            self._metadata_cache[key] = meta
        return meta

    def _compute_yaw_weights(self) -> None:
        yaws = []
        for p in self._image_paths:
            meta = self._get_meta(p)
            if meta is not None and meta.yaw is not None:
                yaws.append(abs(meta.yaw))
            else:
                yaws.append(0.0)
        if not yaws:
            return
        max_yaw = max(yaws)
        if max_yaw < 1e-4:
            self._yaw_weights = [1.0 / len(yaws)] * len(yaws)
            return
        max_yaw += 1e-6
        self._yaw_weights = [1.0 - (y / max_yaw) * 0.7 for y in yaws]
        total = sum(self._yaw_weights)
        if total < 1e-8:
            self._yaw_weights = [1.0 / len(yaws)] * len(yaws)
            return
        self._yaw_weights = [w / total for w in self._yaw_weights]

    def set_dst_sample_for_ct(self, dst_image: np.ndarray, dst_mask: np.ndarray = None) -> None:
        self._dst_sample_for_ct = dst_image
        self._dst_mask_for_ct = dst_mask

    def __len__(self) -> int:
        return len(self._image_paths)

    @property
    def image_paths(self) -> list[Path]:
        return self._image_paths

    @property
    def yaw_weights(self) -> list[float] | None:
        return self._yaw_weights

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
        em_mask = self._render_em_mask(img.shape[:2], meta)

        if self._augment:
            if self._random_flip and random.randint(0, 9) < 4:
                img = np.fliplr(img).copy()
                full_mask = np.fliplr(full_mask).copy()
                em_mask = np.fliplr(em_mask).copy()

            scale = 1.0 + random.uniform(*self._scale_range)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            full_mask = cv2.warpAffine(full_mask, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            em_mask = cv2.warpAffine(em_mask, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        img = cv2.resize(img, (self._resolution, self._resolution), interpolation=cv2.INTER_AREA)
        full_mask = cv2.resize(full_mask, (self._resolution, self._resolution), interpolation=cv2.INTER_LINEAR)
        em_mask = cv2.resize(em_mask, (self._resolution, self._resolution), interpolation=cv2.INTER_LINEAR)

        if self._random_hsv_power > 0 and self._is_src:
            img = self._apply_random_hsv(img, self._random_hsv_power)

        if self._ct_mode is not None and self._is_src and self._dst_sample_for_ct is not None:
            img_f = img.astype(np.float32) / 255.0
            dst_f = self._dst_sample_for_ct.astype(np.float32) / 255.0
            src_mask_f = full_mask.astype(np.float32) / 255.0
            dst_mask_f = (self._dst_mask_for_ct.astype(np.float32) / 255.0
                         if self._dst_mask_for_ct is not None else None)
            try:
                img_ct = color_transfer(self._ct_mode, img_f, dst_f,
                                        src_mask=src_mask_f, trg_mask=dst_mask_f)
                img = np.clip(img_ct * 255.0, 0, 255).astype(np.uint8)
            except Exception:
                pass

        target_image = img.astype(np.float32) / 255.0

        if self._random_warp:
            warped_image = self._random_warp_image(target_image)
        else:
            warped_image = target_image.copy()

        target_image_t = torch.from_numpy(target_image.transpose(2, 0, 1)).float()
        warped_image_t = torch.from_numpy(warped_image.transpose(2, 0, 1)).float()
        full_mask_t = torch.from_numpy(full_mask.astype(np.float32) / 255.0).unsqueeze(0)
        em_mask_t = torch.from_numpy(em_mask.astype(np.float32) / 255.0).unsqueeze(0)

        return {
            'warped_image': warped_image_t,
            'target_image': target_image_t,
            'target_mask': full_mask_t,
            'target_em_mask': em_mask_t,
        }

    def _render_full_mask(self, shape: tuple, meta: FaceMetadata | None) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if meta is not None and meta.xseg_mask is not None:
            xseg_arr = meta.get_xseg_mask_array(h, w)
            if xseg_arr is not None and np.any(xseg_arr) and np.sum(xseg_arr > 0) / (h * w) > 0.01:
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
            if np.any(mask) and np.sum(mask > 0) / (h * w) > 0.01:
                return mask
            _logger.warning(f"seg_ie_polys rendered empty for {getattr(meta, 'source_filename', '?')}, "
                            f"falling back to landmarks")

        if meta is not None and meta.landmarks_106 is not None and len(meta.landmarks_106) >= 68:
            lm = meta.landmarks_106.astype(np.float32).copy()
            scale_x = w / meta.output_size if meta.output_size > 0 else 1.0
            scale_y = h / meta.output_size if meta.output_size > 0 else 1.0
            lm[:, 0] *= scale_x
            lm[:, 1] *= scale_y
            fill_hull_mask_106(mask, lm, eyebrows_expand_mod=2.0 if meta.face_type == "HEAD" else 1.5)
            if np.any(mask):
                return mask

        cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.38), int(h * 0.48)), 0, 0, 360, 255, -1)
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

    def _random_warp_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        w_work = self._resolution

        rotation = random.uniform(-10, 10)
        scale = random.uniform(1.0 / 1.5, 1.5)
        tx = random.uniform(-0.05, 0.05)
        ty = random.uniform(-0.05, 0.05)

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
            max_offset = cell_size * 0.72
            dx = np.clip(np.random.normal(0, 1, (inner_size, inner_size)) * (cell_size * 0.24),
                         -max_offset, max_offset)
            dy = np.clip(np.random.normal(0, 1, (inner_size, inner_size)) * (cell_size * 0.24),
                         -max_offset, max_offset)
            mapx[1:-1, 1:-1] += dx
            mapy[1:-1, 1:-1] += dy

        half_cell = cell_size // 2
        mapx = cv2.resize(mapx, (w_work + cell_size,) * 2)[half_cell:-half_cell, half_cell:-half_cell].astype(np.float32)
        mapy = cv2.resize(mapy, (w_work + cell_size,) * 2)[half_cell:-half_cell, half_cell:-half_cell].astype(np.float32)

        rmat = cv2.getRotationMatrix2D((w_work // 2, w_work // 2), rotation, scale)
        rmat[:, 2] += (tx * w_work, ty * w_work)

        img = cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        img = cv2.warpAffine(img, rmat, (w_work, w_work), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
        img = np.clip(img, 0.0, 1.0)

        return img

    def _apply_random_hsv(self, img: np.ndarray, power: float) -> np.ndarray:
        hsv = bgr_to_hsv(img).astype(np.float32)
        hsv[:, :, 0] += random.uniform(-power * 30, power * 30)
        hsv[:, :, 1] *= (1.0 + random.uniform(-power, power))
        hsv[:, :, 2] *= (1.0 + random.uniform(-power, power))
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return hsv_to_bgr(hsv.astype(np.uint8))
