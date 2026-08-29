import collections
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from faceswap.core.metadata_manager import MetadataManager, FaceMetadata
from faceswap.shared.file_manager import FileManager
from faceswap.shared.image_utils import bgr_to_hsv, hsv_to_bgr
from faceswap.shared.logger import get_logger

_logger = get_logger("faceset_dataset")


def _random_circle_faded(h: int, w: int) -> np.ndarray:
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    wh_max = max(w, h)
    fade_start = np.random.randint(wh_max)
    fade_end = fade_start + np.random.randint(max(1, wh_max - fade_start))
    if fade_end == fade_start:
        fade_end = fade_start + 1
    yy, xx = np.mgrid[:h, :w]
    dists = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    result = np.clip(1.0 - (dists - fade_start) / (fade_end - fade_start), 0.0, 1.0)
    return result


def _apply_with_spatial_mask(img: np.ndarray, augmented: np.ndarray) -> np.ndarray:
    c_mask = _random_circle_faded(img.shape[0], img.shape[1])
    if img.ndim == 3:
        c_mask = c_mask[:, :, np.newaxis]
    return img * (1.0 - c_mask) + augmented * c_mask


class FacesetDataset(Dataset):
    def __init__(
        self,
        aligned_dir: Path,
        resolution: int = 256,
        augment: bool = True,
        pretrain_mode: bool = False,
        face_type: str = "wf",
    ) -> None:
        self._aligned_dir = Path(aligned_dir)
        self._resolution = resolution
        self._augment = augment
        self._pretrain_mode = pretrain_mode
        self._face_type = face_type
        self._image_paths: list[Path] = []
        raw_meta = MetadataManager.load_all(self._aligned_dir, lightweight=True)
        self._metadata_cache: dict[str, FaceMetadata] = {str(self._aligned_dir / k): v for k, v in raw_meta.items()}
        self._image_cache_max = 512
        self._image_cache: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()

        if pretrain_mode:
            for p in FileManager.find_images(self._aligned_dir):
                self._image_paths.append(p)
            if not self._image_paths:
                raise ValueError(f"No face images found in {self._aligned_dir} for pretrain")
            _logger.info(f"FacesetDataset (pretrain): {len(self._image_paths)} images from {self._aligned_dir}")
        else:
            for p in FileManager.find_images(self._aligned_dir):
                meta = self._metadata_cache.get(str(p))
                if meta is not None and meta.seg_ie_polys is not None:
                    self._image_paths.append(p)
            if not self._image_paths:
                raise ValueError(f"No annotated faces found in {self._aligned_dir}")
            _logger.info(f"FacesetDataset: {len(self._image_paths)} segmented images from {self._aligned_dir}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_metadata_cache'] = {}
        state['_image_cache'] = collections.OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_meta(self, img_path) -> FaceMetadata | None:
        key = str(img_path)
        meta = self._metadata_cache.get(key)
        if meta is not None:
            return meta
        meta = MetadataManager.load(Path(img_path), lightweight=True)
        if meta is not None:
            self._metadata_cache[key] = meta
        return meta

    @classmethod
    def merge(cls, datasets: list["FacesetDataset"]) -> "FacesetDataset":
        merged = cls.__new__(cls)
        merged._aligned_dir = datasets[0]._aligned_dir if datasets else Path(".")
        merged._resolution = datasets[0]._resolution if datasets else 256
        merged._augment = True
        merged._pretrain_mode = datasets[0]._pretrain_mode if datasets else False
        merged._image_paths = []
        merged._metadata_cache = {}
        merged._image_cache_max = 512
        merged._image_cache = collections.OrderedDict()
        for ds in datasets:
            merged._image_paths.extend(ds._image_paths)
            for key, val in ds._metadata_cache.items():
                merged._metadata_cache[key] = val
        return merged

    def __len__(self) -> int:
        return len(self._image_paths)

    @property
    def image_paths(self) -> list[Path]:
        return self._image_paths

    @property
    def metadata_cache(self) -> dict[str, FaceMetadata]:
        return self._metadata_cache

    def set_bg_dataset(self, dataset: "FacesetDataset") -> None:
        self._bg_dataset = dataset

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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = self._image_paths[idx]
        img = self._read_image(img_path)
        if img is None:
            img = np.zeros((self._resolution, self._resolution, 3), dtype=np.uint8)

        if self._pretrain_mode:
            res = self._resolution
            img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
            if self._augment:
                if np.random.random() < 0.5:
                    img = np.fliplr(img).copy()
                rotation = np.random.uniform(-10, 10)
                scale = np.random.uniform(1.0 / 1.1, 1.1)
                tx = np.random.uniform(-0.1, 0.1) * res
                ty = np.random.uniform(-0.1, 0.1) * res
                rmat = cv2.getRotationMatrix2D((res / 2.0, res / 2.0), rotation, scale)
                rmat[:, 2] += (tx, ty)
                img = cv2.warpAffine(img, rmat, (res, res), borderMode=cv2.BORDER_REPLICATE)
                hsv = bgr_to_hsv(img).astype(np.float32)
                h_amount = np.random.randint(0, 11)
                hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-h_amount, h_amount + 1)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] + np.random.uniform(-15, 15), 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] + np.random.uniform(-15, 15), 0, 255)
                img = hsv_to_bgr(hsv.astype(np.uint8))
            img_f = img.astype(np.float32) / 255.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_f = gray.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_f).permute(2, 0, 1)
            target_tensor = torch.from_numpy(gray_f).unsqueeze(0)
            return {
                "image": img_tensor,
                "target": target_tensor,
            }

        meta = self._get_meta(img_path)
        mask = self._render_mask(img.shape[:2], meta)

        if meta is not None and meta.image_to_face_mat is not None and meta.source_kps_5 is not None:
            meta_ft = getattr(meta, 'face_type', None)
            meta_ft_str = meta_ft.value if hasattr(meta_ft, 'value') else str(meta_ft) if meta_ft else None
            if meta_ft_str is not None and meta_ft_str != self._face_type:
                try:
                    from insightface.utils.face_align import estimate_norm
                    kps_5 = meta.source_kps_5.astype(np.float32)
                    M_new = estimate_norm(kps_5, self._resolution, mode=None)
                    M_old = meta.image_to_face_mat.astype(np.float32)
                    M_relative = M_new @ np.linalg.inv(M_old)
                    img = cv2.warpAffine(img, M_relative, (self._resolution, self._resolution),
                                         flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    mask = cv2.warpAffine(mask, M_relative, (self._resolution, self._resolution),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                except Exception:
                    img = cv2.resize(img, (self._resolution, self._resolution), interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(mask, (self._resolution, self._resolution), interpolation=cv2.INTER_LINEAR)
            else:
                img = cv2.resize(img, (self._resolution, self._resolution), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (self._resolution, self._resolution), interpolation=cv2.INTER_LINEAR)
        else:
            img = cv2.resize(img, (self._resolution, self._resolution), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self._resolution, self._resolution), interpolation=cv2.INTER_LINEAR)

        if self._augment:
            img, mask = self._augment_pair(img, mask, exclude_idx=idx)

        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)

        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
        }

    def _render_mask(self, shape: tuple, meta: FaceMetadata) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if meta is None:
            _logger.warning(f"render_mask: meta is None, returning empty mask")
            return mask
        if meta.seg_ie_polys is not None and len(meta.seg_ie_polys) > 0:
            sorted_polys = sorted(meta.seg_ie_polys, key=lambda p: p.get("type", 1) == 1, reverse=True)
            for poly_data in sorted_polys:
                pts_data = poly_data.get("pts", [])
                poly_type = poly_data.get("type", 1)
                if len(pts_data) < 3:
                    continue
                scale_x = w / meta.output_size if meta.output_size != w else 1.0
                scale_y = h / meta.output_size if meta.output_size != h else 1.0
                pts = np.array(pts_data, dtype=np.float32)
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                pts = pts.astype(np.int32)
                if poly_type == 1:
                    cv2.fillPoly(mask, [pts], 255)
                else:
                    cv2.fillPoly(mask, [pts], 0)
        return mask

    def _augment_pair(self, img: np.ndarray, mask: np.ndarray, exclude_idx: int = -1) -> tuple[np.ndarray, np.ndarray]:
        img_f = img.astype(np.float32) / 255.0
        mask_f = mask.astype(np.float32) / 255.0

        if np.random.random() < 0.5:
            bg_result = self._sample_background(exclude_idx=exclude_idx)
            if bg_result is not None:
                bg_f, bg_mask = bg_result
                img_f, mask_f = self._mix_background(img_f, mask_f, bg_f, bg_mask)

        if np.random.random() < 0.5:
            img_f = np.fliplr(img_f).copy()
            mask_f = np.fliplr(mask_f).copy()

        h, w = img_f.shape[:2]
        angle = np.random.uniform(-10, 10)
        scale = np.random.uniform(0.95, 1.05)
        tx = np.random.uniform(-0.05, 0.05) * w
        ty = np.random.uniform(-0.05, 0.05) * h
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[:, 2] += [tx, ty]
        img_f = cv2.warpAffine(img_f, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        mask_f = cv2.warpAffine(mask_f, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        mask_f = np.where(mask_f >= 0.5, 1.0, 0.0).astype(np.float32)

        img_f = np.clip(img_f, 0.0, 1.0)
        mask_f3 = mask_f[:, :, np.newaxis]

        if np.random.random() < 0.5:
            krn = np.random.randint(h // 4, h)
            krn = krn - krn % 2 + 1
            img_f = img_f + cv2.GaussianBlur(img_f * mask_f3, (krn, krn), 0)

        if np.random.random() < 0.5:
            krn = np.random.randint(h // 4, h)
            krn = krn - krn % 2 + 1
            img_f = img_f + cv2.GaussianBlur(img_f * (1.0 - mask_f3), (krn, krn), 0)

        img_f = np.clip(img_f, 0.0, 1.0)

        if np.random.random() < 0.5:
            aug = self._aug_hsv_shift(img_f)
            img_f = _apply_with_spatial_mask(img_f, aug)
        else:
            aug = self._aug_rgb_levels(img_f)
            img_f = _apply_with_spatial_mask(img_f, aug)

        if np.random.random() < 0.5:
            aug = self._aug_sharpen(img_f)
            img_f = _apply_with_spatial_mask(img_f, aug)
        else:
            aug = self._aug_motion_blur(img_f)
            img_f = _apply_with_spatial_mask(img_f, aug)
            aug = self._aug_gaussian_blur(img_f)
            img_f = _apply_with_spatial_mask(img_f, aug)

        if np.random.random() < 0.5:
            aug = self._aug_resize_jitter(img_f)
            img_f = _apply_with_spatial_mask(img_f, aug)

        aug = self._aug_jpeg_compress(img_f)
        img_f = _apply_with_spatial_mask(img_f, aug)

        img_f = np.clip(img_f, 0.0, 1.0)
        img = (img_f * 255.0).astype(np.uint8)
        mask = (mask_f * 255.0).astype(np.uint8)
        return img, mask

    def _sample_background(self, exclude_idx: int = -1) -> tuple[np.ndarray, np.ndarray] | None:
        bg_path = None
        bg_ds = getattr(self, '_bg_dataset', None)
        if bg_ds is not None and len(bg_ds) > 0:
            candidates = list(range(len(bg_ds)))
            bg_idx = candidates[np.random.randint(len(candidates))]
            bg_path = bg_ds._image_paths[bg_idx]
            bg_meta = bg_ds._get_meta(bg_path)
        elif len(self._image_paths) >= 2:
            candidates = [i for i in range(len(self._image_paths)) if i != exclude_idx]
            if not candidates:
                return None
            bg_idx = candidates[np.random.randint(len(candidates))]
            bg_path = self._image_paths[bg_idx]
            bg_meta = self._get_meta(bg_path)
        else:
            return None

        bg = self._read_image(bg_path)
        if bg is None:
            return None
        bg_f = bg.astype(np.float32) / 255.0
        bg_mask = self._render_mask(bg.shape[:2], bg_meta).astype(np.float32) / 255.0
        return bg_f, bg_mask

    def _mix_background(self, img_f: np.ndarray, mask_f: np.ndarray, bg_f: np.ndarray, bg_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img_f.shape[:2]
        bg_f = cv2.resize(bg_f, (w, h), interpolation=cv2.INTER_LINEAR)
        bg_mask = cv2.resize(bg_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        bg_angle = np.random.uniform(-180, 180)
        bg_scale = np.random.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), bg_angle, bg_scale)
        bg_f = cv2.warpAffine(bg_f, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        bg_mask = cv2.warpAffine(bg_mask, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        if np.random.random() < 0.5:
            bg_f = np.fliplr(bg_f).copy()
            bg_mask = np.fliplr(bg_mask).copy()

        bg_mask = np.where(bg_mask >= 0.5, 1.0, 0.0).astype(np.float32)

        if np.random.random() < 0.5:
            bg_f = self._aug_hsv_shift(bg_f)
        else:
            bg_f = self._aug_rgb_levels(bg_f)

        c_mask = 1.0 - (1.0 - bg_mask) * (1.0 - mask_f)
        c_mask3 = c_mask[:, :, np.newaxis]
        rnd = 0.15 + np.random.uniform() * 0.85
        img_f = img_f * c_mask3 + img_f * (1.0 - c_mask3) * rnd + bg_f * (1.0 - c_mask3) * (1.0 - rnd)
        return img_f, mask_f

    def _aug_hsv_shift(self, img_f: np.ndarray) -> np.ndarray:
        img_u8 = (np.clip(img_f, 0, 1) * 255).astype(np.uint8)
        hsv = bgr_to_hsv(img_u8).astype(np.float32)
        hsv[:, :, 0] += np.random.uniform(-15, 15)
        hsv[:, :, 1] *= np.random.uniform(0.7, 1.3)
        hsv[:, :, 2] *= np.random.uniform(0.8, 1.2)
        hsv = np.clip(hsv, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
        result = hsv_to_bgr(hsv).astype(np.float32) / 255.0
        return result

    def _aug_rgb_levels(self, img_f: np.ndarray) -> np.ndarray:
        result = img_f.copy()
        for c in range(3):
            gamma = np.random.uniform(0.7, 1.3)
            result[:, :, c] = np.power(np.clip(result[:, :, c], 0, 1), gamma)
        return result

    def _aug_sharpen(self, img_f: np.ndarray) -> np.ndarray:
        if np.random.random() > 0.25:
            return img_f
        ksize = np.random.choice([3, 5])
        blurred = cv2.GaussianBlur(img_f, (ksize, ksize), 0)
        amount = np.random.uniform(0.5, 1.5)
        result = np.clip(img_f + amount * (img_f - blurred), 0, 1)
        return result

    def _aug_motion_blur(self, img_f: np.ndarray) -> np.ndarray:
        if np.random.random() > 0.25:
            return img_f
        size = np.random.randint(2, 6)
        kernel = np.zeros((size, size), dtype=np.float32)
        if np.random.random() < 0.5:
            kernel[size // 2, :] = 1.0 / size
        else:
            kernel[:, size // 2] = 1.0 / size
        result = cv2.filter2D(img_f, -1, kernel)
        return result

    def _aug_gaussian_blur(self, img_f: np.ndarray) -> np.ndarray:
        if np.random.random() > 0.25:
            return img_f
        ksize = np.random.choice([3, 5])
        result = cv2.GaussianBlur(img_f, (ksize, ksize), 0)
        return result

    def _aug_resize_jitter(self, img_f: np.ndarray) -> np.ndarray:
        h, w = img_f.shape[:2]
        scale = np.random.uniform(0.5, 0.75)
        small_h, small_w = int(h * scale), int(w * scale)
        small = cv2.resize(img_f, (small_w, small_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    def _aug_jpeg_compress(self, img_f: np.ndarray) -> np.ndarray:
        if np.random.random() > 0.25:
            return img_f
        img_u8 = (np.clip(img_f, 0, 1) * 255).astype(np.uint8)
        quality = np.random.randint(30, 95)
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, encoded = cv2.imencode('.jpg', img_u8, encode_param)
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        return _apply_with_spatial_mask(img_f, result)
