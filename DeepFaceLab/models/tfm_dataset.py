from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from DeepFaceLab.core.metadata_manager import MetadataManager, FaceMetadata
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("tfm_dataset")


class TFMDataset(Dataset):
    def __init__(
        self,
        aligned_dir: Path,
        resolution: int = 128,
        is_src: bool = True,
        identity_cache: Optional[dict[str, np.ndarray]] = None,
        augment: bool = True,
        preload: bool = True,
        random_hsv_power: float = 0.0,
        random_warp: bool = True,
        random_flip: bool = True,
        color_transfer: str = "none",
    ) -> None:
        self._aligned_dir = Path(aligned_dir)
        self._resolution = resolution
        self._is_src = is_src
        self._identity_cache = identity_cache or {}
        self._augment = augment
        self._random_hsv_power = random_hsv_power
        self._random_warp = random_warp
        self._random_flip = random_flip
        self._color_transfer = color_transfer
        self._image_paths: list[Path] = FileManager.find_images(self._aligned_dir)
        self._metadata_cache: dict[str, FaceMetadata] = MetadataManager.load_all(self._aligned_dir)

        self._image_cache: dict[str, np.ndarray] = {}
        if preload and len(self._image_paths) <= 2000:
            for p in self._image_paths:
                img = cv2.imread(str(p))
                if img is not None:
                    self._image_cache[p.name] = img
            _logger.info(f"TFMDataset ({'src' if is_src else 'dst'}): preloaded {len(self._image_cache)}/{len(self._image_paths)} images into memory")

        self._mask_cache: dict[str, np.ndarray] = {}
        for p in self._image_paths:
            meta = self._metadata_cache.get(p.name)
            img = self._image_cache.get(p.name)
            if img is None:
                img = cv2.imread(str(p))
            if img is None:
                continue
            if meta is not None and meta.seg_ie_polys is not None:
                self._mask_cache[p.name] = self._render_mask_from_polys(img.shape[:2], meta)
            elif meta is not None and meta.landmarks_106 is not None:
                self._mask_cache[p.name] = self._render_face_mask_from_landmarks(img.shape[:2], meta)
        if self._mask_cache:
            _logger.info(f"TFMDataset ({'src' if is_src else 'dst'}): pre-rendered {len(self._mask_cache)} masks ({sum(1 for p in self._image_paths if self._metadata_cache.get(p.name) and self._metadata_cache[p.name].seg_ie_polys is not None)} XSeg, {sum(1 for p in self._image_paths if self._metadata_cache.get(p.name) and self._metadata_cache[p.name].seg_ie_polys is None and self._metadata_cache[p.name].landmarks_106 is not None)} landmark)")

        if not self._image_paths:
            raise ValueError(f"No face images found in {self._aligned_dir}")

        _logger.info(f"TFMDataset ({'src' if is_src else 'dst'}): {len(self._image_paths)} images from {self._aligned_dir}")

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = self._image_paths[idx]
        img = self._image_cache.get(img_path.name)
        if img is None:
            img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self._resolution, self._resolution, 3), dtype=np.uint8)

        meta = self._metadata_cache.get(img_path.name)
        mask = self._mask_cache.get(img_path.name)
        if mask is None:
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

        if self._augment:
            img, mask = self._augment_pair(img, mask)

        img = cv2.resize(img, (self._resolution, self._resolution))
        mask = cv2.resize(mask, (self._resolution, self._resolution))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        img_tensor = img_tensor * 2.0 - 1.0
        mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)

        identity = self._identity_cache.get(img_path.name)
        if identity is not None:
            identity_tensor = torch.from_numpy(identity.astype(np.float32))
        elif meta is not None and meta.arcface_embedding is not None:
            identity_tensor = torch.from_numpy(meta.arcface_embedding.astype(np.float32))
        else:
            identity_tensor = torch.zeros(512, dtype=torch.float32)

        landmarks_tensor = torch.zeros(106, 2, dtype=torch.float32)
        if meta is not None and meta.landmarks_106 is not None:
            lm = meta.landmarks_106.astype(np.float32)
            scale_x = self._resolution / meta.output_size if meta.output_size != self._resolution else 1.0
            scale_y = self._resolution / meta.output_size if meta.output_size != self._resolution else 1.0
            lm[:, 0] *= scale_x
            lm[:, 1] *= scale_y
            landmarks_tensor = torch.from_numpy(lm)

        yaw_val = 0.0
        if meta is not None and meta.yaw is not None:
            yaw_val = meta.yaw

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "identity": identity_tensor,
            "landmarks": landmarks_tensor,
            "yaw": torch.tensor(yaw_val, dtype=torch.float32),
        }

    @staticmethod
    def _render_mask_from_polys(shape: tuple, meta: FaceMetadata) -> np.ndarray:
        h, w = shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        if meta.seg_ie_polys is None:
            return mask
        for poly_data in meta.seg_ie_polys:
            pts_data = poly_data.get("pts", [])
            poly_type = poly_data.get("type", 1)
            if len(pts_data) < 3:
                continue
            scale_x = w / meta.output_size if meta.output_size != w else 1.0
            scale_y = h / meta.output_size if meta.output_size != h else 1.0
            pts = np.array(pts_data, dtype=np.float32)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            pts = pts.astype(np.int64)
            if poly_type == 1:
                cv2.fillPoly(mask, [pts], 255)
            else:
                cv2.fillPoly(mask, [pts], 0)
        return mask

    @staticmethod
    def _render_face_mask_from_landmarks(shape: tuple, meta: FaceMetadata) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if meta.landmarks_106 is None:
            return np.ones((h, w), dtype=np.uint8) * 255
        lm = meta.landmarks_106.astype(np.float64)
        scale_x = w / meta.output_size if meta.output_size != w else 1.0
        scale_y = h / meta.output_size if meta.output_size != h else 1.0
        lm[:, 0] *= scale_x
        lm[:, 1] *= scale_y
        hull = cv2.convexHull(lm.astype(np.float32))
        hull = hull.astype(np.int64)
        cv2.fillPoly(mask, [hull], 255)
        return mask

    def _augment_pair(self, img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._random_flip and np.random.random() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        if self._random_warp:
            angle = np.random.uniform(-10, 10)
            tx = np.random.uniform(-5, 5)
            ty = np.random.uniform(-5, 5)
            scale = np.random.uniform(0.95, 1.05)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            M[:, 2] += [tx, ty]
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        else:
            angle = np.random.uniform(-15, 15)
            scale = np.random.uniform(0.9, 1.1)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        if self._random_hsv_power > 0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            h_shift = np.random.uniform(-10, 10) * self._random_hsv_power
            s_scale = 1.0 + np.random.uniform(-0.2, 0.2) * self._random_hsv_power
            hsv[:, :, 0] += h_shift
            hsv[:, :, 1] *= s_scale
            hsv = np.clip(hsv, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif np.random.random() < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] += np.random.uniform(-10, 10)
            hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)
            hsv = np.clip(hsv, 0, 179).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if np.random.random() < 0.2:
            ksize = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        if self._color_transfer == "rct":
            img = self._color_transfer_rct(img)
        elif self._color_transfer == "mkl":
            img = self._color_transfer_mkl(img)

        return img, mask

    @staticmethod
    def _color_transfer_rct(img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
        l, a, b = cv2.split(lab)
        l_mean, l_std = l.mean(), l.std() + 1e-6
        a_mean, a_std = a.mean(), a.std() + 1e-6
        b_mean, b_std = b.mean(), b.std() + 1e-6
        l = (l - l_mean) / l_std * 50 + 128
        a = (a - a_mean) / a_std * 50 + 128
        b = (b - b_mean) / b_std * 50 + 128
        lab = cv2.merge([np.clip(l, 0, 255), np.clip(a, 0, 255), np.clip(b, 0, 255)]).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _color_transfer_mkl(img: np.ndarray) -> np.ndarray:
        return img

    @staticmethod
    def build_identity_cache(
        aligned_dir: Path,
        adapter,
    ) -> dict[str, np.ndarray]:
        aligned_dir = Path(aligned_dir)
        cache: dict[str, np.ndarray] = {}
        paths = FileManager.find_images(aligned_dir)
        total = len(paths)
        for idx, img_path in enumerate(paths):
            img = cv2.imread(str(img_path))
            if img is None:
                _logger.warning(f"Cannot read image: {img_path}")
                continue
            try:
                faces = adapter.detect_faces(img, max_num=1)
                if faces and faces[0].embedding is not None:
                    cache[img_path.name] = faces[0].embedding
                else:
                    _logger.warning(f"No embedding for: {img_path.name}")
            except Exception as e:
                _logger.warning(f"Failed to extract identity from {img_path.name}: {e}")
            if (idx + 1) % 100 == 0:
                _logger.info(f"Identity cache: {idx + 1}/{total}")
        _logger.info(f"Identity cache built: {len(cache)}/{total} images")
        return cache

    @staticmethod
    def save_identity_cache(cache: dict[str, np.ndarray], path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), **{k: v for k, v in cache.items()})

    @staticmethod
    def load_identity_cache(path: Path) -> dict[str, np.ndarray]:
        path = Path(path)
        if not path.exists():
            return {}
        data = np.load(str(path), allow_pickle=False)
        cache = {k: data[k] for k in data.files}
        _logger.info(f"Identity cache loaded: {len(cache)} entries from {path}")
        return cache
