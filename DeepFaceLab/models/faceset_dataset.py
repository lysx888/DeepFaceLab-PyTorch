import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from DeepFaceLab.core.metadata_manager import MetadataManager, FaceMetadata
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("faceset_dataset")


class FacesetDataset(Dataset):
    def __init__(
        self,
        aligned_dir: Path,
        resolution: int = 256,
        augment: bool = True,
    ) -> None:
        self._aligned_dir = Path(aligned_dir)
        self._resolution = resolution
        self._augment = augment
        self._image_paths: list[Path] = []
        self._metadata_cache: dict[str, FaceMetadata] = MetadataManager.load_all(self._aligned_dir)

        for p in FileManager.find_images(self._aligned_dir):
            meta = self._metadata_cache.get(p.name)
            if meta is not None and meta.seg_ie_polys is not None:
                self._image_paths.append(p)

        if not self._image_paths:
            raise ValueError(f"No annotated faces found in {self._aligned_dir}")

        _logger.info(f"FacesetDataset: {len(self._image_paths)} annotated images from {self._aligned_dir}")

    @classmethod
    def merge(cls, datasets: list["FacesetDataset"]) -> "FacesetDataset":
        merged = cls.__new__(cls)
        merged._aligned_dir = Path("/")
        merged._resolution = datasets[0]._resolution if datasets else 256
        merged._augment = True
        merged._image_paths = []
        merged._metadata_cache = {}
        for ds in datasets:
            merged._image_paths.extend(ds._image_paths)
            for key, val in ds._metadata_cache.items():
                merged._metadata_cache[str(ds._aligned_dir / key)] = val
        return merged

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = self._image_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self._resolution, self._resolution, 3), dtype=np.uint8)

        meta_key = str(img_path)
        meta = self._metadata_cache.get(meta_key) or self._metadata_cache.get(img_path.name)
        mask = self._render_mask(img.shape[:2], meta)

        if self._augment:
            img, mask = self._augment_pair(img, mask)

        img = cv2.resize(img, (self._resolution, self._resolution))
        mask = cv2.resize(mask, (self._resolution, self._resolution))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)

        img_tensor = img_tensor * 2.0 - 1.0

        return {
            "image": img_tensor,
            "mask": mask_tensor,
        }

    def _render_mask(self, shape: tuple, meta: FaceMetadata) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if meta is None or meta.seg_ie_polys is None:
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

    def _augment_pair(self, img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.random() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        angle = np.random.uniform(-15, 15)
        scale = np.random.uniform(0.9, 1.1)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        if np.random.random() < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] += np.random.uniform(-10, 10)
            hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)
            hsv = np.clip(hsv, 0, 179).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if np.random.random() < 0.2:
            ksize = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        return img, mask
