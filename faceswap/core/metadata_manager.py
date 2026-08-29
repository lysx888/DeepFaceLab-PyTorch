import base64
import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from faceswap.setting import FaceType, LANDMARK_POINTS, KPS5_POINTS
from faceswap.shared.file_manager import FileManager
from faceswap.shared.logger import get_logger

_logger = get_logger("metadata_manager")

MAX_WORKERS = 8
MIN_WORKERS = 2
WORKERS_PER_ITEMS = 100


@dataclass
class FaceMetadata:
    landmarks_106: npt.NDArray[np.int64]
    face_type: FaceType
    source_filename: str
    source_rect: Optional[list[float]] = None
    source_landmarks_106: Optional[npt.NDArray[np.int64]] = None
    image_to_face_mat: Optional[npt.NDArray[np.float32]] = None
    output_size: int = 512
    source_kps_5: Optional[npt.NDArray[np.float32]] = None
    source_face_rect: Optional[list[float]] = None
    landmarks_106_visibility: list[bool] = field(default_factory=lambda: [True] * LANDMARK_POINTS)
    kps_5_visibility: list[bool] = field(default_factory=lambda: [True] * KPS5_POINTS)
    seg_ie_polys: Optional[list[dict]] = None
    xseg_mask: Optional[str] = None
    pose: Optional[list[float]] = None

    @staticmethod
    def encode_xseg_mask(mask: np.ndarray) -> str:
        mask_uint8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
        compressed = zlib.compress(mask_uint8.tobytes())
        return base64.b64encode(compressed).decode("ascii")

    def get_xseg_mask_array(self, h: int = 0, w: int = 0) -> Optional[np.ndarray]:
        if self.xseg_mask is None:
            return None
        try:
            compressed = base64.b64decode(self.xseg_mask)
            raw = zlib.decompress(compressed)
            mask_uint8 = np.frombuffer(raw, dtype=np.uint8).copy()
            mask_uint8 = mask_uint8.reshape(self.output_size, self.output_size)
            if h > 0 and w > 0 and (h != self.output_size or w != self.output_size):
                import cv2
                mask_uint8 = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_LINEAR)
            return mask_uint8
        except Exception as e:
            _logger.warning(f"xseg_mask decode failed: {e}")
            return None

    def to_dict(self) -> dict:
        d: dict = {
            "landmarks_106": self.landmarks_106.astype(np.int64).tolist(),
            "face_type": int(self.face_type),
            "source_filename": self.source_filename,
        }
        if self.source_rect is not None:
            d["source_rect"] = self.source_rect
        if self.source_landmarks_106 is not None:
            d["source_landmarks_106"] = self.source_landmarks_106.astype(np.int64).tolist()
        if self.image_to_face_mat is not None:
            d["image_to_face_mat"] = self.image_to_face_mat.astype(np.float32).tolist()
        d["output_size"] = self.output_size
        if self.source_kps_5 is not None:
            d["source_kps_5"] = self.source_kps_5.astype(np.float32).tolist()
        if self.source_face_rect is not None:
            d["source_face_rect"] = self.source_face_rect
        d["landmarks_106_visibility"] = self.landmarks_106_visibility
        d["kps_5_visibility"] = self.kps_5_visibility
        d["seg_ie_polys"] = self.seg_ie_polys
        d["xseg_mask"] = self.xseg_mask
        if self.pose is not None:
            d["pose"] = self.pose
        return d

    @classmethod
    def from_dict(cls, data: dict, lightweight: bool = False) -> "FaceMetadata":
        lm = np.array(data["landmarks_106"], dtype=np.int64)
        if lm.shape != (LANDMARK_POINTS, 2):
            raise ValueError(f"landmarks_106 shape must be ({LANDMARK_POINTS}, 2), got {lm.shape}")
        ft = FaceType(data["face_type"])
        src_rect = data.get("source_rect")
        src_lm = None if lightweight else (np.array(data["source_landmarks_106"], dtype=np.int64) if "source_landmarks_106" in data else None)
        mat = None if lightweight else (np.array(data["image_to_face_mat"], dtype=np.float32) if "image_to_face_mat" in data else None)
        output_size = data.get("output_size", 512)
        src_kps5 = None if lightweight else (np.array(data["source_kps_5"], dtype=np.float32) if "source_kps_5" in data else None)
        src_face_rect = data.get("source_face_rect")
        lm106_vis = data.get("landmarks_106_visibility", [True] * LANDMARK_POINTS)
        kps5_vis = data.get("kps_5_visibility", [True] * KPS5_POINTS)
        seg = data.get("seg_ie_polys")
        xseg_mask = data.get("xseg_mask")
        pose = data.get("pose")
        return cls(
            landmarks_106=lm,
            face_type=ft,
            source_filename=data["source_filename"],
            source_rect=src_rect,
            source_landmarks_106=src_lm,
            image_to_face_mat=mat,
            output_size=output_size,
            source_kps_5=src_kps5,
            source_face_rect=src_face_rect,
            landmarks_106_visibility=lm106_vis,
            kps_5_visibility=kps5_vis,
            seg_ie_polys=seg,
            xseg_mask=xseg_mask,
            pose=pose,
        )


class MetadataManager:

    @staticmethod
    def sidecar_path(image_path: Path) -> Path:
        return image_path.with_suffix(".json")

    @staticmethod
    def load(image_path: Path, lightweight: bool = False) -> Optional[FaceMetadata]:
        json_path = MetadataManager.sidecar_path(Path(image_path))
        if not json_path.exists():
            return None
        try:
            with open(str(json_path), "r", encoding="utf-8") as f:
                data = json.load(f)
            return FaceMetadata.from_dict(data, lightweight=lightweight)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            _logger.warning(f"Failed to load metadata from {json_path}: {e}")
            return None

    @staticmethod
    def save(image_path: Path, metadata: FaceMetadata) -> None:
        json_path = MetadataManager.sidecar_path(Path(image_path))
        json_str = json.dumps(metadata.to_dict(), ensure_ascii=False)
        FileManager.atomic_write(json_path, json_str)

    @staticmethod
    def _load_json_sidecar(img_path: Path) -> Optional[dict]:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            return None
        try:
            with open(str(json_path), "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError, KeyError):
            return None

    @staticmethod
    def _get_n_workers(n_paths: int) -> int:
        return min(MAX_WORKERS, max(MIN_WORKERS, n_paths // WORKERS_PER_ITEMS))

    @staticmethod
    def load_all(aligned_dir: Path, lightweight: bool = False) -> dict[str, FaceMetadata]:
        aligned_dir = Path(aligned_dir)
        result: dict[str, FaceMetadata] = {}
        if not aligned_dir.exists():
            return result
        _paths = list(FileManager.find_images(aligned_dir))

        def _load_one(img_path):
            data = MetadataManager._load_json_sidecar(img_path)
            if data is None:
                return None
            try:
                return img_path.name, FaceMetadata.from_dict(data, lightweight=lightweight)
            except (ValueError, KeyError):
                return None

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MetadataManager._get_n_workers(len(_paths))) as pool:
            for item in pool.map(_load_one, _paths):
                if item is not None:
                    name, meta = item
                    result[name] = meta

        return result

    @staticmethod
    def save_all(aligned_dir: Path, metadata_map: dict[str, FaceMetadata]) -> None:
        aligned_dir = Path(aligned_dir)
        for filename, meta in metadata_map.items():
            img_path = aligned_dir / filename
            MetadataManager.save(img_path, meta)

    @staticmethod
    def load_yaw_only(aligned_dir: Path) -> dict[str, float]:
        """只从JSON读取yaw值（pose[1]），用于uniform_yaw权重计算。"""
        aligned_dir = Path(aligned_dir)
        result: dict[str, float] = {}
        if not aligned_dir.exists():
            return result
        _paths = list(FileManager.find_images(aligned_dir))

        def _load_one(img_path):
            data = MetadataManager._load_json_sidecar(img_path)
            if data is None:
                return None
            pose = data.get("pose")
            if pose is not None and len(pose) >= 2:
                return img_path.name, float(pose[1])
            return None

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MetadataManager._get_n_workers(len(_paths))) as pool:
            for item in pool.map(_load_one, _paths):
                if item is not None:
                    result[item[0]] = item[1]
        return result

    @staticmethod
    def remove_field(aligned_dir: Path, field_name: str) -> None:
        aligned_dir = Path(aligned_dir)
        for img_path in FileManager.find_images(aligned_dir):
            meta = MetadataManager.load(img_path)
            if meta is None:
                continue
            if field_name == "seg_ie_polys":
                meta.seg_ie_polys = None
            elif field_name == "xseg_mask":
                meta.xseg_mask = None
            elif field_name == "landmarks_106_visibility":
                meta.landmarks_106_visibility = [True] * LANDMARK_POINTS
            elif field_name == "kps_5_visibility":
                meta.kps_5_visibility = [True] * KPS5_POINTS
            else:
                continue
            MetadataManager.save(img_path, meta)

    @staticmethod
    def pack_metadata(aligned_dir: Path, output_path: Path) -> None:
        aligned_dir = Path(aligned_dir)
        output_path = Path(output_path)
        metadata_map = MetadataManager.load_all(aligned_dir)
        pack_data = {}
        for filename, meta in metadata_map.items():
            pack_data[filename] = meta.to_dict()
        json_str = json.dumps(pack_data, ensure_ascii=False)
        FileManager.atomic_write(output_path, json_str)

    @staticmethod
    def unpack_metadata(pack_path: Path, aligned_dir: Path) -> None:
        pack_path = Path(pack_path)
        aligned_dir = Path(aligned_dir)
        with open(str(pack_path), "r", encoding="utf-8") as f:
            pack_data = json.load(f)
        for filename, data in pack_data.items():
            meta = FaceMetadata.from_dict(data)
            img_path = aligned_dir / filename
            MetadataManager.save(img_path, meta)
