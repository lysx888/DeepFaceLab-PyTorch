import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from faceswap.core.metadata_manager import MetadataManager, FaceMetadata
from faceswap.models.face_enhancer import FaceEnhancer
from faceswap.shared.file_manager import FileManager
from faceswap.shared.logger import get_logger

_logger = get_logger("face_tool")


class FaceTool:
    def __init__(self) -> None:
        pass

    def enhance(self, aligned_dir: Path, device: str = "auto") -> int:
        enhancer = FaceEnhancer(device=device)
        return enhancer.process_folder(aligned_dir)

    def pack(self, aligned_dir: Path, output_path: Optional[Path] = None) -> Path:
        aligned_dir = Path(aligned_dir)
        if output_path is None:
            output_path = aligned_dir / "faceset.pak"
        output_path = Path(output_path)

        import json
        import base64
        pack_data = {}
        for img_path in FileManager.find_images(aligned_dir):
            with open(str(img_path), "rb") as f:
                img_bytes = f.read()
            meta = MetadataManager.load(img_path)
            entry = {"image": base64.b64encode(img_bytes).decode("ascii")}
            if meta is not None:
                entry["metadata"] = meta.to_dict()
            pack_data[img_path.name] = entry

        json_str = json.dumps(pack_data, ensure_ascii=False)
        FileManager.atomic_write(output_path, json_str)
        _logger.info(f"Packed faceset to {output_path}")
        return output_path

    def unpack(self, pack_path: Path, aligned_dir: Path) -> int:
        pack_path = Path(pack_path)
        aligned_dir = Path(aligned_dir)
        aligned_dir.mkdir(parents=True, exist_ok=True)

        import json
        import base64
        with open(str(pack_path), "r", encoding="utf-8") as f:
            pack_data = json.load(f)

        count = 0
        for filename, entry in pack_data.items():
            img_bytes = base64.b64decode(entry["image"])
            img_path = aligned_dir / filename
            FileManager.atomic_write(img_path, img_bytes)

            if "metadata" in entry:
                meta = FaceMetadata.from_dict(entry["metadata"])
                MetadataManager.save(img_path, meta)
            count += 1

        _logger.info(f"Unpacked {count} faces to {aligned_dir}")
        return count

    def resize(self, aligned_dir: Path, target_size: int) -> int:
        aligned_dir = Path(aligned_dir)
        count = 0
        for img_path in FileManager.find_images(aligned_dir):
            meta = MetadataManager.load(img_path)
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            scale = target_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            cv2.imwrite(str(img_path), resized)

            if meta is not None:
                s = scale
                meta.landmarks_106 = (meta.landmarks_106.astype(np.float64) * s).astype(np.int64)
                if meta.source_landmarks_106 is not None:
                    meta.source_landmarks_106 = (meta.source_landmarks_106.astype(np.float64) * s).astype(np.int64)
                if meta.source_kps_5 is not None:
                    meta.source_kps_5 = (meta.source_kps_5.astype(np.float64) * s).astype(np.int64)
                if meta.source_rect is not None:
                    meta.source_rect = [meta.source_rect[0] * s, meta.source_rect[1] * s,
                                        meta.source_rect[2] * s, meta.source_rect[3] * s]
                if meta.source_face_rect is not None:
                    r = meta.source_face_rect
                    meta.source_face_rect = [r[0] * s, r[1] * s, r[2] * s, r[3] * s]
                if meta.image_to_face_mat is not None:
                    mat = meta.image_to_face_mat.astype(np.float64).copy()
                    mat[0, 2] *= s
                    mat[1, 2] *= s
                    meta.image_to_face_mat = mat
                MetadataManager.save(img_path, meta)
            count += 1

        _logger.info(f"Resized {count} faces to target size {target_size}")
        return count

    def recover_original_filename(self, aligned_dir: Path) -> int:
        aligned_dir = Path(aligned_dir)
        count = 0
        for img_path in FileManager.find_images(aligned_dir):
            meta = MetadataManager.load(img_path)
            if meta is None:
                continue

            original = meta.source_filename
            if not original:
                continue

            new_path = aligned_dir / original
            if new_path.exists() and new_path != img_path:
                _logger.warning(f"Skip recover: target already exists {new_path.name} (source: {img_path.name})")
                continue

            old_json = MetadataManager.sidecar_path(img_path)
            if img_path != new_path:
                img_path.rename(new_path)
                if old_json.exists():
                    old_json.rename(MetadataManager.sidecar_path(new_path))
                count += 1

        _logger.info(f"Recovered original filenames for {count} faces")
        return count

    def add_landmarks_debug_images(self, aligned_dir: Path, output_dir: Optional[Path] = None) -> int:
        aligned_dir = Path(aligned_dir)
        out_dir = Path(output_dir) if output_dir else aligned_dir / "debug"
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in FileManager.find_images(aligned_dir):
            meta = MetadataManager.load(img_path)
            if meta is None:
                continue

            from faceswap.core.landmarks106 import fill_hull_mask_106
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            lm = meta.landmarks_106.astype(np.float32)
            for pt_idx in range(lm.shape[0]):
                x, y = int(lm[pt_idx, 0]), int(lm[pt_idx, 1])
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

            hull_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            fill_hull_mask_106(hull_mask, lm)
            img[hull_mask > 0] = (img[hull_mask > 0] * 0.7).astype(np.uint8)

            debug_path = out_dir / f"debug_{img_path.name}"
            cv2.imwrite(str(debug_path), img)
            count += 1

        _logger.info(f"Generated {count} debug images with landmarks")
        return count

    def view_aligned_result(self, aligned_dir: Path) -> None:
        aligned_dir = Path(aligned_dir)
        import subprocess
        import sys
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(aligned_dir)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(aligned_dir)])
        else:
            subprocess.Popen(["xdg-open", str(aligned_dir)])

    def metadata_save(self, aligned_dir: Path) -> Path:
        output_path = aligned_dir / "meta.dat"
        MetadataManager.pack_metadata(aligned_dir, output_path)
        return output_path

    def metadata_restore(self, aligned_dir: Path) -> int:
        pack_path = aligned_dir / "meta.dat"
        if not pack_path.exists():
            raise FileNotFoundError(f"Metadata pack not found: {pack_path}")
        MetadataManager.unpack_metadata(pack_path, aligned_dir)
        return 0
