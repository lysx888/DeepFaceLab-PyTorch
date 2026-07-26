import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from DeepFaceLab.core.metadata_manager import MetadataManager, FaceMetadata
from DeepFaceLab.models.face_enhancer import FaceEnhancer
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

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
        import io
        pack_data = {}
        for img_path in FileManager.find_images(aligned_dir):
            with open(str(img_path), "rb") as f:
                img_bytes = f.read()
            meta = MetadataManager.load(img_path)
            entry = {"image": img_bytes.hex()}
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
        with open(str(pack_path), "r", encoding="utf-8") as f:
            pack_data = json.load(f)

        count = 0
        for filename, entry in pack_data.items():
            img_bytes = bytes.fromhex(entry["image"])
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
                meta.landmarks_106 = (meta.landmarks_106.astype(np.float64) * scale).astype(np.int64)
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
                continue

            old_json = MetadataManager._sidecar_path(img_path)
            if img_path != new_path:
                img_path.rename(new_path)
                if old_json.exists():
                    old_json.rename(MetadataManager._sidecar_path(new_path))
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

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            for pt_idx in range(meta.landmarks_106.shape[0]):
                x, y = int(meta.landmarks_106[pt_idx, 0]), int(meta.landmarks_106[pt_idx, 1])
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

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
