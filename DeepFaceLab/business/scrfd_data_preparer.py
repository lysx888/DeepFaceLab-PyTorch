import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from DeepFaceLab.shared.file_manager import FileManager, imwrite_auto
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("scrfd_data_preparer")


@dataclass
class SCRFDDataStats:
    train_images: int = 0
    val_images: int = 0
    total_faces: int = 0
    missing_images: int = 0
    skipped_annotations: int = 0


class SCRFDDataPreparer:
    def __init__(self, workspace_dir: Path):
        self._workspace_dir = Path(workspace_dir)
        self._train_dir = self._workspace_dir / "insightface_train"
        self._scrfd_dir = self._train_dir / "scrfd"
        self._manual_annotated_dir = self._train_dir / "manual_annotated"

    @property
    def scrfd_data_dir(self) -> Path:
        return self._scrfd_dir

    def prepare_from_manual_annotated(self, val_ratio: float = 0.1, seed: int = 42) -> SCRFDDataStats:
        if not self._manual_annotated_dir.exists():
            raise FileNotFoundError(f"manual_annotated目录不存在: {self._manual_annotated_dir}")

        annotations = self._load_manual_annotated()
        if not annotations:
            raise FileNotFoundError(f"manual_annotated目录为空: {self._manual_annotated_dir}")

        random.seed(seed)
        indices = list(range(len(annotations)))
        random.shuffle(indices)
        val_count = max(1, int(len(indices) * val_ratio)) if len(indices) > 1 else 0
        val_set = set(indices[:val_count])

        train_images_dir = self._scrfd_dir / "train" / "images"
        val_images_dir = self._scrfd_dir / "val" / "images"
        FileManager.ensure_dir(train_images_dir)
        FileManager.ensure_dir(val_images_dir)

        train_lines = []
        val_lines = []
        stats = SCRFDDataStats()

        for i, ann in enumerate(annotations):
            is_val = i in val_set
            img_dir = val_images_dir if is_val else train_images_dir
            lines = val_lines if is_val else train_lines

            src_img_path = ann.get("source_image_path")
            if src_img_path and Path(src_img_path).exists():
                src_path = Path(src_img_path)
            else:
                src_filename = ann.get("source_filename", "")
                src_path = self._workspace_dir / "data_src" / src_filename
                if not src_path.exists():
                    for ext in [".png", ".jpg", ".jpeg"]:
                        alt = src_path.with_suffix(ext)
                        if alt.exists():
                            src_path = alt
                            break

            if not src_path.exists():
                stats.missing_images += 1
                _logger.warning(f"Source image not found, skipping: {src_path}")
                continue

            seq = f"{i:06d}"
            dest_ext = src_path.suffix
            dest_name = seq + dest_ext
            dest_path = img_dir / dest_name

            img = cv2.imread(str(src_path))
            if img is None:
                stats.missing_images += 1
                _logger.warning(f"Cannot read image, skipping: {src_path}")
                continue

            imwrite_auto(dest_path, img)

            bbox = ann.get("bbox")
            kps_5 = ann.get("kps_5")

            if bbox is None:
                stats.skipped_annotations += 1
                _logger.warning(f"No bbox in annotation {i}, skipping")
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            kps_str = ""
            if kps_5 and len(kps_5) >= 10:
                kps_str = " " + " ".join(f"{float(v):.1f}" for v in kps_5[:10])

            line = f"{dest_name} 1 {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} 1.0{kps_str}"
            lines.append(line)
            stats.total_faces += 1

            if is_val:
                stats.val_images += 1
            else:
                stats.train_images += 1

        if train_lines:
            train_label_path = self._scrfd_dir / "train" / "labelv2.txt"
            with open(str(train_label_path), "w", encoding="utf-8") as f:
                f.write("\n".join(train_lines) + "\n")

        if val_lines:
            val_label_path = self._scrfd_dir / "val" / "labelv2.txt"
            with open(str(val_label_path), "w", encoding="utf-8") as f:
                f.write("\n".join(val_lines) + "\n")

        _logger.info(f"SCRFD data prepared: train={stats.train_images}, val={stats.val_images}, faces={stats.total_faces}")
        return stats

    def prepare_from_widerface(self, widerface_dir: Path, val_ratio: float = 0.1, seed: int = 42) -> SCRFDDataStats:
        widerface_dir = Path(widerface_dir)
        if not widerface_dir.exists():
            raise FileNotFoundError(f"WIDERFace目录不存在: {widerface_dir}")

        annotation_file = widerface_dir / "wider_face_train_bbx_gt.txt"
        if not annotation_file.exists():
            raise FileNotFoundError(f"WIDERFace标注文件不存在: {annotation_file}")

        with open(str(annotation_file), "r", encoding="utf-8") as f:
            lines = f.readlines()

        train_images_dir = self._scrfd_dir / "train" / "images"
        val_images_dir = self._scrfd_dir / "val" / "images"
        FileManager.ensure_dir(train_images_dir)
        FileManager.ensure_dir(val_images_dir)

        stats = SCRFDDataStats()
        train_lines = []
        val_lines = []
        idx = 0

        i = 0
        while i < len(lines):
            img_name = lines[i].strip()
            i += 1
            if not img_name:
                continue

            img_path = widerface_dir / "images" / img_name
            if not img_path.exists():
                stats.missing_images += 1
                if i < len(lines):
                    try:
                        num_faces = int(lines[i].strip())
                        i += 1 + num_faces
                    except ValueError:
                        pass
                continue

            if i >= len(lines):
                break

            try:
                num_faces = int(lines[i].strip())
            except ValueError:
                continue
            i += 1

            faces = []
            for _ in range(num_faces):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                i += 1
                if len(parts) >= 4:
                    x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    faces.append((x, y, x + w, y + h))

            if not faces:
                continue

            random.seed(seed + idx)
            is_val = random.random() < val_ratio
            img_dir = val_images_dir if is_val else train_images_dir
            lines_list = val_lines if is_val else train_lines

            seq = f"{idx:06d}"
            dest_ext = img_path.suffix
            dest_name = seq + dest_ext
            dest_path = img_dir / dest_name

            img = cv2.imread(str(img_path))
            if img is None:
                stats.missing_images += 1
                continue

            imwrite_auto(dest_path, img)

            face_strs = []
            for x1, y1, x2, y2 in faces:
                kps_str = " " + " ".join(["0.0"] * 10)
                face_strs.append(f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} 1.0{kps_str}")

            line = f"{dest_name} {len(faces)} " + " ".join(face_strs)
            lines_list.append(line)
            stats.total_faces += len(faces)

            if is_val:
                stats.val_images += 1
            else:
                stats.train_images += 1
            idx += 1

        if train_lines:
            with open(str(self._scrfd_dir / "train" / "labelv2.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(train_lines) + "\n")

        if val_lines:
            with open(str(self._scrfd_dir / "val" / "labelv2.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(val_lines) + "\n")

        _logger.info(f"SCRFD data from WIDERFace: train={stats.train_images}, val={stats.val_images}")
        return stats

    def validate_data(self) -> SCRFDDataStats:
        stats = SCRFDDataStats()
        for split in ["train", "val"]:
            label_path = self._scrfd_dir / split / "labelv2.txt"
            images_dir = self._scrfd_dir / split / "images"
            if not label_path.exists():
                continue
            with open(str(label_path), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    img_name = parts[0]
                    if (images_dir / img_name).exists():
                        if split == "train":
                            stats.train_images += 1
                        else:
                            stats.val_images += 1
                    else:
                        stats.missing_images += 1
                    if len(parts) >= 3:
                        try:
                            stats.total_faces += int(parts[1])
                        except ValueError:
                            pass
        _logger.info(f"SCRFD data validation: train={stats.train_images}, val={stats.val_images}, missing={stats.missing_images}")
        return stats

    def _load_manual_annotated(self) -> list[dict]:
        annotations = []
        if not self._manual_annotated_dir.exists():
            return annotations
        for json_path in sorted(self._manual_annotated_dir.glob("*.json")):
            try:
                with open(str(json_path), "r", encoding="utf-8") as f:
                    ann = json.load(f)
                ann["_json_path"] = str(json_path)
                annotations.append(ann)
            except Exception as e:
                _logger.warning(f"Failed to load annotation {json_path}: {e}")
        return annotations
