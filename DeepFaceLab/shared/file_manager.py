import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("file_manager")


def imwrite_auto(path: Path, img: np.ndarray, jpg_quality: int = 100, png_compression: int = 0) -> bool:
    path = Path(path)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    elif path.suffix.lower() == ".png":
        return cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    else:
        return cv2.imwrite(str(path), img)


class FileManager:

    @staticmethod
    def atomic_write(target_path: Path, data: bytes | str, encoding: str = "utf-8") -> None:
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target_path.parent),
            prefix=".tmp_",
            suffix=target_path.suffix,
        )
        try:
            if isinstance(data, str):
                with os.fdopen(fd, "w", encoding=encoding) as f:
                    f.write(data)
            else:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
            shutil.move(tmp_path, str(target_path))
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @staticmethod
    def ensure_dir(path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def safe_delete_dir(path: Path, workspace_root: Path) -> None:
        path = Path(path).resolve()
        workspace_root = Path(workspace_root).resolve()
        if not str(path).startswith(str(workspace_root)):
            raise ValueError(f"Cannot delete directory outside workspace: {path}")
        if path.exists():
            shutil.rmtree(str(path))
            _logger.info(f"Deleted directory: {path}")

    @staticmethod
    def validate_workspace_path(path: Path, workspace_root: Path) -> bool:
        try:
            resolved = Path(path).resolve()
            workspace_resolved = Path(workspace_root).resolve()
            return str(resolved).startswith(str(workspace_resolved))
        except (OSError, ValueError):
            return False

    @staticmethod
    def find_images(directory: Path, extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> list[Path]:
        directory = Path(directory)
        if not directory.exists():
            return []
        return sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )

    @staticmethod
    def find_videos(directory: Path, extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm")) -> list[Path]:
        directory = Path(directory)
        if not directory.exists():
            return []
        return sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )

    @staticmethod
    def get_unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1
