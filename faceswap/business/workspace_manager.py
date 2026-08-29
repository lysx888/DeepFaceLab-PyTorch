from pathlib import Path
from typing import Optional

from faceswap.setting import (
    WORKSPACE_DIR,
    DATA_SRC_DIR, DATA_DST_DIR,
    DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR,
    DATA_DST_ALIGNED_DEBUG_DIR,
    DATA_DST_MERGED_DIR, DATA_DST_MERGED_MASK_DIR,
    MODEL_DIR,
    DATA_SRC_VIDEO_PATTERN, DATA_DST_VIDEO_PATTERN,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from faceswap.shared.file_manager import FileManager
from faceswap.shared.logger import get_logger

_logger = get_logger("workspace_manager")


class WorkspaceManager:
    def __init__(self, workspace_dir: Optional[Path] = None) -> None:
        self._workspace_dir = Path(workspace_dir) if workspace_dir else WORKSPACE_DIR

    @property
    def workspace_dir(self) -> Path:
        return self._workspace_dir

    @property
    def data_src_dir(self) -> Path:
        return self._workspace_dir / "data_src"

    @property
    def data_dst_dir(self) -> Path:
        return self._workspace_dir / "data_dst"

    @property
    def data_src_aligned_dir(self) -> Path:
        return self._workspace_dir / "data_src" / "aligned"

    @property
    def data_dst_aligned_dir(self) -> Path:
        return self._workspace_dir / "data_dst" / "aligned"

    @property
    def data_dst_aligned_debug_dir(self) -> Path:
        return self._workspace_dir / "data_dst" / "aligned_debug"

    @property
    def data_dst_merged_dir(self) -> Path:
        return self._workspace_dir / "data_dst" / "merged"

    @property
    def data_dst_merged_mask_dir(self) -> Path:
        return self._workspace_dir / "data_dst" / "merged_mask"

    @property
    def model_dir(self) -> Path:
        return self._workspace_dir / "model"

    def find_src_video(self) -> Optional[Path]:
        return self._find_video("data_src")

    def find_dst_video(self) -> Optional[Path]:
        return self._find_video("data_dst")

    def _find_video(self, prefix: str) -> Optional[Path]:
        for ext in SUPPORTED_VIDEO_EXTENSIONS:
            path = self._workspace_dir / f"{prefix}{ext}"
            if path.exists():
                return path
        return None

    def ensure_structure(self) -> None:
        dirs = [
            self.data_src_dir,
            self.data_src_aligned_dir,
            self.data_dst_dir,
            self.data_dst_aligned_dir,
            self.data_dst_aligned_debug_dir,
            self.data_dst_merged_dir,
            self.data_dst_merged_mask_dir,
            self.model_dir,
        ]
        for d in dirs:
            FileManager.ensure_dir(d)
        _logger.info(f"Workspace structure ensured at {self._workspace_dir}")

    def clear_workspace(self) -> None:
        if self.check_training_running():
            raise RuntimeError("Cannot clear workspace: training task is running.")

        preserved_files = set()
        for video in [self.find_src_video(), self.find_dst_video()]:
            if video:
                preserved_files.add(video)

        dirs_to_clear = [
            self.data_src_dir,
            self.data_dst_dir,
            self.model_dir,
        ]
        for d in dirs_to_clear:
            if d.exists():
                FileManager.safe_delete_dir(d, self._workspace_dir)

        self.ensure_structure()

        for vf in preserved_files:
            if not vf.exists():
                import shutil
                target = self._workspace_dir / vf.name
                if vf != target:
                    shutil.copy2(str(vf), str(target))

        _logger.info("Workspace cleared and rebuilt.")

    def check_training_running(self) -> bool:
        lock_file = self.model_dir / ".training_lock"
        return lock_file.exists()
