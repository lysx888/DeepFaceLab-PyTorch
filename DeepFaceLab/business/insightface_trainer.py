import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from DeepFaceLab.setting import (
    INSIGHTFACE_TRAIN_DIR, INSIGHTFACE_SCRFD_DIR, INSIGHTFACE_SYNTHETICS_DIR,
    INSIGHTFACE_OUTPUT_DIR, WORKSPACE_DIR, INSIGHTFACE_DIR,
)
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("insightface_trainer")

_SCRFD_PROGRESS_PATTERN = re.compile(
    r"Epoch\s+\[(\d+)\]\[(\d+)/(\d+)\].*?loss:\s+([\d.]+)"
)
_SYNTHETICS_PROGRESS_PATTERN = re.compile(
    r"Epoch (\d+).*?train_loss_step.*?([\d.]+)"
)
_SYNTHETICS_VAL_PATTERN = re.compile(
    r"Epoch (\d+).*?val_loss_step.*?([\d.]+)"
)


@dataclass
class SCRFDTrainConfig:
    config_name: str = "dfl_scrfd_10g_bnkps"
    gpus: list[int] = field(default_factory=lambda: [0])
    resume_from: Optional[str] = None


@dataclass
class SyntheticsTrainConfig:
    backbone: str = "resnet50d"
    batch_size: int = 64
    num_gpus: int = 1
    max_epochs: int = 80


@dataclass
class TrainMeta:
    model_type: str
    train_start_time: str
    dataset_path: str
    framework: str
    framework_version: str = ""
    pytorch_version: str = ""
    cuda_version: str = ""
    gpu_count: int = 0
    train_samples: int = 0
    val_samples: int = 0
    checkpoint_path: str = ""
    config_file: str = ""
    backbone: str = ""


@dataclass
class SCRFDProgress:
    epoch: int = 0
    iter_cur: int = 0
    iter_total: int = 1
    loss: float = 0.0


@dataclass
class SyntheticsProgress:
    epoch: int = 0
    loss: float = 0.0
    val_loss: float = 0.0


class InsightFaceTrainer:
    def __init__(self, workspace_dir: Optional[Path] = None):
        self._workspace_dir = Path(workspace_dir) if workspace_dir else WORKSPACE_DIR
        self._train_dir = self._workspace_dir / "insightface_train"
        self._scrfd_dir = self._train_dir / "scrfd"
        self._synthetics_dir = self._train_dir / "synthetics"
        self._output_dir = self._train_dir / "output"
        self._process_manager = None
        self._current_model_type: Optional[str] = None
        self._train_start_time: Optional[datetime] = None
        self._scrfd_progress = SCRFDProgress()
        self._synthetics_progress = SyntheticsProgress()
        self._on_progress: Optional[Callable] = None

    @property
    def train_dir(self) -> Path:
        return self._train_dir

    @property
    def scrfd_dir(self) -> Path:
        return self._scrfd_dir

    @property
    def synthetics_dir(self) -> Path:
        return self._synthetics_dir

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def scrfd_progress(self) -> SCRFDProgress:
        return self._scrfd_progress

    @property
    def synthetics_progress(self) -> SyntheticsProgress:
        return self._synthetics_progress

    def init_train_dirs(self) -> dict[str, Path]:
        dirs = {
            "train_root": self._train_dir,
            "scrfd_train_images": self._scrfd_dir / "train" / "images",
            "scrfd_val_images": self._scrfd_dir / "val" / "images",
            "synthetics_root": self._synthetics_dir,
            "output_scrfd": self._output_dir / "scrfd",
            "output_synthetics": self._output_dir / "synthetics",
        }
        for name, d in dirs.items():
            FileManager.ensure_dir(d)
        _logger.info(f"Training directories initialized: {self._train_dir}")
        return dirs

    def _check_scrfd_prerequisites(self):
        train_images = self._scrfd_dir / "train" / "images"
        val_images = self._scrfd_dir / "val" / "images"
        label_file = self._scrfd_dir / "train" / "labelv2.txt"
        if not train_images.exists():
            raise FileNotFoundError(f"SCRFD训练数据目录不存在: {train_images}\n请先执行数据准备命令")
        if not label_file.exists():
            raise FileNotFoundError(f"SCRFD标注文件不存在: {label_file}\n请先执行数据准备命令")
        train_imgs = list(train_images.iterdir())
        if not train_imgs:
            raise FileNotFoundError(f"SCRFD训练图片目录为空: {train_images}\n请先执行数据准备命令")
        try:
            import mmdet
        except ImportError:
            raise RuntimeError("mmdetection未安装，请执行: pip install mmdet")

    def _check_synthetics_prerequisites(self):
        annot_file = self._synthetics_dir / "annot.pkl"
        if not self._synthetics_dir.exists():
            raise FileNotFoundError(f"2d106训练数据目录不存在: {self._synthetics_dir}\n请先执行数据准备命令")
        if not annot_file.exists():
            raise FileNotFoundError(f"2d106标注文件不存在: {annot_file}\n请先执行数据准备命令")
        try:
            import pytorch_lightning
        except ImportError:
            raise RuntimeError("pytorch_lightning未安装，请执行: pip install pytorch-lightning")

    def train_scrfd(
        self,
        config: Optional[SCRFDTrainConfig] = None,
        on_progress: Optional[Callable] = None,
    ) -> None:
        from DeepFaceLab.core.training_process_manager import TrainingProcessManager

        if config is None:
            config = SCRFDTrainConfig()

        self._check_scrfd_prerequisites()

        if self.is_training_running():
            raise RuntimeError("已有训练进程正在运行")

        self._process_manager = TrainingProcessManager()
        self._current_model_type = "scrfd"
        self._train_start_time = datetime.now()
        self._scrfd_progress = SCRFDProgress()
        self._on_progress = on_progress

        command = self._build_scrfd_train_command(config)
        output_dir = self._output_dir / "scrfd"
        FileManager.ensure_dir(output_dir)

        versions = self._get_versions()
        meta = TrainMeta(
            model_type="scrfd",
            train_start_time=self._train_start_time.isoformat(),
            dataset_path=str(self._scrfd_dir),
            framework="mmdetection",
            framework_version=versions.get("framework_version", ""),
            pytorch_version=versions.get("pytorch_version", ""),
            cuda_version=versions.get("cuda_version", ""),
            gpu_count=len(config.gpus),
            config_file=config.config_name,
            checkpoint_path=str(output_dir),
        )
        self._write_train_meta(output_dir, meta)

        _logger.info(f"Starting SCRFD training: {' '.join(command)}")
        self._process_manager.start_process(
            command,
            on_stdout=self._on_scrfd_stdout,
            on_stderr=self._on_scrfd_stderr,
        )

    def train_synthetics(
        self,
        config: Optional[SyntheticsTrainConfig] = None,
        on_progress: Optional[Callable] = None,
    ) -> None:
        from DeepFaceLab.core.training_process_manager import TrainingProcessManager

        if config is None:
            config = SyntheticsTrainConfig()

        self._check_synthetics_prerequisites()

        if self.is_training_running():
            raise RuntimeError("已有训练进程正在运行")

        self._process_manager = TrainingProcessManager()
        self._current_model_type = "synthetics"
        self._train_start_time = datetime.now()
        self._synthetics_progress = SyntheticsProgress()
        self._on_progress = on_progress

        command = self._build_synthetics_train_command(config)
        output_dir = self._output_dir / "synthetics"
        FileManager.ensure_dir(output_dir)

        versions = self._get_versions()
        meta = TrainMeta(
            model_type="synthetics",
            train_start_time=self._train_start_time.isoformat(),
            dataset_path=str(self._synthetics_dir),
            framework="pytorch_lightning",
            framework_version=versions.get("framework_version", ""),
            pytorch_version=versions.get("pytorch_version", ""),
            cuda_version=versions.get("cuda_version", ""),
            gpu_count=config.num_gpus,
            backbone=config.backbone,
            checkpoint_path=str(output_dir),
        )
        self._write_train_meta(output_dir, meta)

        _logger.info(f"Starting 2d106 training: {' '.join(command)}")
        self._process_manager.start_process(
            command,
            on_stdout=self._on_synthetics_stdout,
            on_stderr=self._on_synthetics_stderr,
        )

    def stop_training(self) -> Optional[int]:
        if self._process_manager is None or not self._process_manager.is_running():
            _logger.info("No training process running.")
            return None
        exit_code = self._process_manager.stop_process()
        _logger.info(f"Training stopped (exit code: {exit_code})")
        return exit_code

    def is_training_running(self) -> bool:
        return self._process_manager is not None and self._process_manager.is_running()

    def _build_scrfd_train_command(self, config: SCRFDTrainConfig) -> list[str]:
        scrfd_root = INSIGHTFACE_DIR / "detection" / "scrfd"
        train_script = scrfd_root / "tools" / "train.py"
        config_file = scrfd_root / "configs" / "scrfd" / f"{config.config_name}.py"
        work_dir = self._output_dir / "scrfd"

        cmd = [
            sys.executable,
            str(train_script),
            str(config_file),
            "--work-dir", str(work_dir),
        ]

        if config.gpus:
            gpu_ids = ",".join(str(g) for g in config.gpus)
            cmd.extend(["--gpu-ids", gpu_ids])

        if config.resume_from:
            cmd.extend(["--resume-from", config.resume_from])

        return cmd

    def _build_synthetics_train_command(self, config: SyntheticsTrainConfig) -> list[str]:
        synthetics_root = INSIGHTFACE_DIR / "alignment" / "synthetics"
        train_script = synthetics_root / "trainer_synthetics.py"

        cmd = [
            sys.executable,
            str(train_script),
            "--root", str(self._synthetics_dir),
            "--backbone", config.backbone,
            "--batch_size", str(config.batch_size),
            "--num-gpus", str(config.num_gpus),
        ]

        return cmd

    def _parse_scrfd_log_line(self, line: str) -> Optional[SCRFDProgress]:
        match = _SCRFD_PROGRESS_PATTERN.search(line)
        if not match:
            return None
        self._scrfd_progress.epoch = int(match.group(1))
        self._scrfd_progress.iter_cur = int(match.group(2))
        self._scrfd_progress.iter_total = int(match.group(3))
        self._scrfd_progress.loss = float(match.group(4))
        return self._scrfd_progress

    def _parse_synthetics_log_line(self, line: str) -> Optional[SyntheticsProgress]:
        train_match = _SYNTHETICS_PROGRESS_PATTERN.search(line)
        if train_match:
            self._synthetics_progress.epoch = int(train_match.group(1))
            self._synthetics_progress.loss = float(train_match.group(2))
            return self._synthetics_progress

        val_match = _SYNTHETICS_VAL_PATTERN.search(line)
        if val_match:
            self._synthetics_progress.val_loss = float(val_match.group(2))
            return self._synthetics_progress

        return None

    def _on_scrfd_stdout(self, line: str) -> None:
        progress = self._parse_scrfd_log_line(line)
        if progress and self._on_progress:
            self._on_progress(progress)
        _logger.info(f"[SCRFD stdout] {line}")

    def _on_scrfd_stderr(self, line: str) -> None:
        _logger.warning(f"[SCRFD stderr] {line}")

    def _on_synthetics_stdout(self, line: str) -> None:
        progress = self._parse_synthetics_log_line(line)
        if progress and self._on_progress:
            self._on_progress(progress)
        _logger.info(f"[2d106 stdout] {line}")

    def _on_synthetics_stderr(self, line: str) -> None:
        _logger.warning(f"[2d106 stderr] {line}")

    def _write_train_meta(self, output_subdir: Path, meta: TrainMeta):
        import json
        meta_path = output_subdir / "train_meta.json"
        meta_dict = {
            "model_type": meta.model_type,
            "train_start_time": meta.train_start_time,
            "dataset_path": meta.dataset_path,
            "framework": meta.framework,
            "framework_version": meta.framework_version,
            "pytorch_version": meta.pytorch_version,
            "cuda_version": meta.cuda_version,
            "gpu_count": meta.gpu_count,
            "train_samples": meta.train_samples,
            "val_samples": meta.val_samples,
            "checkpoint_path": meta.checkpoint_path,
            "config_file": meta.config_file,
            "backbone": meta.backbone,
        }
        FileManager.ensure_dir(output_subdir)
        with open(str(meta_path), "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, ensure_ascii=False, indent=2)
        _logger.info(f"Train meta written: {meta_path}")

    def _get_versions(self) -> dict[str, str]:
        import torch
        versions = {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "cpu",
        }
        try:
            import mmdet
            versions["framework_version"] = mmdet.__version__
        except ImportError:
            pass
        try:
            import pytorch_lightning
            versions["framework_version"] = pytorch_lightning.__version__
        except ImportError:
            pass
        return versions
