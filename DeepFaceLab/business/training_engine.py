import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("training_engine")


@dataclass
class TrainingConfig:
    resolution: int = 128
    batch_size: int = 4
    learning_rate: float = 1e-4
    use_amp: bool = True
    save_interval_min: float = 15.0
    preview_interval_min: float = 15.0
    num_workers: int = 0
    pin_memory: bool = True
    use_compile: bool = True
    ddp_backend: Optional[str] = None


@dataclass
class TrainingState:
    iteration: int = 0
    loss_src: float = 0.0
    loss_dst: float = 0.0
    elapsed_min: float = 0.0


class TrainingEngine:

    def __init__(
        self,
        model_dir: Path,
        config: TrainingConfig,
        device: "torch.device",
    ) -> None:
        import torch
        self._model_dir = Path(model_dir)
        self._config = config
        self._device = device
        self._model_src: Optional[torch.nn.Module] = None
        self._model_dst: Optional[torch.nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scaler: Optional[torch.cuda.amp.GradScaler] = None
        self._state = TrainingState()
        self._start_time: float = 0.0
        self._last_save_time: float = 0.0
        self._last_preview_time: float = 0.0

    def setup(
        self,
        model_src: "torch.nn.Module",
        model_dst: "torch.nn.Module",
        optimizer: "torch.optim.Optimizer",
    ) -> None:
        import torch
        self._model_src = model_src.to(self._device)
        self._model_dst = model_dst.to(self._device)
        self._optimizer = optimizer

        if self._config.use_amp and self._device.type == "cuda":
            self._scaler = torch.cuda.amp.GradScaler()

        if self._config.use_compile:
            try:
                self._model_src = torch.compile(self._model_src)
                self._model_dst = torch.compile(self._model_dst)
                _logger.info("Models compiled with torch.compile().")
            except Exception as e:
                _logger.warning(f"torch.compile() failed, skipping: {e}")

        self._start_time = time.time()
        self._last_save_time = self._start_time
        self._last_preview_time = self._start_time

    def train_step(
        self,
        batch_src: dict[str, "torch.Tensor"],
        batch_dst: dict[str, "torch.Tensor"],
    ) -> TrainingState:
        import torch

        self._optimizer.zero_grad(set_to_none=True)

        if self._config.use_amp and self._scaler is not None:
            with torch.cuda.amp.autocast():
                loss_src = self._forward_src(batch_src)
                loss_dst = self._forward_dst(batch_dst)
                loss = loss_src + loss_dst

            self._scaler.scale(loss).backward()
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            loss_src = self._forward_src(batch_src)
            loss_dst = self._forward_dst(batch_dst)
            loss = loss_src + loss_dst
            loss.backward()
            self._optimizer.step()

        self._state.iteration += 1
        self._state.loss_src = loss_src.item() if isinstance(loss_src, torch.Tensor) else float(loss_src)
        self._state.loss_dst = loss_dst.item() if isinstance(loss_dst, torch.Tensor) else float(loss_dst)
        self._state.elapsed_min = (time.time() - self._start_time) / 60.0

        return self._state

    def _forward_src(self, batch: dict[str, "torch.Tensor"]) -> "torch.Tensor":
        raise NotImplementedError("Subclass must implement _forward_src")

    def _forward_dst(self, batch: dict[str, "torch.Tensor"]) -> "torch.Tensor":
        raise NotImplementedError("Subclass must implement _forward_dst")

    def should_save(self) -> bool:
        elapsed = (time.time() - self._last_save_time) / 60.0
        return elapsed >= self._config.save_interval_min

    def should_preview(self) -> bool:
        elapsed = (time.time() - self._last_preview_time) / 60.0
        return elapsed >= self._config.preview_interval_min

    def save_model(self, model_name: str) -> None:
        import torch
        self._model_dir.mkdir(parents=True, exist_ok=True)

        if self._model_src is not None:
            src_path = self._model_dir / f"{model_name}_decoder_src.pt"
            FileManager.atomic_write(src_path, self._serialize_model(self._model_src))

        if self._model_dst is not None:
            dst_path = self._model_dir / f"{model_name}_decoder_dst.pt"
            FileManager.atomic_write(dst_path, self._serialize_model(self._model_dst))

        if self._optimizer is not None:
            opt_path = self._model_dir / f"{model_name}_optimizer.pt"
            FileManager.atomic_write(opt_path, self._serialize_optimizer(self._optimizer))

        config_path = self._model_dir / f"{model_name}_config.json"
        import json
        config_str = json.dumps(self._config.__dict__, default=str, ensure_ascii=False)
        FileManager.atomic_write(config_path, config_str)

        self._last_save_time = time.time()
        _logger.info(f"Model '{model_name}' saved at iteration {self._state.iteration}")

    def load_model(self, model_name: str) -> bool:
        import torch
        config_path = self._model_dir / f"{model_name}_config.json"
        if not config_path.exists():
            return False

        if self._model_src is not None:
            src_path = self._model_dir / f"{model_name}_decoder_src.pt"
            if src_path.exists():
                state = torch.load(str(src_path), map_location=self._device, weights_only=True)
                self._model_src.load_state_dict(state)

        if self._model_dst is not None:
            dst_path = self._model_dir / f"{model_name}_decoder_dst.pt"
            if dst_path.exists():
                state = torch.load(str(dst_path), map_location=self._device, weights_only=True)
                self._model_dst.load_state_dict(state)

        if self._optimizer is not None:
            opt_path = self._model_dir / f"{model_name}_optimizer.pt"
            if opt_path.exists():
                state = torch.load(str(opt_path), map_location=self._device, weights_only=True)
                self._optimizer.load_state_dict(state)

        _logger.info(f"Model '{model_name}' loaded for resume training.")
        return True

    @staticmethod
    def _serialize_model(model: "torch.nn.Module") -> bytes:
        import io
        import torch
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.getvalue()

    @staticmethod
    def _serialize_optimizer(optimizer: "torch.optim.Optimizer") -> bytes:
        import io
        import torch
        buf = io.BytesIO()
        torch.save(optimizer.state_dict(), buf)
        return buf.getvalue()

    def generate_preview(
        self,
        sample_src: "torch.Tensor",
        sample_dst: "torch.Tensor",
    ) -> npt.NDArray[np.uint8]:
        import torch
        with torch.inference_mode():
            pred_src = self._model_src(sample_src.unsqueeze(0).to(self._device))
            pred_dst = self._model_dst(sample_dst.unsqueeze(0).to(self._device))

        from DeepFaceLab.shared.image_utils import ImageUtils
        src_np = ImageUtils.tensor_to_numpy(pred_src.squeeze(0).cpu())
        dst_np = ImageUtils.tensor_to_numpy(pred_dst.squeeze(0).cpu())
        preview = np.concatenate([src_np, dst_np], axis=1)
        self._last_preview_time = time.time()
        return preview

    def get_summary_text(self) -> str:
        s = self._state
        return (
            f"Iteration: {s.iteration} | "
            f"Loss SRC: {s.loss_src:.6f} | Loss DST: {s.loss_dst:.6f} | "
            f"Elapsed: {s.elapsed_min:.1f} min"
        )

    @property
    def state(self) -> TrainingState:
        return self._state

    @property
    def config(self) -> TrainingConfig:
        return self._config

    @property
    def device(self) -> "torch.device":
        return self._device
