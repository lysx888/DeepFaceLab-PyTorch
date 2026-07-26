from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("base_model")


class BaseModel(ABC):
    def __init__(self, name: str, resolution: int = 128):
        self._name = name
        self._resolution = resolution
        self._encoder: Optional[nn.Module] = None
        self._decoder_src: Optional[nn.Module] = None
        self._decoder_dst: Optional[nn.Module] = None
        self._inter: Optional[nn.Module] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def resolution(self) -> int:
        return self._resolution

    @abstractmethod
    def get_encoder(self) -> nn.Module:
        ...

    @abstractmethod
    def get_decoder_src(self) -> nn.Module:
        ...

    @abstractmethod
    def get_decoder_dst(self) -> nn.Module:
        ...

    def get_inter(self) -> Optional[nn.Module]:
        return None

    @abstractmethod
    def forward_src(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def forward_dst(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def get_config(self) -> dict:
        return {
            "name": self._name,
            "resolution": self._resolution,
        }

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "BaseModel":
        ...

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        import json
        config_str = json.dumps(self.get_config(), ensure_ascii=False)
        FileManager.atomic_write(model_dir / f"{self._name}_config.json", config_str)

        if self._encoder is not None:
            FileManager.atomic_write(model_dir / f"{self._name}_encoder.pt", self._serialize(self._encoder))

        if self._decoder_src is not None:
            FileManager.atomic_write(model_dir / f"{self._name}_decoder_src.pt", self._serialize(self._decoder_src))

        if self._decoder_dst is not None:
            FileManager.atomic_write(model_dir / f"{self._name}_decoder_dst.pt", self._serialize(self._decoder_dst))

        if self._inter is not None:
            FileManager.atomic_write(model_dir / f"{self._name}_inter.pt", self._serialize(self._inter))

        _logger.info(f"Model '{self._name}' saved to {model_dir}")

    def load(self, model_dir: Path, device: torch.device = None) -> bool:
        model_dir = Path(model_dir)
        config_path = model_dir / f"{self._name}_config.json"
        if not config_path.exists():
            return False

        map_loc = device if device else "cpu"

        enc_path = model_dir / f"{self._name}_encoder.pt"
        if enc_path.exists() and self._encoder is not None:
            self._encoder.load_state_dict(torch.load(str(enc_path), map_location=map_loc, weights_only=True))

        src_path = model_dir / f"{self._name}_decoder_src.pt"
        if src_path.exists() and self._decoder_src is not None:
            self._decoder_src.load_state_dict(torch.load(str(src_path), map_location=map_loc, weights_only=True))

        dst_path = model_dir / f"{self._name}_decoder_dst.pt"
        if dst_path.exists() and self._decoder_dst is not None:
            self._decoder_dst.load_state_dict(torch.load(str(dst_path), map_location=map_loc, weights_only=True))

        inter_path = model_dir / f"{self._name}_inter.pt"
        if inter_path.exists() and self._inter is not None:
            self._inter.load_state_dict(torch.load(str(inter_path), map_location=map_loc, weights_only=True))

        _logger.info(f"Model '{self._name}' loaded from {model_dir}")
        return True

    @staticmethod
    def _serialize(module: nn.Module) -> bytes:
        import io
        buf = io.BytesIO()
        torch.save(module.state_dict(), buf)
        return buf.getvalue()

    def parameters_src(self) -> list:
        params = []
        if self._encoder is not None:
            params += list(self._encoder.parameters())
        if self._decoder_src is not None:
            params += list(self._decoder_src.parameters())
        if self._inter is not None:
            params += list(self._inter.parameters())
        return params

    def parameters_dst(self) -> list:
        params = []
        if self._encoder is not None:
            params += list(self._encoder.parameters())
        if self._decoder_dst is not None:
            params += list(self._decoder_dst.parameters())
        if self._inter is not None:
            params += list(self._inter.parameters())
        return params
