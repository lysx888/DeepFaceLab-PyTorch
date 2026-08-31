import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from faceswap.shared.logger import get_logger

_logger = get_logger("config")

_DEFAULT_CONFIG = {
    "workspace_dir": str(Path(__file__).resolve().parent.parent.parent / "workspace"),
    "device": "auto",
    "cuda_alloc_conf": "backend:cudaMallocAsync",
    "num_workers": 0,
    "pin_memory": True,
    "use_amp": True,
    "use_compile": True,
}


class Config:
    _instance: Optional["Config"] = None
    _config: dict[str, Any]
    _config_path: Optional[Path]

    def __new__(cls, config_path: Optional[Path] = None) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(config_path)
        return cls._instance

    def _init(self, config_path: Optional[Path] = None) -> None:
        self._config = dict(_DEFAULT_CONFIG)
        self._config_path = config_path
        if config_path is not None:
            self._load(config_path)
        self._apply_runtime_settings()

    def _load(self, config_path: Path) -> None:
        config_path = Path(config_path)
        if config_path.exists():
            try:
                with open(str(config_path), "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                self._config.update(loaded)
                _logger.info(f"Loaded config from {config_path}")
            except (json.JSONDecodeError, OSError) as e:
                _logger.warning(f"Failed to load config from {config_path}: {e}")

    def _apply_runtime_settings(self) -> None:
        cuda_alloc_conf = self._config.get("cuda_alloc_conf")
        if cuda_alloc_conf and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc_conf

    def save(self, config_path: Optional[Path] = None) -> None:
        path = config_path or self._config_path
        if path is None:
            _logger.warning("No config path specified, cannot save.")
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
        _logger.info(f"Saved config to {path}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._config[key] = value

    @property
    def workspace_dir(self) -> Path:
        return Path(self._config["workspace_dir"])

    @property
    def device(self) -> str:
        return self._config["device"]

    @property
    def use_amp(self) -> bool:
        return self._config["use_amp"]

    @property
    def num_workers(self) -> int:
        return self._config["num_workers"]

    @property
    def pin_memory(self) -> bool:
        return self._config["pin_memory"]

    @property
    def use_compile(self) -> bool:
        return self._config["use_compile"]

    def resolve_device(self) -> "torch.device":
        import torch
        device_str = self._config["device"]
        if device_str == "auto":
            if torch.cuda.is_available():
                device_str = "cuda"
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                device_str = "xpu"
            else:
                device_str = "cpu"
        return torch.device(device_str)

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


def auto_select_device() -> "torch.device":
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu:0')
    return torch.device('cpu')


def is_gpu_available() -> bool:
    import torch
    return torch.cuda.is_available() or (hasattr(torch, 'xpu') and torch.xpu.is_available())


def is_gpu_device(device: "torch.device") -> bool:
    return device.type in ('cuda', 'xpu')


def get_device_name(device: "torch.device" = None) -> str:
    import torch
    if device is None:
        device = auto_select_device()
    if device.type == 'cuda':
        return torch.cuda.get_device_name(device)
    if device.type == 'xpu':
        return torch.xpu.get_device_name(device)
    return "CPU"


def get_device_memory_mb(device: "torch.device" = None) -> int:
    import torch
    if device is None:
        device = auto_select_device()
    if device.type == 'cuda':
        return torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)
    if device.type == 'xpu':
        return torch.xpu.get_device_properties(device).total_memory // (1024 * 1024)
    return 0


def get_device_free_memory_mb(device: "torch.device" = None) -> int:
    import torch
    if device is None:
        device = auto_select_device()
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        return free // (1024 * 1024)
    if device.type == 'xpu':
        free, _ = torch.xpu.mem_get_info(device)
        return free // (1024 * 1024)
    return 0


def get_num_workers(device: "torch.device" = None) -> int:
    if device is None:
        device = auto_select_device()
    if not is_gpu_device(device):
        return 0

    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or (os.cpu_count() or 1)
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        physical = os.cpu_count() or 1
        avail_gb = 8.0

    max_by_cpu = max(1, physical // 2)
    mem_per_worker = 2.0
    if sys.platform == "win32":
        try:
            import psutil
            swap_total_gb = psutil.swap_memory().total / (1024 ** 3)
            if swap_total_gb < 8:
                mem_per_worker = 8.0
        except Exception:
            pass
    max_by_mem = max(1, int(avail_gb / mem_per_worker))
    return min(max_by_cpu, max_by_mem, 8)
