"""
vram_manager.py - 统一显存/内存管理模块

基于 ComfyUI model_management.py 的核心逻辑，提取适合本项目的独立实现。
支持: VRAM 状态检测、dtype 选择、内存预算、模型加载调度、OOM 处理。

设计原则:
- 延迟导入 torch，避免模块导入时卡顿
- 线程安全（全局状态用锁保护）
- 适配 6GB 低显存场景
"""

from __future__ import annotations

import gc
import threading
import weakref
from contextlib import contextmanager
from enum import Enum
from typing import Callable

import psutil

from faceswap.shared.logger import get_logger

_logger = get_logger(__name__)


_torch = None
_torch_imported = False
_init_lock = threading.Lock()
_initialized = False


def _ensure_torch():
    global _torch, _torch_imported
    if _torch_imported:
        return _torch
    import torch
    _torch = torch
    _torch_imported = True
    return torch


class VRAMState(Enum):
    NO_VRAM = 0
    LOW_VRAM = 1
    NORMAL_VRAM = 2
    HIGH_VRAM = 3


vram_state: VRAMState = VRAMState.NORMAL_VRAM
total_vram: int = 0
total_ram: int = 0
vram_headroom: float = 0.5

EXTRA_RESERVED_VRAM = 600 * 1024 * 1024
MIN_WEIGHT_MEMORY_RATIO = 0.0
LOW_VRAM_THRESHOLD = 4 * 1024 * 1024 * 1024
NORMAL_VRAM_THRESHOLD = 8 * 1024 * 1024 * 1024

_loaded_models: list = []
_models_lock = threading.Lock()


def initialize(force_low_vram: bool = False, force_no_vram: bool = False,
               headroom_gb: float = 0.5):
    """
    初始化 VRAM 管理器，检测显存并设置状态。
    必须在使用其他功能前调用一次。
    """
    global vram_state, total_vram, total_ram, vram_headroom, _initialized

    with _init_lock:
        if _initialized and not force_low_vram and not force_no_vram:
            return

        torch = _ensure_torch()
        vram_headroom = headroom_gb

        total_ram = psutil.virtual_memory().total

        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(0)
                total_vram = total
            except Exception:
                total_vram = 0
        else:
            total_vram = 0

        if force_no_vram or total_vram == 0:
            vram_state = VRAMState.NO_VRAM
        elif force_low_vram:
            vram_state = VRAMState.LOW_VRAM
        elif total_vram >= NORMAL_VRAM_THRESHOLD:
            vram_state = VRAMState.NORMAL_VRAM
        elif total_vram >= LOW_VRAM_THRESHOLD:
            vram_state = VRAMState.LOW_VRAM
        else:
            vram_state = VRAMState.NO_VRAM

        _initialized = True


def is_initialized() -> bool:
    return _initialized


def get_vram_state() -> VRAMState:
    return vram_state


def get_torch_device():
    """获取当前主设备"""
    from faceswap.shared.config import auto_select_device
    return auto_select_device()


def is_cuda_available() -> bool:
    from faceswap.shared.config import is_gpu_available
    return is_gpu_available()


def get_total_vram(dev=None) -> int:
    """总显存(bytes)"""
    torch = _ensure_torch()
    if not torch.cuda.is_available():
        return 0
    if dev is None:
        dev = torch.device("cuda")
    try:
        _, total = torch.cuda.mem_get_info(dev)
        return total
    except Exception:
        return 0


def get_free_vram(dev=None) -> int:
    """空闲显存(bytes)，含 torch 缓存可回收部分"""
    torch = _ensure_torch()
    if not torch.cuda.is_available():
        return 0
    if dev is None:
        dev = torch.device("cuda")
    try:
        free, total = torch.cuda.mem_get_info(dev)
        reserved = torch.cuda.memory_reserved(dev)
        allocated = torch.cuda.memory_allocated(dev)
        torch_reclaimable = max(0, reserved - allocated)
        return free + torch_reclaimable
    except Exception:
        return 0


def get_total_ram() -> int:
    """总内存(bytes)"""
    return psutil.virtual_memory().total


def get_free_ram() -> int:
    """可用内存(bytes)"""
    return psutil.virtual_memory().available


def get_ram_usage_percent() -> float:
    """内存使用百分比(0-100)"""
    return psutil.virtual_memory().percent


def get_vram_usage_percent(dev=None) -> float:
    """显存使用百分比(0-100)"""
    torch = _ensure_torch()
    if not torch.cuda.is_available():
        return 0.0
    if dev is None:
        dev = torch.device("cuda")
    try:
        free, total = torch.cuda.mem_get_info(dev)
        if total == 0:
            return 0.0
        return (total - free) / total * 100.0
    except Exception:
        return 0.0


def should_use_fp16(device=None, model_params=0) -> bool:
    """
    判断是否应该使用 fp16。
    基于 GPU 架构和显存压力决策。
    """
    torch = _ensure_torch()
    if device is None:
        device = get_torch_device()
    if device.type == "cpu":
        return False
    if device.type == "mps":
        return True
    if not torch.cuda.is_available():
        return False
    try:
        caps = torch.cuda.get_device_capability(device)
        major, minor = caps
        if major >= 8:
            return True
        if major < 6:
            return False
        if major == 6 and minor == 1:
            total = get_total_vram(device)
            return total > 6 * 1024 * 1024 * 1024
        if major == 7:
            return True
        return False
    except Exception:
        return False


def should_use_bf16(device=None, model_params=0) -> bool:
    """判断是否应该使用 bf16"""
    torch = _ensure_torch()
    if device is None:
        device = get_torch_device()
    if device.type == "cpu":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        caps = torch.cuda.get_device_capability(device)
        major, minor = caps
        if major >= 8:
            return True
        return False
    except Exception:
        return False


def supports_fp8_compute(device=None) -> bool:
    """是否支持 fp8 计算"""
    torch = _ensure_torch()
    if device is None:
        device = get_torch_device()
    if device.type != "cuda":
        return False
    try:
        caps = torch.cuda.get_device_capability(device)
        major, minor = caps
        if major > 9:
            return True
        if major == 9:
            return True
        if major == 8 and minor >= 9:
            torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
            return torch_version >= (2, 3)
        return False
    except Exception:
        return False


def pick_weight_dtype(dtype, fallback_dtype, device=None):
    """
    选择权重 dtype：如果设备不支持 dtype 则降级到 fallback。
    """
    torch = _ensure_torch()
    if device is None:
        device = get_torch_device()

    if dtype == fallback_dtype:
        return dtype

    if dtype == torch.float32:
        return dtype

    if dtype in (torch.float16, torch.bfloat16):
        if device.type == "cpu":
            if dtype == torch.float16:
                return fallback_dtype
            return dtype
        return dtype

    return fallback_dtype


def dtype_size(dtype) -> int:
    """dtype 每元素字节数"""
    torch = _ensure_torch()
    if dtype == torch.float32:
        return 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return 2
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        return 1
    return 4


def minimum_inference_memory() -> int:
    """推理所需的最小预留显存(bytes)"""
    return 800 * 1024 * 1024 + EXTRA_RESERVED_VRAM


def extra_reserved_memory() -> int:
    """额外预留显存(bytes)"""
    return EXTRA_RESERVED_VRAM


def cleanup_memory(aggressive: bool = False):
    """
    释放内存和显存缓存。
    aggressive=True 时执行完整 gc 循环。
    """
    torch = _ensure_torch()
    if aggressive:
        gc.collect()
        gc.collect(2)
    else:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def soft_empty_cache():
    """软清理显存缓存（仅 empty_cache，不 gc）"""
    torch = _ensure_torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize():
    """同步 CUDA 设备"""
    torch = _ensure_torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def is_oom(exception) -> bool:
    """判断异常是否为 OOM"""
    torch = _ensure_torch()
    if isinstance(exception, getattr(torch.cuda, "OutOfMemoryError", RuntimeError)):
        return True
    if isinstance(exception, RuntimeError):
        msg = str(exception).lower()
        if "out of memory" in msg or "cuda error" in msg and "memory" in msg:
            return True
    return False


def module_size(module) -> int:
    """计算 nn.Module 的参数总大小(bytes)"""
    torch = _ensure_torch()
    total = 0
    for param in module.parameters():
        total += param.nelement() * param.element_size()
    for buffer in module.buffers():
        total += buffer.nelement() * buffer.element_size()
    return total


def module_to(module, device, dtype=None, non_blocking=False):
    """
    将模块移动到指定设备/dtype，返回移动的字节数。
    """
    torch = _ensure_torch()
    if dtype is not None:
        module = module.to(dtype=dtype)
    if non_blocking and device.type == "cuda":
        module = module.to(device=device, non_blocking=True)
    else:
        module = module.to(device=device)
    return module


class LoadedModel:
    """
    已加载模型的注册条目。
    跟踪模型大小、设备、引用状态。
    """

    def __init__(self, model, name: str = "", size: int = 0):
        self.name = name
        self.size = size if size > 0 else module_size(model) if hasattr(model, 'parameters') else 0
        self._model_ref = weakref.ref(model) if model is not None else None
        self.device = get_torch_device()
        self.currently_used = True

    @property
    def model(self):
        if self._model_ref is None:
            return None
        return self._model_ref()

    def is_alive(self) -> bool:
        return self.model is not None

    def model_memory(self) -> int:
        return self.size

    def __repr__(self):
        return f"LoadedModel(name={self.name!r}, size={self.size / 1024**3:.2f}GB, device={self.device})"


def register_model(model, name: str = "", size: int = 0) -> LoadedModel:
    """注册一个已加载的模型到全局跟踪表"""
    entry = LoadedModel(model, name, size)
    with _models_lock:
        _loaded_models.append(entry)
    return entry


def unregister_model(entry: LoadedModel):
    """从全局跟踪表移除模型"""
    with _models_lock:
        if entry in _loaded_models:
            _loaded_models.remove(entry)


def loaded_models() -> list[LoadedModel]:
    """获取当前已注册的模型列表（副本）"""
    with _models_lock:
        return list(_loaded_models)


def cleanup_models():
    """清理已死亡的模型条目"""
    with _models_lock:
        dead = [m for m in _loaded_models if not m.is_alive()]
        for m in dead:
            _loaded_models.remove(m)
    if dead:
        cleanup_memory()


def offloaded_memory(device=None) -> int:
    """计算不在指定设备上的模型总内存"""
    if device is None:
        device = get_torch_device()
    with _models_lock:
        return sum(m.size for m in _loaded_models if m.is_alive() and m.device != device)


def free_memory(memory_required: int, device=None, keep_loaded: list = None):
    """
    尝试释放显存到满足 memory_required。
    通过 gc + empty_cache 清理，不主动卸载模型（由调用方负责）。
    """
    if keep_loaded is None:
        keep_loaded = []
    cleanup_models()
    cleanup_memory(aggressive=True)
    if device is not None:
        free = get_free_vram(device)
        if free < memory_required:
            cleanup_memory(aggressive=True)


def estimate_model_memory(model_path, dtype=None) -> int:
    """根据文件大小和 dtype 估算模型内存"""
    torch = _ensure_torch()
    import os
    if not os.path.exists(model_path):
        return 0
    file_size = os.path.getsize(model_path)
    if dtype is not None:
        return file_size
    return file_size


def get_memory_info() -> dict:
    """获取当前内存/显存状态摘要"""
    info = {
        "vram_state": vram_state.name,
        "total_vram_gb": total_vram / 1024**3,
        "total_ram_gb": total_ram / 1024**3,
        "free_vram_gb": get_free_vram() / 1024**3,
        "free_ram_gb": get_free_ram() / 1024**3,
        "vram_usage_percent": get_vram_usage_percent(),
        "ram_usage_percent": get_ram_usage_percent(),
        "loaded_models_count": len(loaded_models()),
    }
    return info


def format_memory_info() -> str:
    """格式化内存信息为字符串"""
    info = get_memory_info()
    return (
        f"VRAM: {info['vram_state']} | "
        f"显存 {info['vram_usage_percent']:.1f}% "
        f"(空闲 {info['free_vram_gb']:.1f}/{info['total_vram_gb']:.1f}GB) | "
        f"内存 {info['ram_usage_percent']:.1f}% "
        f"(空闲 {info['free_ram_gb']:.1f}/{info['total_ram_gb']:.1f}GB) | "
        f"已加载模型 {info['loaded_models_count']}"
    )


@contextmanager
def inference_context(device=None):
    """
    推理上下文管理器：
    - 进入时清理缓存
    - 退出时清理缓存
    - 自动检测 OOM 并重试
    """
    torch = _ensure_torch()
    if device is None:
        device = get_torch_device()
    cleanup_memory()
    try:
        with torch.no_grad():
            yield
    except Exception as e:
        if is_oom(e):
            cleanup_memory(aggressive=True)
        raise
    finally:
        cleanup_memory()


@contextmanager
def cuda_device_context(device):
    """临时切换 CUDA 设备上下文"""
    torch = _ensure_torch()
    if device.type == "cuda":
        prev = torch.cuda.current_device()
        try:
            torch.cuda.set_device(device)
            yield
        finally:
            torch.cuda.set_device(prev)
    else:
        yield


def auto_select_dtype(model_params: int = 0, device=None) -> "torch.dtype":
    """
    根据设备能力和显存压力自动选择最佳 dtype。
    优先级: bf16 > fp16 > fp32
    """
    torch = _ensure_torch()
    if device is None:
        device = get_torch_device()

    if device.type == "cpu":
        return torch.float32

    if should_use_bf16(device, model_params):
        return torch.bfloat16
    if should_use_fp16(device, model_params):
        return torch.float16
    return torch.float32


def auto_select_offload_strategy(model_size: int, device=None) -> str:
    """
    根据模型大小和可用显存选择 offload 策略。
    返回: "none" | "sequential" | "layer" | "full"
    """
    if device is None:
        device = get_torch_device()

    if device.type == "cpu":
        return "none"

    free_vram = get_free_vram(device)
    inference_mem = minimum_inference_memory()
    available_for_model = max(0, free_vram - inference_mem)

    if model_size <= available_for_model * 0.8:
        return "none"
    if model_size <= available_for_model * 2:
        return "sequential"
    if model_size <= total_ram * 0.7:
        return "layer"
    return "full"


def log_memory_status(callback: Callable[[str], None] | None = None, prefix: str = ""):
    """记录/输出当前内存状态"""
    msg = f"{prefix}{format_memory_info()}" if prefix else format_memory_info()
    _logger.info(msg)
    if callback:
        callback(msg)
