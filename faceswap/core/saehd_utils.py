import os
import sys
import torch
import torch.nn as nn

from faceswap.shared.logger import get_logger

_logger = get_logger("saehd_utils")

TORCH_COMPILE_BACKEND_HIGH_VRAM = "inductor"
TORCH_COMPILE_BACKEND_LOW_VRAM = "aot_eager"
TORCH_COMPILE_FALLBACK = "aot_eager"
LOW_VRAM_THRESHOLD_GB = 8.0


def _is_windows_gbk_locale() -> bool:
    if sys.platform != "win32":
        return False
    if os.environ.get("PYTHONUTF8", "0") == "1":
        return False
    try:
        import locale
        enc = locale.getpreferredencoding(False).lower()
        return enc in ("cp936", "gbk", "gb2312", "gb18030")
    except Exception:
        return False


def _select_compile_backend() -> str:
    from faceswap.shared.config import is_gpu_available, get_device_memory_mb, auto_select_device
    if not is_gpu_available():
        return TORCH_COMPILE_BACKEND_LOW_VRAM
    vram_gb = get_device_memory_mb(auto_select_device()) / 1024
    if vram_gb < LOW_VRAM_THRESHOLD_GB:
        _logger.warning(f"torch.compile: VRAM={vram_gb:.1f}GB < {LOW_VRAM_THRESHOLD_GB}GB, "
                        f"using '{TORCH_COMPILE_BACKEND_LOW_VRAM}'. "
                        f"警告: aot_eager仅做AOT编译不做算子融合, 无实际加速, "
                        f"建议关闭torch.compile或使用8GB+显存(inductor后端)")
        return TORCH_COMPILE_BACKEND_LOW_VRAM
    if _is_windows_gbk_locale():
        _logger.info(f"torch.compile: VRAM={vram_gb:.1f}GB but Windows GBK locale detected, "
                     f"using '{TORCH_COMPILE_BACKEND_LOW_VRAM}' (inductor needs PYTHONUTF8=1)")
        return TORCH_COMPILE_BACKEND_LOW_VRAM
    _logger.info(f"torch.compile: VRAM={vram_gb:.1f}GB >= {LOW_VRAM_THRESHOLD_GB}GB, "
                 f"using '{TORCH_COMPILE_BACKEND_HIGH_VRAM}' (inductor,最强优化)")
    return TORCH_COMPILE_BACKEND_HIGH_VRAM


def apply_torch_compile(model, enable: bool = True) -> None:
    if not enable:
        return
    try:
        major, minor = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
        if major < 2:
            _logger.info("torch.compile skipped: PyTorch < 2.0")
            return
    except Exception:
        _logger.info("torch.compile skipped: cannot determine PyTorch version")
        return
    modules_dict = getattr(model, '_modules_dict', {})
    if not modules_dict:
        _logger.info("torch.compile skipped: no sub-modules found")
        return
    backend = _select_compile_backend()
    encoding_warned = False
    compiled_count = 0
    for name, module in list(modules_dict.items()):
        if not isinstance(module, nn.Module):
            continue
        compiled_mod = None
        for be in (backend, TORCH_COMPILE_FALLBACK):
            try:
                compiled_mod = torch.compile(module, backend=be, dynamic=False)
                break
            except UnicodeDecodeError as e:
                if be == backend and not encoding_warned:
                    encoding_warned = True
                    _logger.warning(
                        f"torch.compile '{be}' failed with encoding error ({e}). "
                        f"This is a known Windows Chinese-locale bug. "
                        f"Set PYTHONUTF8=1 to fix. Falling back to '{TORCH_COMPILE_FALLBACK}'."
                    )
            except Exception as e:
                if be == backend:
                    _logger.info(f"torch.compile backend '{be}' failed for {name} ({e}), trying fallback")
                else:
                    _logger.warning(f"torch.compile fallback also failed for {name} ({e}), keeping eager")
        if compiled_mod is not None:
            setattr(model, name, compiled_mod)
            modules_dict[name] = compiled_mod
            compiled_count += 1
    if compiled_count > 0:
        _logger.info(f"torch.compile enabled: {compiled_count} sub-modules ({backend})")
