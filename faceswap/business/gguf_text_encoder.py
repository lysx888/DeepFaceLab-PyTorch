"""
gguf_text_encoder.py - 用 GGUF 加载 Qwen2.5-VL text_encoder

复用 ComfyUI-GGUF 的反量化逻辑，按需反量化（forward 时才反量化权重）。
权重保持量化格式（GGMLTensor），省内存。
"""

from __future__ import annotations

from faceswap.shared.logger import get_logger

_logger = get_logger("gguf_text_encoder")
import torch
import torch.nn as nn

from .gguf_utils.ops import GGMLOps, GGMLLayer
from .gguf_utils.loader import gguf_text_encoder_loader
from .gguf_utils.dequant import dequantize_tensor, is_quantized


def _replace_with_ggml(module: nn.Module):
    """递归替换 Linear/Embedding/LayerNorm 为 GGMLOps 版本（支持按需反量化）"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, GGMLOps.Linear):
            new = GGMLOps.Linear(child.in_features, child.out_features, child.bias is not None)
            module.add_module(name, new)
        elif isinstance(child, nn.Embedding) and not isinstance(child, GGMLOps.Embedding):
            new = GGMLOps.Embedding(child.num_embeddings, child.embedding_dim)
            module.add_module(name, new)
        elif isinstance(child, nn.LayerNorm) and not isinstance(child, GGMLOps.LayerNorm):
            new = GGMLOps.LayerNorm(child.normalized_shape, child.eps)
            module.add_module(name, new)
        else:
            _replace_with_ggml(child)


def _fix_rope_inv_freq(model: nn.Module):
    """
    修复 RoPE inv_freq buffer (meta device 创建时未初始化)。
    RoPE forward 直接使用 self.inv_freq, 必须用正确公式填充:
      - VisionRoPE: 1/(theta^(arange(0,dim,2)/dim))
      - LanguageRoPE: 调用 compute_default_rope_parameters
    """
    fixed = 0
    for mod in model.modules():
        for bn in list(mod._buffers.keys()):
            b = mod._buffers[bn]
            if b is None or b.device.type != "meta":
                continue
            if bn == "inv_freq" and hasattr(mod, "dim") and hasattr(mod, "theta"):
                dim, theta = mod.dim, mod.theta
                mod._buffers[bn] = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
                fixed += 1
            elif bn in ("inv_freq", "original_inv_freq") and hasattr(mod, "config") and hasattr(mod, "compute_default_rope_parameters"):
                inv_freq, _ = mod.compute_default_rope_parameters(mod.config, device=torch.device("cpu"))
                mod._buffers[bn] = inv_freq if bn == "inv_freq" else inv_freq.clone()
                fixed += 1
    if fixed:
        _logger.info(f"修复 {fixed} 个 RoPE inv_freq buffer")


def _pre_dequant_iq_weights(model: nn.Module, dtype: torch.dtype):
    """
    预反量化 IQ 类型权重 (无原生 PyTorch 反量化实现的类型)。
    这些类型每次 forward 会回退到 numpy 极慢反量化, 预反量化后消除回退。
    有原生实现的类型 (Q2_K, Q3_K 等) 保持量化, 节省内存。
    """
    import gguf
    from .gguf_utils.dequant import dequantize_functions, dequantize_tensor, is_quantized

    iq_types = set()
    count = 0
    t0 = __import__("time").time()

    with torch.no_grad():
        for module in model.modules():
            for attr_name in ["weight", "bias"]:
                tensor = getattr(module, attr_name, None)
                if tensor is None or not is_quantized(tensor):
                    continue
                qtype = getattr(tensor, "tensor_type", None)
                if qtype not in dequantize_functions:
                    iq_types.add(getattr(qtype, "name", repr(qtype)))
                    dequant = dequantize_tensor(tensor, dtype=dtype)
                    if attr_name == "weight":
                        setattr(module, attr_name, torch.nn.Parameter(dequant, requires_grad=False))
                    else:
                        setattr(module, attr_name, dequant)
                    count += 1

    if count:
        elapsed = __import__("time").time() - t0
        _logger.info(f"预反量化 {count} 个 IQ 类型权重 ({', '.join(sorted(iq_types))}) 在 {elapsed:.1f}s")


def load_qwen2vl_gguf(
    gguf_path: str,
    config_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    mmproj_search_dirs: list[str] | None = None,
    mmproj_path: str | None = None,
    pre_dequantize: bool = False,
) -> nn.Module:
    """
    从 GGUF 加载 Qwen2.5-VL text_encoder

    gguf_path: text encoder GGUF 文件路径
    config_dir: 包含 config.json 的目录
    dtype: 计算精度
    device: 初始设备
    mmproj_search_dirs: 额外搜索 mmproj 文件的目录列表
    mmproj_path: 显式指定 mmproj 文件路径（覆盖搜索）
    pre_dequantize: 预反量化所有权重到 dtype（省 forward 时间，占更多内存）

    返回 Qwen2_5_VLForConditionalGeneration 模型，权重为量化格式（按需反量化）
    """
    from transformers import Qwen2_5_VLForConditionalGeneration
    from transformers import Qwen2_5_VLConfig

    _logger.info(f"Loading Qwen2.5-VL GGUF: {gguf_path}")
    _logger.info(f"Loading GGUF: {gguf_path}")

    try:
        sd = gguf_text_encoder_loader(gguf_path, mmproj_search_dirs=mmproj_search_dirs, mmproj_path=mmproj_path)
    except Exception as e:
        import traceback
        _logger.info(f"ERROR in gguf_text_encoder_loader:")
        traceback.print_exc()
        raise

    _logger.info(f"state_dict: {len(sd)} tensors")

    none_keys = [k for k, v in sd.items() if v is None]
    if none_keys:
        _logger.info(f"None values: {none_keys[:10]}")
        for k in none_keys:
            sd.pop(k)

    for k, v in list(sd.items()):
        if not is_quantized(v) and v.is_floating_point() and v.dtype != dtype:
            sd[k] = v.to(dtype)

    _logger.info("loading config...")
    config = Qwen2_5_VLConfig.from_pretrained(config_dir)

    _logger.info("creating model (meta device)...")
    with torch.device("meta"):
        model = Qwen2_5_VLForConditionalGeneration(config)

    _logger.info("replacing layers with GGMLOps (on meta)...")
    _replace_with_ggml(model)

    _logger.info("loading state_dict with assign=True (incremental alloc)...")
    try:
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    except Exception as e:
        import traceback
        _logger.info(f"assign=True failed, fallback to to_empty+load: {e}")
        traceback.print_exc()
        _logger.info("fallback: to_empty(cpu)...")
        model.to_empty(device="cpu")
        _logger.info("fallback: replacing layers again...")
        _replace_with_ggml(model)
        _logger.info("fallback: loading state_dict...")
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)

    if missing:
        _logger.info(f"Missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        _logger.info(f"Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    _logger.info("GGUF text_encoder loaded successfully")

    # 修复 RoPE inv_freq (meta device 创建时未初始化, load_state_dict 不会覆盖)
    _fix_rope_inv_freq(model)

    # 预反量化 IQ 类型权重 (无原生 PyTorch 实现, 每次 forward 用 numpy 极慢)
    _pre_dequant_iq_weights(model, dtype)

    if pre_dequantize:
        _logger.info(f"Pre-dequantizing all weights to {dtype}...")
        import time
        t0 = time.time()
        count = 0
        with torch.no_grad():
            for module in model.modules():
                for attr_name in ["weight", "bias"]:
                    tensor = getattr(module, attr_name, None)
                    if tensor is not None and is_quantized(tensor):
                        dequant = dequantize_tensor(tensor, dtype=dtype)
                        setattr(module, attr_name, torch.nn.Parameter(dequant, requires_grad=False))
                        count += 1
        _logger.info(f"Pre-dequant {count} tensors in {time.time()-t0:.1f}s")

    return model
