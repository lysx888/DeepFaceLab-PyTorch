"""
loader.py - GGUF 文件加载器

基于 ComfyUI-GGUF loader.py 提取关键函数。
读取 GGUF 文件为 state_dict（GGMLTensor 量化张量）。
"""

import warnings
from faceswap.shared.logger import get_logger

_logger = get_logger("gguf_loader")
import torch
import gguf
import re
import os

from .ops import GGMLTensor
from .dequant import is_quantized, dequantize_tensor


def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]].item())
    else:
        raise TypeError(f"Unknown field type {field_type}")


def get_gguf_metadata(reader):
    metadata = {}
    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if len(field.types) == 1:
                if field.types[0] == gguf.GGUFValueType.STRING:
                    metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
                elif field.types[0] == gguf.GGUFValueType.INT32:
                    metadata[field_name] = int(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.F32:
                    metadata[field_name] = float(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.BOOL:
                    metadata[field_name] = bool(field.parts[field.data[-1]])
        except:
            continue
    return metadata


def gguf_sd_loader(path, handle_prefix="model.diffusion_model.", is_text_model=False):
    """Read state dict as fake tensors (GGMLTensor)"""
    reader = gguf.GGUFReader(path)

    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    arch_str = get_field(reader, "general.architecture", str)

    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)

        if len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            state_dict[sd_key] = dequantize_tensor(state_dict[sd_key], dtype=torch.float32)

        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    _logger.info("gguf qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()))

    qsd = {k: v for k, v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    extra = {
        "arch_str": arch_str,
        "metadata": get_gguf_metadata(reader)
    }
    return (state_dict, extra)


LLAMA_SD_MAP = {
    "blk.": "model.language_model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_v_norm.": "self_attn.v_norm.",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "token_embd": "model.language_model.embed_tokens",
    "output_norm": "model.language_model.norm",
    "output.weight": "lm_head.weight",
}

CLIP_VISION_SD_MAP = {
    "mm.": "model.visual.merger.mlp.",
    "v.post_ln.": "model.visual.merger.ln_q.",
    "v.patch_embd": "model.visual.patch_embed.proj",
    "v.blk.": "model.visual.blocks.",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "attn_out.": "attn.proj.",
    "ln1.": "norm1.",
    "ln2.": "norm2.",
}


def sd_map_replace(raw_sd, key_map):
    sd = {}
    for k, v in raw_sd.items():
        for s, d in key_map.items():
            k = k.replace(s, d)
        sd[k] = v
    return sd


def strip_quant_suffix(name):
    pattern = r"[-_]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$"
    match = re.search(pattern, name, re.IGNORECASE)
    if match:
        name = name[:match.start()]
    return name


def gguf_mmproj_loader(path, extra_search_dirs=None, explicit_path=None):
    """Load mmproj GGUF for Qwen2.5-VL visual model"""
    _logger.info("Attempting to find mmproj file for text encoder...")

    if explicit_path and os.path.isfile(explicit_path):
        _logger.info(f"Using explicit mmproj: {explicit_path}")
        target = [explicit_path]
    else:
        tenc_fname = os.path.basename(path)
        tenc = os.path.splitext(tenc_fname)[0].lower()
        tenc = strip_quant_suffix(tenc)

        search_dirs = [os.path.dirname(path)]
        if extra_search_dirs:
            search_dirs.extend(extra_search_dirs)

        target = []
        for root in search_dirs:
            if not os.path.isdir(root):
                continue
            for fname in os.listdir(root):
                name, ext = os.path.splitext(fname)
                if ext.lower() != ".gguf":
                    continue
                if "mmproj" not in name.lower():
                    continue
                candidate = os.path.join(root, fname)
                if candidate not in target:
                    target.append(candidate)

        if len(target) == 0:
            _logger.error(f"Can't find mmproj file for '{tenc_fname}'! Searched: {search_dirs}")
            return {}
        if len(target) > 1:
            bf16_targets = [t for t in target if "bf16" in os.path.basename(t).lower()]
            if bf16_targets:
                target = [bf16_targets[0]]
                _logger.info(f"Multiple mmproj found, using BF16: {os.path.basename(target[0])}")
            else:
                _logger.warning(f"Ambiguous mmproj, using first: {os.path.basename(target[0])}")

        _logger.info(f"Using mmproj '{os.path.basename(target[0])}' for '{tenc_fname}'.")
    vsd, _ = gguf_sd_loader(target[0], is_text_model=True)
    _logger.info(f"mmproj raw tensors: {len(vsd)}")

    if "merger.ln_q.weight" in vsd and "v.post_ln.weight" not in vsd:
        _logger.info("New mmproj format: renaming merger.ln_q -> v.post_ln")
        vsd["v.post_ln.weight"] = vsd.pop("merger.ln_q.weight")

    if "v.position_embd.weight" in vsd:
        _logger.info("Removing v.position_embd (not in transformers model)")
        del vsd["v.position_embd.weight"]

    if "v.patch_embd.weight.1" in vsd:
        _logger.info("Concat patch_embd 4D -> 5D")
        w1 = dequantize_tensor(vsd.pop("v.patch_embd.weight"), dtype=torch.float32)
        w2 = dequantize_tensor(vsd.pop("v.patch_embd.weight.1"), dtype=torch.float32)
        vsd["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)

    vsd = sd_map_replace(vsd, CLIP_VISION_SD_MAP)
    _logger.info(f"mmproj after key map: {len(vsd)}")

    if "model.visual.blocks.0.attn_q.weight" in vsd:
        _logger.info("Merging split Q/K/V -> qkv")
        attns = {}
        keys_to_remove = []
        for k, v in vsd.items():
            if any(x in k for x in ["attn_q", "attn_k", "attn_v"]):
                k_attn, k_name = k.rsplit(".attn_", 1)
                k_attn += ".attn.qkv." + k_name.split(".")[-1]
                if k_attn not in attns:
                    attns[k_attn] = {}
                attns[k_attn][k_name] = dequantize_tensor(
                    v, dtype=(torch.bfloat16 if is_quantized(v) else torch.float16)
                )
                keys_to_remove.append(k)

        for k in keys_to_remove:
            del vsd[k]

        for k, v in attns.items():
            suffix = k.split(".")[-1]
            vsd[k] = torch.cat([
                v[f"q.{suffix}"],
                v[f"k.{suffix}"],
                v[f"v.{suffix}"],
            ], dim=0)
        del attns

    _logger.info(f"mmproj final tensors: {len(vsd)}")
    return vsd


def gguf_text_encoder_loader(path, mmproj_search_dirs=None, mmproj_path=None):
    """
    Load Qwen2.5-VL text encoder GGUF + mmproj.
    Returns merged state_dict with llama.cpp keys mapped to model.* keys.
    mmproj_search_dirs: extra directories to search for mmproj file
    mmproj_path: explicit path to mmproj file (overrides search)
    """
    sd, extra = gguf_sd_loader(path, is_text_model=True)
    arch = extra.get("arch_str", None)
    _logger.info(f"arch={arch}, tensors={len(sd)}")

    if arch in {"llama", "qwen2vl", "qwen3", "qwen3vl", "gemma3"}:
        temb_key = "token_embd.weight"
        if temb_key in sd:
            _logger.info(f"temb type={type(sd[temb_key])}, shape={getattr(sd[temb_key], 'shape', 'N/A')}")
        if temb_key in sd and sd[temb_key].shape[0] >= (64 * 1024):
            _logger.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)

        _logger.info(f"applying LLAMA_SD_MAP...")
        sd = sd_map_replace(sd, LLAMA_SD_MAP)
        _logger.info(f"after map: {len(sd)} tensors")

        if arch == "qwen2vl":
            _logger.info(f"loading mmproj, search_dirs={mmproj_search_dirs}")
            vsd = gguf_mmproj_loader(path, extra_search_dirs=mmproj_search_dirs, explicit_path=mmproj_path)
            _logger.info(f"mmproj: {len(vsd)} tensors")
            sd.update(vsd)
            _logger.info(f"merged: {len(sd)} tensors")

    return sd
