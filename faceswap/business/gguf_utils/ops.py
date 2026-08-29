"""
ops.py - GGUF 按需反量化操作

基于 ComfyUI-GGUF ops.py 改写，去掉 comfy 依赖，用原生 torch.nn。
权重保持量化格式（GGMLTensor），forward 时按需反量化。
"""

import gguf
import torch

from .dequant import dequantize_tensor, is_quantized


class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """
    def __init__(self, *args, tensor_type=None, tensor_shape=None, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape if tensor_shape is not None else self.size()
        self.patches = patches

    def __new__(cls, *args, tensor_type=None, tensor_shape=None, patches=[], **kwargs):
        # 兼容 accelerate 的 offload：允许不带 tensor_type/tensor_shape 创建
        # 这种情况下创建的是普通张量的包装，后续会被赋值
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        try:
            return super().copy_(*args, **kwargs)
        except Exception:
            pass

    def new_empty(self, size, *args, **kwargs):
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
            new_tensor,
            tensor_type=getattr(self, "tensor_type", None),
            tensor_shape=size,
            patches=getattr(self, "patches", []).copy()
        )

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


class GGMLLayer(torch.nn.Module):
    """
    Base layer for de-quantizing on the fly
    """
    comfy_cast_weights = True
    dequant_dtype = None
    largest_layer = False
    torch_compatible_tensor_types = {None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f"{prefix}weight"), state_dict.get(f"{prefix}bias")
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self, torch.nn.Linear):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if isinstance(self, torch.nn.Embedding) and weight is not None and weight.shape[0] >= (64 * 1024):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k[prefix_len:] == "weight":
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)

        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix + "weight")

        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        # 返回实际的量化张量数据，而不是 meta tensor
        # 这样才能与 accelerate 的 cpu_offload / state_dict 兼容
        # 注意：这会增加 state_dict 的内存占用，但对于 offload 是必要的
        if keep_vars:
            destination[prefix + "weight"] = self.weight
        else:
            destination[prefix + "weight"] = self.weight.detach().clone()
        if self.bias is not None:
            if keep_vars:
                destination[prefix + "bias"] = self.bias
            else:
                destination[prefix + "bias"] = self.bias.detach().clone()

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)
        return weight

    def cast_bias_weight(self, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        if self.bias is not None:
            bias = self.get_weight(self.bias.to(device), dtype)
            bias = bias.to(bias_dtype)

        weight = self.get_weight(self.weight.to(device), dtype)
        weight = weight.to(dtype)

        return weight, bias

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)
        return out

    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError


class GGMLOps:
    """
    Dequantize weights on the fly before doing the compute.
    用原生 torch.nn 层替换 comfy.ops。
    """
    class Linear(GGMLLayer, torch.nn.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward(self, input):
            if self.is_ggml_quantized():
                weight, bias = self.cast_bias_weight(input)
                return torch.nn.functional.linear(input, weight, bias)
            if self.weight.dtype != input.dtype:
                weight = self.weight.to(input.dtype)
                bias = self.bias.to(input.dtype) if self.bias is not None else None
                return torch.nn.functional.linear(input, weight, bias)
            return torch.nn.functional.linear(input, self.weight, self.bias)

    class Conv2d(GGMLLayer, torch.nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            self.stride = stride if isinstance(stride, tuple) else (stride, stride)
            self.padding = padding if isinstance(padding, tuple) else (padding, padding)
            self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
            self.groups = groups
            self.weight = None
            self.bias = None

        def _conv_forward(self, input, weight, bias):
            return torch.nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        def forward(self, input):
            if self.is_ggml_quantized():
                weight, bias = self.cast_bias_weight(input)
                return self._conv_forward(input, weight, bias)
            return self._conv_forward(input, self.weight, self.bias)

    class Embedding(GGMLLayer, torch.nn.Embedding):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.scale_grad_by_freq = scale_grad_by_freq
            self.sparse = sparse
            self.weight = None
            self.bias = None

        def forward(self, input, out_dtype=None):
            if self.is_ggml_quantized():
                output_dtype = out_dtype
                if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                    out_dtype = None
                weight, _bias = self.cast_bias_weight(self, device=input.device, dtype=out_dtype)
                return torch.nn.functional.embedding(
                    input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
                ).to(dtype=output_dtype)
            return torch.nn.functional.embedding(
                input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
            )

    class LayerNorm(GGMLLayer, torch.nn.LayerNorm):
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            self.weight = None
            self.bias = None

        def forward(self, input):
            if self.is_ggml_quantized():
                if self.weight is None:
                    return torch.nn.functional.layer_norm(input, self.normalized_shape, eps=self.eps)
                weight, bias = self.cast_bias_weight(input)
                return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
            return torch.nn.functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    class GroupNorm(GGMLLayer, torch.nn.GroupNorm):
        def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.num_groups = num_groups
            self.num_channels = num_channels
            self.eps = eps
            self.affine = affine
            self.weight = None
            self.bias = None

        def forward(self, input):
            if self.is_ggml_quantized():
                weight, bias = self.cast_bias_weight(input)
                return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)
            return torch.nn.functional.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
