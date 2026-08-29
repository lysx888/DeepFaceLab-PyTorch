import pickle
import time
from pathlib import Path

import numpy as np
import torch

from faceswap.shared.logger import get_logger
from faceswap.models.xseg_model import XSegNet

_logger = get_logger(__name__)


def _load_pickle_dict(npy_path: Path) -> dict:
    if not npy_path.exists():
        raise FileNotFoundError(f"DFL权重文件不存在: {npy_path}")
    try:
        with open(npy_path, 'rb') as f:
            d = pickle.load(f)
    except pickle.UnpicklingError as e:
        raise ValueError(f"DFL权重文件反序列化失败: {e}") from e
    if not isinstance(d, dict):
        raise TypeError(f"DFL权重文件结构异常: 期望dict, 实际{type(d).__name__}")
    return d


def _map_key_name(key: str) -> str:
    return key.replace('/', '.').replace(':0', '')


def _transpose_weight(value: np.ndarray) -> np.ndarray:
    if value.ndim == 4:
        return np.transpose(value, (3, 2, 0, 1)).astype(np.float32)
    elif value.ndim == 2:
        return value.T.astype(np.float32)
    else:
        return value.astype(np.float32)


def load_dfl_xseg_weights(npy_path: Path, model: XSegNet) -> tuple[list[str], list[str]]:
    start = time.perf_counter()
    dfl_dict = _load_pickle_dict(npy_path)

    converted_state_dict: dict[str, torch.Tensor] = {}
    for dfl_key, w_val in dfl_dict.items():
        torch_key = _map_key_name(dfl_key)
        converted = _transpose_weight(w_val)
        converted_state_dict[torch_key] = torch.from_numpy(converted.copy())

    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)

    elapsed = time.perf_counter() - start
    _logger.debug(
        f"DFL权重转换完成: 转换{len(converted_state_dict)}个键, "
        f"缺失键:{missing_keys}, 意外键:{unexpected_keys}, 耗时{elapsed:.2f}s"
    )
    return missing_keys, unexpected_keys
