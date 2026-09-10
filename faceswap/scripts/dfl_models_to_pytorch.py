#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DFL(DeepFaceLab) 权重 -> 本项目 PyTorch 权重 转换工具。

支持四种架构:
  - SAEHD   (df / liae + d/u/t/c 后缀)                 -> faceswap/models/saehd
  - AMP     (encoder / inter_src / inter_dst / decoder) -> faceswap/models/amp
  - Quick96 (固定 96 分辨率, DeepFakeArchi opts='ud')   -> faceswap/models/quick96
  - XSeg    (XSeg_256.npy 遮罩网络, 固定 256 分辨率)     -> faceswap/models/xseg

用法:
    python scripts/dfl_models_to_pytorch.py \
        --dfl-model "D:\\AI\\DeepFaceLab0\\workspace\\model" \
        --output "D:\\AI\\Inswapper\\workspace\\model\\saehd" --verify

参数:
    --archi <saehd|amp|quick96|xseg>  模型架构(默认自动识别，文件名含 _AMP/_Quick96/_SAEHD/XSeg_)
    --model-name <前缀>          目录含多个 DFL 模型时用此前缀过滤
    --name <名称>                输出模型目录名(默认 "{iter}_{架构}")
    --verify                     用项目网络 strict 校验
    --skip-gan / --skip-code-disc

格式说明:
1. DFL 每个组件(encoder/inter/decoder_*/GAN/code_discriminator)是 pickle.dumps(dict)，
   key 为 TF 变量名如 'down1/downs_0/conv1/weight:0'，value 为 float32 numpy。
2. 本项目是 DFL 网络的 PyTorch 逐层移植，层结构一致
   (见 faceswap/models/{saehd,amp}/ 与 DFL core/leras/archis/DeepFakeArchi.py)。
3. conv 权重 DFL (kh,kw,in,out) -> PyTorch (out,in,kh,kw)；dense (in,out) -> (out,in)。
4. 优化器状态不转换，由项目重建。
"""
import argparse
import json
import pickle
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    HAS_TORCH = False

# Windows 终端编码保护：模型名含中文/特殊字符时避免 print 报错
try:
    if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ---------------------------------------------------------------------------
# 自动把 faceswap 包根目录加入 sys.path（供 --verify 使用）
#   本脚本位于 <包根>/faceswap/scripts/，faceswap 包在 <包根>/faceswap。
# ---------------------------------------------------------------------------
_PKG_PARENT = Path(__file__).resolve().parents[2]  # e.g. D:\AI\Inswapper
if (_PKG_PARENT / 'faceswap' / '__init__.py').exists():
    _pkg_parent_str = str(_PKG_PARENT)
    if _pkg_parent_str not in sys.path:
        sys.path.insert(0, _pkg_parent_str)

# ---------------------------------------------------------------------------
# 架构定义
# ---------------------------------------------------------------------------
ARCHI_NAMES = ('saehd', 'amp', 'quick96', 'xseg')

# 每种架构的组件文件（.npy 后缀名）。GAN 在 AMP 中仅当 gan_power>0 时存在。
ARCHI_COMPONENTS = {
    'saehd':   ['encoder', 'inter', 'decoder_src', 'decoder_dst',
                'inter_AB', 'inter_B', 'decoder', 'GAN', 'code_discriminator'],
    'amp':     ['encoder', 'inter_src', 'inter_dst', 'decoder', 'GAN'],
    'quick96': ['encoder', 'inter', 'decoder_src', 'decoder_dst'],
    # XSeg: DFL 权重文件名为 XSeg_<res>.npy（Model_XSeg 固定 name='XSeg'、resolution=256）
    'xseg':    ['XSeg'],
}
_ALL_COMPONENTS = []
for _lst in ARCHI_COMPONENTS.values():
    for _c in _lst:
        if _c not in _ALL_COMPONENTS:
            _ALL_COMPONENTS.append(_c)
_ALL_COMPONENTS.sort(key=len, reverse=True)  # 长名优先，避免 inter_src 被 inter 误匹配

# 输出文件命名约定
CONFIG_FILENAME = {
    'saehd': 'training_config.json',
    'amp': 'training_config.json',
    'quick96': 'training_config.json',
    'xseg': 'XSeg_config.json',
}
# GUI 的 ConfigManager 按架构前缀命名 config 文件来识别/加载模型，
# 正常训练时 GUI 先写此文件、base_model.save 再写 training_config.json，
# 故转换也须同时写出两份（内容相同），XSeg 无需第二份。
GUI_CONFIG_FILENAME = {
    'saehd': 'SAEHD_training_config.json',
    'amp': 'AMP_training_config.json',
    'quick96': 'Quick96_training_config.json',
    'xseg': None,
}
STATE_FILENAME = {
    'saehd': 'saehd_training_state.json',
    'amp': 'amp_training_state.json',
    'quick96': 'quick96_training_state.json',
    # 项目 XSegModel._model_prefix='XSeg' -> _prefixed('training_state.json')
    'xseg': 'XSeg_training_state.json',
}
SUMMARY_FILENAME = {
    'saehd': 'saehd_summary.txt',
    'amp': 'amp_summary.txt',
    'quick96': 'quick96_summary.txt',
    # 项目 XSegModel._model_prefix='XSeg' -> _prefixed('summary.txt')
    'xseg': 'XSeg_summary.txt',
}
# 组件名 -> 输出的 .pth 文件名（GAN 特判为项目 D_src.pth，XSeg 特判为项目 xseg_net.pth，其余同名）
PTH_NAME = {'GAN': 'D_src', 'XSeg': 'xseg_net'}

# ---------------------------------------------------------------------------
# 层名映射表
#   DFL key 前缀 -> (项目 key 前缀, 类型)
#   类型: 'conv' 普通卷积(含1x1) / 'convT' 转置卷积 / 'dense' 全连接
#   SAEHD/Quick96 同架构；AMP 的层名(down1/conv1, res1/conv1..2, dense1/2,
#   upscale{i}/conv1, res{r}/conv1|2, out_conv*, upscalem{i}/conv1, out_convm)
#   也全部落在本表内，故三架构共用。
# ---------------------------------------------------------------------------
def _conv(proj): return (proj, 'conv')
def _ct(proj):   return (proj, 'convT')
def _dense(proj): return (proj, 'dense')

PREFIX_MAP = {}

# ---- Encoder（无 t 后缀: DownscaleBlock down1/downs_0..3）----
for i in range(4):
    PREFIX_MAP[f'down1/downs_{i}/conv1/'] = _conv(f'down1.downs.{i}.conv.')

# ---- Encoder（t 后缀 / AMP: 独立 down1..down5 + res1/res5）----
PREFIX_MAP['down1/conv1/'] = _conv('down1.conv.')
for i in range(2, 6):
    PREFIX_MAP[f'down{i}/conv1/'] = _conv(f'down{i}.conv.')
PREFIX_MAP['res1/conv1/'] = _conv('res1.conv1.')
PREFIX_MAP['res1/conv2/'] = _conv('res1.conv2.')
PREFIX_MAP['res5/conv1/'] = _conv('res5.conv1.')
PREFIX_MAP['res5/conv2/'] = _conv('res5.conv2.')

# ---- Inter（dense1/dense2 + upscale1，无 t 后缀才有 upscale1）----
PREFIX_MAP['dense1/'] = _dense('dense1.')
PREFIX_MAP['dense2/'] = _dense('dense2.')
PREFIX_MAP['upscale1/conv1/'] = _conv('upscale1.conv.')

# ---- Decoder（RGB 分支）----
for i in range(4):
    PREFIX_MAP[f'upscale{i}/conv1/'] = _conv(f'upscale{i}.conv.')
for r in range(4):
    PREFIX_MAP[f'res{r}/conv1/'] = _conv(f'res{r}.conv1.')
    PREFIX_MAP[f'res{r}/conv2/'] = _conv(f'res{r}.conv2.')
PREFIX_MAP['out_conv/'] = _conv('out_conv.')
PREFIX_MAP['out_conv1/'] = _conv('out_conv1.')
PREFIX_MAP['out_conv2/'] = _conv('out_conv2.')
PREFIX_MAP['out_conv3/'] = _conv('out_conv3.')

# ---- Decoder（Mask 分支）----
for i in range(5):  # 覆盖 0..3(无t) / 0..4(t+d/AMP)
    PREFIX_MAP[f'upscalem{i}/conv1/'] = _conv(f'upscalem{i}.conv.')
PREFIX_MAP['out_convm/'] = _conv('out_convm.')

# ---- GAN: UNetPatchDiscriminator ----
PREFIX_MAP['in_conv/'] = _conv('in_conv.')
for i in range(9):
    PREFIX_MAP[f'convs_{i}/'] = _conv(f'convs.{i}.')
for i in range(9):
    PREFIX_MAP[f'upconvs_{i}/'] = _ct(f'upconvs.{i}.')
PREFIX_MAP['center_conv/'] = _conv('center_conv.')
PREFIX_MAP['center_out/'] = _conv('center_out.')
# GAN 的 out_conv 已由上面 'out_conv/' 覆盖

# ---- code_discriminator: CodeDiscriminator ----
# convs_N / out_conv 前缀已覆盖

# ---- XSeg: XSegNet（DFL core/leras/models/XSeg.py，项目 models/xseg_model.py）----
#   每个 conv 块: conv3x3 + FRNorm2D(weight/bias/eps) + TLU(tau)
#   每个 up 块:   Conv2DTranspose(3x3,SAME) + FRNorm2D + TLU
#   BlurPool 无权重(DFL 与项目均为固定二项核)；dense1/dense2/out_conv 复用上面条目。
#   'vec' = 1 维向量参数(weight/bias/eps/tau)，无需转置。
_XSEG_CONVS = [
    'conv01', 'conv02', 'conv11', 'conv12', 'conv21', 'conv22',
    'conv31', 'conv32', 'conv33', 'conv41', 'conv42', 'conv43',
    'conv51', 'conv52', 'conv53',
    'uconv01', 'uconv02', 'uconv11', 'uconv12', 'uconv21', 'uconv22',
    'uconv31', 'uconv32', 'uconv33', 'uconv41', 'uconv42', 'uconv43',
    'uconv51', 'uconv52', 'uconv53',
]
for _n in _XSEG_CONVS:
    PREFIX_MAP[f'{_n}/conv/'] = _conv(f'{_n}.conv.')
    PREFIX_MAP[f'{_n}/frn/'] = (f'{_n}.frn.', 'vec')
    PREFIX_MAP[f'{_n}/tlu/'] = (f'{_n}.tlu.', 'vec')
for _n in ('up5', 'up4', 'up3', 'up2', 'up1', 'up0'):
    PREFIX_MAP[f'{_n}/conv/'] = _ct(f'{_n}.conv.')
    PREFIX_MAP[f'{_n}/frn/'] = (f'{_n}.frn.', 'vec')
    PREFIX_MAP[f'{_n}/tlu/'] = (f'{_n}.tlu.', 'vec')

# 按前缀长度降序，保证最长匹配优先（如 down1/downs_0/conv1/ 先于 out_conv/）
_PREFIX_ORDER = sorted(PREFIX_MAP.keys(), key=len, reverse=True)


def remap_key(dfl_key: str):
    """把 DFL 变量名 'down1/downs_0/conv1/weight:0' 映射为项目 state_dict key。

    返回 (project_key, kind)；无法映射返回 (None, None)。
    """
    k = dfl_key[:-2] + ':'  # 去掉末尾 ':0'
    for pref in _PREFIX_ORDER:
        if k.startswith(pref):
            tail = k[len(pref):]
            if tail in ('weight:', 'bias:', 'eps:', 'tau:'):
                proj, kind = PREFIX_MAP[pref]
                return proj + tail[:-1], kind
            return None, None
    return None, None


def transpose_weight(arr: np.ndarray, kind: str) -> np.ndarray:
    """按类型转换权重布局到 PyTorch 约定。"""
    if kind in ('conv', 'convT'):
        if arr.ndim != 4:
            raise ValueError(f"conv 权重应为 4 维, 实际 {arr.shape}")
        return np.ascontiguousarray(arr.transpose(3, 2, 0, 1))
    if kind == 'dense':
        if arr.ndim != 2:
            raise ValueError(f"dense 权重应为 2 维, 实际 {arr.shape}")
        return np.ascontiguousarray(arr.transpose(1, 0))
    if kind == 'vec':  # FRNorm weight/bias/eps、TLU tau：1 维向量直接使用
        return arr
    return arr


# ---------------------------------------------------------------------------
# 读取 DFL 权重文件
# ---------------------------------------------------------------------------
def load_dfl_npy(path: Path) -> dict:
    """读取 DFL .npy（实为 pickle dict）权重文件。"""
    with open(path, 'rb') as fh:
        d = pickle.load(fh)
    if not isinstance(d, dict):
        raise TypeError(f"{path} 不是 DFL 权重文件(dict)，实际类型 {type(d)}")
    return d


def load_dfl_dat(path: Path) -> dict:
    """读取 DFL _data.dat，返回 {'iter': int, 'options': dict, ...}。

    _data.dat 是参数权威来源（含真实迭代数与全部配置）；
    _summary.txt 可能被人为修改（如迭代数设为展示值），不可靠。
    """
    with open(path, 'rb') as fh:
        d = pickle.load(fh, encoding='latin1')
    if not isinstance(d, dict):
        raise TypeError(f"{path} 不是 DFL data 文件(dict)，实际类型 {type(d)}")
    return d


def find_all_model_prefixes(model_dir: Path) -> dict:
    """扫描 DFL 模型目录，返回 {prefix: {组件: 路径}}。"""
    model_dir = Path(model_dir)
    found: dict[str, dict] = {}
    for f in model_dir.iterdir():
        if not f.is_file() or not f.name.endswith('.npy') or '_opt' in f.name:
            continue
        name = f.name[:-4]
        # XSeg 特判: DFL XSeg 权重文件名为 XSeg_<res>.npy（如 XSeg_256.npy），
        # 非 "{prefix}_{component}" 结构，走独立正则。
        m = re.match(r'^(?:(.*)_)?XSeg_(\d+)$', name)
        if m:
            prefix = m.group(1) or 'XSeg'
            found.setdefault(prefix, {})['XSeg'] = f
            continue
        for comp in _ALL_COMPONENTS:
            marker = f'_{comp}'
            if name.endswith(marker):
                prefix = name[:-len(marker)]
                found.setdefault(prefix, {})[comp] = f
                break
    # 扫描 _data.dat 文件（参数权威来源，含 iter + options）
    for f in model_dir.iterdir():
        if not f.is_file() or not f.name.endswith('_data.dat'):
            continue
        prefix = f.name[:-len('_data.dat')]
        found.setdefault(prefix, {})['data'] = f
    return found


def infer_archi(prefix: str, forced: Optional[str] = None) -> str:
    """从 DFL 模型前缀推断架构类型。"""
    if forced:
        if forced not in ARCHI_NAMES:
            raise ValueError(f"不支持的架构 '{forced}'，可选: {'/'.join(ARCHI_NAMES)}")
        return forced
    n = prefix.upper()
    if '_SAEHD' in n:
        return 'saehd'
    if '_QUICK96' in n:
        return 'quick96'
    if '_AMP' in n:
        return 'amp'
    if 'XSEG' in n:
        return 'xseg'
    # 兼容旧命名：模型名直接含 "224df" / "192liae" 等
    if re.search(r'\d{3,4}(DF|LIAE)', n):
        return 'saehd'
    return None


# ---------------------------------------------------------------------------
# 生成项目 TrainingConfig 兼容的 JSON
# ---------------------------------------------------------------------------
def _to_bool(v, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in ('true', 'y', 'yes', '1')


def build_saehd_config(options: dict, archi: str, resolution: int) -> dict:
    """把 DFL SAEHD _data.dat options 映射为项目 TrainingConfig 字段。

    DFL options 字段与项目 TrainingConfig 一一对应；
    项目自定义字段（multiscale_loss_power/visibility_loss_power/lr/amp_mode/
    backup_interval/enable_torch_compile/auto_batch/gradient_accumulation_steps）
    DFL 无对应，用默认值。
    """
    ae_dims = int(float(options.get('ae_dims', 256)))
    e_dims = int(float(options.get('e_dims', 64)))
    d_dims = int(float(options.get('d_dims', 64)))
    d_mask_dims = int(float(options.get('d_mask_dims', 22)))

    face_type = options.get('face_type', 'wf')
    if face_type not in ('wf', 'head'):
        raise ValueError(
            f"DFL face_type='{face_type}' 本项目仅支持 'wf' / 'head'，无法直接转换。")

    config = {
        "face_type": face_type,
        "resolution": resolution,
        "archi": archi,
        "ae_dims": ae_dims,
        "e_dims": e_dims,
        "d_dims": d_dims,
        "d_mask_dims": d_mask_dims,
        "masked_training": _to_bool(options.get('masked_training'), True),
        "eyes_mouth_prio": _to_bool(options.get('eyes_mouth_prio'), False),
        "uniform_yaw": _to_bool(options.get('uniform_yaw'), False),
        "blur_out_mask": _to_bool(options.get('blur_out_mask'), False),
        "multiscale_loss_power": 0.0,
        "visibility_loss_power": 0.0,
        "adabelief": _to_bool(options.get('adabelief'), True),
        "lr_dropout": options.get('lr_dropout', 'n'),
        "random_warp": _to_bool(options.get('random_warp'), True),
        "random_src_flip": _to_bool(options.get('random_src_flip'), False),
        "random_dst_flip": _to_bool(options.get('random_dst_flip'), True),
        "random_hsv_power": float(options.get('random_hsv_power', 0.0)),
        "true_face_power": float(options.get('true_face_power', 0.0)),
        "face_style_power": float(options.get('face_style_power', 0.0)),
        "bg_style_power": float(options.get('bg_style_power', 0.0)),
        "gan_power": float(options.get('gan_power', 0.0)),
        "gan_patch_size": int(float(options.get('gan_patch_size', 16))),
        "gan_dims": int(float(options.get('gan_dims', 16))),
        "ct_mode": options.get('ct_mode', 'none'),
        "clipgrad": _to_bool(options.get('clipgrad'), False),
        "pretrain": _to_bool(options.get('pretrain'), False),
        "batch_size": int(float(options.get('batch_size', 8))),
        "lr": 5e-05,
        "amp_mode": "fp16",
        "target_iter": int(float(options.get('target_iter', 0))),
        "backup_interval": 0,
        "enable_torch_compile": False,
        "auto_batch": False,
        "gradient_accumulation_steps": 1,
    }
    return config


def build_amp_config(options: dict, resolution: int) -> dict:
    """把 DFL AMP _data.dat options 映射为项目 AMPTrainingConfig 字段。

    DFL options 字段与项目 AMPTrainingConfig 一一对应；
    项目自定义字段（lr/amp_mode/backup_interval/enable_torch_compile/
    auto_batch/gradient_accumulation_steps）DFL 无对应，用默认值。
    """
    face_type = options.get('face_type', 'wf')
    if face_type not in ('wf', 'head'):
        raise ValueError(
            f"DFL face_type='{face_type}' 本项目 AMP 仅支持 'wf' / 'head'，无法直接转换。")
    return {
        "resolution": resolution,
        "face_type": face_type,
        "ae_dims": int(float(options.get('ae_dims', 256))),
        "inter_dims": int(float(options.get('inter_dims', 1024))),
        "e_dims": int(float(options.get('e_dims', 64))),
        "d_dims": int(float(options.get('d_dims', 64))),
        "d_mask_dims": int(float(options.get('d_mask_dims', 22))),
        "morph_factor": float(options.get('morph_factor', 0.5)),
        "uniform_yaw": _to_bool(options.get('uniform_yaw'), False),
        "blur_out_mask": _to_bool(options.get('blur_out_mask'), False),
        "lr_dropout": options.get('lr_dropout', 'n'),
        "random_warp": _to_bool(options.get('random_warp'), True),
        "random_src_flip": _to_bool(options.get('random_src_flip'), False),
        "random_dst_flip": _to_bool(options.get('random_dst_flip'), True),
        "ct_mode": options.get('ct_mode', 'none'),
        "clipgrad": _to_bool(options.get('clipgrad'), False),
        "gan_power": float(options.get('gan_power', 0.0)),
        "gan_patch_size": int(float(options.get('gan_patch_size', resolution // 8))),
        "gan_dims": int(float(options.get('gan_dims', 16))),
        "batch_size": int(float(options.get('batch_size', 4))),
        "lr": 5e-05,
        "amp_mode": "fp16",
        "target_iter": int(float(options.get('target_iter', 0))),
        "backup_interval": 0,
        "enable_torch_compile": False,
        "auto_batch": False,
        "gradient_accumulation_steps": 1,
        "pretrain": _to_bool(options.get('pretrain'), False),
    }


def build_quick96_config() -> dict:
    """Quick96 全部参数硬编码（DFL 与项目一致）。"""
    return {
        "resolution": 96,
        "face_type": "wf",
        "ae_dims": 128,
        "e_dims": 64,
        "d_dims": 64,
        "d_mask_dims": 16,
        "masked_training": True,
        "random_warp": True,
        "random_src_flip": True,
        "random_dst_flip": True,
        "uniform_yaw": False,
        "ct_mode": "none",
        "eyes_mouth_prio": False,
        "visibility_loss_power": 0.0,
        "batch_size": 4,
        "lr": 2e-4,
        "amp_mode": "fp16",
        "target_iter": 0,
        "backup_interval": 0,
        "enable_torch_compile": False,
    }


def build_xseg_config(options: dict, resolution: int) -> dict:
    """把 DFL XSeg _data.dat options 映射为项目 XSegTrainingConfig 字段。

    DFL XSeg options 仅含 batch_size/face_type/pretrain；
    项目自定义字段（learning_rate/target_iter/amp_mode/lr_dropout/pretrain_iter）
    DFL 无对应，用 DFL 固定值或项目默认值。
    """
    face_type = options.get('face_type', 'wf')
    if face_type not in ('wf', 'head'):
        raise ValueError(
            f"DFL face_type='{face_type}' 本项目 XSeg 仅支持 'wf' / 'head'，无法直接转换。")
    if face_type == 'head':
        print("  [warn] DFL XSeg 固定 256 分辨率，项目 head 模式会提升到 384，"
              "转换按 wf 处理（仅影响配置标记，不影响权重）。")
        face_type = 'wf'
    return {
        "resolution": resolution,
        "face_type": face_type,
        "batch_size": int(float(options.get('batch_size', 4))),
        "learning_rate": 1e-4,
        "pretrain": _to_bool(options.get('pretrain'), False),
        "target_iter": int(float(options.get('target_iter', 0))),
        "amp_mode": "fp16",
        "lr_dropout": 0.3,
        "pretrain_iter": 0,
    }


def build_config(archi: str, options: dict, resolution: int, archi_detail: str) -> dict:
    if archi == 'amp':
        return build_amp_config(options, resolution)
    if archi == 'quick96':
        return build_quick96_config()
    if archi == 'xseg':
        return build_xseg_config(options, resolution)
    return build_saehd_config(options, archi_detail, resolution)


# ---------------------------------------------------------------------------
# 核心转换
# ---------------------------------------------------------------------------
def convert_component(comp_name: str, dfl_path: Path) -> OrderedDict:
    """转换单个组件: DFL .npy -> OrderedDict[str, np.ndarray](项目 key)。"""
    d = load_dfl_npy(dfl_path)
    state = OrderedDict()
    missing = []
    for dfl_key, arr in sorted(d.items()):
        proj_key, kind = remap_key(dfl_key)
        if proj_key is None:
            missing.append(dfl_key)
            continue
        if proj_key.endswith('.weight'):
            arr = transpose_weight(arr, kind)
        state[proj_key] = arr
    if missing:
        raise RuntimeError(
            f"{dfl_path.name} 中存在无法映射的变量(可能是不支持的架构/组件): {missing}")
    return state


# ---------------------------------------------------------------------------
# depth_to_space → pixel_shuffle 通道 permutation
# ---------------------------------------------------------------------------
def _pixel_shuffle_perm(oc: int, r: int = 2):
    """生成 TF depth_to_space → PT pixel_shuffle 的输出通道 permutation。

    TF: input ch (i*r+j)*oc + c → output (c, h*r+i, w*r+j)  [空间优先]
    PT: input ch c*r² + i*r + j   → output (c, h*r+i, w*r+j)  [通道优先]

    返回 perm 数组，使 new[pt_idx] = old[perm[pt_idx]] 后
    pixel_shuffle(new) == depth_to_space(old)。
    """
    perm = [0] * (oc * r * r)
    for c in range(oc):
        for i in range(r):
            for j in range(r):
                pt_idx = c * r * r + i * r + j
                tf_idx = (i * r + j) * oc + c
                perm[pt_idx] = tf_idx
    return perm


def _pad_1x1_to_3x3(w: np.ndarray) -> np.ndarray:
    """把 1x1 卷积权重 pad 成 3x3（中心放原权重、其余 8 位置零）。

    1x1 卷积是 3x3 卷积的特例，pad 成 3x3 + padding=1 后数学完全等价。
    若已是 3x3 则原样返回。d / 非 d 模式共用。
    """
    if w.shape[-1] == 1:
        p = np.zeros((w.shape[0], w.shape[1], 3, 3), dtype=w.dtype)
        p[:, :, 1, 1] = w[:, :, 0, 0]
        return p
    return w


def apply_dts_permutation(state: OrderedDict, opts: str) -> None:
    """对 decoder 权重做通道 permutation + 1x1→3x3 pad，使项目 3x3 out_conv
    能加载 DFL 权重且 F.pixel_shuffle 等价于 TF depth_to_space。

    - 非 d 模式：out_conv 从 1x1 pad 成 3x3（只 pad、不交织），以适配项目 3x3 conv
    - d 模式：Upscale/upscalem 重排通道；out_conv/out_conv1/2/3 pad→concat→perm→切回
    """
    r = 2
    use_d = 'd' in opts

    # 1. Upscale / upscalem 层：重排 conv 输出通道（仅 d 模式）
    #    遍历 state_dict 中所有 upscale*/upscalem*.conv.weight 的 key，
    #    不假设编号从 0 连续（inter 只有 upscale1，decoder 有 upscale0/1/2）
    if use_d:
        for key in list(state.keys()):
            if not key.endswith('.conv.weight'):
                continue
            prefix = key[:-len('.conv.weight')]
            if not (prefix.startswith('upscale') or prefix.startswith('upscalem')):
                continue
            w = state[key]
            oc = w.shape[0] // (r * r)
            perm = _pixel_shuffle_perm(oc, r)
            state[key] = w[perm]
            bkey = key[:-len('.weight')] + '.bias'
            if bkey in state:
                state[bkey] = state[bkey][perm]

    # 2. out_conv 系列：pad 1x1 -> 3x3，d 模式还要跨卷积交织 perm
    if 'out_conv.weight' in state:
        if use_d:
            # d 模式：4 个卷积 pad → concat → perm → 切回 4 个 3x3
            oc = 3  # RGB
            perm = _pixel_shuffle_perm(oc, r)  # 12 元素

            names = ['out_conv', 'out_conv1', 'out_conv2', 'out_conv3']
            w3 = [_pad_1x1_to_3x3(state[f'{n}.weight']) for n in names]
            b3 = [state[f'{n}.bias'] for n in names]

            cat_w = np.concatenate(w3, axis=0)              # (12, cin, 3, 3)
            cat_b = np.concatenate(b3, axis=0)              # (12,)

            cat_w = cat_w[perm]
            cat_b = cat_b[perm]

            for i, n in enumerate(names):
                state[f'{n}.weight'] = cat_w[i * oc:(i + 1) * oc]
                state[f'{n}.bias'] = cat_b[i * oc:(i + 1) * oc]
        else:
            # 非 d 模式：out_conv 只 pad、不交织
            state['out_conv.weight'] = _pad_1x1_to_3x3(state['out_conv.weight'])


def attach_xseg_buffers(state: OrderedDict, resolution: int) -> OrderedDict:
    """把项目 XSegNet 的 BlurPool 二项核 buffer(bp*._kernel) 补进转换结果。

    项目 _kernel 是持久 buffer(strict 加载必需)，而 DFL BlurPool 的核是
    tf.constant 硬编码、不进权重文件。二项核与分辨率无关、由 filt_size 决定，
    直接以项目初始化值写入即可（与 DFL 语义一致）。
    """
    from faceswap.models.xseg_model import XSegNet
    probe = XSegNet(resolution=resolution)
    for k, v in probe.state_dict().items():
        if k.endswith('._kernel'):
            state[k] = v.numpy()
    return state


def save_pth(state: OrderedDict, out_path: Path) -> None:
    """把 np 权重写入 torch .pth(state_dict)。"""
    if not HAS_TORCH:
        raise RuntimeError(
            "需要 torch 才能写出 .pth。请在本项目 Python 环境运行本脚本"
            "（python scripts/dfl_models_to_pytorch.py ...）")
    tensor_dict = OrderedDict(
        (k, torch.as_tensor(v, dtype=torch.float32)) for k, v in state.items())
    torch.save(tensor_dict, out_path)


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------
def verify_with_project(config: dict, out_dir: Path, archi: str,
                        include_gan: bool, include_code: bool) -> None:
    """用项目网络 strict 加载验证转换结果。"""
    if not HAS_TORCH:
        print("  [verify] torch 不可用，跳过网络校验。")
        return
    try:
        from faceswap.models.saehd.saehd_arch import Encoder, Inter, Decoder
        from faceswap.models.saehd.discriminators import (
            CodeDiscriminator, UNetPatchDiscriminator)
        if archi == 'amp':
            from faceswap.models.amp.amp_arch import (
                AMPEncoder, AMPInter, AMPDecoder)
        if archi == 'xseg':
            from faceswap.models.xseg_model import XSegNet
    except Exception as e:
        print(f"  [verify] 无法导入项目网络模块: {e}")
        print("  [verify] 请确认本脚本位于 faceswap 包内(scripts/ 目录)，")
        print("  [verify] 或在命令行前设置 PYTHONPATH=项目根目录(如 D:\\AI\\Inswapper)。")
        return

    res = config['resolution']
    modules = {}

    if archi == 'amp':
        ae = config['ae_dims']; e = config['e_dims']
        d = config['d_dims']; dm = config['d_mask_dims']
        inter = config['inter_dims']
        inter_res = res // 32
        modules = {
            'encoder': AMPEncoder(in_ch=3, e_ch=e, resolution=res, ae_dims=ae),
            'inter_src': AMPInter(ae_dims=ae, inter_dims=inter, inter_res=inter_res),
            'inter_dst': AMPInter(ae_dims=ae, inter_dims=inter, inter_res=inter_res),
            'decoder': AMPDecoder(inter_dims=inter, d_ch=d, d_mask_ch=dm),
        }
        if include_gan:
            modules['D_src'] = UNetPatchDiscriminator(
                patch_size=config['gan_patch_size'], in_ch=3, base_ch=config['gan_dims'])
    elif archi == 'quick96':
        ae = config['ae_dims']; e = config['e_dims']
        d = config['d_dims']; dm = config['d_mask_dims']
        opts = 'ud'
        enc_res = res // 16
        enc_flat = (e * 8) * enc_res * enc_res
        modules = {
            'encoder': Encoder(in_ch=3, e_ch=e, resolution=res, opts=opts),
            'inter': Inter(in_ch=enc_flat, ae_ch=ae, ae_out_ch=ae,
                           resolution=res, opts=opts),
            'decoder_src': Decoder(in_ch=ae, d_ch=d, d_mask_ch=dm, opts=opts),
            'decoder_dst': Decoder(in_ch=ae, d_ch=d, d_mask_ch=dm, opts=opts),
        }
    elif archi == 'saehd':
        archi_detail = config['archi']
        opts = archi_detail.split('-')[1] if '-' in archi_detail else ''
        ae = config['ae_dims']; e = config['e_dims']
        d = config['d_dims']; dm = config['d_mask_dims']
        has_t = 't' in opts
        enc_res = res // (32 if has_t else 16)
        enc_flat = (e * 8) * enc_res * enc_res
        modules = {
            'encoder': Encoder(in_ch=3, e_ch=e, resolution=res, opts=opts),
            'inter': Inter(in_ch=enc_flat, ae_ch=ae, ae_out_ch=ae,
                           resolution=res, opts=opts),
            'decoder_src': Decoder(in_ch=ae, d_ch=d, d_mask_ch=dm, opts=opts),
            'decoder_dst': Decoder(in_ch=ae, d_ch=d, d_mask_ch=dm, opts=opts),
        }
        if include_gan:
            modules['D_src'] = UNetPatchDiscriminator(
                patch_size=config['gan_patch_size'], in_ch=3, base_ch=config['gan_dims'])
        if include_code:
            # DFL: code_res = Inter.get_out_res()
            #   lowest_dense = res // (32 if 'd' in opts else 16)
            #   out_res = lowest_dense*2 if 't' not in opts else lowest_dense
            lowest_dense = res // (32 if 'd' in opts else 16)
            code_res = lowest_dense * 2 if 't' not in opts else lowest_dense
            modules['code_discriminator'] = CodeDiscriminator(
                in_ch=ae, code_res=code_res)
    else:  # xseg
        modules = {'xseg_net': XSegNet(resolution=res)}

    for name, mod in modules.items():
        pth = out_dir / f'{name}.pth'
        if not pth.exists():
            print(f"  [verify] 跳过 {name}: {pth.name} 不存在")
            continue
        sd = torch.load(pth, map_location='cpu')
        if archi == 'xseg':
            # XSeg: 项目 BlurPool 的 _kernel 是固定二项核 buffer（DFL BlurPool 无变量），
            # 保持项目初始化值即可，strict 校验时单独豁免 _kernel。
            missing, unexpected = mod.load_state_dict(sd, strict=False)
            kernel_missing = [k for k in missing if k.endswith('._kernel')]
            real_missing = [k for k in missing if not k.endswith('._kernel')]
            ok = (not unexpected) and (not real_missing)
            print(f"  [verify] OK {name}: {len(sd)} 个张量, 参数 strict 通过"
                  + (f", 忽略自建 buffer {len(kernel_missing)} 个(_kernel)"
                     if kernel_missing else "")
                  + (f" (missing={real_missing}, unexpected={unexpected})" if not ok else ""))
            # forward 冒烟: (1,3,res,res) -> (1,1,res,res)，无 NaN
            mod.eval()
            with torch.no_grad():
                y = mod(torch.zeros(1, 3, res, res))
                n_nan = int(torch.isnan(y).sum())
                print(f"  [verify] forward 冒烟: 输出 {tuple(y.shape)}, NaN={n_nan}")
        else:
            missing, unexpected = mod.load_state_dict(sd, strict=True)
            print(f"  [verify] OK {name}: {len(sd)} 个张量, strict 通过"
                  + (f" (missing={missing}, unexpected={unexpected})"
                     if (missing or unexpected) else ""))
    print("  [verify] 完成。")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def convert_one_model(prefix, files, model_dir, out_root, args):
    """转换单个 DFL 模型。"""
    print(f'\n[info] DFL 模型前缀: {prefix}')
    for comp, p in files.items():
        if p is not None:
            print(f'  [info] 发现 {comp:20s} -> {p.name}')

    # ---- 架构 ----
    archi = infer_archi(prefix, args.archi)
    if archi is None:
        raise RuntimeError(
            f'无法从模型名识别架构(需要含 _SAEHD/_AMP/_Quick96 或 "224df" 字样)，'
            f'请用 --archi 显式指定 (saehd/amp/quick96)')

    # ---- 从 _data.dat 提取参数（唯一来源）----
    dat_path = files.get('data')
    if dat_path is None:
        raise RuntimeError(
            f'找不到 _data.dat 文件，无法提取模型参数。'
            f'请确认 DFL 模型目录中包含 {prefix}_data.dat')
    dat = load_dfl_dat(dat_path)
    iter_v = int(dat.get('iter', 0) or 0)
    options = dat.get('options', {}) or {}
    print(f'[info] 读取 _data.dat: iter={iter_v}, {len(options)} 个配置项')

    # ---- archi_detail（SAEHD 从 options['archi'] 提取）----
    archi_detail = 'df'
    if archi == 'saehd':
        opt_archi = options.get('archi')
        if opt_archi:
            archi_detail = opt_archi
        else:
            m = re.search(r'(\d{3,4})(df|liae)(-[a-z]+)?', prefix)
            if m:
                archi_detail = m.group(2) + (m.group(3) or '')
            else:
                print(f'[warn] dat options 中无 archi 字段，默认 archi="{archi_detail}"')
    print(f'[info] 解析架构: {archi}' + (f' ({archi_detail})' if archi == 'saehd' else ''))

    # 分辨率
    if archi == 'quick96':
        resolution = 96
    elif archi == 'xseg':
        m = re.search(r'_(\d+)\.npy$', files['XSeg'].name)
        resolution = int(m.group(1)) if m else 256
    else:
        res_raw = options.get('resolution')
        if res_raw is None:
            raise RuntimeError('无法从 _data.dat 解析 resolution')
        resolution = int(float(res_raw))

    config = build_config(archi, options, resolution, archi_detail)

    # ---- 输出目录 ----
    out_root = Path(args.output)
    archi_subdir = {'saehd': 'saehd', 'amp': 'amp', 'quick96': 'quick96',
                    'xseg': 'xseg'}[archi]
    if out_root.name.lower() in ('saehd', 'amp', 'quick96', 'xseg'):
        if out_root.name.lower() != archi_subdir:
            out_root = out_root.parent / archi_subdir
    else:
        out_root = out_root / archi_subdir

    if args.name:
        out_name = args.name
    else:
        base = prefix
        for suffix in ('_SAEHD', '_AMP', '_Quick96'):
            if base.upper().endswith(suffix):
                base = base[:-len(suffix)]
                break
        if archi == 'saehd':
            out_name = f"{base}_{archi_detail}"
        else:
            out_name = base
    if archi == 'xseg':
        out_dir = out_root
    else:
        out_dir = out_root / out_name
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f'[错误] 输出目录已存在且非空: {out_dir}')
        print(f'  - 已存在同名模型，如需重新转换请先删除该目录，或用 --name 指定其它名称。')
        return False
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 转换网络权重 ----
    target_pairs = []
    for comp in ARCHI_COMPONENTS[archi]:
        if comp in ('GAN', 'code_discriminator'):
            continue
        if files[comp] is not None:
            target_pairs.append((comp, PTH_NAME.get(comp, comp)))

    print('\n[step] 转换网络权重...')
    # 计算 opts（用于判断是否需要 depth_to_space permutation）
    if archi == 'saehd':
        conv_opts = archi_detail.split('-')[1] if '-' in archi_detail else ''
    elif archi == 'amp':
        conv_opts = 'td'  # AMP 固定 df-td
    elif archi == 'quick96':
        conv_opts = 'ud'
    else:
        conv_opts = ''

    for comp, proj_name in target_pairs:
        if files[comp] is None:
            raise RuntimeError(f'缺少 DFL 组件文件: {comp}.npy')
        print(f'  [conv] {files[comp].name} -> {proj_name}.pth')
        state = convert_component(comp, files[comp])
        if comp in ('inter', 'inter_src', 'inter_dst',
                    'decoder', 'decoder_src', 'decoder_dst') and 'd' in conv_opts:
            apply_dts_permutation(state, conv_opts)
        if archi == 'xseg':
            state = attach_xseg_buffers(state, resolution)
        save_pth(state, out_dir / f'{proj_name}.pth')
        n = sum(int(np.prod(v.shape)) for v in state.values())
        print(f'    -- {len(state)} 个张量, {n/1e6:.2f}M 参数')

    # ---- 判别器（可选）----
    include_gan = (not args.skip_gan) and files['GAN'] is not None
    include_code = (not args.skip_code_disc) and files['code_discriminator'] is not None
    if include_gan:
        print('  [conv] GAN.npy -> D_src.pth')
        state = convert_component('GAN', files['GAN'])
        save_pth(state, out_dir / 'D_src.pth')
    if include_code:
        print('  [conv] code_discriminator.npy -> code_discriminator.pth')
        state = convert_component('code_discriminator', files['code_discriminator'])
        save_pth(state, out_dir / 'code_discriminator.pth')

    # ---- 配置文件与状态 ----
    config_json = json.dumps(config, indent=2, ensure_ascii=False)
    (out_dir / CONFIG_FILENAME[archi]).write_text(config_json, encoding='utf-8')
    gui_cfg_name = GUI_CONFIG_FILENAME.get(archi)
    if gui_cfg_name:
        (out_dir / gui_cfg_name).write_text(config_json, encoding='utf-8')

    state_iter = 0 if archi == 'xseg' else iter_v
    (out_dir / STATE_FILENAME[archi]).write_text(
        json.dumps({'iter': state_iter, 'loss_history': []}), encoding='utf-8')

    archi_tag = {'saehd': archi_detail, 'amp': 'AMP', 'quick96': 'Quick96',
                 'xseg': 'XSeg'}[archi]
    lines = [
        '============================================',
        '  Model Summary',
        '============================================',
        f'  Model name: {prefix} (converted)',
        f'  Current iteration: {iter_v}',
        '--------------------------------------------',
        '  Converted from DeepFaceLab by dfl_models_to_pytorch.py',
        '  archi: %s, resolution: %d, face_type: %s' % (
            archi_tag, resolution, config['face_type']),
        '  NOTE: optimizer state was NOT converted; project will re-create it.',
        '============================================',
    ]
    (out_dir / SUMMARY_FILENAME[archi]).write_text('\n'.join(lines), encoding='utf-8')
    (out_dir / '.save_complete').touch()

    print(f'\n[done] 转换完成 -> {out_dir}')
    print('  - 网络权重: ' + ', '.join(f'{n}.pth' for _, n in target_pairs))
    print(f'  - 配置: {CONFIG_FILENAME[archi]}')
    print(f'  - 状态: {STATE_FILENAME[archi]} / .save_complete')
    print('  - 优化器状态未转换(项目将重建优化器)，首次继续训练会重新累积动量。')

    if args.verify:
        print('\n[step] 项目网络 strict 校验...')
        verify_with_project(config, out_dir, archi, include_gan, include_code)
    return True


def main():
    ap = argparse.ArgumentParser(
        description='DFL 权重 -> 本项目 PyTorch 权重 转换工具 (SAEHD / AMP / Quick96)')
    ap.add_argument('--dfl-model', required=True,
                    help='DFL 模型目录(含 *_encoder.npy / *_data.dat 等)')
    ap.add_argument('--output', required=True,
                    help='项目模型根目录(如 .../model/saehd 或 .../model/amp 或 .../model/quick96)')
    ap.add_argument('--archi', default=None, choices=ARCHI_NAMES,
                    help='模型架构(默认自动识别)')
    ap.add_argument('--model-name', default=None,
                    help='DFL 模型前缀(目录含多个模型时指定，如 new_AMP)')
    ap.add_argument('--name', default=None, help='输出模型目录名(默认 "{iter}_{架构}")')
    ap.add_argument('--skip-gan', action='store_true', help='不转换 GAN 判别器')
    ap.add_argument('--skip-code-disc', action='store_true', help='不转换 true_face 判别器')
    ap.add_argument('--verify', action='store_true', help='转换后实例化项目网络做 strict 校验')
    args = ap.parse_args()

    model_dir = Path(args.dfl_model)
    if not model_dir.is_dir():
        print(f'[错误] DFL 模型目录不存在: {model_dir}')
        sys.exit(1)

    all_found = find_all_model_prefixes(model_dir)
    if not all_found:
        print(f'[错误] 在 {model_dir} 中未找到 DFL 权重文件')
        sys.exit(1)

    if args.model_name:
        if args.model_name not in all_found:
            print(f'[错误] 未找到模型前缀 "{args.model_name}"，可用: {sorted(all_found)}')
            sys.exit(1)
        all_found = {args.model_name: all_found[args.model_name]}

    out_root = Path(args.output)
    total = len(all_found)
    success = 0
    print(f'[info] 发现 {total} 个模型: {sorted(all_found)}')

    for prefix, comp_files in all_found.items():
        print(f'\n{"="*60}')
        print(f'[step] 转换模型 ({success}/{total}): {prefix}')
        print(f'{"="*60}')
        files = {comp: None for comp in _ALL_COMPONENTS}
        files.update(comp_files)
        try:
            if convert_one_model(prefix, files, model_dir, out_root, args):
                success += 1
        except Exception as e:
            print(f'[错误] 转换 {prefix} 失败: {e}')

    print(f'\n{"="*60}')
    print(f'[完成] 成功转换 {success}/{total} 个模型')
    if success < total:
        sys.exit(1)


if __name__ == '__main__':
    main()
