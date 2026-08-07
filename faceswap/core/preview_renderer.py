from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def mask_to_3ch(mask: np.ndarray) -> np.ndarray:
    return np.repeat(mask[..., None], 3, axis=-1)


def apply_mask_black_bg(image: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    return to_uint8(image.astype(np.float32) * mask3)


def alpha_blend(pred: np.ndarray, orig: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    return to_uint8(pred.astype(np.float32) * mask3 + orig.astype(np.float32) * (1.0 - mask3))


@dataclass
class PreviewData:
    S: np.ndarray
    D: np.ndarray
    SS: np.ndarray
    DD: np.ndarray
    SD: np.ndarray
    tgt_srcm: np.ndarray
    tgt_dstm: np.ndarray
    SSM: np.ndarray
    DDM: np.ndarray
    SDM: np.ndarray
    WS: Optional[np.ndarray] = None
    WD: Optional[np.ndarray] = None
    SS_w: Optional[np.ndarray] = None
    DD_w: Optional[np.ndarray] = None
    SD_w: Optional[np.ndarray] = None
    face_type: str = 'wf'
    sd_rct: Optional[np.ndarray] = None


def render_mode_raw(d: PreviewData) -> np.ndarray:
    return np.concatenate([d.S, d.SS, d.D, d.DD, d.SD], axis=1)


def render_mode_masked(d: PreviewData) -> np.ndarray:
    DDM3 = mask_to_3ch(d.DDM)
    SDM3 = mask_to_3ch(d.SDM)
    sd_mask = DDM3 * SDM3 if d.face_type != 'head' else SDM3
    return np.concatenate([
        apply_mask_black_bg(d.S, mask_to_3ch(d.tgt_srcm)),
        d.SS,
        apply_mask_black_bg(d.D, mask_to_3ch(d.tgt_dstm)),
        apply_mask_black_bg(d.DD, DDM3),
        apply_mask_black_bg(d.SD, sd_mask),
    ], axis=1)


def render_mode_warped(d: PreviewData) -> np.ndarray:
    if d.WS is not None and d.SS_w is not None:
        return np.concatenate([d.WS, d.SS_w, d.WD, d.DD_w, d.SD_w], axis=1)
    if d.WS is not None:
        return np.concatenate([d.WS, d.SS, d.WD, d.DD, d.SD], axis=1)
    return np.concatenate([d.S, d.SS, d.D, d.DD, d.SD], axis=1)


def render_mode_merged(d: PreviewData) -> np.ndarray:
    DDM3 = mask_to_3ch(d.DDM)
    SSM3 = mask_to_3ch(d.SSM)
    if d.face_type != 'head':
        dst_merge_mask = mask_to_3ch(d.tgt_dstm) * DDM3
        src_merge_mask = mask_to_3ch(d.tgt_srcm) * SSM3
    else:
        dst_merge_mask = mask_to_3ch(d.tgt_dstm)
        src_merge_mask = mask_to_3ch(d.tgt_srcm)
    ss_comp = alpha_blend(d.SS, d.S, src_merge_mask)
    dd_comp = alpha_blend(d.DD, d.D, dst_merge_mask)
    sd_img = d.sd_rct if d.sd_rct is not None else d.SD
    sd_comp = alpha_blend(sd_img, d.D, dst_merge_mask)
    return np.concatenate([d.S, ss_comp, d.D, dd_comp, sd_comp], axis=1)


SECTION_RENDERERS = {
    "原图预览": render_mode_raw,
    "遮罩下": render_mode_masked,
    "原始输入": render_mode_warped,
    "合并预览": render_mode_merged,
}


def render_all_sections(d: PreviewData) -> dict[str, np.ndarray]:
    return {name: fn(d) for name, fn in SECTION_RENDERERS.items()}
