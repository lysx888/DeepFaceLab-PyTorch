# -*- coding: utf-8 -*-
"""
非刚性形变增强模块 (咀嚼/吸吮/鼓腮/张嘴/遮挡) — 完整三件套
============================================================
组成:
  1. TPSDeform       : TPS 平滑形变 (下半脸/嘴/全脸), 模拟咀嚼/吸吮/鼓腮
  2. MouthOpenDeform : 嘴部大动作 (张嘴), 上下唇沿垂直方向开合
  3. CutoutOcclusion : 嘴/下巴区域遮挡块 (模拟食物), 并更新 visibility
  4. DeformAug       : 组合类, 按概率触发三者, 统一接口 (img, lm, vis) -> (img, lm, vis)

方向约定 (重要, 已修正):
  - RBF 学习"位置 -> 位移 off" (off = dst - src)
  - 图像: out(q) = img(q - off(q))   -> 内容向 +off 方向移动
  - 标签: nl = lm + off(lm)          -> landmark 与图像同步向 +off 移动
  - 两者一致, 保证形变后点贴合真实形变脸

验证:
  - scripts/verify_finetune_deform.py 复跑 (-68% 逻辑验证)
  - scripts/verify_deform_match.py    模板匹配自检: 图像内容实际位移 == landmark 位移
"""
import math

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

# insightface 106 点嘴部索引 (20 点)
MOUTH_IDX = [65, 66, 62, 70, 69, 57, 60, 54,
             52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55]


class _TPSField:
    """共享: 从控制点位移插值全图位移场 + 应用 (图像/标签方向一致)"""

    @staticmethod
    def build_field(size, src, dst, smoothing=0.0):
        """src/dst: (N,2) 控制点源/目标. 返回 (map_x, map_y, off_x, off_y).
        map 语义: 图像 out(q)=img(q-off(q)); landmark nl=lm+off(lm)
        smoothing>0: 对控制点过密/近似共线导致的病态矩阵加正则
        """
        dxx = dst[:, 0] - src[:, 0]
        dyy = dst[:, 1] - src[:, 1]
        rbfx = RBFInterpolator(src, dxx, kernel='thin_plate_spline', smoothing=smoothing)
        rbfy = RBFInterpolator(src, dyy, kernel='thin_plate_spline', smoothing=smoothing)
        yy, xx = np.mgrid[0:size, 0:size]
        grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
        off_x = rbfx(grid).reshape(size, size)
        off_y = rbfy(grid).reshape(size, size)
        map_x = (xx - off_x).astype(np.float32)   # 反向采样: 内容向 +off 移动
        map_y = (yy - off_y).astype(np.float32)
        return map_x, map_y, off_x, off_y, rbfx, rbfy

    @staticmethod
    def apply(img, lm, map_x, map_y, rbfx, rbfy):
        warped = cv2.remap(img, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        if lm is not None and len(lm):
            off = np.stack([rbfx(lm.astype(np.float64)),
                            rbfy(lm.astype(np.float64))], axis=1)
            new_lm = lm + off.astype(np.float32)
            return warped, new_lm
        return warped, None


class TPSDeform:
    """TPS 平滑形变: 随机控制点位移, 模拟咀嚼/吸吮/鼓腮"""

    def __init__(self, size=192, n_ctrl=14, strength_range=(2.0, 12.0),
                 region='lower_face', margin=8, seed=None):
        self.size = size
        self.n_ctrl = n_ctrl
        self.strength_range = strength_range
        self.region = region
        self.margin = margin
        self.rng = np.random.default_rng(seed)

    def _sample_ctrl(self):
        H = W = self.size
        m = self.margin
        pts = []
        xs = np.linspace(m, W - m, 5)
        ys = np.linspace(m, H - m, 5)
        for x in xs:
            for y in ys:
                if x in (m, W - m) or y in (m, H - m):
                    pts.append([x, y])
        if self.region == 'mouth':
            x = self.rng.uniform(W * 0.25, W * 0.75, self.n_ctrl)
            y = self.rng.uniform(H * 0.55, H * 0.78, self.n_ctrl)
        elif self.region == 'full':
            x = self.rng.uniform(W * 0.12, W * 0.88, self.n_ctrl)
            y = self.rng.uniform(H * 0.12, H * 0.88, self.n_ctrl)
        else:  # lower_face
            x = self.rng.uniform(W * 0.15, W * 0.85, self.n_ctrl)
            y = self.rng.uniform(H * 0.40, H * 0.95, self.n_ctrl)
        for i in range(self.n_ctrl):
            pts.append([x[i], y[i]])
        return np.asarray(pts, dtype=np.float64)

    def deform(self, img, lm=None):
        m = self.margin
        W = self.size
        src = self._sample_ctrl()
        dst = src.copy()
        for i in range(len(src)):
            if src[i, 0] in (m, W - m) or src[i, 1] in (m, W - m):
                continue
            ang = self.rng.uniform(0, 2 * math.pi)
            mag = self.rng.uniform(*self.strength_range)
            dst[i, 0] += mag * math.cos(ang)
            dst[i, 1] += mag * math.sin(ang)
        map_x, map_y, _, _, rbfx, rbfy = _TPSField.build_field(self.size, src, dst)
        return _TPSField.apply(img, lm, map_x, map_y, rbfx, rbfy)


class MouthOpenDeform:
    """嘴部大动作: 上下唇沿垂直方向开合, 模拟张嘴/咀嚼.

    实现: 手动构造垂直开合位移场 (非 RBF 外推, 避免远处发散).
      位移场 off(p) = sign(y - y_mid) * open_amt * 距离衰减 * 横向权重
      - 嘴部中心 y_mid 以上向上、以下向下 (张嘴)
      - 距嘴中心越远衰减到 0 (眼睛/额头完全不动)
      - 横向远离嘴中心(嘴角侧)权重低 (中央张开大, 嘴角小)
      图像/标签方向约定与 TPSDeform 一致 (内容向 +off 移动)
    """

    def __init__(self, size=192, open_range=(3.0, 12.0), seed=None):
        self.size = size
        self.open_range = open_range
        self.rng = np.random.default_rng(seed)

    def deform(self, img, lm):
        H = W = self.size
        mouth = lm[MOUTH_IDX]
        y_mid = float(np.median(mouth[:, 1]))
        x_mid = float(np.median(mouth[:, 0]))
        open_amt = self.rng.uniform(*self.open_range)
        # 嘴部影响半径
        R = max(16.0, float(np.abs(mouth[:, 1] - y_mid).max() * 2.5))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt((xx - x_mid) ** 2 + (yy - y_mid) ** 2)
        w = np.clip(1.0 - d / R, 0, 1) ** 1.5          # 距离衰减
        wx = np.clip(1.0 - np.abs(xx - x_mid) / (R * 0.8), 0.3, 1.0)  # 横向权重
        dy_field = np.sign(yy - y_mid) * open_amt * w * wx
        dy_field = cv2.GaussianBlur(dy_field, (0, 0), 2.0).astype(np.float32)
        # 图像: 内容向 +off 移动 -> map = q - off (off 只有 y 分量)
        map_x = xx
        map_y = yy - dy_field
        warped = cv2.remap(img, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        # landmark: nl = lm + off(lm)
        if lm is not None and len(lm):
            lx = np.clip(np.round(lm[:, 0]).astype(int), 0, W - 1)
            ly = np.clip(np.round(lm[:, 1]).astype(int), 0, H - 1)
            off_lm = dy_field[ly, lx]
            nl = lm.copy()
            nl[:, 1] += off_lm
            return warped, nl
        return warped, None


class CutoutOcclusion:
    """嘴/下巴区域遮挡块 (模拟食物/手), 并更新 visibility"""

    def __init__(self, size=192, region_xy=(0.15, 0.85, 0.50, 0.95),
                 min_r=0.10, max_r=0.28, seed=None):
        self.size = size
        self.region_xy = region_xy  # (x0, x1, y0, y1) 归一化
        self.min_r = min_r
        self.max_r = max_r
        self.rng = np.random.default_rng(seed)

    def apply(self, img, lm, vis=None):
        H = W = self.size
        x0, x1, y0, y1 = self.region_xy
        cx = self.rng.uniform(x0, x1) * W
        cy = self.rng.uniform(y0, y1) * H
        rx = self.rng.uniform(self.min_r, self.max_r) * W
        ry = rx * self.rng.uniform(0.7, 1.2)
        ang = self.rng.uniform(0, math.pi)
        # 掩码
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)),
                    math.degrees(ang), 0, 360, 255, -1)
        # 填充: 肤色系/食物色 + 纹理噪声
        color = self.rng.integers(70, 200, 3).tolist()
        noise = np.random.default_rng(self.rng.integers(0, 2**32)).normal(0, 18, (H, W, 3))
        occ = np.zeros((H, W, 3), dtype=np.float32)
        occ[:, :, 0] = color[0]
        occ[:, :, 1] = color[1]
        occ[:, :, 2] = color[2]
        occ += noise
        occ = np.clip(occ, 0, 255).astype(np.uint8)
        img = img.copy()
        img[mask > 0] = occ[mask > 0]
        # 更新 visibility
        if lm is not None and len(lm) and vis is not None:
            vis = vis.copy()
            inside = mask[np.clip(lm[:, 1].astype(int), 0, H - 1),
                          np.clip(lm[:, 0].astype(int), 0, W - 1)] > 0
            vis[inside] = False
        return img, lm, vis


class DeformAug:
    """组合增强: TPS 形变 + 嘴部大动作 + Cutout 遮挡, 按概率触发"""

    def __init__(self, size=192, p_tps=0.30, p_mouth=0.25, p_cutout=0.25,
                 seed=None):
        rng = np.random.default_rng(seed)
        self.p_tps = p_tps
        self.p_mouth = p_mouth
        self.p_cutout = p_cutout
        self.tps = TPSDeform(size=size, seed=rng.integers(0, 2**31))
        self.mouth = MouthOpenDeform(size=size, seed=rng.integers(0, 2**31))
        self.cutout = CutoutOcclusion(size=size, seed=rng.integers(0, 2**31))
        self.rng = rng

    def __call__(self, img, lm, vis=None):
        """img: (H,W,3) BGR; lm: (N,2); vis: (N,) bool 或 None.
        返回 (img, lm, vis)"""
        if vis is None:
            vis = np.ones(len(lm), dtype=bool)
        if self.rng.random() < self.p_tps:
            img, lm = self.tps.deform(img, lm)
        if self.rng.random() < self.p_mouth:
            img, lm = self.mouth.deform(img, lm)
        if self.rng.random() < self.p_cutout:
            img, lm, vis = self.cutout.apply(img, lm, vis)
        return img, lm, vis


# ---- 独立演示 ----
if __name__ == '__main__':
    import json
    from pathlib import Path
    SRC = Path(r'D:\AI\Inswapper\workspace\insightface_train\manual_annotated\097c614774cf50965bec0337521d7d16.json')
    OUT = Path(r'D:\AI\Inswapper\faceswap\scripts\deform_demo')
    OUT.mkdir(parents=True, exist_ok=True)
    ann = json.loads(SRC.read_text(encoding='utf-8'))
    img0 = cv2.imread(str(SRC.with_suffix('.jpg')))
    lm = np.asarray(ann['landmarks_106'], dtype=np.float32)
    bbox = np.asarray(ann['bbox'], dtype=np.float32)
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    center = np.array([(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2], dtype=np.float32)
    scale = 192 / (max(w, h) * 1.5)
    M = np.array([[scale, 0, 96 - center[0] * scale],
                  [0, scale, 96 - center[1] * scale]], dtype=np.float32)
    aligned = cv2.warpAffine(img0, M, (192, 192), flags=cv2.INTER_LINEAR, borderValue=0)
    gt = np.zeros_like(lm)
    gt[:, 0] = M[0, 0] * lm[:, 0] + M[0, 1] * lm[:, 1] + M[0, 2]
    gt[:, 1] = M[1, 0] * lm[:, 0] + M[1, 1] * lm[:, 1] + M[1, 2]

    def draw(aug, name, lm_in=gt):
        img_w, nl = aug.deform(aligned, lm_in)
        vis = img_w.copy()
        for (x, y) in nl.astype(int):
            cv2.circle(vis, (x, y), 2, (255, 200, 0), -1)
        cv2.imwrite(str(OUT / name), vis)

    draw(TPSDeform(size=192, region='lower_face', strength_range=(3.0, 12.0), seed=7), 'tps_lower.png')
    draw(MouthOpenDeform(size=192, open_range=(6.0, 10.0), seed=7), 'mouth_open.png')
    # Cutout: 画遮挡 + 遮挡点(红)
    cut = CutoutOcclusion(size=192, seed=7)
    img_c, lm_c, vis_c = cut.apply(aligned, gt, np.ones(106, dtype=bool))
    visimg = img_c.copy()
    for i, (x, y) in enumerate(lm_c.astype(int)):
        cv2.circle(visimg, (x, y), 2, (0, 0, 255) if not vis_c[i] else (255, 200, 0), -1)
    cv2.imwrite(str(OUT / 'cutout.png'), visimg)
    print('演示已保存到', OUT)
