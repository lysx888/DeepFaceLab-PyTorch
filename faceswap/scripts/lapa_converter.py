import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from faceswap.shared.logger import get_logger

_logger = get_logger("lapa_converter")

_LAPA_TO_IF = [0] * 106
for _l, _i in zip([46, 45, 44, 43, 42, 50, 49, 48, 47], [101, 105, 104, 103, 102, 97, 98, 99, 100]):
    _LAPA_TO_IF[_l] = _i
for _l, _i in zip([33, 34, 35, 36, 37, 38, 39, 40, 41], [43, 48, 49, 51, 50, 46, 47, 45, 44]):
    _LAPA_TO_IF[_l] = _i
for _l, _i in zip([79, 78, 77, 76, 75, 82, 81, 80], [93, 96, 94, 95, 89, 90, 87, 91]):
    _LAPA_TO_IF[_l] = _i
for _l, _i in zip([83, 105], [88, 92]):
    _LAPA_TO_IF[_l] = _i
for _l, _i in zip([66, 67, 68, 69, 70, 71, 72, 73], [35, 41, 40, 42, 39, 37, 33, 36]):
    _LAPA_TO_IF[_l] = _i
for _l, _i in zip([74, 104], [34, 38]):
    _LAPA_TO_IF[_l] = _i
for _l, _i in zip([51, 52, 53, 54], [72, 73, 74, 86]):
    _LAPA_TO_IF[_l] = _i
for _l, _i in zip([55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65], [75, 76, 77, 78, 79, 80, 85, 84, 83, 82, 81]):
    _LAPA_TO_IF[_l] = _i
for _l, _i in zip([96, 97, 98, 99, 100, 101, 102, 103], [65, 66, 62, 70, 69, 57, 60, 54]):
    _LAPA_TO_IF[_l] = _i
for _l, _i in zip([84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95], [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55]):
    _LAPA_TO_IF[_l] = _i
for _l, _i in zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                   [1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0, 24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17]):
    _LAPA_TO_IF[_l] = _i
LAPA_TO_IF = np.array(_LAPA_TO_IF, dtype=np.int64)
IF_TO_LAPA = np.empty_like(LAPA_TO_IF)
IF_TO_LAPA[LAPA_TO_IF] = np.arange(len(LAPA_TO_IF))

_KPS5_RIGHT_EYEBALL = [34, 38]
_KPS5_LEFT_EYEBALL = [88, 92]
_KPS5_NOSE_TIP = 86
_KPS5_RIGHT_MOUTH = 52
_KPS5_LEFT_MOUTH = 61


def _extract_kps5(if_pts: np.ndarray) -> np.ndarray:
    kps = np.zeros((5, 2), dtype=np.float32)
    kps[0] = if_pts[_KPS5_RIGHT_EYEBALL].mean(axis=0)
    kps[1] = if_pts[_KPS5_LEFT_EYEBALL].mean(axis=0)
    kps[2] = if_pts[_KPS5_NOSE_TIP]
    kps[3] = if_pts[_KPS5_RIGHT_MOUTH]
    kps[4] = if_pts[_KPS5_LEFT_MOUTH]
    return kps


def _read_lapa_landmarks(txt_path: Path) -> np.ndarray | None:
    try:
        with open(str(txt_path), "r") as f:
            n = int(f.readline())
            pts = []
            for _ in range(n):
                x, y = map(float, f.readline().split())
                pts.append([x, y])
            return np.array(pts, dtype=np.float32)
    except Exception as e:
        _logger.warning(f"读取LaPa标注失败 {txt_path}: {e}")
        return None


_LAPA_LABEL_OCCLUDED = {0, 10}


def _infer_visibility_from_label(
    if_pts: np.ndarray, label_path: Path
) -> np.ndarray:
    label_img = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if label_img is None:
        return np.ones(len(if_pts), dtype=bool)
    h, w = label_img.shape[:2]
    vis = np.ones(len(if_pts), dtype=bool)
    for i, (x, y) in enumerate(if_pts):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            if int(label_img[iy, ix]) in _LAPA_LABEL_OCCLUDED:
                vis[i] = False
    return vis


def convert_lapa_to_training(
    lapa_dir: str | Path,
    output_dir: str | Path,
    splits: list[str] | None = None,
    margin_ratio: float = 0.1,
) -> int:
    lapa_dir = Path(lapa_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if splits is None:
        splits = ["train", "val"]

    count = 0
    for split in splits:
        img_dir = lapa_dir / split / "images"
        lm_dir = lapa_dir / split / "landmarks"
        label_dir = lapa_dir / split / "labels"
        if not img_dir.exists() or not lm_dir.exists():
            _logger.warning(f"LaPa {split} 目录不存在，跳过")
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue

            lm_path = lm_dir / (img_path.stem + ".txt")
            if not lm_path.exists():
                continue

            lapa_pts = _read_lapa_landmarks(lm_path)
            if lapa_pts is None or lapa_pts.shape != (106, 2):
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            if_pts = lapa_pts[IF_TO_LAPA]

            x1, y1 = if_pts.min(axis=0)
            x2, y2 = if_pts.max(axis=0)
            bw = x2 - x1
            bh = y2 - y1
            margin = max(bw, bh) * margin_ratio
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            bbox = [float(x1), float(y1), float(x2), float(y2)]

            dst_img = output_dir / f"lapa_{split}_{img_path.name}"
            shutil.copy2(str(img_path), str(dst_img))

            kps_5 = _extract_kps5(if_pts)

            visibility = np.ones(106, dtype=bool)
            label_path = label_dir / (img_path.stem + ".png")
            if label_path.exists():
                visibility = _infer_visibility_from_label(if_pts, label_path)

            ann = {
                "landmarks_106": if_pts.tolist(),
                "bbox": bbox,
                "kps_5": kps_5.tolist(),
                "landmarks_106_visibility": visibility.tolist(),
            }
            json_path = dst_img.with_suffix(".json")
            with open(str(json_path), "w", encoding="utf-8") as f:
                json.dump(ann, f)

            count += 1

    _logger.info(f"LaPa转换完成: {count} 样本 -> {output_dir}")
    return count
