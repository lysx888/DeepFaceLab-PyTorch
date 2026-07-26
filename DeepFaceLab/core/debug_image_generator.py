import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("debug_image_generator")

_LANDMARK_GROUPS_106 = [
    ("left_brow", [101, 105, 104, 103, 102, 97, 98, 99, 100]),
    ("right_brow", [43, 48, 49, 51, 50, 46, 47, 45, 44]),
    ("left_eye", [93, 96, 94, 95, 89, 90, 87, 91]),
    ("right_eye", [35, 41, 40, 42, 39, 37, 33, 36]),
    ("nose_bridge", [72, 73, 74, 86]),
    ("nose", [75, 76, 77, 78, 79, 80, 85, 84, 83, 82, 81]),
    ("inner_lip", [65, 66, 62, 70, 69, 57, 60, 54]),
    ("outer_lip", [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55]),
    ("jaw", [1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0, 24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17]),
]

_GROUP_COLORS = {
    "left_brow": (80, 255, 80),
    "right_brow": (80, 80, 255),
    "left_eye": (80, 255, 255),
    "right_eye": (255, 80, 80),
    "nose_bridge": (255, 255, 80),
    "nose": (255, 80, 255),
    "inner_lip": (255, 80, 200),
    "outer_lip": (80, 160, 255),
    "jaw": (200, 200, 200),
}

_LINE_CONNECTIONS = {
    "left_brow": [(101, 105), (105, 104), (104, 103), (103, 102), (102, 97), (97, 98), (98, 99), (99, 100), (100, 101)],
    "right_brow": [(43, 48), (48, 49), (49, 51), (51, 50), (50, 46), (46, 47), (47, 45), (45, 44), (44, 43)],
    "left_eye": [(93, 96), (96, 94), (94, 95), (95, 89), (89, 90), (90, 87), (87, 91), (91, 93)],
    "right_eye": [(35, 41), (41, 40), (40, 42), (42, 39), (39, 37), (37, 33), (33, 36), (36, 35)],
    "nose_bridge": [(72, 73), (73, 74), (74, 86)],
    "nose": [(75, 76), (76, 77), (77, 78), (78, 79), (79, 80), (80, 85), (85, 84), (84, 83), (83, 82), (82, 81)],
    "inner_lip": [(65, 66), (66, 62), (62, 70), (70, 69), (69, 57), (57, 60), (60, 54), (54, 65)],
    "outer_lip": [(52, 64), (64, 63), (63, 71), (71, 67), (67, 68), (68, 61), (61, 58), (58, 59), (59, 53), (53, 56), (56, 55), (55, 52)],
    "jaw": [(1, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 0), (0, 24), (24, 23), (23, 22), (22, 21), (21, 20), (20, 19), (19, 18), (18, 32), (32, 31), (31, 30), (30, 29), (29, 28), (28, 27), (27, 26), (26, 25), (25, 17)],
}

_CLOSED_GROUPS = {"left_brow", "right_brow", "left_eye", "right_eye", "inner_lip", "outer_lip"}

_HULL_PARTS_IDX = {
    "r_jaw": [1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8],
    "l_jaw": [8, 0, 24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17],
    "r_brow": [43, 48, 49, 51, 50, 46, 47, 45, 44],
    "l_brow": [97, 102, 103, 104, 105, 101, 100, 99, 98],
    "nose_bridge": [72, 73, 74, 86],
    "nose": [75, 76, 77, 78, 79, 80, 85, 84, 83, 82, 81],
    "chin": [8],
    "outer_lip": [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55],
}


def draw_debug_image(
    source_img: np.ndarray,
    landmarks_106: np.ndarray,
    transform_mat: Optional[np.ndarray] = None,
    output_size: int = 512,
    face_rect: Optional[tuple] = None,
    draw_hull: bool = True,
) -> np.ndarray:
    debug_img = source_img.copy()
    lm = landmarks_106.astype(np.float32)

    if face_rect is not None:
        rx, ry, rw, rh = [int(v) for v in face_rect]
        cv2.rectangle(debug_img, (rx, ry), (rx + rw, ry + rh), (212, 120, 0), 2)

    for gname, idxs in _LANDMARK_GROUPS_106:
        color = _GROUP_COLORS.get(gname, (0, 255, 0))
        lines = _LINE_CONNECTIONS.get(gname, [])
        for i1, i2 in lines:
            if i1 < len(lm) and i2 < len(lm):
                p1 = (int(lm[i1, 0]), int(lm[i1, 1]))
                p2 = (int(lm[i2, 0]), int(lm[i2, 1]))
                cv2.line(debug_img, p1, p2, color, 1)
        for idx in idxs:
            if idx < len(lm):
                cv2.circle(debug_img, (int(lm[idx, 0]), int(lm[idx, 1])), 2, color, -1)

    if draw_hull:
        hull_mask = np.zeros(debug_img.shape[:2], dtype=np.float32)
        r_jaw = lm[_HULL_PARTS_IDX["r_jaw"]]
        l_jaw = lm[_HULL_PARTS_IDX["l_jaw"]]
        r_brow = lm[_HULL_PARTS_IDX["r_brow"]]
        l_brow = lm[_HULL_PARTS_IDX["l_brow"]]
        nose_bridge = lm[_HULL_PARTS_IDX["nose_bridge"]]
        nose = lm[_HULL_PARTS_IDX["nose"]]
        chin = lm[_HULL_PARTS_IDX["chin"]]
        outer_lip = lm[_HULL_PARTS_IDX["outer_lip"]]
        parts = [
            np.concatenate([r_jaw, r_brow[:1]]),
            np.concatenate([l_jaw, l_brow[-1:]]),
            np.concatenate([r_brow[:3], chin]),
            np.concatenate([l_brow[-3:], chin]),
            np.concatenate([r_brow[3:], l_brow[:4], chin]),
            np.concatenate([r_brow, nose_bridge[:1], nose, chin]),
            np.concatenate([l_brow, nose_bridge[:1], nose, chin]),
            np.concatenate([nose_bridge, nose]),
            np.concatenate([nose, outer_lip, chin]),
        ]
        for item in parts:
            hull = cv2.convexHull(item).astype(np.int32)
            cv2.fillConvexPoly(hull_mask, hull, 1.0)
        debug_img[hull_mask > 0] = (debug_img[hull_mask > 0] * 0.5).astype(np.uint8)

    return debug_img


def save_debug_image(
    debug_dir: Path,
    stem: str,
    source_img: np.ndarray,
    landmarks_106: np.ndarray,
    transform_mat: Optional[np.ndarray] = None,
    output_size: int = 512,
    face_rect: Optional[tuple] = None,
    draw_hull: bool = True,
    quality: int = 95,
    fmt: str = "jpg",
) -> np.ndarray:
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_img = draw_debug_image(source_img, landmarks_106, transform_mat, output_size, face_rect, draw_hull)
    ext = f".{fmt.lower().lstrip('.')}"
    debug_path = debug_dir / (stem + ext)
    from DeepFaceLab.shared.file_manager import imwrite_auto
    imwrite_auto(debug_path, debug_img, jpg_quality=quality)
    return debug_img
