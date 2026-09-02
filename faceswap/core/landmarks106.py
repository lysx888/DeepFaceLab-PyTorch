import cv2
import numpy as np


LANDMARK_GROUPS_LaPa_106 = [
    ("left_eyebrow", [46, 45, 44, 43, 42, 50, 49, 48, 47]),
    ("right_eyebrow", [33, 34, 35, 36, 37, 38, 39, 40, 41]),
    ("left_eye", [79, 78, 77, 76, 75, 82, 81, 80]),
    ("left_eyeball", [83, 105]),
    ("right_eye", [66, 67, 68, 69, 70, 71, 72, 73]),
    ("right_eyeball", [74, 104]),
    ("nose_bridge", [51, 52, 53, 54]),
    ("nose", [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]),
    ("inner_lip", [96, 97,98, 99, 100,101, 102, 103]),
    ("outer_lip", [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    ("jaw_cheek", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]),
]

LANDMARK_GROUPS_WFLW_98 = [
    ("left_eyebrow", [46, 45, 44, 43, 42, 50, 49, 48, 47]),
    ("right_eyebrow", [33, 34, 35, 36, 37, 38, 39, 40, 41]),
    ("left_eye", [72, 71, 70, 69, 68, 75, 74, 73]),
    ("left_eyeball", [97]),
    ("right_eye", [60, 61, 62, 63, 64, 65, 66, 67]),
    ("right_eyeball", [96]),
    ("nose_bridge", [51, 52, 53, 54]),
    ("nose", [55, 56, 57, 58, 59]),
    ("inner_lip", [88, 89,90, 91, 92,93, 94, 95]),
    ("outer_lip", [76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87]),
    ("jaw_cheek", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]),
]

LANDMARK_GROUPS_106 = [
    ("left_eyebrow", [101, 105, 104, 103, 102, 97, 98, 99, 100]),
    ("right_eyebrow", [43, 48, 49, 51, 50, 46, 47, 45, 44]),
    ("left_eye", [93, 96, 94, 95, 89, 90, 87, 91]),
    ("left_eyeball", [88, 92]),
    ("right_eye", [35, 41, 40, 42, 39, 37, 33, 36]),
    ("right_eyeball", [34, 38]),
    ("nose_bridge", [72, 73, 74, 86]),
    ("nose", [75, 76, 77, 78, 79, 80, 85, 84, 83, 82, 81]),
    ("inner_lip", [65, 66, 62, 70, 69, 57, 60, 54]),
    ("outer_lip", [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55]),
    ("jaw_cheek", [1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0, 24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17]),
]

LINE_CONNECTIONS_106 = {
    "left_eyebrow": [(101, 105), (105, 104), (104, 103), (103, 102), (102, 97), (97, 98), (98, 99), (99, 100), (100, 101)],
    "right_eyebrow": [(43, 48), (48, 49), (49, 51), (51, 50), (50, 46), (46, 47), (47, 45), (45, 44), (44, 43)],
    "left_eye": [(93, 96), (96, 94), (94, 95), (95, 89), (89, 90), (90, 87), (87, 91), (91, 93)],
    "right_eye": [(35, 41), (41, 40), (40, 42), (42, 39), (39, 37), (37, 33), (33, 36), (36, 35)],
    "nose_bridge": [(72, 73), (73, 74), (74, 86)],
    "nose": [(75, 76), (76, 77), (77, 78), (78, 79), (79, 80), (80, 85), (85, 84), (84, 83), (83, 82), (82, 81)],
    "inner_lip": [(65, 66), (66, 62), (62, 70), (70, 69), (69, 57), (57, 60), (60, 54), (54, 65)],
    "outer_lip": [(52, 64), (64, 63), (63, 71), (71, 67), (67, 68), (68, 61), (61, 58), (58, 59), (59, 53), (53, 56), (56, 55), (55, 52)],
    "jaw_cheek": [(1, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 0), (0, 24), (24, 23), (23, 22), (22, 21), (21, 20), (20, 19), (19, 18), (18, 32), (32, 31), (31, 30), (30, 29), (29, 28), (28, 27), (27, 26), (26, 25), (25, 17)],
}


def _expand_eyebrows_106(lm: np.ndarray, mod: float) -> np.ndarray:
    lm = lm.copy().astype(np.float32)
    r_eye_outer = lm[35]
    l_eye_outer = lm[93]
    r_jaw_start = lm[1]
    l_jaw_end = lm[17]
    ml_pnt = (r_eye_outer + r_jaw_start) / 2
    mr_pnt = (l_eye_outer + l_jaw_end) / 2
    ql_pnt = (r_eye_outer + ml_pnt) / 2
    qr_pnt = (l_eye_outer + mr_pnt) / 2
    bot_r = np.array([r_eye_outer, lm[41], lm[40], lm[42], lm[39]])
    bot_l = np.array([lm[93], lm[96], lm[94], lm[95], l_eye_outer])
    top_r = lm[np.array([43, 48, 49, 51, 50])]
    top_l = lm[np.array([102, 103, 104, 105, 101])]
    expanded_r = top_r + mod * 0.5 * (top_r - bot_r)
    expanded_l = top_l + mod * 0.5 * (top_l - bot_l)
    lm[np.array([43, 48, 49, 51, 50])] = expanded_r
    lm[np.array([102, 103, 104, 105, 101])] = expanded_l
    return lm


def fill_hull_mask_106(mask: np.ndarray, lm: np.ndarray, eyebrows_expand_mod: float = 1.0) -> None:
    if len(lm) < 68:
        hull_pts = cv2.convexHull(lm.astype(np.float32))
        cv2.fillConvexPoly(mask, hull_pts.astype(np.int32), 255)
        return

    if len(lm) >= 106:
        lmrks = _expand_eyebrows_106(lm, eyebrows_expand_mod)
        r_jaw = lmrks[np.array([1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0, 43])]
        l_jaw = lmrks[np.array([0, 24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17, 101])]
        r_cheek = lmrks[np.array([43, 48, 49, 0])]
        l_cheek = lmrks[np.array([104, 105, 101, 0])]
        nose_ridge = lmrks[np.array([49, 51, 50, 102, 103, 104, 0])]
        r_eye = lmrks[np.array([43, 48, 49, 51, 50, 72, 77, 78, 79, 80, 85, 84, 83, 0])]
        l_eye = lmrks[np.array([102, 103, 104, 105, 101, 72, 77, 78, 79, 80, 85, 84, 83, 0])]
        nose = lmrks[np.array([72, 73, 74, 86, 77, 78, 79, 80, 85, 84, 83])]
    else:
        r_jaw = lm[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 17])]
        l_jaw = lm[np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 26])]
        r_cheek = lm[np.array([17, 18, 19, 8])]
        l_cheek = lm[np.array([24, 25, 26, 8])]
        nose_ridge = lm[np.array([19, 20, 21, 22, 8])]
        r_eye = lm[np.array([17, 18, 19, 20, 21, 22, 27, 28, 31, 32, 33, 34, 35, 8])]
        l_eye = lm[np.array([22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 8])]
        nose = lm[np.array([27, 28, 29, 31, 32, 33, 34, 35])]

    parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]
    for part in parts:
        hull = cv2.convexHull(part.astype(np.float32))
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)


def get_hull_mask_106(shape: tuple, lm106: np.ndarray, eyebrows_expand_mod: float = 1.0) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    fill_hull_mask_106(mask, lm106, eyebrows_expand_mod)
    return mask
