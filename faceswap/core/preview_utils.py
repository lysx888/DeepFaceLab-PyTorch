from typing import Optional

import cv2
import numpy as np

from faceswap.shared.image_utils import bgr_to_rgb
from faceswap.shared.logger import get_logger

_logger = get_logger("preview_utils")


def get_font(size: int = 12):
    try:
        from PIL import ImageFont
        candidates = [
            "consola.ttf", "Consolas.ttf",
            "DejaVuSansMono.ttf", "dejavusansmono.ttf",
            "LiberationMono-Regular.ttf",
            "arial.ttf", "Arial.ttf",
            "NotoSansMono-Regular.ttf",
        ]
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()
    except ImportError:
        return None


def draw_loss_chart(
    width: int,
    height: int,
    loss_history: list[tuple[int, float, ...]],
    loss_range: int = 0,
    loss_names: Optional[list[str]] = None,
    loss_colors: Optional[list[tuple[int, int, int]]] = None,
) -> np.ndarray:
    chart = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(6):
        y = int(i * (height - 1) / 5)
        chart[y, :] = (40, 40, 40)
    if len(loss_history) < 2:
        return chart

    range_labels = ["all", "last 1k", "last 100"]
    range_limits = [0, 1000, 100]
    limit = range_limits[loss_range] if loss_range < len(range_limits) else 0
    history = loss_history[-limit:] if limit > 0 else loss_history

    iters = [h[0] for h in history]
    num_curves = len(history[0]) - 1
    if loss_names is None:
        loss_names = [f"loss{i}" for i in range(num_curves)]
    if loss_colors is None:
        default_colors = [(0, 180, 255), (0, 255, 120), (255, 200, 0), (200, 100, 255)]
        loss_colors = [default_colors[i % len(default_colors)] for i in range(num_curves)]

    all_losses = []
    for ci in range(num_curves):
        all_losses.extend(h[1 + ci] for h in history)
    abs_max = np.mean(sorted(all_losses)[len(all_losses) // 5:]) * 2
    if abs_max <= 0:
        abs_max = 1.0

    for i in range(6):
        y = int(i * (height - 1) / 5)
        chart[y, :] = (40, 40, 40)

    for ci in range(num_curves):
        losses = [h[1 + ci] for h in history]
        color = loss_colors[ci]
        lh_len = len(losses)
        prev_y = None
        for col in range(width):
            idx = int(col * (lh_len - 1) / max(width - 1, 1))
            val = losses[idx]
            y = int(np.clip((val / abs_max) * (height - 1), 0, height - 1))
            row = height - y - 1
            chart[row, col] = color
            if prev_y is not None:
                y0, y1 = min(prev_y, row), max(prev_y, row)
                for r in range(y0, y1 + 1):
                    chart[r, col] = color
            prev_y = row

    last_iter = iters[-1] if iters else 0
    range_label = range_labels[loss_range] if loss_range < len(range_labels) else "all"

    try:
        from PIL import Image, ImageDraw
        font = get_font(12)
        chart_rgb = bgr_to_rgb(chart)
        pil_img = Image.fromarray(chart_rgb)
        draw = ImageDraw.Draw(pil_img)
        draw.text((4, 2), f"Iter: {last_iter}  Range: {range_label}", fill=(180, 180, 180), font=font)
        x_offset = 4
        for ci in range(num_curves):
            val = history[-1][1 + ci] if history else 0
            label = f"{loss_names[ci]}: {val:.4f}"
            rgb_color = loss_colors[ci][::-1]
            draw.text((x_offset, height - 16), label, fill=rgb_color, font=font)
            x_offset += len(label) * 8 + 20
        result_bgr = np.array(pil_img)
        return result_bgr[:, :, ::-1].copy()
    except ImportError:
        return chart


def draw_head_bar(
    width: int,
    lines: list[str],
    line_height: int = 20,
    font_size: int = 14,
) -> np.ndarray:
    head_h = line_height * len(lines)
    head = np.zeros((head_h, width, 3), dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw
        font = get_font(font_size)
        pil_img = Image.fromarray(head)
        draw = ImageDraw.Draw(pil_img)
        for i, line in enumerate(lines):
            y = i * line_height + 2
            draw.text((6, y), line, fill=(200, 200, 200), font=font)
        head_rgb = np.array(pil_img, dtype=np.uint8)
        return head_rgb[:, :, ::-1].copy()
    except ImportError:
        return head


def compose_preview(
    section_rows: list[np.ndarray],
    head_lines: list[str],
    loss_history: list[tuple[int, float, ...]],
    loss_range: int = 0,
    loss_names: Optional[list[str]] = None,
    loss_colors: Optional[list[tuple[int, int, int]]] = None,
    chart_height: int = 100,
) -> np.ndarray:
    if not section_rows:
        return np.zeros((128, 640, 3), dtype=np.uint8)
    for i, row in enumerate(section_rows):
        if row.dtype != np.uint8:
            from faceswap.shared.logger import get_logger
            get_logger("preview_utils").warning(f"section_row[{i}] dtype={row.dtype}, converting")
            section_rows[i] = np.clip(row, 0, 255).astype(np.uint8)
    preview_bgr = np.vstack(section_rows)
    h, w = preview_bgr.shape[:2]
    head = draw_head_bar(w, head_lines, line_height=22, font_size=15)
    chart = draw_loss_chart(w, chart_height, loss_history, loss_range,
                            loss_names=loss_names, loss_colors=loss_colors)
    final = np.vstack([head, chart, preview_bgr])
    return np.clip(final, 0, 255).astype(np.uint8)
