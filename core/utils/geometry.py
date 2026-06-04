"""Lightweight geometry helpers for bounding-box maths."""
from __future__ import annotations

from typing import Sequence, Tuple

Box = Sequence[float]  # (x1, y1, x2, y2)


def box_center(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box[:4]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_box(point: Tuple[float, float], box: Box) -> bool:
    px, py = point
    x1, y1, x2, y2 = box[:4]
    return x1 <= px <= x2 and y1 <= py <= y2


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two boxes."""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clamp_box(box: Box, width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp a box to image bounds and return integer coordinates."""
    x1, y1, x2, y2 = box[:4]
    x1 = int(max(0, min(x1, width - 1)))
    y1 = int(max(0, min(y1, height - 1)))
    x2 = int(max(0, min(x2, width - 1)))
    y2 = int(max(0, min(y2, height - 1)))
    return x1, y1, x2, y2
