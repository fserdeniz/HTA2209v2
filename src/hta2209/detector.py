"""
Lightweight AI-assisted color and edge analysis using k-means and Canny (RGB).
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple

ColorName = Tuple[str, Tuple[int, int, int]]  # (name, RGB)


def _label_color(rgb: np.ndarray) -> str:
    r, g, b = [int(x) for x in rgb]
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv
    if v < 40:
        return "black"
    if v > 220 and s < 30:
        return "white"
    if s < 40:
        return "gray"
    if h < 10 or h >= 170:
        return "red"
    if 10 <= h < 25:
        return "orange"
    if 25 <= h < 35:
        return "yellow"
    if 35 <= h < 85:
        return "green"
    if 85 <= h < 125:
        return "blue"
    if 125 <= h < 150:
        return "purple"
    return "pink"


def dominant_colors(frame_rgb: np.ndarray, k: int = 3) -> List[ColorName]:
    """Returns up to k dominant colors (name, RGB) sorted by pixel share."""
    if frame_rgb.size == 0:
        return []
    small = cv2.resize(frame_rgb, (160, 120), interpolation=cv2.INTER_AREA)
    pixels = small.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _retval, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(counts)[::-1]
    result: List[ColorName] = []
    for idx in order:
        rgb = centers[idx].astype(np.uint8).tolist()
        result.append((_label_color(np.array(rgb, dtype=np.uint8)), tuple(int(x) for x in rgb)))
    return result


def analyze_frame(frame_rgb: np.ndarray) -> Tuple[np.ndarray, List[ColorName]]:
    """
    Returns (overlay_rgb, dominant_colors) where overlay has edge highlights and color labels.
    """
    if frame_rgb.size == 0:
        return frame_rgb, []
    overlay = frame_rgb.copy()
    edges = cv2.Canny(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY), 80, 160)
    overlay[edges > 0] = (255, 0, 0)  # red edges

    colors = dominant_colors(frame_rgb, k=3)
    y = 20
    for name, rgb in colors:
        cv2.rectangle(overlay, (10, y - 12), (30, y + 2), rgb, thickness=-1)
        cv2.putText(overlay, f"{name}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
    return overlay, colors
