"""
Lightweight AI-assisted color and edge analysis using k-means and Canny (RGB).
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from ultralytics import YOLO

    _YOLO_IMPORT_ERROR = None
except Exception as exc:  # optional dependency
    YOLO = None  # type: ignore
    _YOLO_IMPORT_ERROR = exc

ColorName = Tuple[str, Tuple[int, int, int]]  # (name, RGB)
YoloDetection = Dict[str, object]


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


class YoloDetector:
    """Lazy-loaded YOLOv8 detector wrapper."""

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 5,
        imgsz: int = 320,
    ) -> None:
        self.model_path = str(model_path)
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.imgsz = imgsz
        self._model = None
        self.last_error: Optional[str] = None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if YOLO is None:
            self.last_error = f"ultralytics not available: {_YOLO_IMPORT_ERROR}"
            return False
        path = Path(self.model_path)
        if not path.exists():
            self.last_error = f"YOLO model not found: {path}"
            return False
        try:
            self._model = YOLO(str(path))
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = f"YOLO load failed: {exc}"
            return False

    def detect(self, frame_bgr: np.ndarray, class_name: Optional[str] = None) -> List[YoloDetection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if not self._load():
            return []
        try:
            results = self._model.predict(
                source=frame_bgr,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                imgsz=self.imgsz,
                device="cpu",
                verbose=False,
            )
        except Exception as exc:
            self.last_error = f"YOLO predict failed: {exc}"
            return []
        self.last_error = None
        if not results:
            return []
        res = results[0]
        names = getattr(res, "names", {}) or {}
        if isinstance(names, (list, tuple)):
            names = {idx: name for idx, name in enumerate(names)}
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        wanted = class_name.lower() if class_name else None

        detections: List[YoloDetection] = []
        try:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
        except Exception:
            xyxy = None
            confs = None
            classes = None

        if xyxy is not None and confs is not None and classes is not None:
            for idx in range(len(xyxy)):
                class_id = int(classes[idx])
                name = names.get(class_id, str(class_id))
                if wanted and str(name).lower() != wanted:
                    continue
                detections.append(
                    {
                        "xyxy": xyxy[idx].tolist(),
                        "conf": float(confs[idx]),
                        "class_id": class_id,
                        "class_name": name,
                    }
                )
            return detections

        # Fallback: iterate box objects
        for box in boxes:
            try:
                coords = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                name = names.get(class_id, str(class_id))
                conf = float(box.conf[0])
            except Exception:
                continue
            if wanted and str(name).lower() != wanted:
                continue
            detections.append(
                {
                    "xyxy": coords,
                    "conf": conf,
                    "class_id": class_id,
                    "class_name": name,
                }
            )
        return detections


def yolo_best_target(
    detections: Sequence[YoloDetection],
    frame_shape: Tuple[int, int],
    class_name: Optional[str] = None,
) -> Optional[Tuple[int, int, int, float, float, str]]:
    if not detections:
        return None
    wanted = class_name.lower() if class_name else None
    height, width = frame_shape
    frame_area = float(max(1, width * height))
    best = None
    best_area = -1.0
    best_conf = -1.0
    for det in detections:
        name = str(det.get("class_name", ""))
        if wanted and name.lower() != wanted:
            continue
        coords = det.get("xyxy")
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in coords]
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        area = box_w * box_h
        conf = float(det.get("conf", 0.0))
        if area > best_area or (area == best_area and conf > best_conf):
            best = (x1, y1, x2, y2, area, name)
            best_area = area
            best_conf = conf
    if best is None:
        return None
    x1, y1, x2, y2, area, name = best
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    mask_ratio = float(area / frame_area)
    depth_norm = 1.0 / max(1.0, min(area, frame_area))
    depth_norm = max(0.0, min(1.0, depth_norm * 1000.0))
    return (cx, cy, int(area), depth_norm, mask_ratio, name)
