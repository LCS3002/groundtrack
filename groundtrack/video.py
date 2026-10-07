"""Small OpenCV video helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def video_info(path: str | Path, fps_override: float | None = None) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {path}")
    fps = float(fps_override or cap.get(cv2.CAP_PROP_FPS) or 0)
    if not fps or fps > 1000:
        raise ValueError(f"{path}: video reports no usable frame rate; set `fps:` in the config.")
    info = VideoInfo(str(path), fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def read_frame(path: str | Path, frame: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame} from {path}")
    return img


def point_in_polygon(points: np.ndarray, polygon) -> np.ndarray:
    """Vectorised test of (N, 2) pixel points against a polygon [[x, y], ...]."""
    poly = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
                     for x, y in np.asarray(points, float).reshape(-1, 2)], bool)
