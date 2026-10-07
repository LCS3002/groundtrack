"""Optional lens undistortion from a checkerboard calibration."""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class Lens:
    K: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int]   # (width, height)
    rms_px: float | None = None

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """Raw pixel coords (N, 2) -> undistorted pixel coords in the same camera matrix."""
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return pts.reshape(0, 2)
        out = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        return out.reshape(-1, 2)

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        return cv2.undistort(img, self.K, self.dist)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "K": self.K.tolist(), "dist": self.dist.ravel().tolist(),
            "image_size": list(self.image_size), "rms_px": self.rms_px,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Lens":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(d["K"], float), np.array(d["dist"], float),
                   tuple(d["image_size"]), d.get("rms_px"))


def _frames_from_source(source: str, every_s: float = 1.0):
    p = Path(source)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT)
    elif p.suffix.lower() in IMAGE_EXT or any(ch in source for ch in "*?"):
        files = sorted(Path(f) for f in glob.glob(source))
    else:
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(round(fps * every_s)))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield f"frame{i}", frame
            i += 1
        cap.release()
        return
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            yield f.name, img


def calibrate_lens(source: str, pattern: tuple[int, int], square_mm: float = 25.0,
                   every_s: float = 1.0, log=print) -> Lens:
    """Calibrate from checkerboard images (folder / glob) or a video of the board.

    pattern: inner corners (columns, rows), e.g. (9, 6) for a 10x7-square board.
    Film/photograph the board with the SAME phone lens, zoom and resolution as the footage.
    """
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    obj_pts, img_pts, size = [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for name, img in _frames_from_source(source, every_s):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            log(f"  skip {name}: different resolution")
            continue
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        obj_pts.append(objp)
        img_pts.append(corners)
    log(f"checkerboard found in {len(img_pts)} images")
    if len(img_pts) < 8:
        raise RuntimeError("Need the checkerboard detected in >= 8 views (vary angle & position).")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    log(f"lens calibration RMS reprojection error: {rms:.3f} px (good: < 0.5 px)")
    return Lens(K=K, dist=dist.ravel(), image_size=tuple(size), rms_px=float(rms))
