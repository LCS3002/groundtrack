"""Camera-motion compensation: align every frame to the calibration (reference) frame.

The ground homography is only valid for the frame it was clicked on. If the camera moves
(handheld, a wobbly railing, a knocked tripod), every frame is registered to that reference
frame with ORB features + RANSAC on the static background, and foot points are mapped into
reference-frame pixels before projection. Moving people/cars are outliers to RANSAC.

For a camera that mostly *rotates* (handheld, pan/tilt) the frame-to-frame mapping is an
exact homography whatever the scene depth; small translations of a few cm add errors of the
same few cm on the ground.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

WORK_WIDTH = 960      # features are matched on a downscaled grey frame
MIN_INLIERS = 40


class FrameRegistrar:
    def __init__(self, reference_bgr: np.ndarray, n_features: int = 3000):
        self.scale = min(1.0, WORK_WIDTH / reference_bgr.shape[1])
        self.orb = cv2.ORB_create(n_features, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        g = self._gray(reference_bgr)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(g, None)
        if self.ref_des is None or len(self.ref_kp) < 100:
            raise ValueError("Reference frame has too little texture to register against.")
        S = np.diag([self.scale, self.scale, 1.0])
        self._S, self._Sinv = S, np.linalg.inv(S)
        self.last = np.eye(3)

    def _gray(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.scale < 1:
            g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return g

    def register(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        """3x3 mapping pixels of this frame -> pixels of the reference frame, and #inliers.

        On failure returns the last good homography with inliers = 0.
        """
        kp, des = self.orb.detectAndCompute(self._gray(frame_bgr), None)
        if des is None or len(kp) < MIN_INLIERS:
            return self.last, 0
        pairs = self.matcher.knnMatch(des, self.ref_des, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < 0.8 * rest[0].distance]
        if len(good) < MIN_INLIERS:
            return self.last, 0
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0, maxIters=3000)
        n_in = int(mask.sum()) if mask is not None else 0
        if H is None or n_in < MIN_INLIERS:
            return self.last, 0
        H = self._Sinv @ H @ self._S          # back to full-resolution pixels
        self.last = H / H[2, 2]
        return self.last, n_in


def save_registration(path: Path, frames, Hs, inliers, reference_frame: int) -> None:
    np.savez_compressed(path, frames=np.asarray(frames, np.int64), H=np.asarray(Hs, np.float64),
                        inliers=np.asarray(inliers, np.int32), reference_frame=reference_frame)


def load_registration(path: Path):
    d = np.load(path)
    return dict(zip(d["frames"].tolist(), d["H"])), int(d["reference_frame"]), d["inliers"]


def apply_registration(frames: np.ndarray, uv: np.ndarray, reg: dict) -> np.ndarray:
    """Map per-row pixel coords into the reference frame (rows whose frame has no H -> NaN)."""
    out = np.full_like(uv, np.nan, dtype=float)
    for f in np.unique(frames):
        H = reg.get(int(f))
        if H is None:
            continue
        sel = frames == f
        p = np.column_stack([uv[sel], np.ones(sel.sum())]) @ H.T
        out[sel] = p[:, :2] / p[:, 2:3]
    return out


def drift_px(H: np.ndarray, w: int, h: int) -> float:
    """Largest displacement of the frame corners/centre under H (how far the camera moved)."""
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]]).reshape(-1, 1, 2)
    return float(np.linalg.norm(cv2.perspectiveTransform(c, H) - c, axis=2).max())
