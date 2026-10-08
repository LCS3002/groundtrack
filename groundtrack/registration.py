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
FUSE_S = 0.5          # time constant: below it trust frame-to-frame flow, above it the reference


def _box_mask(shape, boxes, scale, pad=0.15):
    """255 everywhere except inside (padded) detection boxes: moving people are not scenery."""
    m = np.full(shape, 255, np.uint8)
    if boxes is None:
        return m
    for x1, y1, x2, y2 in np.asarray(boxes, float).reshape(-1, 4):
        w, h = (x2 - x1) * pad, (y2 - y1) * pad
        cv2.rectangle(m, (int((x1 - w) * scale), int((y1 - h) * scale)),
                      (int((x2 + w) * scale), int((y2 + h) * scale)), 0, -1)
    return m


class FrameRegistrar:
    """Per-frame camera-motion estimate, fused from two sources:

    * absolute: ORB features matched against the reference frame. No drift, but each frame
      is estimated on its own, so it jitters by a few pixels;
    * relative: sparse optical flow from the previous frame. Very smooth and precise, but
      drifts when chained over many frames.

    `finalize()` combines them (complementary filter on the frame-corner trajectories): the
    chained flow gives the fast motion, the reference matches correct the slow drift.
    """

    def __init__(self, reference_bgr: np.ndarray, n_features: int = 2000, abs_every: int = 5):
        self.scale = min(1.0, WORK_WIDTH / reference_bgr.shape[1])
        self.h, self.w = reference_bgr.shape[:2]
        self.orb = cv2.ORB_create(n_features, fastThreshold=10)
        # approximate (LSH) matching: ~10x faster than brute force for binary ORB descriptors
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), dict(checks=64))
        self.abs_every = max(1, int(abs_every))
        self.k = 0
        g = self._gray(reference_bgr)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(g, None)
        if self.ref_des is None or len(self.ref_kp) < 100:
            raise ValueError("Reference frame has too little texture to register against.")
        S = np.diag([self.scale, self.scale, 1.0])
        self._S, self._Sinv = S, np.linalg.inv(S)
        self.last = np.eye(3)
        self.prev_gray = None
        self.prev_mask = None
        self.absolute: list[np.ndarray] = []
        self.abs_ok: list[bool] = []
        self.relative: list[np.ndarray | None] = []
        self.inliers: list[int] = []

    def _gray(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.scale < 1:
            g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return g

    def _absolute(self, g, mask):
        kp, des = self.orb.detectAndCompute(g, mask)
        if des is None or len(kp) < MIN_INLIERS:
            return None, 0
        pairs = self.matcher.knnMatch(des, self.ref_des, k=2)
        good = [pr[0] for pr in pairs if len(pr) == 2 and pr[0].distance < 0.8 * pr[1].distance]
        if len(good) < MIN_INLIERS:
            return None, 0
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 1.5, maxIters=3000)
        n_in = int(inl.sum()) if inl is not None else 0
        if H is None or n_in < MIN_INLIERS:
            return None, 0
        H = self._Sinv @ H @ self._S
        return H / H[2, 2], n_in

    def _relative(self, g, mask):
        """Homography mapping this frame -> previous frame, from sparse optical flow."""
        if self.prev_gray is None:
            return None
        pts = cv2.goodFeaturesToTrack(self.prev_gray, 800, 0.01, 8, mask=self.prev_mask)
        if pts is None or len(pts) < MIN_INLIERS:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, g, pts, None,
                                              winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < MIN_INLIERS:
            return None
        H, inl = cv2.findHomography(nxt[ok], pts[ok], cv2.RANSAC, 1.0, maxIters=2000)
        if H is None or inl.sum() < MIN_INLIERS:
            return None
        H = self._Sinv @ H @ self._S
        return H / H[2, 2]

    def register(self, frame_bgr: np.ndarray, boxes=None) -> tuple[np.ndarray, int]:
        """Record this frame's motion; returns (current frame -> reference estimate, inliers).

        boxes: detections (x1, y1, x2, y2) in frame pixels, masked out of the matching.
        """
        g = self._gray(frame_bgr)
        mask = _box_mask(g.shape, boxes, self.scale)
        R = self._relative(g, mask)
        self.relative.append(R)
        # the (slower) match against the reference only every `abs_every` frames: the flow
        # chain covers the frames in between and finalize() interpolates the drift correction
        A, n_in = (self._absolute(g, mask) if self.k % self.abs_every == 0 else (None, 0))
        self.k += 1
        self.abs_ok.append(A is not None)
        if A is None:
            A = self.last @ R if R is not None else self.last
        self.last = A
        self.absolute.append(A)
        self.inliers.append(n_in)
        self.prev_gray, self.prev_mask = g, mask
        return A, n_in

    def finalize(self, fps: float) -> list[np.ndarray]:
        """Fused, smooth frame -> reference homographies (see class docstring)."""
        from scipy.ndimage import gaussian_filter1d

        n = len(self.absolute)
        if n == 0:
            return []
        c = np.float32([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]]).reshape(-1, 1, 2)

        def corners(H):
            return cv2.perspectiveTransform(c, H).reshape(-1)

        P_abs = np.array([corners(A) for A in self.absolute])
        chain = [self.absolute[0]]
        for k in range(1, n):
            R = self.relative[k]
            if R is None:  # no flow: fall back to the step implied by the reference matches
                R = np.linalg.inv(self.absolute[k - 1]) @ self.absolute[k]
            chain.append(chain[-1] @ R)
        P_chain = np.array([corners(C) for C in chain])
        ok = np.array(self.abs_ok, bool)
        diff = P_abs - P_chain
        if (~ok).any() and ok.any():  # frames without a reference match: interpolate the drift
            idx = np.arange(n)
            for j in range(diff.shape[1]):
                diff[~ok, j] = np.interp(idx[~ok], idx[ok], diff[ok, j])
        drift = gaussian_filter1d(diff, sigma=max(1.0, FUSE_S * fps), axis=0, mode="nearest")
        fused = P_chain + drift
        out = []
        for k in range(n):
            H = cv2.getPerspectiveTransform(c.reshape(-1, 2), fused[k].reshape(-1, 2)
                                            .astype(np.float32))
            out.append(H / H[2, 2])
        return out


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


def measure_camera_motion(video, reference_frame: int = 0, samples: int = 8) -> float:
    """Largest displacement (px) of the frame corners from the reference frame, sampled over
    the clip: a quick check whether stabilisation is needed at all."""
    from .video import read_frame, video_info

    info = video_info(video)
    reg = FrameRegistrar(read_frame(video, reference_frame), abs_every=1)
    worst = 0.0
    for f in np.linspace(0, max(info.n_frames - 2, 0), samples).astype(int):
        reg.prev_gray = None                      # independent samples: no flow chain
        H, n = reg._absolute(reg._gray(read_frame(video, int(f))), None)
        if H is not None:
            worst = max(worst, drift_px(H, info.width, info.height))
    return worst
