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
MAX_STEP_DEG = 4.0    # rotation model: a larger turn between two frames is a failed fit


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


def _bearings(pts: np.ndarray, Kinv: np.ndarray) -> np.ndarray:
    p = np.column_stack([pts, np.ones(len(pts))]) @ Kinv.T
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R with a ~ R b (unit vectors as rows)."""
    U, _, Vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def rotation_ransac(a: np.ndarray, b: np.ndarray, thr_rad: float, iters: int = 200,
                    rng=None) -> tuple[np.ndarray | None, int]:
    """Robust rotation between two sets of bearings (a ~ R b): 2-point RANSAC + refit."""
    n = len(a)
    if n < MIN_INLIERS:
        return None, 0
    rng = rng or np.random.default_rng(0)
    best = None
    for _ in range(iters):
        i = rng.choice(n, 2, replace=False)
        R = _kabsch(a[i], b[i])
        inl = np.einsum("ij,ij->i", a, b @ R.T) > np.cos(thr_rad)
        if best is None or inl.sum() > best.sum():
            best = inl
    if best.sum() < MIN_INLIERS:
        return None, 0
    R = _kabsch(a[best], b[best])
    inl = np.einsum("ij,ij->i", a, b @ R.T) > np.cos(thr_rad)
    if inl.sum() < MIN_INLIERS:
        return None, 0
    return _kabsch(a[inl], b[inl]), int(inl.sum())


class RotationRegistrar(FrameRegistrar):
    """For a camera turning on the spot (a handheld pan): the motion to the reference frame
    as a pure rotation with the calibrated focal length, H = K R K^-1.

    Three parameters instead of eight: chained over a long pan the estimate cannot blow up
    in perspective, and frames that no longer overlap the reference are carried by the
    frame-to-frame rotations (drift corrected wherever the reference is seen again)."""

    def __init__(self, reference_bgr: np.ndarray, focal_px: float, n_features: int = 2000,
                 abs_every: int = 5):
        super().__init__(reference_bgr, n_features, abs_every)
        self.K = np.array([[focal_px, 0, self.w / 2], [0, focal_px, self.h / 2], [0, 0, 1.0]])
        self.Kinv = np.linalg.inv(self.K)
        self.Ks_inv = np.linalg.inv(self._S @ self.K)       # working-resolution intrinsics
        self.thr = 2.0 / (focal_px * self.scale)            # ~2 working px, in radians
        self.rng = np.random.default_rng(0)
        self.last = np.eye(3)

    def _absolute(self, g, mask):
        kp, des = self.orb.detectAndCompute(g, mask)
        if des is None or len(kp) < MIN_INLIERS:
            return None, 0
        pairs = self.matcher.knnMatch(des, self.ref_des, k=2)
        good = [pr[0] for pr in pairs if len(pr) == 2 and pr[0].distance < 0.8 * pr[1].distance]
        if len(good) < MIN_INLIERS:
            return None, 0
        cur = _bearings(np.float32([kp[m.queryIdx].pt for m in good]), self.Ks_inv)
        ref = _bearings(np.float32([self.ref_kp[m.trainIdx].pt for m in good]), self.Ks_inv)
        return rotation_ransac(ref, cur, self.thr, rng=self.rng)        # ref <- cur

    def _relative(self, g, mask):
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
        prev = _bearings(pts.reshape(-1, 2)[ok], self.Ks_inv)
        cur = _bearings(nxt.reshape(-1, 2)[ok], self.Ks_inv)
        R, _ = rotation_ransac(prev, cur, self.thr, rng=self.rng)       # prev <- cur
        if R is not None and np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])) > MAX_STEP_DEG:
            return None             # a hand cannot turn that fast between frames: a bad fit
        return R

    def register(self, frame_bgr: np.ndarray, boxes=None) -> tuple[np.ndarray, int]:
        g = self._gray(frame_bgr)
        mask = _box_mask(g.shape, boxes, self.scale)
        R = self._relative(g, mask)
        self.relative.append(R)
        A, n_in = (self._absolute(g, mask) if self.k % self.abs_every == 0 else (None, 0))
        self.k += 1
        self.abs_ok.append(A is not None)
        if A is None:
            A = self.last @ R if R is not None else self.last
        self.last = A
        self.absolute.append(A)
        self.inliers.append(n_in)
        self.prev_gray, self.prev_mask = g, mask
        return self.K @ A @ self.Kinv, n_in

    def finalize(self, fps: float) -> list[np.ndarray]:
        """Chained rotations, drift-corrected toward the reference matches (smoothly)."""
        from scipy.ndimage import gaussian_filter1d

        n = len(self.absolute)
        if n == 0:
            return []
        chain = [self.absolute[0]]
        for k in range(1, n):
            R = self.relative[k]
            if R is None:           # no usable flow: keep the previous step's motion
                R = self.relative[k - 1] if k > 1 and self.relative[k - 1] is not None                     else np.eye(3)
            chain.append(chain[-1] @ R)
        ok = np.array(self.abs_ok, bool)
        corr = np.zeros((n, 3))
        for k in np.flatnonzero(ok):            # rotation taking the chain onto the match
            corr[k] = cv2.Rodrigues(self.absolute[k] @ chain[k].T)[0].ravel()
        if ok.any() and (~ok).any():
            idx = np.arange(n)
            for j in range(3):
                corr[~ok, j] = np.interp(idx[~ok], idx[ok], corr[ok, j])
        corr = gaussian_filter1d(corr, sigma=max(1.0, FUSE_S * fps), axis=0, mode="nearest")
        out = []
        for k in range(n):
            R = cv2.Rodrigues(corr[k].reshape(3, 1))[0] @ chain[k]
            H = self.K @ R @ self.Kinv
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
    the clip: a quick check whether stabilisation is needed at all.

    A frame that cannot be matched to the reference at all counts as moved (inf): a handheld
    pan quickly leaves too little overlap to match, which is not a static camera."""
    from .video import read_frame, video_info

    info = video_info(video)
    reg = FrameRegistrar(read_frame(video, reference_frame), abs_every=1)
    worst = 0.0
    for f in np.linspace(0, max(info.n_frames - 2, 0), samples).astype(int):
        if int(f) == int(reference_frame):
            continue
        reg.prev_gray = None                      # independent samples: no flow chain
        H, n = reg._absolute(reg._gray(read_frame(video, int(f))), None)
        if H is None:
            return float("inf")
        worst = max(worst, drift_px(H, info.width, info.height))
    return worst


def register_run(video, raw, reference_frame: int, start: int, end: int, stride: int,
                 fps: float, out_path: Path, extra_boxes=None, focal_px: float | None = None,
                 log=print) -> dict:
    """Camera motion of an existing run, without detecting again: every processed frame is
    registered to the reference frame (the run's boxes masked out) -> registration.npz.
    With the calibrated focal length the motion is modelled as a rotation (handheld pans)."""
    import pandas as pd
    from tqdm import tqdm

    from .video import read_frame

    cols = ["frame", "x1", "y1", "x2", "y2"]
    allb = raw[cols] if extra_boxes is None else pd.concat([raw[cols], extra_boxes[cols]])
    boxes = {f: g[cols[1:]].to_numpy(float) for f, g in allb.groupby("frame")}
    ref = read_frame(video, reference_frame)
    reg = RotationRegistrar(ref, focal_px) if focal_px else FrameRegistrar(ref)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, Hs, inl = [], [], []
    for f in tqdm(range(start, end), desc="stabilize", unit="frame"):
        ok, img = cap.read()
        if not ok:
            break
        if (f - start) % stride:
            continue
        H, n = reg.register(img, boxes.get(f))
        frames.append(f)
        Hs.append(H)
        inl.append(n)
    cap.release()
    Hs = reg.finalize(fps / max(stride, 1)) if frames else []
    save_registration(out_path, frames, Hs, inl, reference_frame)
    h, w = reg.h, reg.w
    drift = [drift_px(H, w, h) for H in Hs] or [0.0]
    failed = int(sum(1 for k, ok in enumerate(reg.abs_ok) if k % reg.abs_every == 0 and not ok))
    log(f"stabilize: camera moved up to {max(drift):.1f} px from the reference frame; "
        f"{failed} frames could not be matched to it (flow chain used)")
    return {"reference_frame": reference_frame, "max_drift_px": round(max(drift), 1),
            "registration_failures": failed}
