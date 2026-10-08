"""Image -> ground-plane homography: fitting, error reporting, (de)serialisation.

Internally the homography maps pixels to *local* metres (world minus a rounded origin),
because British National Grid coordinates (~500 000, ~180 000) would otherwise waste
most of the floating point precision of the 3x3 matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

WARN_RMSE_M = 0.5
SPREAD_WARN = 0.3   # sqrt(smaller / larger extent) of the clicked map points


def point_spread(world: np.ndarray) -> float:
    """0 = all points on one line, 1 = spread equally in both directions."""
    w = np.asarray(world, float).reshape(-1, 2)
    if len(w) < 3:
        return 0.0
    ev = np.linalg.eigvalsh(np.cov((w - w.mean(axis=0)).T))
    return float(np.sqrt(max(ev[0], 0) / max(ev[1], 1e-12)))


@dataclass
class Homography:
    H: np.ndarray                         # 3x3, pixel -> local metres
    origin: tuple[float, float]           # world coords of local (0, 0)
    image_size: tuple[int, int] | None = None   # (width, height) of the calibration frame
    undistorted: bool = False             # pixel coords are lens-undistorted
    crs: str = "EPSG:27700"
    points: list[dict] = field(default_factory=list)
    rmse_m: float | None = None
    max_error_m: float | None = None
    loo_rmse_m: float | None = None
    w_sign: float = 1.0                   # sign of the projective w for ground points
    reference_frame: int = 0              # video frame the points were clicked on

    # -- projection -----------------------------------------------------------
    def to_world(self, px: np.ndarray) -> np.ndarray:
        """Pixel (N, 2) -> world (N, 2). Points above the horizon become NaN."""
        px = np.asarray(px, float).reshape(-1, 2)
        ph = np.column_stack([px, np.ones(len(px))]) @ self.H.T
        w = ph[:, 2]
        valid = (np.sign(w) == np.sign(self.w_sign)) & (np.abs(w) > 1e-12)
        out = np.full((len(px), 2), np.nan)
        out[valid] = ph[valid, :2] / w[valid, None] + np.asarray(self.origin)
        return out

    def to_pixel(self, world: np.ndarray) -> np.ndarray:
        world = np.asarray(world, float).reshape(-1, 2) - np.asarray(self.origin)
        Hinv = np.linalg.inv(self.H)
        ph = np.column_stack([world, np.ones(len(world))]) @ Hinv.T
        return ph[:, :2] / ph[:, 2:3]

    def camera(self) -> dict | None:
        """Estimate the camera from the homography (cached).

        Assumes square pixels, no skew and the principal point at the image centre (true
        enough for phone video). Returns focal length (px), camera ground position (E, N) and
        height above the ground plane (m), or None if the geometry is degenerate (e.g. the
        camera looks straight down, where the focal length is unobservable).
        """
        if not hasattr(self, "_camera"):
            self._camera = estimate_camera(self.H, self.image_size, self.origin)
        return self._camera

    def world_H(self) -> np.ndarray:
        """3x3 mapping pixels straight to world coordinates (for reference / other tools)."""
        T = np.array([[1, 0, self.origin[0]], [0, 1, self.origin[1]], [0, 0, 1]], float)
        return T @ self.H

    # -- io ---------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        d = {
            "description": "pixel -> world homography. world = origin + dehomog(H_local @ [u, v, 1])",
            "crs": self.crs,
            "origin": list(self.origin),
            "H_local": self.H.tolist(),
            "H_world": self.world_H().tolist(),
            "w_sign": self.w_sign,
            "image_size": list(self.image_size) if self.image_size else None,
            "reference_frame": self.reference_frame,
            "undistorted": self.undistorted,
            "rmse_m": self.rmse_m,
            "max_error_m": self.max_error_m,
            "loo_rmse_m": self.loo_rmse_m,
            "camera_estimate": self.camera(),
            "camera_params": getattr(self, "camera_params", None),
            "points": self.points,
        }
        Path(path).write_text(json.dumps(d, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Homography":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        h_obj = cls(
            H=np.array(d["H_local"], float),
            origin=tuple(d["origin"]),
            image_size=tuple(d["image_size"]) if d.get("image_size") else None,
            undistorted=bool(d.get("undistorted", False)),
            crs=d.get("crs", "EPSG:27700"),
            points=d.get("points", []),
            rmse_m=d.get("rmse_m"),
            max_error_m=d.get("max_error_m"),
            loo_rmse_m=d.get("loo_rmse_m"),
            w_sign=float(d.get("w_sign", 1.0)),
            reference_frame=int(d.get("reference_frame", 0)),
        )
        if d.get("camera_params"):
            h_obj.camera_params = d["camera_params"]
        return h_obj


def estimate_camera(H: np.ndarray, image_size, origin=(0.0, 0.0)) -> dict | None:
    """Recover focal length and camera centre from a pixel -> ground homography.

    G = H^-1 maps ground (x, y, 1) to pixels and equals lambda * K [r1 r2 t]. With
    K = [[f, 0, cx], [0, f, cy], [0, 0, 1]], the orthonormality of r1, r2 gives f; then the
    camera centre is C = -R^T t.
    """
    if not image_size:
        return None
    cx, cy = image_size[0] / 2, image_size[1] / 2
    G = np.linalg.inv(H)
    A = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]]) @ G
    # scale so that pixel-unit and unit-less rows are comparable (conditioning)
    A = np.diag([1 / cx, 1 / cx, 1.0]) @ A
    A /= np.linalg.norm(A[:, :2])
    a1, a2 = A[:, 0], A[:, 1]
    # with w = (cx / f)^2:  r1.r2 = 0     ->  p1 * w + q1 = 0
    #                       |r1| = |r2|   ->  p2 * w + q2 = 0
    # solved jointly in least squares; a constraint that is degenerate for this view
    # (e.g. camera looking exactly along grid north) has p ~ q ~ 0 and drops out
    p = np.array([a1[0] * a2[0] + a1[1] * a2[1], a1[0] ** 2 + a1[1] ** 2 - a2[0] ** 2 - a2[1] ** 2])
    q = np.array([a1[2] * a2[2], a1[2] ** 2 - a2[2] ** 2])
    if p @ p < 1e-18:
        return None
    w = -(p @ q) / (p @ p)
    if not (w > 0 and np.isfinite(w)):
        return None
    f = float(cx / np.sqrt(w))
    A = np.diag([cx, cx, 1.0]) @ A
    Kinv = np.linalg.inv(np.array([[f, 0, 0], [0, f, 0], [0, 0, 1.0]]))
    m1, m2, m3 = Kinv @ A[:, 0], Kinv @ A[:, 1], Kinv @ A[:, 2]
    lam = 2.0 / (np.linalg.norm(m1) + np.linalg.norm(m2))
    r1, r2, t = m1 * lam, m2 * lam, m3 * lam
    r1 /= np.linalg.norm(r1)
    r2 = r2 - r1 * (r1 @ r2)
    r2 /= np.linalg.norm(r2)
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    C = -R.T @ t
    height = abs(float(C[2]))
    if not (0.1 < height < 10_000) or f < 50:
        return None
    return {
        "focal_px": round(f, 1),
        "hfov_deg": round(float(np.degrees(2 * np.arctan(image_size[0] / (2 * f)))), 1),
        "E": float(C[0] + origin[0]), "N": float(C[1] + origin[1]),
        "height_m": round(height, 2),
        "note": "estimated from the homography assuming a centred principal point; a sanity "
                "check, not a survey",
    }


def _w(H: np.ndarray, px: np.ndarray) -> np.ndarray:
    return (np.column_stack([px, np.ones(len(px))]) @ H.T)[:, 2]


def fit_homography(px: np.ndarray, world: np.ndarray, ransac_thresh_m: float = 1.0,
                   image_size=None, undistorted=False, crs="EPSG:27700") -> Homography:
    """Fit pixel -> world homography with RANSAC and report per-point errors in metres.

    px, world: (N, 2), N >= 4 (6-8 well spread points recommended).
    """
    px = np.asarray(px, float).reshape(-1, 2)
    world = np.asarray(world, float).reshape(-1, 2)
    n = len(px)
    if n < 4 or len(world) != n:
        raise ValueError(f"Need >= 4 matching point pairs, got {n} pixel / {len(world)} world")
    origin = tuple(np.round(world.mean(axis=0)))
    local = world - np.asarray(origin)

    if n >= 5:
        H, mask = cv2.findHomography(px, local, cv2.RANSAC, ransac_thresh_m, maxIters=5000,
                                     confidence=0.999)
        if H is None or mask.sum() < 4:
            H, mask = cv2.findHomography(px, local, 0)
            mask = np.ones((n, 1), np.uint8)
    else:
        H, mask = cv2.findHomography(px, local, 0)
        mask = np.ones((n, 1), np.uint8)
    if H is None:
        raise ValueError("Homography fit failed: points may be collinear or duplicated.")
    inlier = mask.ravel().astype(bool)
    # Least-squares refit on the inliers only (RANSAC's model is the minimal-set one).
    if inlier.sum() >= 4:
        H2, _ = cv2.findHomography(px[inlier], local[inlier], 0)
        if H2 is not None:
            H = H2
    H = H / H[2, 2]
    w_sign = float(np.sign(np.median(_w(H, px[inlier])))) or 1.0

    h = Homography(H=H, origin=origin, image_size=tuple(image_size) if image_size else None,
                   undistorted=undistorted, crs=crs, w_sign=w_sign)
    proj = h.to_world(px)
    err = np.linalg.norm(proj - world, axis=1)
    # a point the fit puts behind the camera has no finite position: count it as a huge error
    err = np.where(np.isfinite(err), err, np.inf)
    h.spread = point_spread(world)

    # Leave-one-out error: refit without each inlier and measure the error at that point.
    # With only 6-8 points the in-sample residual flatters the fit; LOO is the honest number.
    loo = np.full(n, np.nan)
    idx = np.flatnonzero(inlier)
    if len(idx) >= 5:
        for i in idx:
            keep = idx[idx != i]
            Hk, _ = cv2.findHomography(px[keep], local[keep], 0)
            if Hk is None:
                continue
            hk = Homography(H=Hk / Hk[2, 2], origin=origin, w_sign=w_sign)
            loo[i] = np.linalg.norm(hk.to_world(px[i:i + 1])[0] - world[i])

    h.points = [
        {
            "id": i + 1, "u": float(px[i, 0]), "v": float(px[i, 1]),
            "E": float(world[i, 0]), "N": float(world[i, 1]),
            "error_m": float(err[i]) if np.isfinite(err[i]) else None, "loo_error_m": None if np.isnan(loo[i]) else float(loo[i]),
            "inlier": bool(inlier[i]),
        }
        for i in range(n)
    ]
    h.rmse_m = float(np.sqrt(np.mean(err[inlier] ** 2)))
    h.max_error_m = float(err[inlier].max())
    if not np.isfinite(h.rmse_m):
        h.rmse_m = h.max_error_m = float("inf")
    h.loo_rmse_m = float(np.sqrt(np.nanmean(loo[idx] ** 2))) if np.isfinite(loo).any() else None
    return h


def report(h: Homography, warn_m: float = WARN_RMSE_M) -> str:
    lines = ["  #        u        v            E            N   err (m)   LOO (m)  inlier"]
    for p in h.points:
        loo = f"{p['loo_error_m']:9.3f}" if p.get("loo_error_m") is not None else "        -"
        lines.append(
            f"{p['id']:3d} {p['u']:8.1f} {p['v']:8.1f} {p['E']:12.2f} {p['N']:12.2f} "
            f"{(p['error_m'] if p['error_m'] is not None else float('inf')):9.3f} {loo} "
            f"{'yes' if p['inlier'] else 'NO':>7}"
        )
    lines.append(f"RMSE (inliers): {h.rmse_m:.3f} m   max: {h.max_error_m:.3f} m"
                 + (f"   leave-one-out RMSE: {h.loo_rmse_m:.3f} m" if h.loo_rmse_m else ""))
    outliers = [p["id"] for p in h.points if not p["inlier"]]
    if outliers:
        lines.append(f"WARNING: points {outliers} rejected as outliers by RANSAC. Re-check those clicks.")
    if h.rmse_m > warn_m:
        lines.append(f"WARNING: RMSE {h.rmse_m:.2f} m > {warn_m} m. Positions will be unreliable; "
                     "re-pick points (spread them across the whole ground area, use sharp "
                     "features such as kerb corners, road markings or drain covers).")
    elif h.loo_rmse_m and h.loo_rmse_m > 2 * warn_m:
        lines.append(f"WARNING: leave-one-out RMSE {h.loo_rmse_m:.2f} m is high. The fit depends "
                     "heavily on individual points; add more well-spread points.")
    behind = [p["id"] for p in h.points if p["error_m"] is None]
    if behind:
        lines.append(f"WARNING: points {behind} land behind the camera with this fit. The points "
                     "do not pin down the ground plane: see the spread warning below, or "
                     "re-check those clicks.")
    sp = getattr(h, "spread", None)
    if sp is not None and sp < SPREAD_WARN:
        lines.append(f"WARNING: the points lie almost on one line (spread {sp:.2f}, want > "
                     f"{SPREAD_WARN}). Add points well to the left and right of that line, near "
                     "and far, or the projection away from it is guesswork.")
    if len(h.points) < 6:
        lines.append("NOTE: fewer than 6 points; 6-8 well-spread points are recommended.")
    cam = h.camera()
    if cam:
        lines.append(f"Camera estimate (sanity check): {cam['height_m']:.1f} m above the ground at "
                     f"E {cam['E']:.1f}, N {cam['N']:.1f}; focal {cam['focal_px']:.0f} px "
                     f"(horizontal FOV {cam['hfov_deg']:.0f} deg). If the height or position "
                     "is clearly wrong, the calibration is too.")
    return "\n".join(lines)
