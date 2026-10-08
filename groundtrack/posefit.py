"""Fit a physical camera (known rough position) to clicked point pairs.

A free homography has 8 unknowns and is easily thrown by a few imperfect clicks. A real
pinhole camera whose position you roughly know (building, floor) has only yaw, tilt, roll
and focal length left, plus a small position / height correction. That is far better
conditioned, and points that do not fit (rooftops, mis-clicks) stand out clearly.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .demo import SyntheticCamera
from .homography import Homography


def pose_homography(E, N, height, yaw, pitch, f_px, size, roll=0.0, origin=None) -> Homography:
    """Pixel -> world homography of a pinhole camera standing at (E, N), `height` m up."""
    origin = np.round([E, N]) if origin is None else np.asarray(origin, float)
    cam = SyntheticCamera(width=size[0], height=size[1], f=f_px, height_m=height,
                          pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll,
                          cam_local=(E - origin[0], N - origin[1]), origin=origin)
    T = np.array([[1, 0, origin[0]], [0, 1, origin[1]], [0, 0, 1.0]])
    H = np.linalg.inv(cam.ground_homography() @ T)          # pixel -> local metres
    H /= H[2, 2]
    w = (H @ np.array([size[0] / 2, size[1] * 0.75, 1.0]))[2]  # a point on the ground
    return Homography(H=H, origin=tuple(origin), image_size=tuple(size),
                      w_sign=float(np.sign(w)) or 1.0)


def _project(params, world, size, origin):
    yaw, pitch, roll, logf, dE, dN, height = params
    cam = SyntheticCamera(width=size[0], height=size[1], f=float(np.exp(logf)),
                          height_m=height, pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll,
                          cam_local=(dE, dN), origin=origin)
    return cam.ground_to_pixel(world)


def fit_camera(px, world, size, E, N, height, yaw0, pitch0, f0, pos_tol_m=60.0,
               height_tol_m=20.0, loss_scale_px=6.0):
    """Robust fit. Returns (homography, params dict, per-point pixel residuals)."""
    px = np.asarray(px, float)
    world = np.asarray(world, float)
    origin = np.array([E, N], float)
    x0 = np.array([yaw0, pitch0, 0.0, np.log(f0), 0.0, 0.0, height])
    lo = [yaw0 - 40, 0.5, -10, np.log(300), -pos_tol_m, -pos_tol_m, height - height_tol_m]
    hi = [yaw0 + 40, 60, 10, np.log(30000), pos_tol_m, pos_tol_m, height + height_tol_m]

    # weak prior: the camera is near the stated position / height (1 sigma = half the
    # tolerance, weighted like a few pixels) - decisive with few clicks, negligible with many
    sig_pos, sig_h = max(pos_tol_m / 2, 0.1), max(height_tol_m / 2, 0.1)

    def resid(p):
        r = (_project(p, world, size, origin) - px).ravel()
        prior = loss_scale_px * np.array([p[4] / sig_pos, p[5] / sig_pos, (p[6] - height) / sig_h])
        return np.concatenate([r, prior])

    best = None
    for dy in (-15, 0, 15):          # a few starts: the yaw guess may be off
        for fp in (0.6, 1.0, 1.6):
            s = x0.copy()
            s[0] += dy
            s[3] = np.log(f0 * fp)
            s = np.clip(s, lo, hi)
            sol = least_squares(resid, s, bounds=(lo, hi), loss="soft_l1",
                                f_scale=loss_scale_px, max_nfev=4000)
            if best is None or sol.cost < best.cost:
                best = sol
    p = best.x
    res = np.linalg.norm((_project(p, world, size, origin) - px), axis=1)
    params = {"yaw_deg": p[0], "tilt_deg": p[1], "roll_deg": p[2], "focal_px": float(np.exp(p[3])),
              "E": E + p[4], "N": N + p[5], "height_m": p[6],
              "hfov_deg": float(np.degrees(2 * np.arctan(size[0] / 2 / np.exp(p[3]))))}
    h = pose_homography(params["E"], params["N"], params["height_m"], params["yaw_deg"],
                        params["tilt_deg"], params["focal_px"], size, roll=params["roll_deg"])
    return h, params, res


def _project_params(p: dict, world, size):
    cam = SyntheticCamera(width=size[0], height=size[1], f=p["focal_px"], height_m=p["height_m"],
                          pitch_deg=p["tilt_deg"], yaw_deg=p["yaw_deg"], roll_deg=p["roll_deg"],
                          cam_local=(0.0, 0.0), origin=np.array([p["E"], p["N"]]))
    return cam.ground_to_pixel(world)


def camera_calibration(px, world, size, prior: dict, outlier_m: float = 3.0):
    """Homography from clicked pairs + a known camera position (``camera_position`` in the
    site config: E, N, height_m, optional position_tol_m, height_tol_m, hfov_deg).

    Fits the physical camera (yaw, tilt, roll, zoom, small position/height correction) and
    returns a Homography carrying the usual per-point report. Needs >= 3 pairs.
    """
    px = np.asarray(px, float).reshape(-1, 2)
    world = np.asarray(world, float).reshape(-1, 2)
    if len(px) < 3:
        raise ValueError("Need at least 3 point pairs with a known camera position")
    E, N, H = float(prior["E"]), float(prior["N"]), float(prior["height_m"])
    d = world - [E, N]
    yaw0 = float(np.degrees(np.arctan2(d[:, 0].mean(), d[:, 1].mean())))
    dist = float(np.median(np.linalg.norm(d, axis=1)))
    tilt0 = float(np.degrees(np.arctan2(H, max(dist, 1.0))))
    hfov = float(prior.get("hfov_deg", 65.0))
    f0 = size[0] / 2 / np.tan(np.radians(hfov / 2))
    h, p, res = fit_camera(px, world, size, E, N, H, yaw0, tilt0, f0,
                           pos_tol_m=float(prior.get("position_tol_m", 15.0)),
                           height_tol_m=float(prior.get("height_tol_m", max(1.0, 0.1 * H))))
    def _errors(hh):
        e = np.linalg.norm(hh.to_world(px) - world, axis=1)
        return np.where(np.isfinite(e), e, np.inf)

    err = _errors(h)
    fin = np.isfinite(err)
    inl = err <= max(outlier_m, 3 * np.median(err[fin]) if fin.any() else outlier_m)
    if 3 <= inl.sum() < len(px):  # refit on the clicks that agree (drops rooftops, mis-clicks)
        h, p, _ = fit_camera(px[inl], world[inl], size, E, N, H, p["yaw_deg"], p["tilt_deg"],
                             p["focal_px"], pos_tol_m=float(prior.get("position_tol_m", 15.0)),
                             height_tol_m=float(prior.get("height_tol_m", max(1.0, 0.1 * H))))
        res = np.linalg.norm(_project_params(p, world, size) - px, axis=1)
        err = _errors(h)
    h.points = [{"id": i + 1, "u": float(u), "v": float(v), "E": float(e), "N": float(n),
                 "error_m": float(er) if np.isfinite(er) else None, "loo_error_m": None,
                 "inlier": bool(ok), "pixel_error": float(r)}
                for i, ((u, v), (e, n), er, ok, r) in enumerate(zip(px, world, err, inl, res))]
    good = err[inl & np.isfinite(err)]
    h.rmse_m = float(np.sqrt(np.mean(good ** 2))) if len(good) else float("inf")
    h.max_error_m = float(good.max()) if len(good) else float("inf")
    h.spread = 1.0  # a physical camera cannot degenerate like a free homography
    h.camera_params = {k: float(v) for k, v in p.items()}
    return h
