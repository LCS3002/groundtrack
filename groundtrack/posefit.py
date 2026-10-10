"""Fit a physical camera (known rough position) to clicked point pairs.

A free homography has 8 unknowns and is easily thrown by a few imperfect clicks. A real
pinhole camera whose position you roughly know (building, floor) has only yaw, tilt, roll
and focal length left, plus a small position / height correction. That is far better
conditioned, and points that do not fit (rooftops, mis-clicks) stand out clearly.

With a terrain model (terrain.py) every clicked point sits at its real height (a terrace,
a step), so clicks on different levels agree instead of pulling the camera apart.
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


def _project(params, world, size, origin, z=None):
    yaw, pitch, roll, logf, dE, dN, height = params
    cam = SyntheticCamera(width=size[0], height=size[1], f=float(np.exp(logf)),
                          height_m=height, pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll,
                          cam_local=(dE, dN), origin=origin)
    if z is None:
        return cam.ground_to_pixel(world)
    return cam.project(np.column_stack([np.asarray(world, float).reshape(-1, 2), z]))


def fit_camera(px, world, size, E, N, height, yaw0, pitch0, f0, pos_tol_m=60.0,
               height_tol_m=20.0, loss_scale_px=6.0, world_z=None):
    """Robust fit. Returns (homography, params dict, per-point pixel residuals).

    world_z: height of each point above the datum (terrain); None = all on the plane."""
    px = np.asarray(px, float)
    world = np.asarray(world, float)
    z = None if world_z is None else np.asarray(world_z, float)
    origin = np.array([E, N], float)
    x0 = np.array([yaw0, pitch0, 0.0, np.log(f0), 0.0, 0.0, height])
    lo = [yaw0 - 40, -10, -10, np.log(300), -pos_tol_m, -pos_tol_m, height - height_tol_m]
    hi = [yaw0 + 40, 60, 10, np.log(30000), pos_tol_m, pos_tol_m, height + height_tol_m]

    # weak prior: the camera is near the stated position / height (1 sigma = half the
    # tolerance, weighted like a few pixels) - decisive with few clicks, negligible with many
    sig_pos, sig_h = max(pos_tol_m / 2, 0.1), max(height_tol_m / 2, 0.1)

    def resid(p):
        r = (_project(p, world, size, origin, z) - px).ravel()
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
    res = np.linalg.norm((_project(p, world, size, origin, z) - px), axis=1)
    params = {"yaw_deg": p[0], "tilt_deg": p[1], "roll_deg": p[2], "focal_px": float(np.exp(p[3])),
              "E": E + p[4], "N": N + p[5], "height_m": p[6],
              "hfov_deg": float(np.degrees(2 * np.arctan(size[0] / 2 / np.exp(p[3]))))}
    h = pose_homography(params["E"], params["N"], params["height_m"], params["yaw_deg"],
                        params["tilt_deg"], params["focal_px"], size, roll=params["roll_deg"])
    h.camera_params = {k: float(v) for k, v in params.items()}
    return h, params, res


def _project_params(p: dict, world, size, z=None):
    cam = SyntheticCamera(width=size[0], height=size[1], f=p["focal_px"], height_m=p["height_m"],
                          pitch_deg=p["tilt_deg"], yaw_deg=p["yaw_deg"], roll_deg=p["roll_deg"],
                          cam_local=(0.0, 0.0), origin=np.array([p["E"], p["N"]]))
    if z is None:
        return cam.ground_to_pixel(world)
    return cam.project(np.column_stack([np.asarray(world, float).reshape(-1, 2), z]))


def pose_from_clicks(px, world, size, origin, world_z=None,
                     hfovs=(30, 40, 50, 60, 70, 80, 95, 110), ground=None) -> dict | None:
    """Camera pose from the clicked pairs alone (perspective-n-point), no position needed.

    Tried for a range of zooms; the one that reprojects the clicks best wins. Returns camera
    parameters like fit_camera's (E, N, height above the datum, yaw, tilt, roll, focal) plus
    'median_px', or None. Needs 5+ pairs. ground: local (E, N) -> ground height above the
    datum (a terrain); the camera must stand above the ground where it is, not the datum."""
    import cv2

    px = np.asarray(px, np.float64).reshape(-1, 2)
    if len(px) < 5:
        return None
    z = np.zeros(len(px)) if world_z is None else np.asarray(world_z, float)
    obj = np.column_stack([np.asarray(world, float) - origin, z]).astype(np.float64)
    best = None
    for hf in hfovs:
        f = size[0] / 2 / np.tan(np.radians(hf / 2))
        K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
        for flag in (cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE):
            try:
                ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, px, K, None, flags=flag,
                                                         reprojectionError=15.0,
                                                         iterationsCount=1000)
            except cv2.error:
                continue
            if not ok or inl is None or len(inl) < 4:
                continue
            i = inl.ravel()
            try:
                rvec, tvec = cv2.solvePnPRefineLM(obj[i], px[i], K, None, rvec, tvec)
            except cv2.error:
                pass
            R, _ = cv2.Rodrigues(rvec)
            C = (-R.T @ tvec).ravel()
            depth = (obj - C) @ R[2]
            g = 0.0 if ground is None else float(np.nan_to_num(ground(C[None, :2])[0]))
            if C[2] - g <= 0.3 or (depth <= 0).any():
                continue                         # under the ground / points behind the camera
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
            err = np.linalg.norm(proj.reshape(-1, 2) - px, axis=1)
            score = float(np.median(err)) + 50.0 * (1 - len(i) / len(px))
            if best is None or score < best[0]:
                fwd, right = R[2], R[0]
                yaw = np.degrees(np.arctan2(fwd[0], fwd[1]))
                tilt = np.degrees(np.arcsin(np.clip(-fwd[2], -1, 1)))
                r0 = np.array([np.cos(np.radians(yaw)), -np.sin(np.radians(yaw)), 0.0])
                d0 = np.cross(fwd, r0)
                roll = np.degrees(np.arctan2(right @ d0, right @ r0))
                best = (score, {"E": origin[0] + C[0], "N": origin[1] + C[1], "height_m": C[2],
                                "yaw_deg": yaw, "tilt_deg": tilt, "roll_deg": roll,
                                "focal_px": f, "median_px": float(np.median(err))})
    return best[1] if best else None


def camera_calibration(px, world, size, prior: dict, outlier_m: float = 3.0, terrain=None):
    """Homography from clicked pairs + a known camera position (``camera_position`` in the
    site config: E, N, height_m, optional position_tol_m, height_tol_m, hfov_deg).

    Fits the physical camera (yaw, tilt, roll, zoom, small position/height correction) and
    returns a Homography carrying the usual per-point report. Needs >= 3 pairs.

    terrain: a terrain.Terrain. The camera height is then above the ground at the camera
    position (prior ``ground_z_m`` overrides that datum), and the ground follows the model.
    """
    px = np.asarray(px, float).reshape(-1, 2)
    world = np.asarray(world, float).reshape(-1, 2)
    if len(px) < 3:
        raise ValueError("Need at least 3 point pairs with a known camera position")
    E, N, H = float(prior["E"]), float(prior["N"]), float(prior["height_m"])
    z0 = world_z = None
    if terrain is not None:
        z0 = prior.get("ground_z_m")
        if z0 is None:
            z0 = terrain.height(np.array([[E, N]]))[0]
        if not np.isfinite(z0):
            z0 = float(np.nanmedian(terrain.height(world)))
        z0 = float(z0)
        world_z = terrain.height(world) - z0
        world_z = np.where(np.isfinite(world_z), world_z, 0.0)

    def fit(*a, world_z=None, datum=None, **k):
        hh, pp, rr = fit_camera(*a, world_z=world_z, **k)
        if terrain is not None:
            hh.set_terrain(terrain, z0 if datum is None else datum)
        return hh, pp, rr

    d = world - [E, N]
    yaw0 = float(np.degrees(np.arctan2(d[:, 0].mean(), d[:, 1].mean())))
    dist = float(np.median(np.linalg.norm(d, axis=1)))
    tilt0 = float(np.degrees(np.arctan2(H, max(dist, 1.0))))
    hfov = float(prior.get("hfov_deg", 65.0))
    f0 = size[0] / 2 / np.tan(np.radians(hfov / 2))
    h, p, res = fit(px, world, size, E, N, H, yaw0, tilt0, f0, world_z=world_z,
                    pos_tol_m=float(prior.get("position_tol_m", 15.0)),
                    height_tol_m=float(prior.get("height_tol_m", max(1.0, 0.1 * H))))
    def _errors(hh):
        e = np.linalg.norm(hh.to_world(px) - world, axis=1)
        return np.where(np.isfinite(e), e, np.inf)

    err = _errors(h)
    fin = np.isfinite(err)
    inl = err <= max(outlier_m, 3 * np.median(err[fin]) if fin.any() else outlier_m)
    if 3 <= inl.sum() < len(px):  # refit on the clicks that agree (drops rooftops, mis-clicks)
        h, p, _ = fit(px[inl], world[inl], size, E, N, H, p["yaw_deg"], p["tilt_deg"],
                      p["focal_px"], pos_tol_m=float(prior.get("position_tol_m", 15.0)),
                      height_tol_m=float(prior.get("height_tol_m", max(1.0, 0.1 * H))),
                      world_z=None if world_z is None else world_z[inl])
        res = np.linalg.norm(_project_params(p, world, size, world_z) - px, axis=1)
        err = _errors(h)
    moved = None
    if len(px) >= 5 and (not np.isfinite(err).all() or
                         np.sqrt(np.mean(err[inl & np.isfinite(err)] ** 2)) > 2.0
                         or inl.mean() < 0.6):
        # the clicks don't fit a camera near the configured spot: find the camera from the
        # clicks alone, then fit again around it; keep whichever explains the clicks better
        ground = None if terrain is None else \
            (lambda xy: terrain.height(np.asarray(xy) + [E, N]) - z0)
        start = pose_from_clicks(px, world, size, np.array([E, N]), world_z, ground=ground)
        if start is not None and start["median_px"] < 25:
            # the datum moves with the camera: the ground where it actually stood
            dz, wz2, z0b = 0.0, world_z, None
            if terrain is not None:
                zc = float(terrain.height(np.array([[start["E"], start["N"]]]))[0])
                if np.isfinite(zc):
                    dz, z0b = zc - z0, zc
                    wz2 = world_z - dz
            hs = start["height_m"] - dz
            h2, p2, _ = fit(px, world, size, start["E"], start["N"], hs,
                            start["yaw_deg"], start["tilt_deg"], start["focal_px"],
                            pos_tol_m=5.0, height_tol_m=max(1.0, 0.2 * hs),
                            world_z=wz2, datum=z0b)
            e2 = _errors(h2)
            fin2 = np.isfinite(e2)
            inl2 = e2 <= max(outlier_m, 3 * np.median(e2[fin2]) if fin2.any() else outlier_m)
            good1 = err[inl & np.isfinite(err)]
            good2 = e2[inl2 & fin2]
            if inl2.sum() > inl.sum() or (inl2.sum() == inl.sum() and len(good2)
                                          and np.sqrt(np.mean(good2 ** 2))
                                          < np.sqrt(np.mean(good1 ** 2))):
                h, p, err, inl = h2, p2, e2, inl2
                res = np.linalg.norm(_project_params(p, world, size, wz2) - px, axis=1)
                moved = float(np.hypot(p["E"] - E, p["N"] - N))
    h.points = [{"id": i + 1, "u": float(u), "v": float(v), "E": float(e), "N": float(n),
                 "error_m": float(er) if np.isfinite(er) else None, "loo_error_m": None,
                 "inlier": bool(ok), "pixel_error": float(r)}
                for i, ((u, v), (e, n), er, ok, r) in enumerate(zip(px, world, err, inl, res))]
    good = err[inl & np.isfinite(err)]
    h.rmse_m = float(np.sqrt(np.mean(good ** 2))) if len(good) else float("inf")
    h.max_error_m = float(good.max()) if len(good) else float("inf")
    # a physical camera cannot degenerate like a free homography, but points along one line
    # still leave zoom against tilt loose: report it (the picker warns live)
    from .homography import point_spread

    h.spread = point_spread(world[inl])
    h.camera_params = {k: float(v) for k, v in p.items()}
    if moved is not None:
        h.camera_moved_m = moved     # the configured position was this far off
    return h
