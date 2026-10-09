"""Calibration check and refinement from the people in the footage.

Every walking person is a measuring stick of known size (adults ~1.7 m). With a calibration
we can compute how tall each detected person would have to be to fill their box: the
camera matrix gives where a point 1 m above their feet appears, so the box top fixes the
height. If the calibration is right the median is ~1.7 m; a wrong tilt / zoom / camera
height shows up immediately (e.g. "people are 2.4 m tall").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .homography import Homography

PERSON_HEIGHT_M = 1.70


def camera_matrix(h: Homography) -> np.ndarray | None:
    """3x4 projection for local ground coords (E - origin E, N - origin N, height)."""
    if not h.image_size:
        return None
    cam = getattr(h, "camera_params", None) or h.camera()
    if not cam:
        return None
    w, hh = h.image_size
    K = np.array([[cam["focal_px"], 0, w / 2], [0, cam["focal_px"], hh / 2], [0, 0, 1.0]])
    G = np.linalg.inv(h.H)                       # local ground (x, y, 1) -> pixel
    Kinv = np.linalg.inv(K)
    s = 1.0 / np.linalg.norm(Kinv @ G[:, 0])
    r1, r2 = s * (Kinv @ G[:, 0]), s * (Kinv @ G[:, 1])
    r3 = np.cross(r1, r2)
    P = np.column_stack([G[:, 0], G[:, 1], (K @ r3) / s, G[:, 2]])
    # 'up' must move a point up the image (smaller v) for a ground point in view
    probe = h.plane_to_world(np.array([[w / 2, hh * 0.8]]))[0] - np.asarray(h.origin)
    if np.isfinite(probe).all():
        def v_at(z):
            x = P @ np.array([probe[0], probe[1], z, 1.0])
            return x[1] / x[2]
        if v_at(1.0) > v_at(0.0):
            P[:, 2] *= -1
    return P


def implied_heights(foot_px: np.ndarray, top_px: np.ndarray, h: Homography,
                    P: np.ndarray | None = None) -> np.ndarray:
    """Height (m) a person standing at each foot pixel must have to reach the top pixel."""
    P = camera_matrix(h) if P is None else P
    if P is None:
        return np.full(len(foot_px), np.nan)
    gw = h.to_world(foot_px)
    g = gw - np.asarray(h.origin)
    # the foot stands on the ground (a terrace or a step with a terrain model); solve for
    # the height above it
    X = np.column_stack([g, h.ground_height(gw), np.ones(len(g))])
    a, c = X @ P[1], X @ P[2]
    b, d = P[1, 2], P[2, 2]
    v = np.asarray(top_px, float)[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (a - v * c) / (v * d - b)
    return np.where(np.isfinite(g).all(axis=1), z, np.nan)


def people_check(raw: pd.DataFrame, h: Homography, registration: dict | None = None,
                 max_range_m: float | None = None, min_box_px: float = 25.0) -> dict | None:
    """Median implied height of clearly visible, detected people (None if too few)."""
    foot, top = people_observations(raw, registration, min_box_px)
    if len(foot) < 30:
        return None
    z = implied_heights(foot, top, h)
    if max_range_m:
        cam = getattr(h, "camera_params", None) or h.camera()
        if cam:
            gw = h.to_world(foot)
            far = np.hypot(gw[:, 0] - cam["E"], gw[:, 1] - cam["N"]) > max_range_m
            z = np.where(far, np.nan, z)
    z = z[np.isfinite(z) & (z > 0.3) & (z < 5.0)]
    if len(z) < 30:
        return None
    med = float(np.median(z))
    return {"implied_person_height_m": round(med, 2),
            "iqr_m": [round(float(np.percentile(z, 25)), 2), round(float(np.percentile(z, 75)), 2)],
            "n": int(len(z)),
            "scale_error_pct": round(100 * (med / PERSON_HEIGHT_M - 1), 1),
            "note": "median height people would need to have under this calibration; "
                    "~1.7 m means the calibration is consistent"}


def people_observations(raw: pd.DataFrame, registration: dict | None = None,
                        min_box_px: float = 25.0):
    """(foot_px, top_px) of clearly visible detected people, in calibration-frame pixels."""
    from .registration import apply_registration
    from .trajectories import recover_feet

    r = raw.copy()
    r["predicted"] = r["predicted"].astype(str).str.lower().isin(["true", "1"])
    feet, squished = recover_feet(r, 60)
    m = ((r["class"] == "person") & ~r["predicted"] & ~squished
         & ((r["y2"] - r["y1"]) >= min_box_px) & (r["y1"] > 2)).to_numpy()
    r = r[m]
    foot = feet[m]
    top = np.column_stack([(r["x1"] + r["x2"]) / 2, r["y1"]]).astype(float)
    if registration:
        fr = r["frame"].to_numpy()
        foot = apply_registration(fr, foot, registration)
        top = apply_registration(fr, top, registration)
    ok = np.isfinite(foot).all(axis=1) & np.isfinite(top).all(axis=1)
    return foot[ok], top[ok]
