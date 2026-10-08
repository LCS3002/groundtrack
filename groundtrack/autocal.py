"""Calibration without clicking.

1. Same spot (`transfer`): a clip filmed from a position that is already calibrated inherits
   that calibration. From one camera position, any two views (other direction, other zoom)
   are related by a single image homography, so this is exact geometry, not a guess: match
   the two frames, chain the homographies, refit the physical camera.
2. Roads (`fit_to_roads`): with the camera position known, vehicles must drive on roads.
   The camera (direction, tilt, roll, zoom) is solved so that the tracked vehicles land on
   the OpenStreetMap road lines. No clicks and no image matching, so old aerials, blur and
   haze don't matter.

Both return a Homography like the point picker does, and `quality_checks` tests any
calibration against what the footage itself says (people ~1.7 m, vehicles on roads,
plausible speeds) so a bad automatic result is caught instead of silently used.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .homography import Homography, fit_homography

USER_AGENT = "groundtrack (architecture research tool)"
OVERPASS = "https://overpass-api.de/api/interpreter"


# --------------------------------------------------------------------------- 1. same spot
def match_frames(dst_bgr: np.ndarray, src_bgr: np.ndarray, max_dim: int = 1600):
    """Homography taking dst pixels to src pixels (same camera position), with inlier stats.

    Returns (H, n_inliers, inlier_ratio) or (None, 0, 0)."""
    def prep(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = min(1.0, max_dim / max(g.shape))
        if s < 1:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cv2.createCLAHE(2.0, (8, 8)).apply(g), s

    g1, s1 = prep(dst_bgr)
    g2, s2 = prep(src_bgr)
    sift = cv2.SIFT_create(nfeatures=6000)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None, 0, 0.0
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    good = [m for m, n in (p for p in matcher.knnMatch(d1, d2, k=2) if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        return None, len(good), 0.0
    a = np.float32([k1[m.queryIdx].pt for m in good]) / s1
    b = np.float32([k2[m.trainIdx].pt for m in good]) / s2
    H, mask = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 4.0, maxIters=10000, confidence=0.999)
    if H is None:
        return None, 0, 0.0
    n = int(mask.sum())
    return H, n, n / len(good)


def transfer(src: Homography, src_frame: np.ndarray, dst_frame: np.ndarray,
             camera_prior: dict | None = None, min_inliers: int = 40, log=print):
    """Calibration for dst_frame from a calibrated src_frame filmed from the same spot.

    Returns (Homography or None, info dict)."""
    H, n, ratio = match_frames(dst_frame, src_frame)
    info = {"method": "same-spot transfer", "inliers": n, "inlier_ratio": round(ratio, 3)}
    if H is None or n < min_inliers or ratio < 0.2:
        info["reason"] = f"frames don't match well enough ({n} inliers, {ratio:.0%})"
        return None, info
    h_dst, w_dst = dst_frame.shape[:2]
    h_src, w_src = src_frame.shape[:2]
    # synthetic correspondences: a grid over the new frame, through the old calibration
    gu, gv = np.meshgrid(np.linspace(0.03, 0.97, 32) * w_dst, np.linspace(0.03, 0.97, 18) * h_dst)
    px = np.column_stack([gu.ravel(), gv.ravel()])
    q = cv2.perspectiveTransform(px.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    inside = (q[:, 0] > 0) & (q[:, 0] < w_src) & (q[:, 1] > 0) & (q[:, 1] < h_src)
    world = src.to_world(q)
    ok = inside & np.isfinite(world).all(axis=1)
    cam = getattr(src, "camera_params", None) or src.camera()
    if cam and ok.any():   # drop the far fringe (near the horizon one pixel is many metres)
        d = np.hypot(world[:, 0] - cam["E"], world[:, 1] - cam["N"])
        ok &= d < np.nanpercentile(d[ok], 95) * 1.5
    info["shared_points"] = int(ok.sum())
    if ok.sum() < 12:
        info["reason"] = "the two views share too little ground"
        return None, info
    px, world = px[ok], world[ok]
    if camera_prior:
        from .posefit import camera_calibration

        prior = dict(camera_prior)
        src_cam = getattr(src, "camera_params", None)
        if src_cam:             # the old fit is a better starting point than the config
            prior.update({k: src_cam[k] for k in ("E", "N", "height_m")})
            prior["position_tol_m"] = min(float(prior.get("position_tol_m", 15)), 5.0)
        h = camera_calibration(px, world, (w_dst, h_dst), prior)
    else:
        h = fit_homography(px, world, image_size=(w_dst, h_dst))
    h.image_size = (w_dst, h_dst)
    info["fit_rmse_m"] = round(float(h.rmse_m), 3)
    log(f"  same spot: {n} matching features ({ratio:.0%}), {int(ok.sum())} shared ground points,"
        f" refit error {h.rmse_m:.2f} m")
    return h, info


# --------------------------------------------------------------------------- 2. roads
ROAD_TYPES = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|"
              "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street|"
              "bus_guideway|busway")


MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link",
         "trunk_link", "primary_link", "secondary_link", "tertiary_link"}


class Road(np.ndarray):
    """(n, 2) EPSG:27700 polyline with its OSM highway class in `.kind`."""

    def __new__(cls, xy, kind=""):
        obj = np.asarray(xy, float).view(cls)
        obj.kind = kind
        return obj

    def __array_finalize__(self, obj):
        self.kind = getattr(obj, "kind", "")


def osm_roads(E: float, N: float, radius_m: float = 1500.0, timeout: float = 60.0,
              cache: Path | None = None) -> list[np.ndarray]:
    """Road centrelines around (E, N) from OpenStreetMap, as (n, 2) arrays in EPSG:27700.

    Only the bounding box of the search area is sent to the Overpass API."""
    from rasterio.warp import transform

    if cache is not None and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data and isinstance(data[0], dict):
            return [Road(np.asarray(w["xy"], float), w["kind"]) for w in data]
    lon, lat = transform("EPSG:27700", "EPSG:4326",
                         [E - radius_m, E + radius_m], [N - radius_m, N + radius_m])
    bbox = f"{lat[0]:.6f},{lon[0]:.6f},{lat[1]:.6f},{lon[1]:.6f}"
    q = f'[out:json][timeout:50];way["highway"~"^({ROAD_TYPES})$"]({bbox});out geom;'
    req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    ways = []
    for el in data.get("elements", []):
        g = el.get("geometry") or []
        if len(g) < 2:
            continue
        xs, ys = transform("EPSG:4326", "EPSG:27700", [p["lon"] for p in g], [p["lat"] for p in g])
        ways.append(Road(np.column_stack([xs, ys]), (el.get("tags") or {}).get("highway", "")))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([{"kind": w.kind, "xy": w.round(2).tolist()} for w in ways]),
                         encoding="utf-8")
    return ways


class RoadDistance:
    """Distance (m) from any ground point to the nearest road line, from a raster."""

    def __init__(self, ways: list[np.ndarray], E: float, N: float, radius_m: float,
                 res: float = 1.0, minor_penalty_m: float = 0.0):
        from scipy.ndimage import distance_transform_edt

        self.x0, self.y1, self.res = E - radius_m, N + radius_m, res
        n = int(2 * radius_m / res) + 1
        self.n = n
        dist = []
        angle = np.zeros((n, n), np.float32)          # road direction (deg, mod 180) per pixel
        for major in (True, False):
            mask = np.ones((n, n), np.uint8)
            for w in ways:
                if minor_penalty_m and (getattr(w, "kind", "") in MAJOR) != major:
                    continue
                pts = np.round(np.column_stack([(w[:, 0] - self.x0) / res,
                                                (self.y1 - w[:, 1]) / res])).astype(np.int32)
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(mask, tuple(map(int, a)), tuple(map(int, b)), 0, 1)
                    ang = np.degrees(np.arctan2(-(b[1] - a[1]), b[0] - a[0])) % 180
                    cv2.line(angle, tuple(map(int, a)), tuple(map(int, b)), float(ang), 1)
            if not minor_penalty_m:
                d, idx = distance_transform_edt(mask, return_indices=True)
                dist = [d * res]
                self.nearest = idx
                break
            d, idx = distance_transform_edt(mask, return_indices=True)
            dist.append((d * res + (0 if major else minor_penalty_m), idx))
        if minor_penalty_m:
            (dm, im), (dn, inn) = dist
            use_major = dm <= dn
            self.d = np.where(use_major, dm, dn)
            self.nearest = np.where(use_major[None], im, inn)
        else:
            self.d = dist[0]
        self.angle = angle

    def _rc(self, xy):
        return (self.y1 - xy[:, 1]) / self.res, (xy[:, 0] - self.x0) / self.res

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates

        r, c = self._rc(xy)
        out = map_coordinates(self.d, [r, c], order=1, mode="nearest")
        far = (c < 0) | (r < 0) | (c > self.n - 1) | (r > self.n - 1)
        return np.where(far | ~np.isfinite(out), 200.0, out)

    def direction(self, xy: np.ndarray) -> np.ndarray:
        """Direction (deg from +E, mod 180) of the road nearest to each point."""
        r, c = self._rc(xy)
        ri = np.clip(np.round(r), 0, self.n - 1).astype(int)
        ci = np.clip(np.round(c), 0, self.n - 1).astype(int)
        nr, nc = self.nearest[0][ri, ci], self.nearest[1][ri, ci]
        return self.angle[nr, nc]


class VehicleSamples(tuple):
    """(uv, ids, times) - unpacks like a tuple; .left / .right / .cls hold the box-bottom
    corners and the class for the size cue."""

    def __new__(cls, uv, ids, ts):
        obj = super().__new__(cls, (uv, ids, ts))
        obj.left = obj.right = np.zeros((0, 2))
        obj.cls = np.zeros(0, object)
        return obj


# typical footprints (length, width) in m: a box's bottom edge on the ground spans about
# width*|cos a| + length*|sin a|, a = angle between the vehicle's heading and the view ray
VEHICLE_SIZE = {"car": (4.5, 1.8), "bus": (11.0, 2.55), "truck": (8.0, 2.5),
                "motorcycle": (2.1, 0.8)}


def vehicle_track_samples(raw: pd.DataFrame, registration: dict | None = None,
                          max_tracks: int = 300, per_track: int = 12, seed: int = 0):
    """Foot pixels of moving vehicle tracks: a few samples per track, their track ids and
    times (s)."""
    from .registration import apply_registration

    r = raw[raw["class"].isin(["car", "bus", "truck", "motorcycle"])].copy()
    r = r[~r["predicted"].astype(str).str.lower().isin(["true", "1"])]
    if r.empty:
        return VehicleSamples(np.zeros((0, 2)), np.zeros(0, int), np.zeros(0))
    r["u"] = (r["x1"] + r["x2"]) / 2
    r["v"] = r["y2"]
    r["ul"], r["ur"] = r["x1"].astype(float), r["x2"].astype(float)
    if registration:
        fr = r["frame"].to_numpy()
        uv = apply_registration(fr, r[["u", "v"]].to_numpy(float), registration)
        lft = apply_registration(fr, r[["ul", "v"]].to_numpy(float), registration)
        rgt = apply_registration(fr, r[["ur", "v"]].to_numpy(float), registration)
        r["u"], r["v"], r["ul"], r["ur"] = uv[:, 0], uv[:, 1], lft[:, 0], rgt[:, 0]
        r["vl"], r["vr"] = lft[:, 1], rgt[:, 1]
    else:
        r["vl"] = r["vr"] = r["v"]
    # moving tracks only: parked cars sit beside roads and would pull the fit off them
    span = r.groupby("track_id").agg(du=("u", lambda s: s.max() - s.min()),
                                     dv=("v", lambda s: s.max() - s.min()), n=("u", "size"))
    moving = span[(np.hypot(span["du"], span["dv"]) > 40) & (span["n"] >= 8)].index
    rng = np.random.default_rng(seed)
    if len(moving) > max_tracks:
        moving = rng.choice(moving, max_tracks, replace=False)
    pts, ids, ts, left, right, cls = [], [], [], [], [], []
    for tid in moving:
        t = r[r["track_id"] == tid].sort_values("frame")
        k = np.linspace(0, len(t) - 1, min(per_track, len(t))).round().astype(int)
        pts.append(t[["u", "v"]].to_numpy(float)[k])
        left.append(t[["ul", "vl"]].to_numpy(float)[k])
        right.append(t[["ur", "vr"]].to_numpy(float)[k])
        ids.append(np.full(len(k), tid))
        ts.append(t["time_s"].to_numpy(float)[k])
        cls.append(np.full(len(k), t["class"].mode().iloc[0]))
    if not pts:
        return VehicleSamples(np.zeros((0, 2)), np.zeros(0, int), np.zeros(0))
    S = VehicleSamples(np.vstack(pts), np.concatenate(ids), np.concatenate(ts))
    S.left, S.right, S.cls = np.vstack(left), np.vstack(right), np.concatenate(cls)
    return S


class RoadObjective:
    """How well a camera puts the tracked vehicles where vehicles can be.

    Four cues, as residuals for least squares:
      road       each sample near an OpenStreetMap road (minor roads cost a few metres more)
      direction  each step of a track runs along the nearest road, not across it
      size       each vehicle's box bottom on the ground matches its footprint at that viewing
                 angle (car 4.5 x 1.8 m, ...): this sets the zoom / scale
      speed      no impossible speeds (slow traffic is normal: queues, lights)
    Parameters p: yaw, tilt, roll (deg), log focal (px), dE, dN (m), height (m).
    """

    def __init__(self, S: VehicleSamples, road: RoadDistance, E: float, N: float, size,
                 speed_band=(0.2, 40.0)):
        self.S, self.road, self.E, self.N = S, road, E, N
        self.w, self.h = size
        uv, ids, ts = S
        self.uv = uv
        same = ids[1:] == ids[:-1]
        self.i0 = np.nonzero(same)[0]
        self.i1 = self.i0 + 1
        self.dts = np.maximum(ts[self.i1] - ts[self.i0], 1e-3)
        self.tracks, self.t_pair = np.unique(ids[self.i0], return_inverse=True)
        self.all_tracks, self.t_all = np.unique(ids, return_inverse=True)
        self.v_lo, self.v_hi = speed_band
        self.L = np.array([VEHICLE_SIZE.get(c, (4.5, 1.8))[0] for c in S.cls])
        self.W = np.array([VEHICLE_SIZE.get(c, (4.5, 1.8))[1] for c in S.cls])
        n = max(len(uv), 1)
        self.w_size = 8.0 * np.sqrt(n / max(len(self.all_tracks), 1))
        self.w_speed = 3.0 * np.sqrt(n / max(len(self.tracks), 1))

    def ground(self, p, pix):
        """Pixels -> ground (E, N) under camera parameters p."""
        yaw, pitch, roll, logf, dE, dN, height = p
        f = math.exp(logf)
        ry, rp, rr = math.radians(yaw), math.radians(pitch), math.radians(roll)
        fwd = np.array([math.sin(ry) * math.cos(rp), math.cos(ry) * math.cos(rp), -math.sin(rp)])
        right = np.array([math.cos(ry), -math.sin(ry), 0.0])
        down = np.cross(fwd, right)
        right, down = (math.cos(rr) * right + math.sin(rr) * down,
                       -math.sin(rr) * right + math.cos(rr) * down)
        x = (pix[:, 0] - self.w / 2) / f
        y = (pix[:, 1] - self.h / 2) / f
        ray = fwd[None] + x[:, None] * right[None] + y[:, None] * down[None]
        t = np.where(ray[:, 2] < -1e-6, height / -np.minimum(ray[:, 2], -1e-6), np.nan)
        return np.column_stack([self.E + dE + t * ray[:, 0], self.N + dN + t * ray[:, 1]])

    def residuals(self, p, robust: float = 4.0) -> np.ndarray:
        g = self.ground(p, self.uv)
        fin = np.isfinite(g).all(axis=1)
        d = np.where(fin, self.road(np.nan_to_num(g, nan=1e9)), 200.0)
        r_road = np.sqrt(2 * robust * (np.sqrt(1 + (d / robust) ** 2) - 1))   # soft-L1
        i0, i1 = self.i0, self.i1
        mv = g[i1] - g[i0]
        # speed: geometric mean per track, only impossible values cost
        step = np.linalg.norm(mv, axis=1) / self.dts
        step = np.where(np.isfinite(step), step, 1e4)
        k = len(self.tracks)
        v = np.exp(np.bincount(self.t_pair, np.log(np.maximum(step, 1e-3)), k)
                   / np.maximum(np.bincount(self.t_pair, minlength=k), 1))
        r_speed = self.w_speed * (np.maximum(0, np.log(v / self.v_hi))
                                  + np.maximum(0, np.log(self.v_lo / v)))
        # direction: along the nearest road
        mid = np.nan_to_num((g[i0] + g[i1]) / 2, nan=1e9)
        ang = np.degrees(np.arctan2(mv[:, 1], mv[:, 0])) % 180
        dd = np.abs((ang - self.road.direction(mid) + 90) % 180 - 90)
        r_dir = 3.0 * np.sin(np.radians(np.where(np.isfinite(dd), dd, 90)))
        # size: box bottom edge on the ground vs the footprint seen from this angle
        meas = np.linalg.norm(self.ground(p, self.S.right) - self.ground(p, self.S.left), axis=1)
        heading = np.full(len(g), np.nan)
        heading[i0] = np.arctan2(mv[:, 1], mv[:, 0])
        ray = g - np.array([self.E + p[4], self.N + p[5]])
        a = heading - np.arctan2(ray[:, 1], ray[:, 0])
        expect = self.L * np.abs(np.sin(a)) + self.W * np.abs(np.cos(a))
        ratio = np.log(np.maximum(meas, 1e-3) / np.maximum(expect, 1e-3))
        use = np.isfinite(ratio)
        n_all = len(self.all_tracks)
        tr = (np.bincount(self.t_all, np.where(use, ratio, 0.0), n_all)
              / np.maximum(np.bincount(self.t_all, use.astype(float), n_all), 1))
        r_size = self.w_size * np.maximum(np.abs(tr) - 0.15, 0.0)
        return np.concatenate([r_road, r_speed, r_dir, r_size])

    def parts(self, p) -> dict:
        """Cost per cue (sum of squares), to see which cue disagrees."""
        r = self.residuals(p)
        cut = np.cumsum([len(self.uv), len(self.tracks), len(self.i0)])
        return {nm: round(float(np.sum(seg ** 2)), 1)
                for nm, seg in zip(("road", "speed", "direction", "size"), np.split(r, cut))}


def fit_to_roads(raw: pd.DataFrame, size, prior: dict, ways: list[np.ndarray],
                 registration: dict | None = None, log=print, radius_m: float = 1500.0,
                 speed_band=(0.2, 40.0), minor_penalty_m: float = 5.0,
                 extra_starts: list[dict] | None = None, clicks=None, click_px: float = 4.0,
                 search: bool = True):
    """Camera from the known position + vehicles-on-roads. Returns (Homography|None, info).

    clicks: optional (pixels, world) point pairs; each counts like `click_px` pixels of
    uncertainty, so a few clicks anchor the position and the vehicles add hundreds of
    constraints on direction, tilt and zoom. search=False only refines extra_starts.

    Search: every direction, tilt and zoom on a small sample of tracks (steps scaled to the
    zoom, so long lenses are not skipped), then the most promising distinct candidates are
    refined on all tracks, with the camera position free within its tolerance."""
    from scipy.optimize import least_squares

    from .posefit import pose_homography

    S = vehicle_track_samples(raw, registration)
    n_tracks = int(len(set(S[1])))
    info = {"method": "vehicles on OpenStreetMap roads", "vehicle_tracks": n_tracks}
    if n_tracks < 15:
        info["reason"] = f"only {n_tracks} moving vehicle tracks (need 15+)"
        return None, info
    E, N, Hc = float(prior["E"]), float(prior["N"]), float(prior["height_m"])
    pos_tol = float(prior.get("position_tol_m", 15.0))
    h_tol = float(prior.get("height_tol_m", max(1.0, 0.06 * Hc)))
    road = RoadDistance(ways, E, N, radius_m, minor_penalty_m=minor_penalty_m)
    w, hh = size
    full = RoadObjective(S, road, E, N, size, speed_band)
    small = RoadObjective(vehicle_track_samples(raw, registration, max_tracks=80, per_track=6,
                                                seed=1), road, E, N, size, speed_band)

    # 1. coarse search on the small sample
    fovs = list(np.geomspace(5, 80, 21)) if search else []
    if prior.get("hfov_deg"):
        fovs.append(float(prior["hfov_deg"]))
    cands = []
    for fov in fovs:
        logf = math.log(w / 2 / math.tan(math.radians(fov / 2)))
        vfov = fov * hh / w
        for pitch in np.arange(1.0, 65.0, float(np.clip(vfov / 3, 0.75, 4.0))):
            for yaw in np.arange(0.0, 360.0, float(np.clip(fov / 3, 1.0, 6.0))):
                p = (float(yaw), float(pitch), 0.0, logf, 0.0, 0.0, Hc)
                cands.append((float(np.mean(small.residuals(p) ** 2)), p, fov))
    cands.sort(key=lambda c: c[0])

    # 2. refine distinct candidates on every track
    lo = [-1e9, 0.3, -12, math.log(150), -pos_tol, -pos_tol, max(0.5, Hc - h_tol)]
    hi = [1e9, 75, 12, math.log(60000), pos_tol, pos_tol, Hc + h_tol]
    sig_pos, sig_h = max(pos_tol / 2, 0.1), max(h_tol / 2, 0.1)

    if clicks is not None:
        from .posefit import _project

        c_px, c_world = (np.asarray(a, float).reshape(-1, 2) for a in clicks)
        w_click = math.sqrt(len(full.uv) / max(len(c_px), 1)) / click_px

    def refine(x0, obj=full, nfev=300):
        def r(p):
            prior_r = 3.0 * np.array([p[4] / sig_pos, p[5] / sig_pos, (p[6] - Hc) / sig_h])
            out = [obj.residuals(p), prior_r]
            if clicks is not None:
                out.append(w_click * (_project(p, c_world, size, np.array([E, N])) - c_px).ravel())
            return np.concatenate(out)
        return least_squares(r, np.clip(np.asarray(x0, float), lo, hi), bounds=(lo, hi),
                             max_nfev=nfev, diff_step=1e-3)

    def distinct(x, others, tol=1.0):
        return not any(abs((x[0] - q[0] + 180) % 360 - 180) < 3 * tol and abs(x[1] - q[1]) < 3 * tol
                       and abs(x[3] - q[3]) < 0.1 * tol for q in others)

    sols = []
    for cam in extra_starts or []:      # e.g. a calibration to test, or a hint
        x0 = [cam["yaw_deg"], cam["tilt_deg"], cam.get("roll_deg", 0.0),
              math.log(cam["focal_px"]), cam["E"] - E, cam["N"] - N, cam["height_m"]]
        info.setdefault("extra_starts", []).append(full.parts(np.clip(x0, lo, hi)))
        sols.append(refine(x0))
    # 2a. quick refinement of many distinct candidates on the small sample
    picked, quick = [], []
    for c, p, fov in (cands if search else []):
        if not distinct(p, [q for q, _ in picked], tol=max(fov / 6, 1.0)):
            continue
        picked.append((p, fov))
        quick.append(refine(p, small, nfev=60))
        if len(picked) >= 40:
            break
    quick.sort(key=lambda s_: s_.cost)
    # 2b. the best few, refined on every track
    done = []
    for q in quick:
        if distinct(q.x, done):
            done.append(q.x)
            sols.append(refine(q.x))
        if len(done) >= 8:
            break
    sols.sort(key=lambda s_: s_.cost)
    p = sols[0].x
    g = full.ground(p, full.uv)
    d = road(np.nan_to_num(g, nan=1e9))
    on_road = float(np.mean(d < 6.0))
    v_tr = np.linalg.norm(g[full.i1] - g[full.i0], axis=1) / full.dts
    # how distinct is the answer? a different solution almost as good = ambiguous
    alt = [s_ for s_ in sols[1:] if abs((s_.x[0] - p[0] + 180) % 360 - 180) > 5
           or abs(s_.x[1] - p[1]) > 3 or abs(s_.x[3] - p[3]) > 0.15]
    margin = (alt[0].cost / max(sols[0].cost, 1e-9)) if alt else float("inf")
    params = {"yaw_deg": float(p[0] % 360), "tilt_deg": float(p[1]), "roll_deg": float(p[2]),
              "focal_px": float(math.exp(p[3])), "E": E + float(p[4]), "N": N + float(p[5]),
              "height_m": float(p[6]),
              "hfov_deg": float(math.degrees(2 * math.atan(w / 2 / math.exp(p[3]))))}
    h = pose_homography(params["E"], params["N"], params["height_m"], params["yaw_deg"],
                        params["tilt_deg"], params["focal_px"], size, roll=params["roll_deg"],
                        origin=np.round([E, N]))
    h.camera_params = params
    h.points = []
    h.rmse_m = float(np.median(d))
    h.max_error_m = float(np.percentile(d, 90))
    h.spread = 1.0
    info.update(on_road_share=round(on_road, 3),
                median_road_distance_m=round(float(np.median(d)), 2),
                median_vehicle_speed_m_s=round(float(np.nanmedian(v_tr)), 2),
                ambiguity_ratio=round(float(margin), 2) if np.isfinite(margin) else None,
                cost_parts=full.parts(p), camera={k: round(v, 3) for k, v in params.items()})
    log(f"  roads: {n_tracks} vehicle tracks, {on_road:.0%} of samples within 6 m of a road, "
        f"looking {params['yaw_deg']:.0f}°, tilt {params['tilt_deg']:.1f}°, field of view "
        f"{params['hfov_deg']:.1f}°")
    return h, info


# --------------------------------------------------------------------------- 4. checks
def quality_checks(h: Homography, raw: pd.DataFrame, cfg, registration: dict | None = None,
                   ways: list[np.ndarray] | None = None, people: bool = True) -> dict:
    """Does this calibration agree with the footage? Returns {checks: [...], verdict}."""
    from .selfcal import people_check
    from .trajectories import max_range_for

    checks = []
    if people and "person" in cfg.classes and (raw["class"] == "person").sum() > 30:
        chk = people_check(raw, h, registration, max_range_for(cfg, h, "people"))
        if chk:
            err = chk["scale_error_pct"]
            checks.append({"name": "people height", "value": f"{chk['implied_person_height_m']} m",
                           "ok": abs(err) <= 12, "note": f"{err:+.0f}% vs 1.70 m"})
    veh = raw["class"].isin(["car", "bus", "truck", "motorcycle"])
    if veh.sum() > 30 and ways:
        uv = vehicle_track_samples(raw, registration)[0]
        cam = h.camera_params or h.camera() or {}
        if len(uv) and cam:
            r = 1500.0
            road = RoadDistance(ways, cam["E"], cam["N"], r)
            g = h.to_world(uv)
            d = road(np.nan_to_num(g, nan=1e9))
            share = float(np.mean(d < 6.0))
            checks.append({"name": "vehicles on roads", "value": f"{share:.0%}",
                           "ok": share >= 0.7, "note": "share of vehicle positions within 6 m "
                                                        "of an OpenStreetMap road"})
    verdict = "ok" if checks and all(c["ok"] for c in checks) else (
        "check" if checks else "no checks possible")
    return {"checks": checks, "verdict": verdict}


# --------------------------------------------------------------------------- orchestration
def _latest_run(cfg) -> Path | None:
    from .layout import Run

    root = (cfg.path("output_dir") or cfg.base_dir / "runs") / cfg.site
    runs = sorted((p for p in root.glob("*") if Run(p).is_run()), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _road_cache(out: Path, prior: dict) -> Path:
    """One road download per camera spot (100 m grid), shared by every clip filmed there."""
    return out.parent / f"osm_roads_{round(prior['E'], -2):.0f}_{round(prior['N'], -2):.0f}.json"


def run_autocalibration(cfg, log=print, run_dir: Path | None = None, device: str = "auto",
                        replace: bool = False, track_frames: int = 900) -> dict:
    """Calibrate a site without clicking, as far as the footage allows. Returns the report.

    1. same spot: another site config in the same folder, already calibrated and filmed
       from the same position -> exact transfer.
    2. vehicles on roads (needs camera_position and moving vehicles): fit the camera to
       OpenStreetMap roads; clicked points, if there are any, are used as well.
    The result goes to the configured homography file if there is none yet (or `replace`),
    otherwise next to it as <name>_auto.json, so a hand calibration is never overwritten.
    """
    from .config import load_config
    from .layout import open_run
    from .pipeline import new_run_dir, stage_track
    from .registration import load_registration
    from .video import read_frame

    out = cfg.path("homography")
    if out is None:
        raise ValueError("Set `homography:` (output path) in the site config")
    prior = cfg.get("camera_position")
    report: dict = {"site": cfg.site, "tried": []}
    video = cfg.path("video")
    frame_idx = Homography.load(out).reference_frame if out.exists() else 0
    frame = read_frame(video, frame_idx)
    size = (frame.shape[1], frame.shape[0])
    h, raw, reg, ways = None, None, None, None

    # 1. same spot ---------------------------------------------------------------------
    best = None
    for other in sorted(cfg.base_dir.glob("*.yaml")):
        try:
            oc = load_config(other)
        except Exception:
            continue
        op, ov = oc.path("homography"), oc.path("video")
        if oc.site == cfg.site or op is None or not op.exists() or op == out \
                or ov is None or not ov.exists():
            continue
        ocam = oc.get("camera_position")
        if prior and ocam and (math.hypot(ocam["E"] - prior["E"], ocam["N"] - prior["N"]) > 30
                               or abs(ocam["height_m"] - prior["height_m"]) > 10):
            continue                                  # a different camera position
        hs = Homography.load(op)
        hh, info = transfer(hs, read_frame(ov, hs.reference_frame), frame, prior,
                            log=lambda *_: None)
        info["from"] = oc.site
        report["tried"].append(info)
        if hh is not None and (best is None or info["inliers"] > best[1]["inliers"]):
            best = (hh, info)
    if best and best[1]["inliers"] >= 100 and best[1]["inlier_ratio"] >= 0.3:
        h, chosen = best
        h.method = f"same spot (from {chosen['from']})"
        h.method_info = chosen
        log(f"same spot as {chosen['from']}: {chosen['inliers']} matching features -> "
            "calibration transferred (exact geometry)")
    elif best:
        log(f"same spot as {best[1]['from']}? only {best[1]['inliers']} matches: not trusted")

    # tracks: needed for the road fit and for the checks
    rd = run_dir or _latest_run(cfg)
    if rd is None and h is None:
        log(f"no tracked run yet: tracking the first {track_frames} frames ...")
        from .device import resolve_device

        rd = new_run_dir(cfg, "autocal")
        stage_track(cfg, rd, resolve_device(device), log=log, max_frames=track_frames)
    if rd is not None:
        run = open_run(rd)
        raw = pd.read_csv(run.raw_tracks)
        if run.registration.exists():
            reg, ref, _ = load_registration(run.registration)
            if h is None:
                frame_idx = ref                      # the tracks are aligned to that frame
    n_veh = 0 if raw is None else int(raw["class"].isin(["car", "bus", "truck"]).sum())

    # 2. vehicles on roads -------------------------------------------------------------
    if h is None:
        if not prior:
            report["reason"] = ('needs the camera position first: groundtrack locate '
                                '"<building>" --floor N -c <site>.yaml --write')
        elif n_veh < 100:
            report["reason"] = ("no moving vehicles to fit to roads: for people-only clips, "
                                "pick 3-4 points (groundtrack calibrate)")
        else:
            from .calibrate import load_points_csv

            ways = osm_roads(prior["E"], prior["N"], cache=_road_cache(out, prior))
            pts_csv = out.with_name(out.stem + "_points.csv")
            clicks = load_points_csv(pts_csv) if pts_csv.exists() else None
            starts = []
            if out.exists() and getattr(Homography.load(out), "camera_params", None):
                starts.append(Homography.load(out).camera_params)
            h, info = fit_to_roads(raw, size, prior, ways, reg, log=log, clicks=clicks,
                                   extra_starts=starts)
            info["with_clicks"] = clicks is not None
            report["tried"].append(info)
            if h is not None:
                h.method = "vehicles on roads" + (" + clicked points" if clicks is not None
                                                  else "")
                h.method_info = {k: v for k, v in info.items() if k != "extra_starts"}
    if h is None:
        log("automatic calibration not possible: " + report.get("reason", "see the report"))
        report["result"] = None
        return report

    # 4. checks ------------------------------------------------------------------------
    if raw is not None:
        if ways is None and prior and n_veh > 100:
            try:
                ways = osm_roads(prior["E"], prior["N"], cache=_road_cache(out, prior))
            except OSError as e:
                log(f"  (OpenStreetMap roads unavailable: {e}; road check skipped)")
                ways = None
        report["quality"] = quality_checks(h, raw, cfg, reg, ways)
        if not report["quality"]["checks"]:
            log("  no automatic check possible for this clip: look at the check image")
        for c in report["quality"]["checks"]:
            log(f"  check {c['name']}: {c['value']} ({c['note']}) -> "
                + ("ok" if c["ok"] else "CHECK"))
    h.reference_frame = int(frame_idx)
    h.image_size = size
    target = out if (replace or not out.exists()) else out.with_name(out.stem + "_auto.json")
    if replace and out.exists():
        import shutil

        shutil.copy2(out, out.with_name(out.stem + ".before_auto.json"))
    h.save(target)
    try:
        from .geo import load_geotiff
        from .overlay_check import map_in_video

        mp, ins = map_in_video(frame, h, load_geotiff(cfg.path("geotiff")))
        blend = frame.copy()
        blend[ins] = (0.5 * frame[ins] + 0.5 * mp[ins]).astype(np.uint8)
        cv2.imwrite(str(target.with_name(target.stem + "_check_video.jpg")), blend)
    except Exception as e:  # the check image is a convenience
        log(f"  (check image skipped: {e})")
    report["result"] = str(target)
    report["camera"] = getattr(h, "camera_params", None)
    target.with_name(target.stem + "_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"saved {target}" + ("" if target == out else
                             f"\n  (your calibration {out.name} is kept; compare, then rerun "
                             "with --replace to use the automatic one)"))
    return report
