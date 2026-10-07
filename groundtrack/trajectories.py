"""Stages 3-4: project boxes to the ground, clean, smooth, and compute motion vectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import Config, Group
from .homography import Homography
from .lens import Lens

# Every output sample has a source; anything but 'detected' is machine-made (predicted=True):
#   predicted    - the tracker's Kalman prediction while the object was not detected
#   stitched     - gap between two tracker fragments re-linked in ground coordinates
#   interpolated - a dropped outlier sample, linearly interpolated
DETECTED, PREDICTED, STITCHED, INTERPOLATED = "detected", "predicted", "stitched", "interpolated"

POINT_COLUMNS = ["track_id", "class", "frame", "time_s", "x", "y", "vx", "vy", "speed",
                 "heading", "predicted", "source", "group"]


# --------------------------------------------------------------------------- classes
def assign_track_classes(raw: pd.DataFrame) -> pd.Series:
    """Confidence-weighted majority vote of the class over each track's detected rows.

    YOLO flickers between car/truck/bus on the same vehicle; a track gets one class.
    """
    det = raw[~raw["predicted"].astype(bool)]
    if det.empty:
        return pd.Series(dtype=object)
    w = det.groupby(["track_id", "class"])["confidence"].sum().reset_index()
    best = w.sort_values("confidence", ascending=False).drop_duplicates("track_id")
    return best.set_index("track_id")["class"]


# --------------------------------------------------------------------------- projection
def foot_points(raw: pd.DataFrame) -> np.ndarray:
    """Bottom-centre of each bbox in pixels."""
    return np.column_stack([(raw["x1"].to_numpy() + raw["x2"].to_numpy()) / 2,
                            raw["y2"].to_numpy()]).astype(float)


def recover_feet(raw: pd.DataFrame, rows_per_window: int = 60, squish_ratio: float = 0.8):
    """Foot points that survive partial occlusion. Returns (uv, recovered_mask).

    When a railing, a sign or another person hides someone's legs, the box gets shorter
    from the bottom and its bottom edge (the "foot point") jumps up the image, so the
    person projects too far away. Per track we keep a robust normal box height (rolling
    80th percentile); a box much shorter than that is "squished". We then check which
    edge moved against the trend of the unsquished boxes: if the bottom moved, the feet
    are rebuilt as top + normal height; if the top moved (a gantry, an umbrella) the
    bottom is real and kept. Tracker predictions are left alone.
    """
    uv = foot_points(raw)
    rec = np.zeros(len(raw), bool)
    if raw.empty:
        return uv, rec
    y1 = raw["y1"].to_numpy(float)
    y2 = raw["y2"].to_numpy(float)
    fr = raw["frame"].to_numpy()
    pred = raw["predicted"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    tids = raw["track_id"].to_numpy()
    order = np.lexsort((fr, tids))
    win = max(5, int(rows_per_window))
    for t in np.unique(tids):
        m = order[tids[order] == t]
        m = m[~pred[m]]
        if len(m) < 5:
            continue
        hgt = y2[m] - y1[m]
        ref = pd.Series(hgt).rolling(win, center=True, min_periods=3).quantile(0.8).to_numpy()
        squished = hgt < squish_ratio * ref
        if not squished.any() or squished.all():
            continue
        good = ~squished
        f = fr[m]
        y1_hat = np.interp(f, f[good], y1[m][good])
        y2_hat = np.interp(f, f[good], y2[m][good])
        bottom_cut = squished & (np.abs(y2[m] - y2_hat) > np.abs(y1[m] - y1_hat))
        idx = m[bottom_cut]
        uv[idx, 1] = y1[idx] + ref[bottom_cut]
        rec[idx] = True
    return uv, rec


def project_to_ground(raw: pd.DataFrame, h: Homography, lens: Lens | None = None,
                      registration: dict | None = None, uv: np.ndarray | None = None
                      ) -> pd.DataFrame:
    """Add u, v (foot pixel in the calibration frame) and gx, gy (ground coords) columns.

    registration: {frame: 3x3} mapping each frame's pixels to the calibration frame's
    (camera-motion compensation); rows of frames without one are dropped.
    """
    if h.undistorted and lens is None:
        raise ValueError("The homography was calibrated on undistorted frames but no `lens:` "
                         "file is configured. Add it, or re-run calibrate without a lens.")
    if lens is not None and not h.undistorted:
        raise ValueError("A `lens:` file is configured but the homography was calibrated on "
                         "raw frames. Re-run calibrate with the lens, or remove `lens:`.")
    df = raw.copy()
    uv = foot_points(df) if uv is None else np.asarray(uv, float)
    if registration is not None:
        from .registration import apply_registration

        uv = apply_registration(df["frame"].to_numpy(), uv, registration)
    if lens is not None:
        uv = lens.undistort_points(uv)
    g = h.to_world(uv)
    df["u"], df["v"] = uv[:, 0], uv[:, 1]
    df["gx"], df["gy"] = g[:, 0], g[:, 1]
    return df[np.isfinite(df["gx"]) & np.isfinite(df["gy"])]


# --------------------------------------------------------------------------- cleaning
def _step_ok(t, x, y, i, j, vmax, tol) -> bool:
    d = np.hypot(x[j] - x[i], y[j] - y[i])
    return d <= vmax * abs(t[j] - t[i]) + tol


def remove_spikes(t, x, y, vmax: float, tol: float, passes: int = 3) -> np.ndarray:
    """Mask of points to keep after removing isolated jumps (out-and-back in one sample)."""
    keep = np.ones(len(t), bool)
    for _ in range(passes):
        idx = np.flatnonzero(keep)
        if len(idx) < 3:
            break
        bad = []
        for k in range(len(idx)):
            i = idx[k]
            prev_i = idx[k - 1] if k > 0 else None
            next_i = idx[k + 1] if k + 1 < len(idx) else None
            if prev_i is not None and next_i is not None:
                if (not _step_ok(t, x, y, prev_i, i, vmax, tol)
                        and not _step_ok(t, x, y, i, next_i, vmax, tol)
                        and _step_ok(t, x, y, prev_i, next_i, vmax, tol)):
                    bad.append(i)
            elif next_i is not None and k + 2 < len(idx):  # first point
                if (not _step_ok(t, x, y, i, next_i, vmax, tol)
                        and _step_ok(t, x, y, next_i, idx[k + 2], vmax, tol)):
                    bad.append(i)
            elif prev_i is not None and k >= 2:  # last point
                if (not _step_ok(t, x, y, prev_i, i, vmax, tol)
                        and _step_ok(t, x, y, idx[k - 2], prev_i, vmax, tol)):
                    bad.append(i)
        if not bad:
            break
        keep[bad] = False
    return keep


def split_segments(t, x, y, vmax: float, tol: float, max_gap_s: float) -> list[np.ndarray]:
    """Split where a step is physically impossible (ID switch) or the time gap is too long."""
    if len(t) == 0:
        return []
    cuts = [0]
    for i in range(1, len(t)):
        if (t[i] - t[i - 1] > max_gap_s) or not _step_ok(t, x, y, i - 1, i, vmax, tol):
            cuts.append(i)
    cuts.append(len(t))
    return [np.arange(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


# --------------------------------------------------------------------------- smoothing
def savgol_window(n: int, window_s: float, dt: float, polyorder: int) -> int:
    """Odd window length in samples, clipped to the track length. 0 = too short to smooth."""
    w = int(round(window_s / dt))
    w = max(w, polyorder + 2)
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    return w if w > polyorder else 0


def smooth_and_differentiate(x: np.ndarray, y: np.ndarray, dt: float, window_s: float,
                             polyorder: int = 2):
    """Savitzky-Golay smoothed positions and their time derivative (m/s)."""
    n = len(x)
    w = savgol_window(n, window_s, dt, polyorder)
    if w == 0:
        if n >= 2:
            return x.copy(), y.copy(), np.gradient(x, dt), np.gradient(y, dt)
        return x.copy(), y.copy(), np.zeros(n), np.zeros(n)
    xs = savgol_filter(x, w, polyorder)
    ys = savgol_filter(y, w, polyorder)
    vx = savgol_filter(x, w, polyorder, deriv=1, delta=dt)
    vy = savgol_filter(y, w, polyorder, deriv=1, delta=dt)
    return xs, ys, vx, vy


def heading_deg(vx: np.ndarray, vy: np.ndarray, min_speed: float) -> np.ndarray:
    """Compass heading, degrees clockwise from grid north. Held while (nearly) stationary."""
    speed = np.hypot(vx, vy)
    hd = (np.degrees(np.arctan2(vx, vy)) + 360.0) % 360.0
    hd = np.where(speed >= min_speed, hd, np.nan)
    s = pd.Series(hd).ffill().bfill()
    return s.to_numpy()


# --------------------------------------------------------------------------- vehicle offset
def away_from_camera(h: Homography, u: np.ndarray, v: np.ndarray, step_px: float = 4.0):
    """Unit ground vector pointing away from the camera at each foot point.

    Exact: radial from the camera's ground position, recovered from the homography.
    Fallback (camera not recoverable): the ground direction of 'image up', which is exact
    on the image's centre column and a close approximation elsewhere.
    """
    p0 = h.to_world(np.column_stack([u, v]))
    cam = h.camera()
    if cam is not None:
        d = p0 - np.array([cam["E"], cam["N"]])
        n = np.linalg.norm(d, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return d / n
    p1 = h.to_world(np.column_stack([u, v - step_px]))
    d = p1 - p0
    n = np.linalg.norm(d, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return d / n


def vehicle_centre(x, y, heading, group: Group, cls: str, h: Homography | None = None,
                   u=None, v=None):
    """Shift foot points (bbox bottom-centre on the ground) to the vehicle centre."""
    half_len = group.offset_for(cls)
    if half_len <= 0:
        return x, y
    hd = np.radians(heading)
    tx, ty = np.sin(hd), np.cos(hd)        # unit travel direction (E, N)
    known = np.isfinite(hd)
    if group.offset_mode == "travel":
        ox = np.where(known, -half_len * tx, 0.0)
        oy = np.where(known, -half_len * ty, 0.0)
        return x + ox, y + oy
    # view mode: the bbox bottom is the footprint point nearest the camera; the centre lies
    # further away along the viewing ray by the rectangle's support distance in that direction
    d = away_from_camera(h, u, v)
    half_w = group.vehicle_width_m / 2
    cos_t = np.abs(d[:, 0] * tx + d[:, 1] * ty)
    sin_t = np.sqrt(np.clip(1 - cos_t ** 2, 0, 1))
    mag = np.where(known, half_len * cos_t + half_w * sin_t, (half_len + half_w) / 2)
    ok = np.isfinite(d).all(axis=1)
    return (np.where(ok, x + mag * d[:, 0], x), np.where(ok, y + mag * d[:, 1], y))


# --------------------------------------------------------------------------- pipeline
def process_tracks(raw: pd.DataFrame, cfg: Config, h: Homography, fps: float, stride: int = 1,
                   lens: Lens | None = None, log=print, registration: dict | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """raw_tracks -> (points, track_summary)."""
    clean = cfg["cleaning"]
    dt = stride / fps
    polyorder = int(clean["smooth_polyorder"])
    raw = raw.copy()
    raw["predicted"] = raw["predicted"].astype(str).str.lower().isin(["true", "1"])
    if raw.empty:
        return pd.DataFrame(columns=POINT_COLUMNS), pd.DataFrame()

    classes = assign_track_classes(raw)
    raw = raw[raw["track_id"].isin(classes.index)].copy()
    raw["class"] = raw["track_id"].map(classes)
    raw = raw[raw["class"].isin(cfg.classes)]
    uv = None
    if clean.get("recover_occluded_feet", True):
        win = int(round(float(clean.get("feet_window_s", 2.0)) * fps / stride))
        uv, rec = recover_feet(raw, win, float(clean.get("squish_ratio", 0.8)))
        if rec.any():
            log(f"  {int(rec.sum())} foot points rebuilt where the lower body was hidden")
    g = project_to_ground(raw, h, lens, registration, uv)
    n_above = len(raw) - len(g)
    if n_above:
        log(f"  {n_above} rows above the horizon / outside the ground plane dropped")

    next_id = int(raw["track_id"].max()) + 1 if len(raw) else 1
    stats = {"spikes": 0, "splits": 0, "stitched": 0, "short": 0, "stationary": 0,
             "interpolated": 0}

    # 1. per tracker id: drop spikes, split at impossible jumps / long gaps
    segments: list[Segment] = []
    for tid, tr in g.groupby("track_id", sort=True):
        tr = tr.sort_values("frame").drop_duplicates("frame")
        cls = tr["class"].iloc[0]
        group = cfg.group_of(cls)
        t = tr["time_s"].to_numpy(float)
        x, y = tr["gx"].to_numpy(float), tr["gy"].to_numpy(float)
        keep = remove_spikes(t, x, y, group.max_speed, group.jump_tolerance_m)
        stats["spikes"] += int((~keep).sum())
        tr = tr[keep]
        t, x, y = t[keep], x[keep], y[keep]
        segs = split_segments(t, x, y, group.max_speed, group.jump_tolerance_m,
                              float(clean["max_gap_s"]))
        stats["splits"] += max(0, len(segs) - 1)
        # the longest segment keeps the original id, the others get fresh ids
        order = sorted(range(len(segs)), key=lambda k: -len(segs[k]))
        for rank, k in enumerate(order):
            new_id = int(tid) if rank == 0 else next_id
            if rank > 0:
                next_id += 1
            segments.append(Segment(new_id, [int(tid)], cls, group.name, tr.iloc[segs[k]]))

    # 2. re-link fragments broken by occlusions, in ground coordinates
    if clean.get("stitch", True):
        n_before = len(segments)
        segments = stitch_segments(segments, cfg, float(clean["stitch_max_gap_s"]))
        stats["stitched"] = n_before - len(segments)

    # 3. smooth, differentiate, summarise
    out_frames, summaries = [], []
    for seg in segments:
        grp = cfg.groups[seg.group]
        if grp.min_displacement_m > 0:
            xy = seg.df[["gx", "gy"]].to_numpy()
            if np.linalg.norm(xy - xy[0], axis=1).max() < grp.min_displacement_m:
                stats["stationary"] += 1
                continue
        res = _process_segment(seg, grp, cfg, h, fps, stride, dt, polyorder)
        if res is None:
            stats["short"] += 1
            continue
        pts, summ = res
        stats["interpolated"] += int((pts["source"] == INTERPOLATED).sum())
        out_frames.append(pts)
        summaries.append(summ)

    log(f"  cleaning: {stats['spikes']} spike points removed, {stats['splits']} track splits, "
        f"{stats['stitched']} occlusion gaps stitched, {stats['short']} segments shorter than "
        f"{clean['min_track_s']} s dropped, {stats['stationary']} stationary (parked) dropped, "
        f"{stats['interpolated']} samples interpolated")
    if not out_frames:
        return pd.DataFrame(columns=POINT_COLUMNS), pd.DataFrame()
    points = pd.concat(out_frames, ignore_index=True)[POINT_COLUMNS]
    summary = pd.DataFrame(summaries).sort_values("track_id").reset_index(drop=True)
    return points, summary


@dataclass
class Segment:
    """A piece of trajectory in ground coordinates (rows: frame, time_s, gx, gy, u, v, ...)."""
    track_id: int
    orig_ids: list[int]
    cls: str
    group: str
    df: pd.DataFrame
    stitched_frames: set = field(default_factory=set)

    @property
    def t0(self) -> float:
        return float(self.df["time_s"].iloc[0])

    @property
    def t1(self) -> float:
        return float(self.df["time_s"].iloc[-1])

    def end_state(self, at_end: bool, fit_s: float = 0.6):
        """Position and velocity at the start or end, from a line fit over fit_s seconds."""
        d = self.df
        t = d["time_s"].to_numpy(float)
        sel = (t >= t[-1] - fit_s) if at_end else (t <= t[0] + fit_s)
        tt, xx, yy = t[sel], d["gx"].to_numpy(float)[sel], d["gy"].to_numpy(float)[sel]
        p = np.array([xx[-1], yy[-1]]) if at_end else np.array([xx[0], yy[0]])
        if len(tt) < 3 or np.ptp(tt) <= 0:
            return p, np.zeros(2)
        vx, vy = np.polyfit(tt, xx, 1)[0], np.polyfit(tt, yy, 1)[0]
        # position from the fit as well: the box edge is noisy right at an occlusion
        tref = tt[-1] if at_end else tt[0]
        px = np.polyval(np.polyfit(tt, xx, 1), tref)
        py = np.polyval(np.polyfit(tt, yy, 1), tref)
        return np.array([px, py]), np.array([vx, vy])


def stitch_segments(segments: list[Segment], cfg: Config, max_gap_s: float) -> list[Segment]:
    """Join a segment that ends to one that starts up to max_gap_s later, if the start lies
    where the first segment's ground velocity predicts (within the group's stitch radius,
    growing with the gap). Greedy by smallest prediction error; one link per end/start."""
    cands = []
    for i, a in enumerate(segments):
        pa, va = a.end_state(at_end=True)
        grp = cfg.groups[a.group]
        for j, b in enumerate(segments):
            if i == j or b.group != a.group:
                continue
            gap = b.t0 - a.t1
            if not (0 < gap <= max_gap_s):
                continue
            pb, vb = b.end_state(at_end=False)
            jump = np.linalg.norm(pb - pa)
            if jump > grp.max_speed * gap + grp.jump_tolerance_m:
                continue
            # predict across the gap with the mean of the two velocities (robust to one bad end)
            err = np.linalg.norm(pb - (pa + 0.5 * (va + vb) * gap))
            if err <= grp.stitch_radius_m * (1 + gap):
                cands.append((err, i, j))
    nxt, has_prev, used_end = {}, set(), set()
    for _err, i, j in sorted(cands):
        if i in used_end or j in has_prev:
            continue
        nxt[i] = j
        used_end.add(i)
        has_prev.add(j)
    if not nxt:
        return segments
    out = []
    for i in range(len(segments)):
        if i in has_prev:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        if len(chain) == 1:
            out.append(segments[i])
            continue
        parts = [segments[k] for k in chain]
        stitched = set()
        for a, b in zip(parts[:-1], parts[1:]):
            fa, fb = int(a.df["frame"].iloc[-1]), int(b.df["frame"].iloc[0])
            stitched |= set(range(fa + 1, fb))
        main = max(parts, key=lambda s: len(s.df))
        df = pd.concat([s.df for s in parts], ignore_index=True)
        df["class"] = main.cls
        out.append(Segment(parts[0].track_id, [o for s in parts for o in s.orig_ids], main.cls,
                           parts[0].group, df,
                           stitched.union(*[s.stitched_frames for s in parts])))
    return out


def _process_segment(seg: Segment, group: Group, cfg: Config, h, fps, stride, dt, polyorder):
    clean = cfg["cleaning"]
    df, tid, cls = seg.df, seg.track_id, seg.cls
    f0, f1 = int(df["frame"].iloc[0]), int(df["frame"].iloc[-1])
    if (f1 - f0) / fps < float(clean["min_track_s"]) or len(df) < 2:
        return None
    frames = np.arange(f0, f1 + 1, stride)
    idx = df.set_index("frame")
    src = np.full(len(frames), INTERPOLATED, dtype=object)
    present = np.isin(frames, idx.index.to_numpy())
    pred = idx["predicted"].reindex(frames).to_numpy()
    src[present] = np.where(pred[present].astype(bool), PREDICTED, DETECTED)
    if seg.stitched_frames:
        src[~present & np.isin(frames, list(seg.stitched_frames))] = STITCHED

    def interp(col):
        return np.interp(frames, idx.index.to_numpy(), idx[col].to_numpy(float))

    x, y, u, v = interp("gx"), interp("gy"), interp("u"), interp("v")
    t = frames / fps
    xs, ys, vx, vy = smooth_and_differentiate(x, y, dt, group.smooth_window_s, polyorder)
    hd = heading_deg(vx, vy, float(clean["min_moving_speed"]))
    xs, ys = vehicle_centre(xs, ys, hd, group, cls, h, u, v)
    speed = np.hypot(vx, vy)

    pts = pd.DataFrame({
        "track_id": tid, "class": cls, "frame": frames, "time_s": np.round(t, 4),
        "x": np.round(xs, 3), "y": np.round(ys, 3), "vx": np.round(vx, 3), "vy": np.round(vy, 3),
        "speed": np.round(speed, 3), "heading": np.round(hd, 1),
        "predicted": src != DETECTED, "source": src, "group": group.name,
    })
    seglen = np.hypot(np.diff(xs), np.diff(ys))
    path = float(seglen.sum())
    straight = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    summ = {
        "track_id": tid, "tracker_ids": "+".join(str(i) for i in seg.orig_ids), "class": cls,
        "group": group.name,
        "start_s": round(float(t[0]), 3), "end_s": round(float(t[-1]), 3),
        "duration_s": round(float(t[-1] - t[0]), 3), "n_points": len(frames),
        "n_predicted": int((src == PREDICTED).sum()),
        "n_stitched": int((src == STITCHED).sum()),
        "n_interpolated": int((src == INTERPOLATED).sum()),
        "path_length_m": round(path, 3), "straight_dist_m": round(straight, 3),
        "straightness": round(straight / path, 4) if path > 0 else np.nan,
        "mean_speed": round(float(speed.mean()), 3),
        "median_speed": round(float(np.median(speed)), 3),
        "max_speed": round(float(speed.max()), 3),
        "start_x": round(float(xs[0]), 3), "start_y": round(float(ys[0]), 3),
        "end_x": round(float(xs[-1]), 3), "end_y": round(float(ys[-1]), 3),
    }
    return pts, summ
