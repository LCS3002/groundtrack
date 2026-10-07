"""Stage 5: CSV / GeoJSON / grid / stats / Houdini exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .colors import ramp_stops
from .config import Config

EPSG = 27700
GEOJSON_CRS = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{EPSG}"}}


# --------------------------------------------------------------------------- points
def write_points_csv(points: pd.DataFrame, path: Path) -> None:
    points.to_csv(path, index=False)


# --------------------------------------------------------------------------- geojson
def _clean_props(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating, float)):
            v = None if not np.isfinite(v) else float(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        out[k] = v
    return out


def tracks_geojson(points: pd.DataFrame, summary: pd.DataFrame) -> dict:
    """One LineString per track, with the per-track summary as attributes."""
    feats = []
    summ = summary.set_index("track_id") if len(summary) else None
    for tid, tr in points.groupby("track_id", sort=True):
        coords = tr[["x", "y"]].to_numpy()
        # drop consecutive duplicates (stationary objects) - keeps the geometry valid & small
        keep = np.r_[True, np.any(np.diff(coords, axis=0) != 0, axis=1)]
        coords = coords[keep]
        if len(coords) < 2:
            continue
        props = {"track_id": int(tid)}
        if summ is not None and tid in summ.index:
            props.update(summ.loc[tid].to_dict())
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": np.round(coords, 3).tolist()},
            "properties": _clean_props(props),
        })
    return {"type": "FeatureCollection", "name": "tracks", "crs": GEOJSON_CRS, "features": feats}


def predicted_segments_geojson(points: pd.DataFrame) -> dict:
    """The 'machine-made gaps': stretches where the tracker predicted with no detection."""
    feats = []
    for tid, tr in points.groupby("track_id", sort=True):
        flag = tr["predicted"].to_numpy(bool)
        if not flag.any():
            continue
        xy = tr[["x", "y"]].to_numpy()
        src = tr["source"].to_numpy()
        t = tr["time_s"].to_numpy()
        # runs of predicted samples, extended by one sample each side to join the real path
        edges = np.flatnonzero(np.diff(np.r_[0, flag.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            lo, hi = max(a - 1, 0), min(b + 1, len(xy))
            seg = xy[lo:hi]
            if len(seg) < 2:
                continue
            feats.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": np.round(seg, 3).tolist()},
                "properties": {
                    "track_id": int(tid), "class": str(tr["class"].iloc[0]),
                    "start_s": float(t[a]), "duration_s": float(t[b - 1] - t[a]),
                    "n_samples": int(b - a),
                    # predicted (tracker Kalman), stitched (re-linked) or interpolated (outlier)
                    "kind": pd.Series(src[a:b]).mode().iloc[0],
                },
            })
    return {"type": "FeatureCollection", "name": "predicted_gaps", "crs": GEOJSON_CRS,
            "features": feats}


def write_geojson(obj: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


# --------------------------------------------------------------------------- grid
def field_grid(points: pd.DataFrame, cell: float, include_predicted: bool = False) -> pd.DataFrame:
    """Aggregate motion vectors onto a world-aligned grid.

    Cells are aligned to multiples of `cell` in EPSG:27700 so grids from different runs of
    the same site overlay exactly. Every sample counts once, so slow movers (more samples
    per cell) weigh more: the field is time-weighted occupancy.
    """
    cols = ["cell_x", "cell_y", "i", "j", "mean_vx", "mean_vy", "mean_speed", "flow_speed",
            "heading", "coherence", "count", "n_tracks"]
    p = points if include_predicted else points[~points["predicted"].astype(bool)]
    if p.empty:
        return pd.DataFrame(columns=cols)
    i = np.floor(p["x"].to_numpy() / cell).astype(np.int64)
    j = np.floor(p["y"].to_numpy() / cell).astype(np.int64)
    df = pd.DataFrame({"i": i, "j": j, "vx": p["vx"].to_numpy(), "vy": p["vy"].to_numpy(),
                       "speed": p["speed"].to_numpy(), "track_id": p["track_id"].to_numpy()})
    agg = df.groupby(["i", "j"]).agg(mean_vx=("vx", "mean"), mean_vy=("vy", "mean"),
                                     mean_speed=("speed", "mean"), count=("vx", "size"),
                                     n_tracks=("track_id", "nunique")).reset_index()
    agg["cell_x"] = (agg["i"] + 0.5) * cell
    agg["cell_y"] = (agg["j"] + 0.5) * cell
    agg["flow_speed"] = np.hypot(agg["mean_vx"], agg["mean_vy"])
    agg["heading"] = (np.degrees(np.arctan2(agg["mean_vx"], agg["mean_vy"])) + 360) % 360
    # 1 = everyone moves the same way, 0 = movement cancels out (crossing / milling flows)
    agg["coherence"] = np.where(agg["mean_speed"] > 0, agg["flow_speed"] / agg["mean_speed"], 0)
    agg = agg[cols].sort_values(["j", "i"]).reset_index(drop=True)
    for c in ("mean_vx", "mean_vy", "mean_speed", "flow_speed", "coherence"):
        agg[c] = agg[c].round(4)
    agg["heading"] = agg["heading"].round(1)
    return agg


# --------------------------------------------------------------------------- stats
def _crossings(points: pd.DataFrame, a, b):
    """Yield (track_id, class, time_s, direction) for crossings of segment a-b.

    direction +1: crossed from the left of a->b to the right.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    for tid, tr in points.groupby("track_id"):
        xy = tr[["x", "y"]].to_numpy()
        t = tr["time_s"].to_numpy()
        side = np.sign(d[0] * (xy[:, 1] - a[1]) - d[1] * (xy[:, 0] - a[0]))
        for k in np.flatnonzero((side[:-1] * side[1:]) < 0):
            p, q = xy[k], xy[k + 1]
            # does the step p->q cross the finite segment a-b?
            r = q - p
            den = r[0] * d[1] - r[1] * d[0]
            if abs(den) < 1e-12:
                continue
            s = ((a[0] - p[0]) * d[1] - (a[1] - p[1]) * d[0]) / den
            u = ((a[0] - p[0]) * r[1] - (a[1] - p[1]) * r[0]) / den
            if 0 <= s <= 1 and 0 <= u <= 1:
                # side +1 = left of a->b, so starting on the left means a left->right crossing
                yield int(tid), tr["class"].iloc[0], float(t[k] + s * (t[k + 1] - t[k])), \
                    int(side[k])


def compute_stats(points: pd.DataFrame, summary: pd.DataFrame, cfg: Config,
                  window_s: tuple[float, float] | None = None) -> dict:
    st = cfg["stats"]
    if window_s is None:
        window_s = (float(points["time_s"].min()), float(points["time_s"].max())) \
            if len(points) else (0.0, 0.0)
    dur_min = max((window_s[1] - window_s[0]) / 60.0, 1e-9)
    n_min = int(np.ceil(dur_min))
    out: dict = {
        "site": cfg.site, "crs": f"EPSG:{EPSG}",
        "window_s": [round(window_s[0], 3), round(window_s[1], 3)],
        "duration_min": round(dur_min, 3),
        "n_tracks": int(len(summary)),
        "predicted_share": round(float(points["predicted"].mean()), 4) if len(points) else 0.0,
        "per_class": {}, "per_group": {}, "count_lines": [],
        "notes": {
            "speeds": "m/s. Track speeds are the mean of the smoothed per-sample speed of each "
                      "track; histograms count each track once.",
            "flow_per_minute": "tracks first seen in each minute of the analysed window",
            "straightness": "straight-line distance / path length (1 = perfectly straight)",
            "heading": "degrees clockwise from British National Grid north",
        },
    }

    def block(s: pd.DataFrame, pts: pd.DataFrame, rng: tuple[float, float] | None):
        b = {
            "n_tracks": int(len(s)),
            "flow_per_min_mean": round(len(s) / dur_min, 3),
            "mean_speed": _r(s["mean_speed"].mean()),
            "median_speed": _r(s["mean_speed"].median()),
            "sample_mean_speed": _r(pts.loc[~pts["predicted"].astype(bool), "speed"].mean()),
            "p85_speed": _r(s["mean_speed"].quantile(0.85)) if len(s) else None,
            "mean_straightness": _r(s["straightness"].mean()),
            "mean_duration_s": _r(s["duration_s"].mean()),
            "mean_path_length_m": _r(s["path_length_m"].mean()),
        }
        minute = np.floor((s["start_s"].to_numpy() - window_s[0]) / 60).astype(int)
        b["flow_per_minute"] = np.bincount(np.clip(minute, 0, max(n_min - 1, 0)),
                                           minlength=n_min).tolist() if len(s) else [0] * n_min
        if rng is not None:
            nb = int(st["histogram_bins"])
            edges = np.linspace(rng[0], rng[1], nb + 1)
            vals = np.clip(s["mean_speed"].to_numpy(), rng[0], rng[1])  # overflow -> last bin
            b["speed_histogram"] = {"bin_edges": np.round(edges, 4).tolist(),
                                    "counts": np.histogram(vals, edges)[0].tolist(),
                                    "note": "last bin includes everything above the range"}
        return b

    for cls, s in summary.groupby("class") if len(summary) else []:
        g = cfg.group_of(cls)
        out["per_class"][cls] = block(s, points[points["class"] == cls], g.speed_range)
    for gname, s in summary.groupby("group") if len(summary) else []:
        out["per_group"][gname] = block(s, points[points["group"] == gname],
                                        cfg.groups[gname].speed_range)
    for line in st.get("count_lines") or []:
        crosses = list(_crossings(points, line["a"], line["b"]))
        pos = [c for c in crosses if c[3] > 0]
        neg = [c for c in crosses if c[3] < 0]
        per_class: dict = {}
        for _, cls, _, d in crosses:
            pc = per_class.setdefault(cls, {"a_to_right": 0, "a_to_left": 0})
            pc["a_to_right" if d > 0 else "a_to_left"] += 1
        out["count_lines"].append({
            "name": line.get("name", f"line{len(out['count_lines']) + 1}"),
            "a": list(line["a"]), "b": list(line["b"]),
            "crossings_left_to_right": len(pos), "crossings_right_to_left": len(neg),
            "per_class": per_class,
            "flow_per_min": round(len(crosses) / dur_min, 3),
            "note": "left/right as seen standing at a looking towards b",
        })
    return out


def _r(v, nd=3):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def stats_table(stats: dict) -> pd.DataFrame:
    """Flatten stats.json into a long CSV: scope, name, metric, value, bin_lo, bin_hi."""
    rows = [("all", "all", "n_tracks", stats["n_tracks"], None, None),
            ("all", "all", "duration_min", stats["duration_min"], None, None),
            ("all", "all", "predicted_share", stats["predicted_share"], None, None)]
    for scope in ("per_class", "per_group"):
        for name, b in stats[scope].items():
            sc = scope.replace("per_", "")
            for k, v in b.items():
                if k == "flow_per_minute":
                    for m, n in enumerate(v):
                        rows.append((sc, name, f"flow_minute_{m:03d}", n, m * 60, (m + 1) * 60))
                elif k == "speed_histogram":
                    e = v["bin_edges"]
                    for c, lo, hi in zip(v["counts"], e[:-1], e[1:]):
                        rows.append((sc, name, "speed_hist", c, lo, hi))
                else:
                    rows.append((sc, name, k, v, None, None))
    for cl in stats["count_lines"]:
        rows.append(("count_line", cl["name"], "crossings_left_to_right",
                     cl["crossings_left_to_right"], None, None))
        rows.append(("count_line", cl["name"], "crossings_right_to_left",
                     cl["crossings_right_to_left"], None, None))
        rows.append(("count_line", cl["name"], "flow_per_min", cl["flow_per_min"], None, None))
    return pd.DataFrame(rows, columns=["scope", "name", "metric", "value", "bin_lo", "bin_hi"])


# --------------------------------------------------------------------------- houdini
HOUDINI_TEMPLATE = r'''"""groundtrack -> Houdini. Paste into a Python SOP (Geometry > Python).

Reads points.csv into points with attributes, joins them into one open polyline per
track_id and colours them (Cd) by speed with the same blue -> red ramp as topdown.png.

Axes: Houdini is Y-up, so Easting -> +X and Northing -> -Z (north is up in the Top view).
Coordinates are relative to ORIGIN (stored as detail attributes origin_E / origin_N) to keep
float32 precision; add ORIGIN back to get British National Grid metres.

Attributes
  point: track_id, class, group, frame, time_s, v (velocity m/s, Houdini axes), speed,
         heading (deg from grid north), predicted (1 = tracker guess through occlusion),
         Cd (speed ramp, darker where predicted)
  prim:  track_id, class, group

MODE = "tracks"  -> every sample, polylines per track (paths / trails)
MODE = "current" -> one point per track alive at the current Houdini time
                    (time_s = $T * TIME_SCALE + TIME_OFFSET), for POP / particle sources.
"""
import csv
import hou

CSV_PATH = __CSV_PATH__
ORIGIN = __ORIGIN__
SPEED_RANGES = __SPEED_RANGES__        # m/s colour range per group
RAMP = __RAMP__                        # (position, sRGB) stops, blue -> red
CD_LINEAR = True                       # convert sRGB ramp to linear for Houdini's colour pipeline
PREDICTED_DIM = 0.45                   # multiply Cd of predicted samples (show machine-made gaps)
MODE = "tracks"
TIME_SCALE = 1.0
TIME_OFFSET = 0.0

node = hou.pwd()
geo = node.geometry()
if node.parm("csv_path") is not None and node.evalParm("csv_path"):
    CSV_PATH = node.evalParm("csv_path")


def _lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def ramp(speed, group):
    lo, hi = SPEED_RANGES.get(group, (0.0, 1.0))
    t = min(max((speed - lo) / max(hi - lo, 1e-9), 0.0), 1.0)
    for (p0, c0), (p1, c1) in zip(RAMP[:-1], RAMP[1:]):
        if t <= p1:
            f = (t - p0) / max(p1 - p0, 1e-9)
            c = [a + (b - a) * f for a, b in zip(c0, c1)]
            break
    else:
        c = list(RAMP[-1][1])
    return [_lin(x) for x in c] if CD_LINEAR else c


with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: (int(r["track_id"]), int(r["frame"])))

if MODE == "current":
    now = hou.time() * TIME_SCALE + TIME_OFFSET
    by_track = {}
    for r in rows:
        by_track.setdefault(int(r["track_id"]), []).append(r)
    picked = []
    for tid, rs in by_track.items():
        if not (float(rs[0]["time_s"]) <= now <= float(rs[-1]["time_s"])):
            continue
        # nearest sample in time (samples are ~1 frame apart)
        picked.append(min(rs, key=lambda r: abs(float(r["time_s"]) - now)))
    rows = picked

for name, default in (("track_id", 0), ("frame", 0), ("predicted", 0)):
    geo.addAttrib(hou.attribType.Point, name, default)
for name in ("time_s", "speed", "heading"):
    geo.addAttrib(hou.attribType.Point, name, 0.0)
for name in ("class", "group"):
    geo.addAttrib(hou.attribType.Point, name, "")
geo.addAttrib(hou.attribType.Point, "v", (0.0, 0.0, 0.0))
geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
geo.addAttrib(hou.attribType.Prim, "track_id", 0)
geo.addAttrib(hou.attribType.Prim, "class", "")
geo.addAttrib(hou.attribType.Prim, "group", "")
geo.addAttrib(hou.attribType.Global, "origin_E", float(ORIGIN[0]))
geo.addAttrib(hou.attribType.Global, "origin_N", float(ORIGIN[1]))

ox, oy = ORIGIN
positions = [hou.Vector3(float(r["x"]) - ox, 0.0, -(float(r["y"]) - oy)) for r in rows]
points = geo.createPoints(positions)
pred = [1 if r["predicted"].strip().lower() in ("true", "1") else 0 for r in rows]
geo.setPointIntAttribValues("track_id", [int(r["track_id"]) for r in rows])
geo.setPointIntAttribValues("frame", [int(r["frame"]) for r in rows])
geo.setPointIntAttribValues("predicted", pred)
geo.setPointFloatAttribValues("time_s", [float(r["time_s"]) for r in rows])
geo.setPointFloatAttribValues("speed", [float(r["speed"]) for r in rows])
geo.setPointFloatAttribValues("heading", [float(r["heading"] or 0) for r in rows])
geo.setPointStringAttribValues("class", [r["class"] for r in rows])
geo.setPointStringAttribValues("group", [r.get("group", "") for r in rows])
vel = []
cd = []
for r, p in zip(rows, pred):
    vel += [float(r["vx"]), 0.0, -float(r["vy"])]
    c = ramp(float(r["speed"]), r.get("group", ""))
    cd += [x * PREDICTED_DIM for x in c] if p else c
geo.setPointFloatAttribValues("v", vel)
geo.setPointFloatAttribValues("Cd", cd)

if MODE == "tracks":
    tracks = {}
    for i, r in enumerate(rows):
        tracks.setdefault(int(r["track_id"]), []).append(i)
    for tid, idx in tracks.items():
        if len(idx) < 2:
            continue
        poly = geo.createPolygon(is_closed=False)
        for i in idx:
            poly.addVertex(points[i])
        poly.setAttribValue("track_id", tid)
        poly.setAttribValue("class", rows[idx[0]]["class"])
        poly.setAttribValue("group", rows[idx[0]].get("group", ""))
'''


HOUDINI_FIELD_TEMPLATE = r'''"""groundtrack -> Houdini: smoothed top-down vector field for particle sims.

Paste into a Python SOP (or exec() this file from one). Set MODE:

  "volume"   volumes named vel.x, vel.y, vel.z (velocity, m/s), density (object-s/m^2) and
             confidence (0..1): plug into POP Advect by Volumes (velocity) - see README.
  "points"   one point per grid cell with v, speed, heading, density, confidence, Cd, pscale:
             arrows / guides (Visualize v, or Copy to Points with a line).
  "sources"  where tracks start (group 'sources') and end (group 'sinks'), with v and time_s:
             emitters for POP Source, matching where people / vehicles really enter.

Axes match houdini_import.py: Easting -> +X, Northing -> -Z, relative to ORIGIN (detail
attributes origin_E / origin_N). The field is a flat slab one cell thick around y = 0.
"""
import csv
import hou

FIELD_FILES = __FIELD_FILES__          # group -> vector_field csv ("all" = every class)
SLICE_FILES = __SLICE_FILES__          # group -> time-sliced field csv (if exported)
POINTS_CSV = __POINTS_CSV__
ORIGIN = __ORIGIN__
CELL = __CELL__                        # grid cell size in metres
SPEED_RANGES = __SPEED_RANGES__
RAMP = __RAMP__

MODE = "volume"                        # "volume" | "points" | "sources"
GROUP = "all"                          # which field: "all" or a group name in FIELD_FILES
FADE_BY_CONFIDENCE = True              # multiply velocity by confidence (soft edges)
USE_TIME_SLICES = False                # animate the field through the recorded time windows
TIME_SCALE = 1.0                       # time_s = $T * TIME_SCALE + TIME_OFFSET
TIME_OFFSET = 0.0
CD_LINEAR = True

node = hou.pwd()
geo = node.geometry()
ox, oy = ORIGIN


def _lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def ramp(speed, lo, hi):
    t = min(max((speed - lo) / max(hi - lo, 1e-9), 0.0), 1.0)
    for (p0, c0), (p1, c1) in zip(RAMP[:-1], RAMP[1:]):
        if t <= p1:
            f = (t - p0) / max(p1 - p0, 1e-9)
            c = [a + (b - a) * f for a, b in zip(c0, c1)]
            break
    else:
        c = list(RAMP[-1][1])
    return [_lin(x) for x in c] if CD_LINEAR else c


def read_field():
    now = hou.time() * TIME_SCALE + TIME_OFFSET
    path = FIELD_FILES[GROUP]
    if USE_TIME_SLICES and SLICE_FILES.get(GROUP):
        path = SLICE_FILES[GROUP]
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if USE_TIME_SLICES and SLICE_FILES.get(GROUP):
        rows = [r for r in rows if float(r["t_start"]) <= now < float(r["t_end"])]
    return rows


geo.addAttrib(hou.attribType.Global, "origin_E", float(ox))
geo.addAttrib(hou.attribType.Global, "origin_N", float(oy))
lo, hi = SPEED_RANGES.get(GROUP, max(SPEED_RANGES.values(), key=lambda r: r[1]))

if MODE == "volume":
    rows = read_field()
    if rows:
        cells = {}
        for r in rows:
            cells[(int(r["i"]), int(r["j"]))] = r
        i0 = min(k[0] for k in cells)
        i1 = max(k[0] for k in cells)
        j0 = min(k[1] for k in cells)
        j1 = max(k[1] for k in cells)
        nx, nz = i1 - i0 + 1, j1 - j0 + 1
        # Houdini z grows southwards (north = -Z): voxel row k holds grid row j = j1 - k
        bbox = hou.BoundingBox(i0 * CELL - ox, -0.5 * CELL, -((j1 + 1) * CELL - oy),
                               (i1 + 1) * CELL - ox, 0.5 * CELL, -(j0 * CELL - oy))
        names = ("vel.x", "vel.y", "vel.z", "density", "confidence")
        data = {n: [0.0] * (nx * nz) for n in names}
        for (i, j), r in cells.items():
            idx = (i - i0) + (j1 - j) * nx           # x varies fastest, then y (=1), then z
            fade = float(r["confidence"]) if FADE_BY_CONFIDENCE else 1.0
            data["vel.x"][idx] = float(r["vx"]) * fade
            data["vel.z"][idx] = -float(r["vy"]) * fade
            data["density"][idx] = float(r["density"])
            data["confidence"][idx] = float(r["confidence"])
        geo.addAttrib(hou.attribType.Prim, "name", "")
        for n in names:
            vol = geo.createVolume(nx, 1, nz, bbox)
            vol.setAttribValue("name", n)
            vol.setAllVoxels(data[n])

elif MODE == "points":
    rows = read_field()
    for name in ("speed", "heading", "density", "confidence", "pscale"):
        geo.addAttrib(hou.attribType.Point, name, 0.0)
    geo.addAttrib(hou.attribType.Point, "v", (0.0, 0.0, 0.0))
    geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
    pts = geo.createPoints([hou.Vector3(float(r["cell_x"]) - ox, 0.0, -(float(r["cell_y"]) - oy))
                            for r in rows])
    v, cd = [], []
    for r in rows:
        fade = float(r["confidence"]) if FADE_BY_CONFIDENCE else 1.0
        v += [float(r["vx"]) * fade, 0.0, -float(r["vy"]) * fade]
        cd += ramp(float(r["speed"]), lo, hi)
    geo.setPointFloatAttribValues("v", v)
    geo.setPointFloatAttribValues("Cd", cd)
    for name in ("speed", "heading", "density", "confidence"):
        geo.setPointFloatAttribValues(name, [float(r[name]) for r in rows])
    geo.setPointFloatAttribValues("pscale", [CELL * float(r["confidence"]) for r in rows])

elif MODE == "sources":
    with open(POINTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    first, last = {}, {}
    for r in rows:
        if r["predicted"].strip().lower() in ("true", "1"):
            continue
        tid = int(r["track_id"])
        if GROUP != "all" and r.get("group") != GROUP:
            continue
        if tid not in first or float(r["time_s"]) < float(first[tid]["time_s"]):
            first[tid] = r
        if tid not in last or float(r["time_s"]) > float(last[tid]["time_s"]):
            last[tid] = r
    geo.addAttrib(hou.attribType.Point, "track_id", 0)
    geo.addAttrib(hou.attribType.Point, "time_s", 0.0)
    geo.addAttrib(hou.attribType.Point, "speed", 0.0)
    geo.addAttrib(hou.attribType.Point, "v", (0.0, 0.0, 0.0))
    geo.addAttrib(hou.attribType.Point, "class", "")
    geo.addAttrib(hou.attribType.Point, "kind", "")
    src_grp = geo.createPointGroup("sources")
    snk_grp = geo.createPointGroup("sinks")
    for kind, d, grp in (("source", first, src_grp), ("sink", last, snk_grp)):
        for tid, r in d.items():
            pt = geo.createPoint()
            pt.setPosition(hou.Vector3(float(r["x"]) - ox, 0.0, -(float(r["y"]) - oy)))
            pt.setAttribValue("track_id", tid)
            pt.setAttribValue("time_s", float(r["time_s"]))
            pt.setAttribValue("speed", float(r["speed"]))
            pt.setAttribValue("v", (float(r["vx"]), 0.0, -float(r["vy"])))
            pt.setAttribValue("class", r["class"])
            pt.setAttribValue("kind", kind)
            grp.add(pt)
'''


def write_houdini_field_script(path: Path, field_files: dict, slice_files: dict, points_csv: Path,
                               origin: tuple[float, float], cell: float, cfg: Config) -> None:
    ranges = {g.name: tuple(g.speed_range) for g in cfg.groups.values()}
    ranges["all"] = max(ranges.values(), key=lambda r: r[1]) if ranges else (0.0, 1.0)
    posix = {k: Path(v).resolve().as_posix() for k, v in field_files.items()}
    sl = {k: Path(v).resolve().as_posix() for k, v in slice_files.items()}
    txt = (HOUDINI_FIELD_TEMPLATE
           .replace("__FIELD_FILES__", repr(posix))
           .replace("__SLICE_FILES__", repr(sl))
           .replace("__POINTS_CSV__", repr(Path(points_csv).resolve().as_posix()))
           .replace("__ORIGIN__", repr((float(origin[0]), float(origin[1]))))
           .replace("__CELL__", repr(float(cell)))
           .replace("__SPEED_RANGES__", repr(ranges))
           .replace("__RAMP__", repr(ramp_stops())))
    Path(path).write_text(txt, encoding="utf-8")


def write_houdini_script(path: Path, points_csv: Path, origin: tuple[float, float],
                         cfg: Config) -> None:
    ranges = {g.name: tuple(g.speed_range) for g in cfg.groups.values()}
    txt = (HOUDINI_TEMPLATE
           .replace("__CSV_PATH__", repr(Path(points_csv).resolve().as_posix()))
           .replace("__ORIGIN__", repr((float(origin[0]), float(origin[1]))))
           .replace("__SPEED_RANGES__", repr(ranges))
           .replace("__RAMP__", repr(ramp_stops())))
    Path(path).write_text(txt, encoding="utf-8")
