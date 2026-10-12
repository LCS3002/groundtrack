"""Several clips of one place on one plan: the composite.

Every run of a site is already in British National Grid on the same aerial, so their tracks
can simply be laid together. The clips were filmed one after another, not at the same
moment: the composite shows typical movement across the time they cover, not one instant.

Density is normalised by observation. Each clip sees part of the ground (its view
footprint, within the range its calibration is trusted) for as long as the clip runs. A
cell's value is person-seconds per m^2 divided by the seconds that cell was in view: the
average number of people per m^2 while it was being watched, so an area two clips filmed is
not twice as busy, and an area nobody filmed is blank rather than empty.

Output folder (same idea as a run's package): images/ frameless plan images and layers,
labels/ legends, scale bar, north arrow, titles, data/ all points + per-clip table + grids,
houdini/ the combined points with the import script.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter

from . import portfolio as pf
from .config import Config
from .field import smooth_field
from .geo import GeoRaster
from .homography import Homography
from .layout import open_run
from .package import (DENSITY_STOPS, _additive_to_rgba, _over, _ramp, _save, flow_layer,
                      legend_png, north_arrow_png, scale_bar_png)
from .trajectories import max_range_for

GRID_M = 1.0                 # composite grid cell (coverage, density)
SMOOTH_M = 1.5               # density smoothing
MIN_SEEN_S = 5.0             # cells watched for less than this stay blank
FAR_CAP_M = 150.0            # a footprint never reaches further than this


@dataclass
class ClipRun:
    name: str
    cfg: Config
    run_dir: Path
    grade: str = "ok"
    note: str = ""
    duration_s: float = 0.0
    points: pd.DataFrame = field(default_factory=pd.DataFrame)
    homography: Homography | None = None


def load_clip(name: str, cfg: Config, run_dir: Path) -> ClipRun | None:
    run = open_run(run_dir)
    if not run.points.exists() or not run.homography_used.exists():
        return None
    h = Homography.load(run.homography_used)
    info = getattr(h, "method_info", None) or {}
    meta = json.loads(run.meta.read_text(encoding="utf-8"))
    fps = float(meta["fps"])
    pts = pd.read_csv(run.points)
    pts["dt"] = int(meta["vid_stride"]) / fps
    return ClipRun(name, cfg, run.root, info.get("grade", "ok"), info.get("note", ""),
                   (int(meta["end_frame"]) - int(meta["start_frame"])) / fps, pts, h)


def footprint(clip: ClipRun, grid) -> np.ndarray:
    """Cells of the composite grid this clip could see (within its trusted range)."""
    x0, y0, nx, ny = grid
    h = clip.homography
    w, hh = h.image_size
    uu, vv = np.meshgrid(np.linspace(0, w - 1, 120), np.linspace(0, hh - 1, 68))
    g = h.to_world(np.column_stack([uu.ravel(), vv.ravel()]))
    cam = h.camera_params
    d = np.hypot(g[:, 0] - cam["E"], g[:, 1] - cam["N"])
    rng = min(max_range_for(clip.cfg, h, "people") or FAR_CAP_M, FAR_CAP_M)
    ok = np.isfinite(d) & (d <= rng)
    m = np.zeros((ny, nx), bool)
    i = np.floor((g[ok, 0] - x0) / GRID_M).astype(int)
    j = np.floor((g[ok, 1] - y0) / GRID_M).astype(int)
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    m[j[inside], i[inside]] = True
    # the samples thin out with distance: close the gaps, but never past the range
    m = binary_fill_holes(binary_closing(m, np.ones((7, 7), bool), iterations=2))
    cx, cy = (cam["E"] - x0) / GRID_M, (cam["N"] - y0) / GRID_M
    jj, ii = np.mgrid[0:ny, 0:nx]
    return m & (np.hypot(ii + 0.5 - cx, jj + 0.5 - cy) * GRID_M <= rng)


AGREE_DEG = 30.0             # flows within this angle count as agreeing


def overlap_agreement(clips: list[ClipRun], pts: pd.DataFrame, cfg: Config, ext, size,
                      images: Path, data: Path, log=print) -> dict:
    """Compare the clips' own people flow fields where their views overlap.

    Each clip gets its field on one shared grid; a cell counts where two or more clips have
    a well-supported, clearly directional flow there (confidence >= 0.5, dominance >= 0.6).
    Writes overlap_cells.csv (per cell: clips, combined vector, direction spread), and
    overlap_pairs.csv (per pair of clips: shared cells, median angle and speed ratio) - a pair
    that disagrees everywhere it overlaps points at a calibration, though different moments
    can also move differently. plan_agreement_layer.png colours the cells."""
    cell = float(cfg["field"]["cell_size_m"])
    sm = float(cfg["field"]["smooth_m"])
    people = pts[(pts["group"] == "people") & ~pts["predicted"].astype(bool)]
    if people.empty:
        return {}
    x, y = people["x"].to_numpy(), people["y"].to_numpy()
    bounds = (int(np.floor((x.min() - 3 * sm) / cell)), int(np.floor((x.max() + 3 * sm) / cell)),
              int(np.floor((y.min() - 3 * sm) / cell)), int(np.floor((y.max() + 3 * sm) / cell)))
    per = {}
    for c in clips:
        p = people[people["clip"] == c.name]
        if p["track_id"].nunique() < 3:
            continue
        f = smooth_field(p, cell, sm, float(p["dt"].median()), bounds=bounds)
        f = f[(f["confidence"] >= 0.5) & (f["dominance"] >= 0.6)]
        per[c.name] = f.set_index(["i", "j"])[["vx", "vy", "speed", "weight"]]
    if len(per) < 2:
        return {}
    allf = pd.concat(per, names=["clip"]).reset_index()
    rows, pairs = [], {}
    for (i, j), g in allf.groupby(["i", "j"]):
        if len(g) < 2:
            continue
        w = g["weight"].to_numpy()
        vx, vy = (g["vx"] * w).sum() / w.sum(), (g["vy"] * w).sum() / w.sum()
        ang = np.degrees(np.arctan2(g["vx"].to_numpy(), g["vy"].to_numpy()))
        mean_ang = np.degrees(np.arctan2(vx, vy))
        spread = float(np.max(np.abs((ang - mean_ang + 180) % 360 - 180)))
        rows.append({"x": (i + 0.5) * cell, "y": (j + 0.5) * cell, "clips": len(g),
                     "clip_names": " ".join(g["clip"]), "vx": round(vx, 3), "vy": round(vy, 3),
                     "speed": round(float(np.hypot(vx, vy)), 3),
                     "direction_spread_deg": round(spread, 1)})
        recs = g.to_dict("records")
        for a in range(len(recs)):
            for b in range(a + 1, len(recs)):
                ra, rb = recs[a], recs[b]
                d = abs((np.degrees(np.arctan2(ra["vx"], ra["vy"]) - np.arctan2(rb["vx"], rb["vy"]))
                         + 180) % 360 - 180)
                key = tuple(sorted((ra["clip"], rb["clip"])))
                pairs.setdefault(key, []).append((d, ra["speed"] / max(rb["speed"], 1e-6)
                                                  if key[0] == ra["clip"] else
                                                  rb["speed"] / max(ra["speed"], 1e-6)))
    if not rows:
        log("  overlap: no cell where two clips both show a clear flow")
        return {}
    cells = pd.DataFrame(rows)
    cells.to_csv(data / "overlap_cells.csv", index=False)
    pr = pd.DataFrame([{"clip_a": a, "clip_b": b, "shared_cells": len(v),
                        "median_angle_deg": round(float(np.median([d for d, _ in v])), 1),
                        "agree_share": round(float(np.mean([d <= AGREE_DEG for d, _ in v])), 2),
                        "median_speed_ratio": round(float(np.median([s for _, s in v])), 2)}
                       for (a, b), v in pairs.items()]).sort_values("shared_cells", ascending=False)
    pr.to_csv(data / "overlap_pairs.csv", index=False)
    # agreement layer: cyan = agree, amber = 30-90 deg apart, red = opposite
    W, H = size
    x0, x1, y0, y1 = ext
    layer = np.zeros((H, W, 4), np.uint8)
    half = max(1, int(round(cell / (x1 - x0) * W / 2)))
    for r in cells.itertuples():
        col = ((38, 182, 218) if r.direction_spread_deg <= AGREE_DEG else
               (240, 180, 60) if r.direction_spread_deg <= 90 else (230, 70, 60))
        cx = int((r.x - x0) / (x1 - x0) * W)
        cy = int((y1 - r.y) / (y1 - y0) * H)
        layer[max(0, cy - half):cy + half, max(0, cx - half):cx + half] = (*col, 200)
    _save(layer, images / "plan_agreement_layer.png")
    share = float((cells["direction_spread_deg"] <= AGREE_DEG).mean())
    log(f"  overlap: {len(cells)} cells seen by 2+ clips with a clear flow; "
        f"{share:.0%} agree within {AGREE_DEG:.0f} deg")
    return {"cells": len(cells), "agree_share": round(share, 2)}


def build_composite(clips: list[ClipRun], raster: GeoRaster | None, out_dir: Path,
                    title: str, subtitle: str = "", credit: str = "", plan_height_px: int = 3000,
                    log=print) -> Path:
    out_dir = Path(out_dir)
    images, labels, data, hou = (out_dir / d for d in ("images", "labels", "data", "houdini"))
    for d in (images, labels, data, hou):
        d.mkdir(parents=True, exist_ok=True)
    # ---- all points, track ids unique across clips
    parts = []
    for k, c in enumerate(clips):
        p = c.points.copy()
        p["clip"], p["quality"] = c.name, c.grade
        p["track_id"] = (k + 1) * 1_000_000 + p["track_id"].astype(int)
        parts.append(p)
    pts = pd.concat(parts, ignore_index=True)
    cfg = clips[0].cfg
    ranges = pf.group_ranges(cfg, pts)
    ext = pf.plan_extent(pts, margin_frac=0.06, aspect=(0.62, 1.6))
    W, H = pf.plan_size(ext, plan_height_px)
    lw = 1.2 * plan_height_px / 2000
    log(f"composite: {len(clips)} clips, {pts['track_id'].nunique()} tracks, "
        f"{len(pts)} samples, plan {W} x {H} px")

    # ---- trails
    L = pf.plan_layers(pts, raster, ext, plan_height_px, ranges, line_px=lw)
    base = L["base"]
    if L["aerial"] is not None:
        _save(L["aerial"], images / "plan_aerial.png")
    _save(base, images / "plan_map.png")
    _save(base + L["trails"], images / "plan_tracks.png")
    _save(_additive_to_rgba(L["trails"]), images / "plan_tracks_layer.png")

    # ---- flow field per group (each on its own speed scale)
    fc = float(cfg["field"]["cell_size_m"])
    for gname, rg in ranges.items():
        g = pts[pts["group"] == gname]
        if g["track_id"].nunique() < 5:
            continue
        fld = smooth_field(g, fc, float(cfg["field"]["smooth_m"]), float(g["dt"].median()))
        fld.to_csv(data / f"field_grid_{gname}.csv", index=False)
        fl = flow_layer(fld, ext, (W, H), fc, rg, line_px=1.5 * plan_height_px / 2000)
        _save(base + fl, images / f"plan_flowfield_{gname}.png")
        _save(_additive_to_rgba(fl), images / f"plan_flowfield_{gname}_layer.png")

    # ---- where views overlap: do the clips agree on the movement?
    if "people" in ranges:
        overlap_agreement(clips, pts, cfg, ext, (W, H), images, data, log)

    # ---- coverage (seconds in view) and density per observed second, on a 1 m grid
    x0, x1, y0, y1 = ext
    nx, ny = int(np.ceil((x1 - x0) / GRID_M)), int(np.ceil((y1 - y0) / GRID_M))
    grid = (x0, y0, nx, ny)
    seen = np.zeros((ny, nx))
    fp = {}
    for c in clips:
        m = footprint(c, grid)
        fp[c.name] = m
        seen += m * c.duration_s
    people = pts[(pts["group"] == "people") & ~pts["predicted"].astype(bool)]
    i = np.floor((people["x"].to_numpy() - x0) / GRID_M).astype(int)
    j = np.floor((people["y"].to_numpy() - y0) / GRID_M).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    occ = np.zeros((ny, nx))
    np.add.at(occ, (j[ok], i[ok]), people["dt"].to_numpy()[ok])          # person-seconds
    s = SMOOTH_M / GRID_M
    dens = np.where(gaussian_filter(seen, s) >= MIN_SEEN_S,
                    gaussian_filter(occ, s) / np.maximum(gaussian_filter(seen, s), 1e-9)
                    / (GRID_M * GRID_M), np.nan)                          # people per m^2
    dmax = float(np.nanpercentile(dens, 99.5)) if np.isfinite(dens).any() else 0.0

    def to_plan(a):                                   # grid (north = last row) -> plan pixels
        return cv2.resize(np.flipud(a).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    t = np.clip(np.nan_to_num(dens / max(dmax, 1e-9)), 0, 1) ** 0.5
    tp = to_plan(t).clip(0, 1)
    seen_p = to_plan((seen >= MIN_SEEN_S).astype(float)) > 0.5
    dl = np.dstack([_ramp(tp, DENSITY_STOPS), np.where(seen_p, np.clip(tp * 1.4 + 0.08, 0, 0.92)
                                                       * 255, 0)]).astype(np.uint8)
    _save(_over(base, dl), images / "plan_density.png")
    _save(dl, images / "plan_density_layer.png")
    smax = float(seen.max())
    tc = to_plan(np.sqrt(seen / max(smax, 1e-9))).clip(0, 1)
    cl = np.dstack([_ramp(tc, DENSITY_STOPS), np.where(tc > 0.01, 200, 0)]).astype(np.uint8)
    _save(_over(base, cl), images / "plan_coverage.png")
    _save(cl, images / "plan_coverage_layer.png")
    # outlines of every clip's view, with its number (labels kept apart from the image)
    fl_img = np.zeros((H, W, 4), np.uint8)
    num_img = np.zeros((H, W, 4), np.uint8)
    for k, (name, m) in enumerate(fp.items(), start=1):
        mp = (to_plan(m.astype(float)) > 0.5).astype(np.uint8)
        cs, _ = cv2.findContours(mp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(fl_img, cs, -1, (235, 235, 230, 220), max(2, int(lw * 1.5)), cv2.LINE_AA)
        cam = clips[k - 1].homography.camera_params
        q = (int((cam["E"] - x0) / (x1 - x0) * W), int((y1 - cam["N"]) / (y1 - y0) * H))
        cv2.circle(num_img, q, int(6 * lw), (255, 255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(num_img, str(k), (q[0] + int(9 * lw), q[1] + int(6 * lw)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * lw, (255, 255, 255, 255), max(1, int(lw)),
                    cv2.LINE_AA)
    _save(fl_img, images / "plan_footprints_layer.png")
    _save(num_img, labels / "camera_numbers_layer.png")

    # ---- labels
    for v in ("light", "dark"):
        for gname, (lo, hi) in ranges.items():
            legend_png(labels / f"legend_speed_{gname}_{v}.png", f"SPEED  ·  {gname.upper()}",
                       f"{lo:g}", f"{hi:g} m/s", pf.speed_rgb(np.linspace(lo, hi, 720), lo, hi), v)
        grad = _ramp(np.linspace(0, 1, 720), DENSITY_STOPS)
        legend_png(labels / f"legend_density_{v}.png", "PEOPLE PER 100 m²  ·  WHILE IN VIEW", "0",
                   f"{dmax * 100:.2g}", grad, v)
        legend_png(labels / f"legend_coverage_{v}.png", "SECONDS IN VIEW", "0", f"{smax:.0f} s",
                   grad, v)
        scale_bar_png(labels / f"scale_bar_{v}.png", pf._nice(x1 - x0), W / (x1 - x0), v)
        north_arrow_png(labels / f"north_arrow_{v}.png", v)
    (labels / "title.txt").write_text("\n".join(x for x in (title, subtitle, credit and
                                                            f"Imagery: {credit}") if x) + "\n",
                                      encoding="utf-8")
    with open(labels / "camera_numbers.txt", "w", encoding="utf-8") as f:
        for k, c in enumerate(clips, start=1):
            f.write(f"{k:2d}  {c.name:16} {c.grade:6} {c.duration_s:6.1f} s  {c.note}\n")

    # ---- data
    pts.drop(columns=["dt"]).to_csv(data / "points_all.csv", index=False)
    rows = []
    for k, c in enumerate(clips, start=1):
        cam = c.homography.camera_params
        g = c.points
        rows.append({"number": k, "clip": c.name, "quality": c.grade, "note": c.note,
                     "duration_s": round(c.duration_s, 1), "tracks": int(g["track_id"].nunique()),
                     "camera_E": round(cam["E"], 1), "camera_N": round(cam["N"], 1),
                     "area_in_view_m2": int(fp[c.name].sum() * GRID_M * GRID_M)})
    pd.DataFrame(rows).to_csv(data / "clips.csv", index=False)
    jj, ii = np.nonzero(seen > 0)
    pd.DataFrame({"x": x0 + (ii + 0.5) * GRID_M, "y": y0 + (jj + 0.5) * GRID_M,
                  "seconds_in_view": seen[jj, ii].round(1),
                  "people_per_m2": np.round(dens[jj, ii], 4)}).to_csv(
        data / "coverage_density_grid.csv", index=False)
    pl = pts[~pts["predicted"].astype(bool)]
    metrics = [("clips", "Clips", len(clips), ""),
               ("observed_s", "Footage, all clips", round(sum(c.duration_s for c in clips)), "s"),
               ("area_m2", "Ground seen", int((seen > 0).sum() * GRID_M * GRID_M), "m²"),
               ("tracks", "Tracks", int(pts["track_id"].nunique()), "")]
    for gname in ranges:
        g = pl[pl["group"] == gname]
        if len(g):
            metrics.append((f"{gname}_tracks", f"{gname} tracks", int(g["track_id"].nunique()), ""))
            metrics.append((f"{gname}_median_speed", f"{gname} median speed",
                            round(float(g["speed"].median()), 2), "m/s"))
    with open(data / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "unit", "label"])
        for key, label, value, unit in metrics:
            w.writerow([key, value, unit, label])

    # ---- houdini: the combined points with the usual import script
    from .exports import write_houdini_script

    pts.drop(columns=["dt"]).to_csv(hou / "points.csv", index=False)
    write_houdini_script(hou / "houdini_import.py", hou / "points.csv",
                         (round(float(np.mean(ext[:2]))), round(float(np.mean(ext[2:])))), cfg,
                         z0=clips[0].homography.ground_z_m)
    log(f"composite -> {out_dir}")
    return out_dir
