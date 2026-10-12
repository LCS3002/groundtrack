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
houdini/ the combined points and fields with the import / field scripts, rhino/ a .3dm (tracks,
field arrows, cameras and view outlines) and the Grasshopper script.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from . import portfolio as pf
from .config import Config
from .field import smooth_field
from .geo import GeoRaster
from .homography import Homography
from .layout import open_run
from .package import (_over, _save, coverage_rgb, density_rgb, flow_layer, group_key_png,
                      legend_png,
                      north_arrow_png, scale_bar_png)
from .colors import group_rgb
from .trajectories import max_range_for

GRID_M = 1.0                 # composite grid cell (coverage, density)
SMOOTH_M = 1.5               # density smoothing
MIN_SEEN_S = 5.0             # cells watched for less than this stay blank
FAR_CAP_M = 150.0            # a footprint never reaches further than this ...
FAR_CAP_PER_HEIGHT = 10.0    # ... or 10x the camera height (a high-rise view sees far)


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
    cap = max(FAR_CAP_M, FAR_CAP_PER_HEIGHT * abs(float(cam.get("height_m") or 0)))
    groups = (clip.points["group"].unique() if "group" in clip.points else None)
    groups = list(groups) if groups is not None and len(groups) else ["people"]
    ranges = [max_range_for(clip.cfg, h, g) or cap for g in groups]
    rng = min(max(ranges), cap)
    ok = np.isfinite(d) & (d <= rng)
    m = np.zeros((ny, nx), np.uint8)
    if ok.sum() >= 3:
        # the ground a camera sees of a plane is convex (the picture's rectangle through a
        # homography, cut by the range circle): fill the hull of the samples, which thin out
        # with distance, instead of leaving rows of dots far away
        q = np.column_stack([(g[ok, 0] - x0) / GRID_M, (g[ok, 1] - y0) / GRID_M])
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(q).astype(np.int32)), 1)
    m = m.astype(bool)
    cx, cy = (cam["E"] - x0) / GRID_M, (cam["N"] - y0) / GRID_M
    jj, ii = np.mgrid[0:ny, 0:nx]
    return m & (np.hypot(ii + 0.5 - cx, jj + 0.5 - cy) * GRID_M <= rng)


AGREE_DEG = 30.0             # flows within this angle count as agreeing


def overlap_agreement(clips: list[ClipRun], pts: pd.DataFrame, cfg: Config, ext, size,
                      images: Path, group: str = "people", log=print
                      ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Compare the clips' own flow fields of one group where their views overlap.

    Each clip gets its field on one shared grid; a cell counts where two or more clips have
    a well-supported, clearly directional flow there (confidence >= 0.5, dominance >= 0.6).
    Returns (cells: clips, combined vector, direction spread per cell; pairs: shared cells,
    median angle and speed ratio per pair of clips) - a pair that disagrees everywhere it
    overlaps points at a calibration, though different moments can also move differently.
    plan_agreement_<group>_layer.png colours the cells."""
    cell = float(cfg["field"]["cell_size_m"])
    sm = float(cfg["field"]["smooth_m"])
    people = pts[(pts["group"] == group) & ~pts["predicted"].astype(bool)]
    if people.empty:
        return None
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
        return None
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
        log(f"  overlap ({group}): no cell where two clips both show a clear flow")
        return None
    cells = pd.DataFrame(rows)
    cells.insert(0, "group", group)
    pr = pd.DataFrame([{"group": group, "clip_a": a, "clip_b": b, "shared_cells": len(v),
                        "median_angle_deg": round(float(np.median([d for d, _ in v])), 1),
                        "agree_share": round(float(np.mean([d <= AGREE_DEG for d, _ in v])), 2),
                        "median_speed_ratio": round(float(np.median([s for _, s in v])), 2)}
                       for (a, b), v in pairs.items()]).sort_values("shared_cells", ascending=False)
    # agreement layer: green = agree, amber = 30-90 deg apart, red = opposite
    W, H = size
    x0, x1, y0, y1 = ext
    layer = np.zeros((H, W, 4), np.uint8)
    half = max(1, int(round(cell / (x1 - x0) * W / 2)))
    for r in cells.itertuples():
        col = ((40, 160, 70) if r.direction_spread_deg <= AGREE_DEG else
               (240, 170, 40) if r.direction_spread_deg <= 90 else (215, 48, 39))
        cx = int((r.x - x0) / (x1 - x0) * W)
        cy = int((y1 - r.y) / (y1 - y0) * H)
        layer[max(0, cy - half):cy + half, max(0, cx - half):cx + half] = (*col, 200)
    _save(layer, images / f"plan_agreement_{group}_layer.png")
    share = float((cells["direction_spread_deg"] <= AGREE_DEG).mean())
    log(f"  overlap ({group}): {len(cells)} cells seen by 2+ clips with a clear flow; "
        f"{share:.0%} agree within {AGREE_DEG:.0f} deg")
    return cells, pr


MASTER_ORDER = ("people", "cycles", "buses", "vehicles")    # drawn bottom -> top


def master_field(pts: pd.DataFrame, groups, cell: float, smooth_m: float) -> pd.DataFrame:
    """Every group's flow on one grid: the pooled velocity of everything that moves (vx, vy,
    speed, ... as in any vector field, weighted by how much each group was seen there), plus
    per-group columns <group>_vx, _vy, _speed, _density, _confidence, _weight and the
    dominant_group (most observed) of each cell."""
    p = pts[~pts["predicted"].astype(bool) & pts["group"].isin(groups)]
    x, y = p["x"].to_numpy(), p["y"].to_numpy()
    m = 3 * smooth_m
    bounds = (int(np.floor((x.min() - m) / cell)), int(np.floor((x.max() + m) / cell)),
              int(np.floor((y.min() - m) / cell)), int(np.floor((y.max() + m) / cell)))
    out = smooth_field(p, cell, smooth_m, float(p["dt"].median()), bounds=bounds)
    out = out.set_index(["i", "j"])
    weights = []
    for g in groups:
        sel = p[p["group"] == g]
        f = smooth_field(sel, cell, smooth_m, float(sel["dt"].median()), bounds=bounds)
        cols = ["vx", "vy", "speed", "density", "confidence", "weight"]
        out = out.join(f.set_index(["i", "j"])[cols].add_prefix(f"{g}_"), how="outer")
        weights.append(f"{g}_weight")
    out = out.reset_index()
    out["cell_x"] = (out["i"] + 0.5) * cell
    out["cell_y"] = (out["j"] + 0.5) * cell
    w = out[weights].fillna(0.0).to_numpy()
    out["dominant_group"] = np.where(w.max(axis=1) > 0,
                                     np.array(groups, dtype=object)[w.argmax(axis=1)], "")
    return out


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
    _save(_over(base, L["trails"]), images / "plan_tracks.png")
    _save(L["trails"], images / "plan_tracks_layer.png")

    # ---- flow field per group (each on its own speed scale)
    fc = float(cfg["field"]["cell_size_m"])
    field_files = {}
    for gname, rg in ranges.items():
        g = pts[pts["group"] == gname]
        if g["track_id"].nunique() < 5:
            continue
        fld = smooth_field(g, fc, float(cfg["field"]["smooth_m"]), float(g["dt"].median()))
        fld.to_csv(data / f"field_grid_{gname}.csv", index=False)
        field_files[gname] = hou / f"vector_field_{gname}.csv"     # for Houdini / Grasshopper
        fld.to_csv(field_files[gname], index=False)
        fl = flow_layer(fld, ext, (W, H), fc, rg, line_px=1.5 * plan_height_px / 2000)
        _save(_over(base, fl), images / f"plan_flowfield_{gname}.png")
        _save(fl, images / f"plan_flowfield_{gname}_layer.png")

    # ---- master field: every group on one grid; image = each group's lines in its colour
    master_groups = []
    if len(field_files) >= 2:
        master_groups = sorted(field_files, key=lambda g: (MASTER_ORDER.index(g)
                                                           if g in MASTER_ORDER else 9, g))
        mf = master_field(pts, master_groups, fc, float(cfg["field"]["smooth_m"]))
        mf.to_csv(data / "field_master.csv", index=False)
        mf.to_csv(hou / "vector_field_master.csv", index=False)
        layer = None
        for gname in master_groups:
            fl = flow_layer(pd.read_csv(field_files[gname]), ext, (W, H), fc, ranges[gname],
                            line_px=1.5 * plan_height_px / 2000, colour=group_rgb(gname))
            layer = fl if layer is None else pf.stack_rgba(layer, fl)
        _save(_over(base, layer), images / "plan_flowfield_master.png")
        _save(layer, images / "plan_flowfield_master_layer.png")
        log(f"  master field: {', '.join(master_groups)} on one grid ({len(mf)} cells)")

    # ---- where views overlap: do the clips agree on the movement? (per group)
    found = [r for r in (overlap_agreement(clips, pts, cfg, ext, (W, H), images, g, log)
                         for g in ranges) if r is not None]
    if found:
        pd.concat([c for c, _ in found]).to_csv(data / "overlap_cells.csv", index=False)
        pd.concat([p for _, p in found]).to_csv(data / "overlap_pairs.csv", index=False)

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
    s = SMOOTH_M / GRID_M

    def to_plan(a):                                   # grid (north = last row) -> plan pixels
        return cv2.resize(np.flipud(a).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    seen_p = to_plan((seen >= MIN_SEEN_S).astype(float)) > 0.5
    dens, dmax = {}, {}
    for gname in ranges:
        g = pts[(pts["group"] == gname) & ~pts["predicted"].astype(bool)]
        if g.empty:
            continue
        i = np.floor((g["x"].to_numpy() - x0) / GRID_M).astype(int)
        j = np.floor((g["y"].to_numpy() - y0) / GRID_M).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        occ = np.zeros((ny, nx))
        np.add.at(occ, (j[ok], i[ok]), g["dt"].to_numpy()[ok])          # e.g. person-seconds
        d = np.where(gaussian_filter(seen, s) >= MIN_SEEN_S,
                     gaussian_filter(occ, s) / np.maximum(gaussian_filter(seen, s), 1e-9)
                     / (GRID_M * GRID_M), np.nan)                        # e.g. people per m^2
        dens[gname] = d
        dmax[gname] = float(np.nanpercentile(d, 99.5)) if np.isfinite(d).any() else 0.0
        t = np.clip(np.nan_to_num(d / max(dmax[gname], 1e-9)), 0, 1) ** 0.5
        tp = to_plan(t).clip(0, 1)
        dl = np.dstack([density_rgb(tp),
                        np.where(seen_p, np.clip(tp * 1.6 + 0.06, 0, 0.88) * 255, 0)]
                       ).astype(np.uint8)
        _save(_over(base, dl), images / f"plan_density_{gname}.png")
        _save(dl, images / f"plan_density_{gname}_layer.png")
    smax = float(seen.max())
    tc = to_plan(np.sqrt(seen / max(smax, 1e-9))).clip(0, 1)
    cl = np.dstack([coverage_rgb(tc), np.where(tc > 0.01, 170, 0)]).astype(np.uint8)
    _save(_over(base, cl), images / "plan_coverage.png")
    _save(cl, images / "plan_coverage_layer.png")
    # outlines of every clip's view, with its number (labels kept apart from the image)
    fl_img = np.zeros((H, W, 4), np.uint8)
    num_img = np.zeros((H, W, 4), np.uint8)
    for k, (name, m) in enumerate(fp.items(), start=1):
        mp = (to_plan(m.astype(float)) > 0.5).astype(np.uint8)
        cs, _ = cv2.findContours(mp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(fl_img, cs, -1, (*pf.INK, 220), max(2, int(lw * 1.5)), cv2.LINE_AA)
        cam = clips[k - 1].homography.camera_params
        q = (int((cam["E"] - x0) / (x1 - x0) * W), int((y1 - cam["N"]) / (y1 - y0) * H))
        tq = (q[0] + int(9 * lw), q[1] + int(6 * lw))
        cv2.circle(num_img, q, int(8 * lw), (255, 255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(num_img, q, int(6 * lw), (*pf.INK, 255), -1, cv2.LINE_AA)
        for col, th in (((255, 255, 255, 255), max(3, int(4 * lw))), ((*pf.INK, 255),
                                                                       max(1, int(lw)))):
            cv2.putText(num_img, str(k), tq, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * lw, col, th,
                        cv2.LINE_AA)
    _save(fl_img, images / "plan_footprints_layer.png")
    _save(num_img, labels / "camera_numbers_layer.png")

    # ---- labels
    for v in ("light", "dark"):
        for gname, (lo, hi) in ranges.items():
            legend_png(labels / f"legend_speed_{gname}_{v}.png", f"SPEED  ·  {gname.upper()}",
                       f"{lo:g}", f"{hi:g} m/s", pf.speed_rgb(np.linspace(lo, hi, 720), lo, hi), v)
        u = np.linspace(0, 1, 720)
        for gname, dm in dmax.items():
            legend_png(labels / f"legend_density_{gname}_{v}.png",
                       f"{gname.upper()} PER 100 m²  ·  WHILE IN VIEW", "0", f"{dm * 100:.2g}",
                       density_rgb(u ** 0.5), v)
        legend_png(labels / f"legend_coverage_{v}.png", "SECONDS IN VIEW", "0", f"{smax:.0f} s",
                   coverage_rgb(u ** 0.5), v)
        if master_groups:
            group_key_png(labels / f"legend_flow_master_{v}.png", "FLOW  ·  ALL GROUPS",
                          master_groups, v)
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
                  **{f"{g}_per_m2": np.round(d[jj, ii], 4) for g, d in dens.items()}}).to_csv(
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

    # ---- houdini + rhino: the combined points and fields with the usual scripts
    from .exports import write_houdini_field_script, write_houdini_script
    from .rhino import write_3dm, write_grasshopper_script

    origin = (round(float(np.mean(ext[:2]))), round(float(np.mean(ext[2:]))))
    z0 = clips[0].homography.ground_z_m
    allp = pts.drop(columns=["dt"])
    allp.to_csv(hou / "points.csv", index=False)
    write_houdini_script(hou / "houdini_import.py", hou / "points.csv", origin, cfg, z0=z0)
    if field_files:
        groups_field = dict(field_files)
        groups_field["all"] = (hou / "vector_field_master.csv" if master_groups
                               else next(iter(field_files.values())))   # default field
        write_houdini_field_script(hou / "houdini_field.py", groups_field, {},
                                   hou / "points.csv", origin, fc, cfg)
    rh = out_dir / "rhino"
    dur = float(allp.groupby("clip")["time_s"].agg(lambda s: s.max() - s.min()).max()) if len(allp) else 0.0
    gh_fields = dict(field_files)
    if master_groups:
        gh_fields["all"] = hou / "vector_field_master.csv"           # the default field
    write_grasshopper_script(rh / "grasshopper_import.py", hou / "points.csv", gh_fields,
                             origin, z0, cfg, fc, dur)
    try:
        cams = [(c.name, c.homography.camera_params["E"], c.homography.camera_params["N"],
                 c.homography.camera_params.get("height_m", 0.0)) for c in clips]
        rings = []
        for name, mk in fp.items():          # each view's outline, back to metres
            cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cs:
                q = cnt.reshape(-1, 2).astype(float)
                if len(q) >= 3:
                    rings.append((name, [(x0 + (u + 0.5) * GRID_M, y0 + (v + 0.5) * GRID_M)
                                         for u, v in q]))
        fl = {k: pd.read_csv(f) for k, f in field_files.items()}
        if master_groups:
            fl["master"] = pd.read_csv(hou / "vector_field_master.csv")
        write_3dm(rh / f"{out_dir.name}.3dm", allp, fl, origin, z0, cfg, fc, cams, rings, title)
    except ImportError:
        log("rhino: rhino3dm not installed (pip install rhino3dm), .3dm skipped")
    log(f"composite -> {out_dir}")
    return out_dir
