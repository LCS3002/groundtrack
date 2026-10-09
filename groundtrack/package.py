"""Frameless images, separate labels, the metrics and the complete plate of a run.

Written into the run folder (see layout.py):
    <site>_plate.png      the complete plate
    images/   insitu_frame, insitu_tracks, plan_aerial, plan_map, plan_tracks, plan_flowfield,
              plan_density: pure images, nothing around the data (no title, frame, axes,
              legend or scale bar). <name>_layer.png = the drawing alone on transparency.
    labels/   legend_speed, legend_density, scale_bar, north_arrow (each _light for dark
              backgrounds and _dark for light ones, transparent), title.txt
    data/     metrics.csv (key figures), metrics.json (figures + all stats + plan geometry)

Every plan_* image and layer shares one extent and size, so they stack exactly. The scale
bar is drawn at that pixel scale: scale it together with the plan images.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import portfolio as pf
from .config import Config
from .geo import GeoRaster
from .layout import open_run

# occupancy ramp: deep blue -> cyan -> white (light on dark)
DENSITY_STOPS = [(0.0, (22, 48, 110)), (0.55, (38, 182, 218)), (1.0, (244, 250, 255))]
INK_VARIANTS = {"light": ((236, 236, 232), (150, 150, 154)),
                "dark": ((22, 22, 24), (96, 96, 100))}


def _ramp(t, stops):
    t = np.clip(np.asarray(t, float), 0, 1)
    xs = [s[0] for s in stops]
    return np.stack([np.interp(t, xs, [s[1][i] for s in stops]) for i in range(3)], axis=-1)


def _additive_to_rgba(layer: np.ndarray) -> np.ndarray:
    """Additive light layer (float RGB) -> straight-alpha RGBA that looks the same over black."""
    rgb = np.clip(layer, 0, 255)
    a = rgb.max(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        col = np.where(a[..., None] > 0, rgb / a[..., None] * 255, 0)
    return np.dstack([col, a]).clip(0, 255).astype(np.uint8)


def _over(base: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.float32) / 255
    return base.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a


def _save(img: np.ndarray, path: Path) -> Path:
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path, optimize=True)
    return path


# --------------------------------------------------------------------------- layers
def flow_layer(field: pd.DataFrame, ext, size, cell: float, speed_range,
               min_confidence: float = 0.5, min_dominance: float = 0.6,
               line_px: float = 1.6) -> np.ndarray:
    """Streamlines of the vector field as an additive light layer (float RGB).

    Drawn only where the field is well supported (confidence) and coherent (dominance:
    most movement in one direction); where streams cross, lines would invent swirls."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    from .field import field_to_arrays

    W, H = size
    if field.empty:
        return np.zeros((H, W, 3), np.float32)
    xs, ys, U, V, S, C = field_to_arrays(field, cell)
    weak = ~(C >= min_confidence) | ~np.isfinite(U) | ~np.isfinite(V)
    if "dominance" in field:
        D = np.ones_like(C)
        D[field["j"].to_numpy() - int(field["j"].min()),
          field["i"].to_numpy() - int(field["i"].min())] = field["dominance"].to_numpy()
        weak |= D < min_dominance
    Um = np.ma.array(np.nan_to_num(U), mask=weak)
    Vm = np.ma.array(np.nan_to_num(V), mask=weak)
    lo, hi = speed_range
    cmap = LinearSegmentedColormap.from_list(
        "groundtrack", [(t, np.array(c) / 255) for t, c in pf.STOPS])
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    pt = 0.72 * line_px                                   # px -> pt at 100 dpi
    if len(xs) > 1 and len(ys) > 1 and (~weak).sum() > 4:
        ax.streamplot(xs, ys, Um, Vm, color=np.nan_to_num(S), cmap=cmap, norm=Normalize(lo, hi),
                      linewidth=pt * (0.45 + 0.9 * np.clip(C, 0, 1)),
                      density=float(np.clip(max(W, H) / 1000, 1.2, 4.0)),
                      arrowsize=0.9 * line_px, arrowstyle="-|>", minlength=0.12)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), np.float32).copy()
    plt.close(fig)
    if rgba.shape[:2] != (H, W):
        rgba = cv2.resize(rgba, (W, H), interpolation=cv2.INTER_AREA)
    lit = rgba[..., :3] * (rgba[..., 3:4] / 255)          # premultiplied = light on black
    return lit + 0.7 * cv2.GaussianBlur(lit, (0, 0), sigmaX=max(2.0, line_px * 2.5))


def density_layer(points: pd.DataFrame, ext, size, dt: float,
                  include_predicted: bool = False) -> tuple[np.ndarray, float]:
    """Occupancy (object-seconds per m^2) as RGBA, and its maximum."""
    from scipy.ndimage import gaussian_filter

    W, H = size
    p = points if include_predicted else points[~points["predicted"].astype(bool)]
    x0, x1, y0, y1 = ext
    span = max(x1 - x0, y1 - y0)
    res = max(0.25, span / 800)                       # metres per density cell
    sigma_m = max(0.5, 2 * res)
    xb = np.linspace(x0, x1, max(2, int(round((x1 - x0) / res))) + 1)
    yb = np.linspace(y0, y1, max(2, int(round((y1 - y0) / res))) + 1)
    Hh, _, _ = np.histogram2d(p["x"], p["y"], bins=[xb, yb])
    D = gaussian_filter(Hh.T * dt, sigma_m / res) / (res * res)
    dmax = float(D.max()) if D.size else 0.0
    if dmax <= 0:
        return np.zeros((H, W, 4), np.uint8), 0.0
    t = np.clip(D / dmax, 0, 1) ** 0.45    # gamma, not log: shapes stay crisp, tails stay dark
    t = cv2.resize(np.flipud(t).astype(np.float32), (W, H),      # north up, plan size
                   interpolation=cv2.INTER_CUBIC).clip(0, 1)
    rgb = _ramp(t, DENSITY_STOPS)
    a = np.clip(t * 1.4, 0, 0.92) * 255
    return np.dstack([rgb, a]).astype(np.uint8), dmax


# --------------------------------------------------------------------------- labels
def _canvas(w, h):
    img = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def legend_png(path: Path, caption: str, lo_text: str, hi_text: str, gradient: np.ndarray,
               variant: str, width: int = 720) -> Path:
    ink, grey = INK_VARIANTS[variant]
    img, d = _canvas(width + 8, 132)
    pf._text(d, (4, 4), caption, 26, "SemiLight", grey, tracking=0.3)
    bar = np.repeat(gradient[None].astype(np.uint8), 8, axis=0)
    img.paste(Image.fromarray(bar).convert("RGBA"), (4, 62))
    pf._text(d, (4, 86), lo_text, 28, "Light", ink)
    hw = d.textlength(hi_text, font=pf._font(28))
    pf._text(d, (4 + width - hw, 86), hi_text, 28, "Light", ink)
    img.save(path)
    return path


def scale_bar_png(path: Path, metres: float, px_per_m: float, variant: str) -> Path:
    ink, _ = INK_VARIANTS[variant]
    L = metres * px_per_m
    img, d = _canvas(L + 12, 90)
    y = 70
    d.line([(6, y), (6 + L, y)], fill=ink, width=3)
    for x in (6, 6 + L):
        d.line([(x, y - 14), (x, y + 14)], fill=ink, width=3)
    pf._text(d, (6, 6), f"{metres:g} M", 28, "SemiLight", ink, tracking=0.25)
    img.save(path)
    return path


def north_arrow_png(path: Path, variant: str) -> Path:
    ink, _ = INK_VARIANTS[variant]
    img, d = _canvas(60, 170)
    cx = 30
    d.line([(cx, 120), (cx, 26)], fill=ink, width=3)
    d.polygon([(cx, 4), (cx - 13, 32), (cx + 13, 32)], fill=ink)
    pf._text(d, (cx - 11, 130), "N", 30, "SemiLight", ink)
    img.save(path)
    return path


# --------------------------------------------------------------------------- metrics
def key_figures(stats: dict, summary: pd.DataFrame, groups, cam: dict | None,
                rmse_m: float | None) -> list[tuple[str, str, object, str]]:
    """(metric, label, value, unit) rows: the figures printed on the plate, and a few more.

    With several groups (people and vehicles in one clip) the per-track figures are given
    per group, as <metric>_<group>."""
    groups = [groups] if isinstance(groups, str) else list(groups)
    rows = [("tracks", "Tracks", int(len(summary)), "count")]
    for g in groups:
        sel = summary[summary["group"] == g] if "group" in summary else summary
        if not len(sel):
            continue
        sfx, lab = ("", "") if len(groups) == 1 else (f"_{g}", f" ({g})")
        if sfx:
            rows.append((f"tracks{sfx}", f"Tracks{lab}", int(len(sel)), "count"))
        med = float(sel["mean_speed"].median())
        rows.append((f"median_speed{sfx}", f"Median speed{lab}", round(med, 2), "m/s"))
        if g == "vehicles":
            rows.append((f"median_speed_kmh{sfx}", f"Median speed{lab}", round(med * 3.6, 1),
                         "km/h"))
        rows.append((f"mean_speed{sfx}", f"Mean speed{lab}",
                     round(float(sel["mean_speed"].mean()), 2), "m/s"))
        rows.append((f"median_straightness{sfx}", f"Straightness{lab}",
                     round(float(sel["straightness"].median()), 3), "0..1"))
        if "path_length_m" in sel:
            rows.append((f"median_path_length{sfx}", f"Median path length{lab}",
                         round(float(sel["path_length_m"].median()), 1), "m"))
    for cl in stats.get("count_lines", []):
        rows.append((f"flow_{cl['name']}", f"Flow {cl['name']}", cl["flow_per_min"], "per min"))
        rows.append((f"crossings_{cl['name']}_l2r", f"{cl['name']} left to right",
                     cl["crossings_left_to_right"], "count"))
        rows.append((f"crossings_{cl['name']}_r2l", f"{cl['name']} right to left",
                     cl["crossings_right_to_left"], "count"))
    w = stats.get("window_s") or [0, 0]
    rows.append(("duration", "Duration", round(float(w[1] - w[0]), 1), "s"))
    rows.append(("predicted_share", "Bridged by prediction", stats.get("predicted_share"), "0..1"))
    if cam:
        rows.append(("camera_height", "Camera height", round(float(cam["height_m"]), 1), "m"))
        rows.append(("camera_hfov", "Camera field of view", round(float(cam["hfov_deg"]), 1),
                     "deg"))
        rows.append(("camera_looking", "Camera looking", pf._compass(cam["yaw_deg"]), ""))
    if rmse_m is not None and np.isfinite(rmse_m):
        rows.append(("calibration_rmse", "Calibration error (RMSE)", round(float(rmse_m), 2), "m"))
    chk = stats.get("calibration_check")
    if chk:
        rows.append(("implied_person_height", "People height check",
                     chk["implied_person_height_m"], "m"))
    return [r for r in rows if r is not None]


# --------------------------------------------------------------------------- build
def plate_text(cfg: Config) -> dict:
    pk = cfg.get("package") or {}
    return {"title": pk.get("title") or cfg.site, "index": str(pk.get("index") or ""),
            "subtitle": pk.get("subtitle") or "",
            "date": pk.get("date") or date.today().strftime("%d.%m.%Y"),
            "credit": pk.get("credit") or ""}


def build_package(cfg: Config, run_dir: Path, raster: GeoRaster | None = None,
                  video: Path | None = None, log=print, plate: bool = True) -> Path:
    """Write the frameless images, labels, metrics and plate (module docstring) of a run."""
    run = open_run(run_dir).make()
    pk = cfg.get("package") or {}
    images, labels = run.images, run.labels
    meta = json.loads(run.meta.read_text(encoding="utf-8"))
    points = pd.read_csv(run.points)
    summary = pd.read_csv(run.tracks)
    stats = json.loads(run.stats_json.read_text(encoding="utf-8"))
    if points.empty:
        log("images / labels: no tracks, nothing to draw")
        return run.root
    ranges = pf.group_ranges(cfg, points)
    groups = list(ranges)
    fps, stride = float(meta["fps"]), int(meta["vid_stride"])
    hom = {}
    if run.homography_used.exists():
        hom = json.loads(run.homography_used.read_text(encoding="utf-8"))
    cam = hom.get("camera_params")
    variants = list(pk.get("label_variants") or ["light", "dark"])

    # ---- plan images: one extent, one size
    ext = pf.plan_extent(points)
    plan_h = int(pk.get("plan_height_px", 2400))
    W, H = pf.plan_size(ext, plan_h)
    lw = 1.3 * plan_h / 2000
    L = pf.plan_layers(points, raster, ext, plan_h, ranges, line_px=lw)
    base = L["base"]
    if L["aerial"] is not None:
        _save(L["aerial"], images / "plan_aerial.png")
    _save(base, images / "plan_map.png")
    _save(base + L["trails"], images / "plan_tracks.png")
    _save(_additive_to_rgba(L["trails"]), images / "plan_tracks_layer.png")
    # one field per group (each on its own speed scale), or the single "all" field
    fields = [(run.field_csv(gn), rg) for gn, rg in ranges.items() if len(ranges) > 1
              and run.field_csv(gn).exists()] or [(run.field_csv(), next(iter(ranges.values())))]
    if all(f.exists() for f, _ in fields):
        flow = sum(flow_layer(pd.read_csv(f), ext, (W, H), float(cfg["field"]["cell_size_m"]),
                              rg, line_px=1.6 * plan_h / 2000) for f, rg in fields)
        _save(base + flow, images / "plan_flowfield.png")
        _save(_additive_to_rgba(flow), images / "plan_flowfield_layer.png")
    dens, dmax = density_layer(points, ext, (W, H), stride / fps)
    _save(_over(base, dens), images / "plan_density.png")
    _save(dens, images / "plan_density_layer.png")

    # ---- in situ (needs the footage)
    video = Path(video or meta.get("video", ""))
    insitu = None
    if video.exists():
        insitu = pf.insitu_layers(video, run.root, cfg, fps, int(meta["height"]),
                                  line_px=1.6 * int(meta["height"]) / 1400)
        _save(insitu["frame_rgb"], images / "insitu_frame.png")
        _save(insitu["base"] + insitu["trails"], images / "insitu_tracks.png")
        _save(_additive_to_rgba(insitu["trails"]), images / "insitu_tracks_layer.png")
    else:
        log(f"images: footage {video} not found, in-situ images skipped")

    # ---- labels (one speed legend per group when several are present)
    den_grad = _ramp(np.linspace(0, 1, 720), DENSITY_STOPS)
    px_per_m = W / (ext[1] - ext[0])
    bar_m = pf._nice(ext[1] - ext[0])
    for v in variants:
        for gname, (lo, hi) in ranges.items():
            spd_grad = pf.speed_rgb(np.linspace(lo, hi, 720), lo, hi)
            hi_txt = f"{hi:g} m/s" + (f"  ·  {hi * 3.6:.0f} km/h" if gname == "vehicles" else "")
            if len(ranges) == 1:
                legend_png(labels / f"legend_speed_{v}.png", "SPEED", f"{lo:g}", hi_txt,
                           spd_grad, v)
            else:
                legend_png(labels / f"legend_speed_{gname}_{v}.png", f"SPEED  ·  {gname.upper()}",
                           f"{lo:g}", hi_txt, spd_grad, v)
        legend_png(labels / f"legend_density_{v}.png", "OCCUPANCY", "low",
                   f"{dmax:.2g} s / m²", den_grad, v)
        scale_bar_png(labels / f"scale_bar_{v}.png", bar_m, px_per_m, v)
        north_arrow_png(labels / f"north_arrow_{v}.png", v)

    figs = key_figures(stats, summary, groups, cam, hom.get("rmse_m"))
    with open(run.metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "unit", "label"])
        for key, label, value, unit in figs:
            w.writerow([key, value, unit, label])
    text = plate_text(cfg)
    if hom.get("terrain"):
        from .terrain import CREDIT

        text["credit"] = "; ".join(x for x in (text["credit"], CREDIT) if x)
    (labels / "title.txt").write_text(
        "\n".join(x for x in (text["title"] + (f"  {text['index']}" if text["index"] else ""),
                              text["subtitle"], text["date"],
                              f"Imagery: {text['credit']}" if text["credit"] else "") if x) + "\n",
        encoding="utf-8")
    info = {
        "site": cfg.site, "title": text,
        "figures": {k: {"value": v, "unit": u, "label": lab} for k, lab, v, u in figs},
        "plan": {"epsg": 27700, "extent_m": {"west": ext[0], "east": ext[1], "south": ext[2],
                                             "north": ext[3]},
                 "size_px": [W, H], "px_per_m": round(px_per_m, 5), "scale_bar_m": bar_m,
                 "density_max_s_per_m2": round(dmax, 4)},
        "insitu": ({"frame": insitu["frame"], "size_px": list(insitu["frame_rgb"].shape[1::-1])}
                   if insitu else None),
        "speed_range_m_s": {gn: list(rg) for gn, rg in ranges.items()},
        "camera": cam,
        "stats": stats,
    }
    run.metrics_json.write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")

    # ---- the complete plate
    if plate and insitu is not None:
        pf.make_plate(cfg, run.root, raster, run.plate(cfg.site), text["title"], text["index"],
                      text["subtitle"], text["date"], credit=text["credit"])
    log(f"images (frameless), labels, metrics and plate -> {run.root}")
    return run.root
