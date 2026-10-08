"""Stage 6: topdown.png, density.png, speed_histogram.png."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from .colors import cmap  # noqa: E402
from .config import Config  # noqa: E402
from .geo import GeoRaster  # noqa: E402

GROUP_LABELS = {"people": "People", "cycles": "Cycles", "vehicles": "Vehicles"}
INK = "#1d1d1f"


# --------------------------------------------------------------------------- helpers
def _extent(points: pd.DataFrame, raster: GeoRaster | None, cfg: Config):
    vis = cfg["visuals"]
    if (vis["extent"] == "geotiff" or points.empty) and raster is not None:
        return raster.left, raster.right, raster.bottom, raster.top
    m = float(vis["margin_m"])
    x0, x1 = points["x"].min() - m, points["x"].max() + m
    y0, y1 = points["y"].min() - m, points["y"].max() + m
    if raster is not None:  # don't run off the map
        x0, x1 = max(x0, raster.left), min(x1, raster.right)
        y0, y1 = max(y0, raster.bottom), min(y1, raster.top)
    return x0, x1, y0, y1


def _basemap(ax, raster: GeoRaster | None, ext, brightness: float):
    if raster is None:
        ax.set_facecolor("#2b2b2e")
        return
    x0, x1, y0, y1 = ext
    c0, r0 = raster.world_to_pixel([[x0, y1]])[0]
    c1, r1 = raster.world_to_pixel([[x1, y0]])[0]
    c0, r0 = max(int(np.floor(c0)), 0), max(int(np.floor(r0)), 0)
    c1 = min(int(np.ceil(c1)), raster.image.shape[1])
    r1 = min(int(np.ceil(r1)), raster.image.shape[0])
    if c1 <= c0 or r1 <= r0:
        ax.set_facecolor("#2b2b2e")
        return
    crop = raster.image[r0:r1, c0:c1].astype(float) / 255
    gray = crop @ np.array([0.299, 0.587, 0.114])
    img = np.clip(gray * brightness, 0, 1)
    (wl, wt), (wr, wb) = raster.pixel_to_world([[c0, r0], [c1, r1]])
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=(wl, wr, wb, wt),
              interpolation="bilinear", zorder=0)


def _figure(ext, n_legends: int, long_side_in: float = 12.0):
    w, h = ext[1] - ext[0], ext[3] - ext[2]
    legend_w = 1.4 * max(n_legends, 1)
    if w >= h:
        fw, fh = long_side_in, long_side_in * h / w
    else:
        fw, fh = long_side_in * w / h, long_side_in
    fh = max(fh, 4.5)
    fig = plt.figure(figsize=(fw + legend_w, fh + 0.9))
    ax = fig.add_axes([0.02, 0.04, fw / (fw + legend_w) - 0.03, (fh) / (fh + 0.9) - 0.02])
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.6)
    return fig, ax


def _nice_length(span: float) -> float:
    target = span / 5
    for v in (0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000):
        if v >= target:
            return v
    return 10000.0


def scale_bar(ax, ext):
    x0, x1, y0, y1 = ext
    L = _nice_length(x1 - x0)
    w, h = x1 - x0, y1 - y0
    bx, by = x0 + 0.04 * w, y0 + 0.05 * h
    bh = 0.012 * h
    ax.add_patch(Rectangle((bx - 0.01 * w, by - 0.035 * h), L + 0.02 * w, 0.075 * h,
                           facecolor="white", alpha=0.85, edgecolor="none", zorder=5))
    for k in range(4):
        ax.add_patch(Rectangle((bx + k * L / 4, by), L / 4, bh,
                               facecolor=INK if k % 2 == 0 else "white",
                               edgecolor=INK, linewidth=0.6, zorder=6))
    ax.text(bx, by - 0.008 * h, "0", ha="center", va="top", fontsize=7, color=INK, zorder=6)
    ax.text(bx + L, by - 0.008 * h, f"{L:g} m", ha="center", va="top", fontsize=7, color=INK,
            zorder=6)


def north_arrow(ax, ext):
    x0, x1, y0, y1 = ext
    w, h = x1 - x0, y1 - y0
    s = 0.07 * min(w, h)
    cx, cy = x1 - 0.06 * w, y1 - 0.06 * h - s
    ax.add_patch(Rectangle((cx - 0.5 * s, cy - 0.25 * s), s, 1.55 * s, facecolor="white",
                           alpha=0.85, edgecolor="none", zorder=5))
    ax.add_patch(FancyArrow(cx, cy, 0, 0.8 * s, width=0.12 * s, head_width=0.45 * s,
                            head_length=0.35 * s, length_includes_head=True, color=INK, zorder=6))
    ax.text(cx, cy + 0.98 * s, "N", ha="center", va="bottom", fontsize=9, weight="bold",
            color=INK, zorder=6)


def _segments(tr: pd.DataFrame):
    xy = tr[["x", "y"]].to_numpy()
    segs = np.stack([xy[:-1], xy[1:]], axis=1)
    spd = (tr["speed"].to_numpy()[:-1] + tr["speed"].to_numpy()[1:]) / 2
    pred = tr["predicted"].to_numpy(bool)
    pred = pred[:-1] | pred[1:]
    return segs, spd, pred


def _colorbar(fig, x, label, rng, km_h: bool):
    cax = fig.add_axes([x, 0.18, 0.018, 0.62])
    sm = plt.cm.ScalarMappable(norm=Normalize(*rng), cmap=cmap())
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_linewidth(0.5)
    ticks = np.linspace(rng[0], rng[1], 6)
    cb.set_ticks(ticks)
    if km_h:
        cb.set_ticklabels([f"{t:g} m/s  ({t * 3.6:.0f} km/h)" for t in ticks])
    else:
        cb.set_ticklabels([f"{t:g} m/s" for t in ticks])
    cb.ax.tick_params(labelsize=7, colors=INK)
    cb.ax.set_title(label, fontsize=8, color=INK, loc="left", pad=8)
    return cb


# --------------------------------------------------------------------------- topdown
def plot_topdown(points: pd.DataFrame, raster: GeoRaster | None, cfg: Config, path: Path,
                 title: str = "") -> None:
    vis = cfg["visuals"]
    ext = _extent(points, raster, cfg)
    groups = [g for g in cfg.groups if g in set(points["group"])] if len(points) else []
    fig, ax = _figure(ext, len(groups))
    _basemap(ax, raster, ext, float(vis["basemap_brightness"]))
    lw = float(vis["line_width"])
    for gi, gname in enumerate(groups):
        g = cfg.groups[gname]
        norm = Normalize(*g.speed_range)
        segs_all, spd_all, pred_all = [], [], []
        for _, tr in points[points["group"] == gname].groupby("track_id"):
            if len(tr) < 2:
                continue
            s, v, p = _segments(tr)
            segs_all.append(s)
            spd_all.append(v)
            pred_all.append(p)
        if not segs_all:
            continue
        segs, spd, pred = (np.concatenate(segs_all), np.concatenate(spd_all),
                           np.concatenate(pred_all))
        solid = LineCollection(segs[~pred], cmap=cmap(), norm=norm, linewidths=lw,
                               alpha=0.85, capstyle="round", zorder=2 + gi)
        solid.set_array(spd[~pred])
        ax.add_collection(solid)
        if vis.get("show_predicted", True) and pred.any():
            dashed = LineCollection(segs[pred], colors="white", linewidths=lw * 0.9,
                                    linestyles=(0, (1.5, 1.5)), alpha=0.9, zorder=3 + gi)
            ax.add_collection(dashed)
        cb_x = 1 - (len(groups) - gi) * (1.4 / fig.get_figwidth()) + 0.01
        _colorbar(fig, cb_x, f"{GROUP_LABELS.get(gname, gname)} speed", g.speed_range,
                  km_h=g.speed_range[1] > 8)
    scale_bar(ax, ext)
    north_arrow(ax, ext)
    n_tracks = points["track_id"].nunique() if len(points) else 0
    sub = f"{n_tracks} tracks · EPSG:27700 British National Grid · grid north"
    if vis.get("show_predicted", True) and len(points) and points["predicted"].any():
        sub += " · white dotted = predicted through occlusion"
    fig.text(0.02, 1 - 0.25 / fig.get_figheight(), title or cfg.site, fontsize=13,
             weight="bold", color=INK, va="top")
    fig.text(0.02, 1 - 0.55 / fig.get_figheight(), sub, fontsize=8, color="#555", va="top")
    fig.savefig(path, dpi=int(vis["dpi"]), facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- density
def plot_density(points: pd.DataFrame, raster: GeoRaster | None, cfg: Config, path: Path,
                 dt: float, include_predicted: bool = False) -> None:
    from scipy.ndimage import gaussian_filter

    vis = cfg["visuals"]
    ext = _extent(points, raster, cfg)
    fig, ax = _figure(ext, 1)
    _basemap(ax, raster, ext, float(vis["basemap_brightness"]) * 0.8)
    p = points if include_predicted else points[~points["predicted"].astype(bool)]
    span = max(ext[1] - ext[0], ext[3] - ext[2])
    res = max(0.25, span / 800)                       # metres per density pixel
    sigma_m = max(0.5, 2 * res)
    xb = np.arange(ext[0], ext[1] + res, res)
    yb = np.arange(ext[2], ext[3] + res, res)
    H, _, _ = np.histogram2d(p["x"], p["y"], bins=[xb, yb])
    # occupancy: object-seconds spent per square metre
    D = gaussian_filter(H.T * dt, sigma_m / res) / (res * res)
    if D.max() > 0:
        lo = max(D.max() * 1e-3, 1e-6)
        norm = LogNorm(vmin=lo, vmax=D.max())
        t = np.clip(np.ma.filled(norm(np.maximum(D, lo)), 0), 0, 1)
        rgba = plt.get_cmap("inferno")(t)
        rgba[..., 3] = np.clip(t * 2.5, 0, 0.92)  # fade out smoothly instead of a hard mask
        ax.imshow(rgba, origin="lower", extent=(xb[0], xb[-1], yb[0], yb[-1]),
                  interpolation="bilinear", zorder=2)
        cax = fig.add_axes([1 - 1.4 / fig.get_figwidth() + 0.01, 0.18, 0.018, 0.62])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="inferno"), cax=cax)
        cb.ax.tick_params(labelsize=7)
        cb.ax.set_title("occupancy\n(object-s / m²)", fontsize=8, loc="left", pad=8)
    scale_bar(ax, ext)
    north_arrow(ax, ext)
    fig.text(0.02, 1 - 0.25 / fig.get_figheight(), f"{cfg.site}: where movement happens",
             fontsize=13, weight="bold", color=INK, va="top")
    fig.text(0.02, 1 - 0.55 / fig.get_figheight(),
             f"time spent per m², all classes, {sigma_m:.1f} m smoothing · EPSG:27700",
             fontsize=8, color="#555", va="top")
    fig.savefig(path, dpi=int(vis["dpi"]), facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- flow field
def plot_flow_field(field: pd.DataFrame, points: pd.DataFrame, raster: GeoRaster | None,
                    cfg: Config, path: Path, cell: float, speed_range, label: str,
                    min_confidence: float = 0.3) -> None:
    """Streamlines of the smoothed vector field over the map: colour = speed, width = data."""
    from .field import field_to_arrays

    vis = cfg["visuals"]
    ext = _extent(points, raster, cfg)
    fig, ax = _figure(ext, 1)
    _basemap(ax, raster, ext, float(vis["basemap_brightness"]) * 0.85)
    if len(field):
        xs, ys, U, V, S, C = field_to_arrays(field, cell)
        mask = C < min_confidence
        U, V = np.ma.array(U, mask=mask | np.isnan(U)), np.ma.array(V, mask=mask | np.isnan(V))
        norm = Normalize(*speed_range)
        span = max(ext[1] - ext[0], ext[3] - ext[2])
        density = float(np.clip(span / (cell * 60), 0.8, 3.0))
        lw = 0.3 + 2.2 * np.clip(C, 0, 1)
        ax.streamplot(xs, ys, U, V, color=np.ma.array(S, mask=mask), cmap=cmap(), norm=norm,
                      linewidth=lw, density=density, arrowsize=0.7, minlength=0.15,
                      broken_streamlines=True, zorder=3)
        _colorbar(fig, 1 - 1.4 / fig.get_figwidth() + 0.01, f"{label} flow speed", speed_range,
                  km_h=speed_range[1] > 8)
    scale_bar(ax, ext)
    north_arrow(ax, ext)
    fig.text(0.02, 1 - 0.25 / fig.get_figheight(), f"{cfg.site}: {label.lower()} flow field",
             fontsize=13, weight="bold", color=INK, va="top")
    fig.text(0.02, 1 - 0.55 / fig.get_figheight(),
             ("velocity of the dominant direction" if cfg["field"].get("direction_bins")
              else "smoothed mean velocity")
             + f", {cfg['field']['smooth_m']:g} m kernel · line width = "
             "amount of data · EPSG:27700", fontsize=8, color="#555", va="top")
    fig.savefig(path, dpi=int(vis["dpi"]), facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- histogram
def plot_speed_histogram(summary: pd.DataFrame, cfg: Config, path: Path) -> None:
    groups = [g for g in cfg.groups if len(summary) and g in set(summary["group"])]
    n = max(len(groups), 1)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.8), squeeze=False)
    nb = int(cfg["stats"]["histogram_bins"])
    for ax, gname in zip(axes[0], groups):
        g = cfg.groups[gname]
        lo, hi = g.speed_range
        edges = np.linspace(lo, hi, nb + 1)
        vals = np.clip(summary.loc[summary["group"] == gname, "mean_speed"].to_numpy(), lo, hi)
        counts, _ = np.histogram(vals, edges)
        centres = (edges[:-1] + edges[1:]) / 2
        colours = cmap()(Normalize(lo, hi)(centres))
        ax.bar(centres, counts, width=np.diff(edges) * 0.92, color=colours, edgecolor="none")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if len(vals):
            mean, med = float(np.mean(vals)), float(np.median(vals))
            ax.axvline(mean, color=INK, lw=1, ls="-")
            ax.axvline(med, color=INK, lw=1, ls=":")
            ax.text(0.98, 0.95, f"n = {len(vals)} tracks\nmean {mean:.2f} m/s\n"
                    f"median {med:.2f} m/s", transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color=INK)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("mean track speed (m/s)" + ("; top axis km/h" if hi > 8 else ""),
                      fontsize=9)
        ax.set_ylabel("tracks", fontsize=9)
        ax.set_title(GROUP_LABELS.get(gname, gname), fontsize=11, loc="left", weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        if hi > 8:
            sec = ax.secondary_xaxis("top", functions=(lambda v: v * 3.6, lambda v: v / 3.6))
            sec.tick_params(labelsize=7)
        ax.tick_params(labelsize=8)
    fig.suptitle(f"{cfg.site}: mean speed per track\nsolid line = mean, dotted = median, "
                 "last bar includes faster tracks", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
