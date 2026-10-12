"""The original look of the first maps (visuals.py), frameless at a plan's pixel size.

Greyscale aerial dimmed to `visuals.basemap_brightness`, tracks and streamlines in the speed
ramp (colors.py, each group on its own scale), occupancy in inferno on a log scale. The figure
is set up like the first maps (12 inches on the long side), so line widths and arrow sizes in
points come out the same relative to the plan, whatever its pixel size. Every function returns
a layer on transparency (straight-alpha RGBA uint8) or a base image (float RGB).
"""

from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

from .colors import cmap  # noqa: E402

LONG_SIDE_IN = 12.0          # the first maps: 12 in on the long side
TRACKS_LEVEL = 1.0           # basemap brightness factors of the first maps, per map
DENSITY_LEVEL = 0.8
FLOW_LEVEL = 0.85


def grey_base(raster, ext, size, brightness: float) -> np.ndarray:
    """The aerial in grey, dimmed (float RGB 0..255); no raster: the first maps' dark grey."""
    W, H = size
    if raster is None:
        return np.full((H, W, 3), (43, 43, 46), np.float32)
    from .portfolio import aerial_rgb

    rgb = aerial_rgb(raster, ext, W, H).astype(np.float32) / 255
    g = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    return np.repeat((np.clip(g * brightness, 0, 1) * 255)[..., None], 3, axis=2)


def _figure(ext, size):
    W, H = size
    dpi = max(W, H) / LONG_SIDE_IN
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    return fig, ax


def _layer(fig, size) -> np.ndarray:
    W, H = size
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), np.uint8).copy()
    plt.close(fig)
    if rgba.shape[:2] != (H, W):
        rgba = cv2.resize(rgba, (W, H), interpolation=cv2.INTER_AREA)
    return rgba


def tracks_layer(points: pd.DataFrame, ext, size, ranges: dict, line_width: float = 1.2,
                 show_predicted: bool = True) -> np.ndarray:
    """Every track as a line in the speed colour of its group's scale (alpha 0.85, round caps);
    stretches predicted through occlusion as white dots, as in the first topdown map."""
    fig, ax = _figure(ext, size)
    for gi, (gname, rng) in enumerate(ranges.items()):
        g = points[points["group"] == gname]
        segs, spd, pred = [], [], []
        for _, tr in g.groupby("track_id"):
            if len(tr) < 2:
                continue
            xy = tr[["x", "y"]].to_numpy()
            segs.append(np.stack([xy[:-1], xy[1:]], axis=1))
            sp = tr["speed"].to_numpy()
            spd.append((sp[:-1] + sp[1:]) / 2)
            p = tr["predicted"].to_numpy(bool)
            pred.append(p[:-1] | p[1:])
        if not segs:
            continue
        segs, spd, pred = np.concatenate(segs), np.concatenate(spd), np.concatenate(pred)
        solid = LineCollection(segs[~pred], cmap=cmap(), norm=Normalize(*rng),
                               linewidths=line_width, alpha=0.85, capstyle="round",
                               zorder=2 + gi)
        solid.set_array(spd[~pred])
        ax.add_collection(solid)
        if show_predicted and pred.any():
            ax.add_collection(LineCollection(segs[pred], colors="white",
                                             linewidths=line_width * 0.9,
                                             linestyles=(0, (1.5, 1.5)), alpha=0.9,
                                             zorder=3 + gi))
    return _layer(fig, size)


SITE_SPACING_M = 1.5     # the first maps' streamline spacing and shortest line, in metres, on
SITE_MIN_LEN_M = 6.0     # a typical single clip (~45 m across); kept for big combined plans
SITE_REF_SPAN_M = 45.0   # line widths thin with the plan size (square root) so lines 1.5 m
                         # apart stay apart on a 250 m plan


def flow_layer(fields, ext, size, cell: float, min_confidence: float = 0.3,
               metres: bool = False) -> np.ndarray:
    """Streamlines of one or more fields [(field, (lo, hi)), ...] as in the first flow-field
    map: colour = speed, line width = how much data supports it. metres: spacing and shortest
    line fixed in metres (SITE_*), as they came out on a single clip, instead of relative to
    the field (matplotlib's density / minlength are fractions of the field's extent)."""
    from .field import field_to_arrays

    fig, ax = _figure(ext, size)
    span = max(ext[1] - ext[0], ext[3] - ext[2])
    density = float(np.clip(span / (cell * 60), 0.8, 3.0))
    minlength = 0.15
    thin = float(np.sqrt(min(1.0, SITE_REF_SPAN_M / span))) if metres else 1.0
    for fld, rng in fields:
        if fld is None or len(fld) == 0:
            continue
        xs, ys, U, V, S, C = field_to_arrays(fld, cell)
        if len(xs) < 2 or len(ys) < 2:
            continue
        if metres:
            span_f = max(xs[-1] - xs[0], ys[-1] - ys[0]) + cell
            density = float(np.clip(span_f / (30 * SITE_SPACING_M), 0.5, 15.0))
            minlength = float(np.clip(SITE_MIN_LEN_M / span_f, 0.01, 0.5))
        mask = C < min_confidence
        if (~mask).sum() <= 4:
            continue
        U = np.ma.array(U, mask=mask | np.isnan(U))
        V = np.ma.array(V, mask=mask | np.isnan(V))
        ax.streamplot(xs, ys, U, V, color=np.ma.array(S, mask=mask), cmap=cmap(),
                      norm=Normalize(*rng), linewidth=thin * (0.3 + 2.2 * np.clip(C, 0, 1)),
                      density=density, arrowsize=0.7 * thin, minlength=minlength,
                      broken_streamlines=True, zorder=3)
    return _layer(fig, size)


def occupancy(points: pd.DataFrame, ext, dt: float, include_predicted: bool = False):
    """Object-seconds per m^2 on the first density map's grid: (grid north-up, res m)."""
    from scipy.ndimage import gaussian_filter

    p = points if include_predicted else points[~points["predicted"].astype(bool)]
    span = max(ext[1] - ext[0], ext[3] - ext[2])
    res = max(0.25, span / 800)
    sigma_m = max(0.5, 2 * res)
    xb = np.arange(ext[0], ext[1] + res, res)
    yb = np.arange(ext[2], ext[3] + res, res)
    Hh, _, _ = np.histogram2d(p["x"], p["y"], bins=[xb, yb])
    D = gaussian_filter(Hh.T * dt, sigma_m / res) / (res * res)
    return np.flipud(D), res


def density_layer(grid: np.ndarray, size, valid: np.ndarray | None = None
                  ) -> tuple[np.ndarray, tuple[float, float]]:
    """A density grid (north-up rows) in inferno on a log scale from max/1000 to max, fading
    out at the low end (alpha), as the first density map. Returns (RGBA, (lo, hi))."""
    W, H = size
    D = np.nan_to_num(np.asarray(grid, float))
    hi = float(D.max()) if D.size else 0.0
    if hi <= 0:
        return np.zeros((H, W, 4), np.uint8), (0.0, 0.0)
    lo = max(hi * 1e-3, 1e-9)
    t = np.clip(np.ma.filled(LogNorm(vmin=lo, vmax=hi)(np.maximum(D, lo)), 0), 0, 1)
    t = cv2.resize(t.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR).clip(0, 1)
    rgba = plt.get_cmap("inferno")(t)
    a = np.clip(t * 2.5, 0, 0.92)
    if valid is not None:
        a = a * (cv2.resize(valid.astype(np.float32), (W, H)) > 0.5)
    rgba[..., 3] = a
    return (rgba * 255).astype(np.uint8), (lo, hi)


def inferno_gradient(n: int = 720) -> np.ndarray:
    return np.asarray(plt.get_cmap("inferno")(np.linspace(0, 1, n)))[:, :3] * 255
