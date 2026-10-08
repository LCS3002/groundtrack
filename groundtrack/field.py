"""Smoothed top-down vector field from the trajectories (for particle sims and flow maps).

Kernel regression on a world-aligned grid: every (non-predicted) sample drops its velocity
into a cell; sums and counts are Gaussian-blurred with the same kernel, and the field is
blurred velocity sum / blurred count. This is a Nadaraya-Watson estimate, so the field is
smooth, keeps real speeds (it is an average, not a sum), and fades out where there is no
data instead of inventing motion there.

Per cell:
  vx, vy, speed, heading   smoothed velocity (m/s, heading clockwise from grid north)
  density                  occupancy, object-seconds per m^2 (same units as density.png)
  weight                   effective number of samples under the kernel
  confidence               1 - exp(-weight / confidence_samples), 0..1: use it to fade
                           forces / emission at the edges of the observed area
  dominance                share of the samples moving in the cell's direction (0..1):
                           1 = one-way flow, ~0.5 = two equal opposite streams

Direction-aware (direction_bins > 0, the default): samples are sorted into heading sectors
and every cell takes the velocity of its dominant direction (with the neighbouring sectors).
Opposite streams close together (two carriageways, a two-way footpath) then keep their
full speed instead of averaging out to a slow band between them. direction_bins = 0 gives
the plain mean velocity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

FIELD_COLUMNS = ["cell_x", "cell_y", "i", "j", "vx", "vy", "speed", "heading", "density",
                 "weight", "confidence", "dominance"]


def _grid_bounds(x, y, cell, pad_m):
    i0 = int(np.floor((np.min(x) - pad_m) / cell))
    i1 = int(np.floor((np.max(x) + pad_m) / cell))
    j0 = int(np.floor((np.min(y) - pad_m) / cell))
    j1 = int(np.floor((np.max(y) + pad_m) / cell))
    return i0, i1, j0, j1


def smooth_field(points: pd.DataFrame, cell: float, smooth_m: float, dt: float,
                 include_predicted: bool = False, confidence_samples: float = 15.0,
                 min_weight: float = 0.05, bounds=None, direction_bins: int = 8) -> pd.DataFrame:
    """Smoothed vector field (one row per cell with any support), see module docstring.

    bounds: optional (i0, i1, j0, j1) so several fields (e.g. time slices) share one grid.
    """
    p = points if include_predicted else points[~points["predicted"].astype(bool)]
    if p.empty:
        return pd.DataFrame(columns=FIELD_COLUMNS)
    x, y = p["x"].to_numpy(float), p["y"].to_numpy(float)
    i0, i1, j0, j1 = bounds or _grid_bounds(x, y, cell, 3 * smooth_m)
    nx, ny = i1 - i0 + 1, j1 - j0 + 1
    ii = np.floor(x / cell).astype(np.int64) - i0
    jj = np.floor(y / cell).astype(np.int64) - j0
    ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
    ii, jj = ii[ok], jj[ok]
    pvx, pvy = p["vx"].to_numpy(float)[ok], p["vy"].to_numpy(float)[ok]
    sigma = max(smooth_m / cell, 1e-6)
    # sum (not mean) normalisation: blur(count) = effective samples under the kernel
    k = 2 * np.pi * sigma ** 2

    def blurred(sel):
        c, sx, sy = (np.zeros((ny, nx)) for _ in range(3))
        np.add.at(c, (jj[sel], ii[sel]), 1.0)
        np.add.at(sx, (jj[sel], ii[sel]), pvx[sel])
        np.add.at(sy, (jj[sel], ii[sel]), pvy[sel])
        return tuple(gaussian_filter(a, sigma, mode="constant") * k for a in (c, sx, sy))

    w, bvx, bvy = blurred(np.ones(len(ii), bool))
    dominance = np.ones((ny, nx))
    if direction_bins and direction_bins > 1:
        # heading is held through stops by the cleaning step; fall back to the velocity
        hd = (p["heading"].to_numpy(float)[ok] if "heading" in p
              else np.degrees(np.arctan2(pvx, pvy)))
        hd = np.where(np.isfinite(hd), hd, np.degrees(np.arctan2(pvx, pvy)))
        nb = int(direction_bins)
        sector = np.floor(((hd + 180.0 / nb) % 360) / (360.0 / nb)).astype(int) % nb
        W, BX, BY = (np.zeros((nb, ny, nx)) for _ in range(3))
        for b in range(nb):
            if (sector == b).any():
                W[b], BX[b], BY[b] = blurred(sector == b)
        # each direction together with its neighbours (a stream on a sector edge stays whole)
        mix = lambda a: a + 0.5 * (np.roll(a, 1, axis=0) + np.roll(a, -1, axis=0))  # noqa: E731
        W2, BX2, BY2 = mix(W), mix(BX), mix(BY)
        best = np.argmax(W2, axis=0)[None]
        wb = np.take_along_axis(W2, best, 0)[0]
        bvx, bvy = np.take_along_axis(BX2, best, 0)[0], np.take_along_axis(BY2, best, 0)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            dominance = np.clip(np.where(w > 0, wb / (w + 1e-12), 0.0), 0, 1)
        wv = wb
    else:
        wv = w
    with np.errstate(invalid="ignore", divide="ignore"):
        vx = np.where(wv > 0, bvx / wv, 0.0)
        vy = np.where(wv > 0, bvy / wv, 0.0)
    density = w / k * dt / (cell * cell)
    conf = 1 - np.exp(-w / max(confidence_samples, 1e-9))

    jj_all, ii_all = np.nonzero(w >= min_weight)
    out = pd.DataFrame({
        "i": ii_all + i0, "j": jj_all + j0,
        "vx": vx[jj_all, ii_all], "vy": vy[jj_all, ii_all],
        "density": density[jj_all, ii_all], "weight": w[jj_all, ii_all],
        "confidence": conf[jj_all, ii_all], "dominance": dominance[jj_all, ii_all],
    })
    out["cell_x"] = (out["i"] + 0.5) * cell
    out["cell_y"] = (out["j"] + 0.5) * cell
    out["speed"] = np.hypot(out["vx"], out["vy"])
    out["heading"] = (np.degrees(np.arctan2(out["vx"], out["vy"])) + 360) % 360
    for c, nd in (("vx", 4), ("vy", 4), ("speed", 4), ("heading", 1), ("density", 5),
                  ("weight", 3), ("confidence", 4), ("dominance", 3), ("cell_x", 3),
                  ("cell_y", 3)):
        out[c] = out[c].round(nd)
    return out[FIELD_COLUMNS].sort_values(["j", "i"]).reset_index(drop=True)


def time_sliced_fields(points: pd.DataFrame, window_s: float, cell: float, smooth_m: float,
                       dt: float, **kw) -> pd.DataFrame:
    """One field per time window (all on the same grid), with t_start / t_end columns."""
    p = points if kw.get("include_predicted") else points[~points["predicted"].astype(bool)]
    if p.empty:
        return pd.DataFrame(columns=["t_start", "t_end", *FIELD_COLUMNS])
    bounds = _grid_bounds(p["x"].to_numpy(), p["y"].to_numpy(), cell, 3 * smooth_m)
    t0, t1 = float(points["time_s"].min()), float(points["time_s"].max())
    frames = []
    for a in np.arange(t0, t1, window_s):
        sub = points[(points["time_s"] >= a) & (points["time_s"] < a + window_s)]
        f = smooth_field(sub, cell, smooth_m, dt, bounds=bounds, **kw)
        if len(f):
            f.insert(0, "t_end", round(a + window_s, 3))
            f.insert(0, "t_start", round(a, 3))
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=["t_start", "t_end", *FIELD_COLUMNS])


def field_to_arrays(field: pd.DataFrame, cell: float):
    """Field rows -> regular arrays (X, Y centres, U, V, speed, confidence) for plotting."""
    i0, i1 = int(field["i"].min()), int(field["i"].max())
    j0, j1 = int(field["j"].min()), int(field["j"].max())
    nx, ny = i1 - i0 + 1, j1 - j0 + 1
    U = np.full((ny, nx), np.nan)
    V = np.full((ny, nx), np.nan)
    C = np.zeros((ny, nx))
    jj, ii = field["j"].to_numpy() - j0, field["i"].to_numpy() - i0
    U[jj, ii], V[jj, ii] = field["vx"], field["vy"]
    C[jj, ii] = field["confidence"]
    xs = (np.arange(i0, i1 + 1) + 0.5) * cell
    ys = (np.arange(j0, j1 + 1) + 0.5) * cell
    return xs, ys, U, V, np.hypot(U, V), C
