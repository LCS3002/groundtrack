"""topdown.mp4: the measured movement animated over the map (needs a calibrated run).

Trails build up on the dimmed GeoTIFF, coloured by speed (same ramp and ranges as
topdown.png), with a dot at each object's current position, a legend, scale bar and clock.
Machine-made samples (predicted / stitched) are drawn as dotted white.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .colors import ramp_rgb
from .config import Config
from .geo import GeoRaster

INK = (29, 29, 31)


def _base_image(raster: GeoRaster | None, ext, out_w: int, brightness: float):
    x0, x1, y0, y1 = ext
    out_h = int(round(out_w * (y1 - y0) / (x1 - x0)))
    out_h += out_h % 2
    if raster is None:
        return np.full((out_h, out_w, 3), 45, np.uint8)
    c0, r0 = raster.world_to_pixel([[x0, y1]])[0]
    c1, r1 = raster.world_to_pixel([[x1, y0]])[0]
    sx, sy = (c1 - c0) / out_w, (r1 - r0) / out_h
    M = np.array([[sx, 0, c0], [0, sy, r0]], np.float32)
    img = cv2.warpAffine(raster.image, M, (out_w, out_h),
                         flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(45, 45, 45))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) * brightness
    return cv2.cvtColor(gray.clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _nice(span):
    for v in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000):
        if v >= span / 5:
            return v
    return 5000


def render_topdown_video(points: pd.DataFrame, raster: GeoRaster | None, cfg: Config,
                         out_path: Path, fps: float, ext, width: int = 1920,
                         speedup: float = 1.0, log=print) -> None:
    from tqdm import tqdm

    if points.empty:
        log("topdown video: no points")
        return
    vis = cfg["visuals"]
    base = _base_image(raster, ext, width, float(vis["basemap_brightness"]))
    H, W = base.shape[:2]
    x0, x1, y0, y1 = ext
    sx, sy = W / (x1 - x0), H / (y1 - y0)

    pts = points.sort_values(["frame", "track_id"]).copy()
    pts["px"] = (pts["x"] - x0) * sx
    pts["py"] = (y1 - pts["y"]) * sy
    ranges = {g.name: g.speed_range for g in cfg.groups.values()}
    lo = pts["group"].map(lambda g: ranges.get(g, (0, 1))[0]).to_numpy(float)
    hi = pts["group"].map(lambda g: ranges.get(g, (0, 1))[1]).to_numpy(float)
    rgb = np.vstack([ramp_rgb(np.array([s]), a, b) for s, a, b in
                     zip(pts["speed"].to_numpy(), lo, hi)]) if len(pts) else np.zeros((0, 3))
    pts["b"], pts["g"], pts["r"] = (rgb[:, 2] * 255), (rgb[:, 1] * 255), (rgb[:, 0] * 255)
    by_frame = {f: g for f, g in pts.groupby("frame")}
    frames = sorted(by_frame)
    th = max(2, int(round(W / 900)))
    out_fps = fps * speedup
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
    comp = base.copy()          # the map never changes: trails are drawn straight onto it
    last: dict[int, tuple] = {}
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    L = _nice(x1 - x0)

    for f in tqdm(range(frames[0], frames[-1] + 1, step), desc="topdown video", unit="frame",
                  mininterval=2.0):
        rows = by_frame.get(f)
        heads = []
        if rows is not None:
            for r in rows.itertuples():
                p = (int(round(r.px)), int(round(r.py)))
                col = (int(r.b), int(r.g), int(r.r))
                prev = last.get(r.track_id)
                if prev is not None and f - prev[0] <= 3 * step:
                    if r.predicted:
                        if (f // step) % 4 < 2:
                            cv2.line(comp, prev[1], p, (235, 235, 235), max(1, th - 1),
                                     cv2.LINE_AA)
                    else:
                        cv2.line(comp, prev[1], p, col, th, cv2.LINE_AA)
                last[r.track_id] = (f, p)
                heads.append((p, col))
        img = comp.copy()
        for p, col in heads:
            cv2.circle(img, p, 2 * th + 1, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(img, p, 2 * th, col, -1, cv2.LINE_AA)
        # clock, scale bar, north arrow, legend
        cv2.putText(img, f"{cfg.site}   t = {f / fps:6.1f} s", (16, 34), font, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        bl = int(L * sx)
        cv2.rectangle(img, (14, H - 44), (30 + bl, H - 12), (240, 240, 240), -1)
        for k in range(4):
            c = INK if k % 2 == 0 else (255, 255, 255)
            cv2.rectangle(img, (22 + k * bl // 4, H - 36), (22 + (k + 1) * bl // 4, H - 28), c, -1)
        cv2.putText(img, f"{L:g} m", (22 + bl - 40, H - 16), font, 0.45, INK, 1, cv2.LINE_AA)
        cv2.arrowedLine(img, (W - 40, 90), (W - 40, 40), (255, 255, 255), 3, cv2.LINE_AA,
                        tipLength=0.35)
        cv2.putText(img, "N", (W - 48, 112), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        _legend(img, cfg, sorted(set(points["group"])))
        writer.write(img)
    writer.release()
    log(f"topdown video -> {out_path}")


def _legend(img, cfg: Config, groups):
    H, W = img.shape[:2]
    bw, bh = 260, 10
    y = H - 60 - 46 * (len(groups) - 1)
    for gname in groups:
        g = cfg.groups.get(gname)
        if g is None:
            continue
        x = W - bw - 30
        cv2.rectangle(img, (x - 10, y - 26), (x + bw + 10, y + bh + 20), (30, 30, 30), -1)
        bar = (ramp_rgb(np.linspace(0, 1, bw), 0, 1)[:, ::-1] * 255).astype(np.uint8)
        img[y:y + bh, x:x + bw] = bar[None]
        cv2.putText(img, f"{gname} speed (m/s)", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{g.speed_range[0]:g}", (x, y + bh + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
        t = f"{g.speed_range[1]:g}"
        cv2.putText(img, t, (x + bw - 8 * len(t), y + bh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 46
