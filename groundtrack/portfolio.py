"""Portfolio plates: a minimal, dark diptych per clip.

Left  IN SITU: the mid-clip frame (desaturated, darkened, people blurred) with every track of
      the clip as a thin glowing light trail, drawn where the camera saw it.
Right PLAN:    the aerial map (dark) with the same tracks in metres, EPSG:27700.
Below: a row of key figures and a hairline speed legend. Typeset in Bahnschrift Light.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .config import Config
from .geo import GeoRaster
from .layout import open_run

BG = (11, 11, 12)
INK = (236, 236, 232)
GREY = (138, 138, 142)
HAIR = (58, 58, 62)
# slow -> fast: deep blue, cyan, white, amber (stays legible as light on black)
STOPS = [(0.0, (52, 92, 178)), (0.35, (38, 182, 218)), (0.72, (232, 246, 252)),
         (1.0, (255, 168, 64))]
FONT_FILES = [r"C:\Windows\Fonts\bahnschrift.ttf", "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def speed_rgb(v, lo, hi) -> np.ndarray:
    t = np.clip((np.asarray(v, float) - lo) / max(hi - lo, 1e-9), 0, 1)
    xs = [s[0] for s in STOPS]
    return np.stack([np.interp(t, xs, [s[1][i] for s in STOPS]) for i in range(3)], axis=-1)


def _font(size: int, weight: str = "Light"):
    for f in FONT_FILES:
        if Path(f).exists():
            font = ImageFont.truetype(f, size)
            try:
                font.set_variation_by_name(weight)
            except (OSError, ValueError, AttributeError):
                pass
            return font
    return ImageFont.load_default()


def _text(draw, xy, text, size, weight="Light", fill=INK, tracking=0.0):
    """Draw text with letter-spacing (tracking in em)."""
    font = _font(size, weight)
    x, y = xy
    if not tracking:
        draw.text((x, y), text, font=font, fill=fill)
        return draw.textlength(text, font=font)
    start = x
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking * size
    return x - start


def _glow_lines(shape, segments, width: float, ss: int | None = None, glow: float = 0.9):
    """Additive light-trail layer from [(p0, p1, rgb), ...] in output pixels."""
    h, w = shape
    if ss is None:  # supersample small images; big ones are fine with antialiased lines
        ss = 2 if h * w <= 6_000_000 else 1
    layer = np.zeros((h * ss, w * ss, 3), np.float32)
    lw = max(1, int(round(width * ss)))
    for p0, p1, c in segments:
        a = (int(round(p0[0] * ss)), int(round(p0[1] * ss)))
        b = (int(round(p1[0] * ss)), int(round(p1[1] * ss)))
        cv2.line(layer, a, b, (float(c[0]), float(c[1]), float(c[2])), lw, cv2.LINE_AA)
    if ss > 1:
        layer = cv2.resize(layer, (w, h), interpolation=cv2.INTER_AREA)
    halo = cv2.GaussianBlur(layer, (0, 0), sigmaX=max(2.0, width * 3))
    return layer + glow * halo


def _darken(rgb: np.ndarray, level: float) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g = np.clip((g - 0.5) * 1.15 + 0.5, 0, 1) * level        # a touch of contrast, then dim
    return np.repeat((g * 255)[..., None], 3, axis=2)


# --------------------------------------------------------------------------- panels
def insitu_layers(video: Path, run_dir: Path, cfg: Config, fps: float, out_h: int,
                  frame: int | None = None, line_px: float = 1.6) -> dict:
    """In-situ layers at out_h: frame (RGB, people blurred, aligned to the calibration
    frame), base (darkened frame, float), trails (additive light layer, float)."""
    from .debug_video import blur_box, trail_samples
    from .registration import apply_registration, load_registration
    from .trajectories import recover_feet
    from .video import read_frame

    run = open_run(run_dir)
    raw = pd.read_csv(run.raw_tracks)
    raw["predicted"] = raw["predicted"].astype(str).str.lower().isin(["true", "1"])
    reg, ref_frame = None, 0
    if run.registration.exists():
        reg, ref_frame, _ = load_registration(run.registration)
    if frame is None:
        frame = int((raw["frame"].min() + raw["frame"].max()) // 2)

    def blurred(f):
        im = read_frame(video, f)
        for r in raw[(raw["frame"] == f) & (raw["class"] == "person")].itertuples():
            blur_box(im, r.x1, r.y1, r.x2, r.y2)
        return im

    img = blurred(frame)
    H, W = img.shape[:2]
    if reg is not None and frame in reg:
        img = cv2.warpPerspective(img, reg[frame], (W, H))
        cover = cv2.warpPerspective(np.full((H, W), 255, np.uint8), reg[frame], (W, H),
                                    flags=cv2.INTER_NEAREST)
        if cover.min() == 0:  # fill what the moved frame no longer covers with the reference
            ref = blurred(ref_frame)
            img[cover == 0] = ref[cover == 0]
    feet, _ = recover_feet(raw, int(2 * fps))
    uv = apply_registration(raw["frame"].to_numpy(), feet, reg) if reg else feet
    pos, speed, lo, hi, _ = trail_samples(raw, cfg, run_dir, fps, uv, window_s=1.0)
    keep = np.isfinite(speed)
    s = out_h / H
    paths = []
    sub = raw.assign(px=pos[:, 0] * s, py=pos[:, 1] * s, sp=speed)[keep]
    for _, t in sub.sort_values("frame").groupby("track_id"):
        paths.append((t[["px", "py"]].to_numpy(), t["sp"].to_numpy(), t["frame"].to_numpy()))
    lo_, hi_ = float(np.nanmin(lo)), float(np.nanmax(hi))
    rgb = cv2.cvtColor(cv2.resize(img, (int(round(W * s)), out_h), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    stride = max(1, int(np.median(np.diff(np.sort(raw["frame"].unique())))))
    trails = _glow_lines(rgb.shape[:2], _segments_from(paths, lo_, hi_, 3 * stride), line_px)
    return {"frame_rgb": rgb, "base": _darken(rgb, 0.42), "trails": trails,
            "speed_range": (lo_, hi_), "frame": frame}


def insitu_panel(video: Path, run_dir: Path, cfg: Config, fps: float, out_h: int,
                 frame: int | None = None, line_px: float = 1.6):
    L = insitu_layers(video, run_dir, cfg, fps, out_h, frame, line_px)
    return np.clip(L["base"] + L["trails"], 0, 255).astype(np.uint8), L["speed_range"], L["frame"]


def _segments_from(paths, lo, hi, max_gap):
    segs = []
    for xy, sp, fr in paths:
        cols = speed_rgb(sp, lo, hi)
        for k in range(1, len(xy)):
            if fr[k] - fr[k - 1] > max_gap:
                continue
            segs.append((xy[k - 1], xy[k], (cols[k - 1] + cols[k]) / 2))
    return segs


def plan_extent(points: pd.DataFrame, margin_frac: float = 0.12, aspect=(0.62, 1.5)):
    """Plan extent (x0, x1, y0, y1) around the data, with a calm aspect ratio."""
    x0, x1 = points["x"].min(), points["x"].max()
    y0, y1 = points["y"].min(), points["y"].max()
    span = max(x1 - x0, y1 - y0)
    m = span * margin_frac
    x0, x1, y0, y1 = x0 - m, x1 + m, y0 - m, y1 + m
    w, h = x1 - x0, y1 - y0
    a = w / h
    if a < aspect[0]:
        d = (aspect[0] * h - w) / 2
        x0, x1 = x0 - d, x1 + d
    elif a > aspect[1]:
        d = (w / aspect[1] - h) / 2
        y0, y1 = y0 - d, y1 + d
    return float(x0), float(x1), float(y0), float(y1)


def aerial_rgb(raster: GeoRaster, ext, out_w: int, out_h: int) -> np.ndarray:
    """The GeoTIFF cut to ext and resampled to out_w x out_h, in colour (uint8 RGB)."""
    x0, x1, y0, y1 = ext
    c0, r0 = raster.world_to_pixel([[x0, y1]])[0]
    c1, r1 = raster.world_to_pixel([[x1, y0]])[0]
    sx, sy = (c1 - c0) / out_w, (r1 - r0) / out_h
    src = raster.image
    if sx > 1.5:  # shrinking a lot: pre-blur so fine texture does not alias
        src = cv2.GaussianBlur(src, (0, 0), sigmaX=0.4 * sx, sigmaY=0.4 * sy)
    M = np.array([[sx, 0, c0], [0, sy, r0]], np.float32)
    img = cv2.warpAffine(src, M, (out_w, out_h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    return img[..., :3].astype(np.uint8)


def plan_size(ext, out_h: int) -> tuple[int, int]:
    return int(round(out_h * (ext[1] - ext[0]) / (ext[3] - ext[2]))), out_h


def plan_layers(points: pd.DataFrame, raster: GeoRaster | None, ext, out_h: int, speed_range,
                line_px: float = 1.3) -> dict:
    """Plan layers at out_h: aerial (RGB or None), base (darkened, float), trails (float)."""
    x0, x1, y0, y1 = ext
    out_w, _ = plan_size(ext, out_h)
    aerial = None
    if raster is not None:
        aerial = aerial_rgb(raster, ext, out_w, out_h)
        base = _darken(aerial, 0.34)
    else:
        base = np.full((out_h, out_w, 3), BG, np.float32)
    sx, sy = out_w / (x1 - x0), out_h / (y1 - y0)
    paths = []
    for _, t in points.sort_values("frame").groupby("track_id"):
        xy = np.column_stack([(t["x"] - x0) * sx, (y1 - t["y"]) * sy])
        paths.append((xy, t["speed"].to_numpy(), t["frame"].to_numpy()))
    stride = max(1, int(np.median(np.diff(np.sort(points["frame"].unique())))))
    trails = _glow_lines((out_h, out_w), _segments_from(paths, *speed_range, 3 * stride), line_px)
    return {"aerial": aerial, "base": base, "trails": trails}


def plan_panel(points: pd.DataFrame, raster: GeoRaster | None, cfg: Config, out_h: int,
               speed_range, margin_frac: float = 0.12, aspect=(0.62, 1.5), line_px: float = 1.3,
               ext=None):
    ext = ext or plan_extent(points, margin_frac, aspect)
    L = plan_layers(points, raster, ext, out_h, speed_range, line_px)
    return np.clip(L["base"] + L["trails"], 0, 255).astype(np.uint8), ext


def _compass(yaw_deg: float) -> str:
    names = ["north", "north-north-east", "north-east", "east-north-east", "east",
             "east-south-east", "south-east", "south-south-east", "south", "south-south-west",
             "south-west", "west-south-west", "west", "west-north-west", "north-west",
             "north-north-west"]
    return names[int(((yaw_deg % 360) + 11.25) // 22.5) % 16]


def _nice(span):
    for v in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500):
        if v >= span / 6:
            return v
    return 1000


# --------------------------------------------------------------------------- plate
def make_plate(cfg: Config, run_dir: Path, raster: GeoRaster | None, out_png: Path,
               title: str, index: str = "", subtitle: str = "", date: str = "",
               panel_h: int = 2000, frame: int | None = None, width: int | None = 3600,
               credit: str = "") -> Path:
    run = open_run(run_dir)
    run_dir = run.root
    meta = json.loads(run.meta.read_text(encoding="utf-8"))
    points = pd.read_csv(run.points)
    summary = pd.read_csv(run.tracks)
    stats = json.loads(run.stats_json.read_text(encoding="utf-8"))
    group = summary["group"].mode().iloc[0]
    g = cfg.groups[group]

    margin, gutter, top, bottom = 200, 90, 520, 400
    if width:  # same plate width for every clip: panels share it in proportion to their aspect
        _, e0 = plan_panel(points, None, cfg, 200, g.speed_range)
        a_left = float(meta["width"]) / float(meta["height"])
        a_right = (e0[1] - e0[0]) / (e0[3] - e0[2])
        panel_h = int((width - 2 * margin - gutter) / (a_left + a_right))
    left, _, frame = insitu_panel(Path(meta["video"]), run_dir, cfg, float(meta["fps"]), panel_h,
                                  frame)
    right, ext = plan_panel(points, raster, cfg, panel_h, g.speed_range)
    W = margin * 2 + left.shape[1] + gutter + right.shape[1]
    H = top + panel_h + bottom
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(Image.fromarray(left), (margin, top))
    rx = margin + left.shape[1] + gutter
    canvas.paste(Image.fromarray(right), (rx, top))
    d = ImageDraw.Draw(canvas)

    # header
    _text(d, (margin, 150), "GROUNDTRACK  —  MOVEMENT STUDY" + (f"  —  {date}" if date else ""),
          26, "SemiLight", GREY, tracking=0.32)
    tw = _text(d, (margin, 205), title.upper(), 104, "Light", INK, tracking=0.06)
    if index:
        _text(d, (margin + tw + 40, 232), index, 54, "Light", GREY, tracking=0.1)
    cam = None
    try:
        cam = json.loads(run.homography_used.read_text(encoding="utf-8"))             .get("camera_params")
    except (OSError, ValueError):
        pass
    if cam:
        subtitle = (subtitle + "  ·  " if subtitle else "") + f"looking {_compass(cam['yaw_deg'])}"
    if subtitle:
        _text(d, (margin, 348), subtitle, 34, "Light", GREY, tracking=0.02)

    # panel captions + hairlines
    for x, w_, cap in ((margin, left.shape[1], "01   IN SITU"),
                       (rx, right.shape[1], "02   PLAN   ·   EPSG:27700")):
        _text(d, (x, top - 52), cap, 22, "SemiLight", GREY, tracking=0.3)
        d.line([(x, top - 16), (x + w_, top - 16)], fill=HAIR, width=2)

    # plan furniture: hairline scale bar + north
    sx = right.shape[1] / (ext[1] - ext[0])
    L = _nice(ext[1] - ext[0])
    by = top + panel_h - 70
    d.line([(rx + 50, by), (rx + 50 + L * sx, by)], fill=INK, width=2)
    for xx in (rx + 50, rx + 50 + L * sx):
        d.line([(xx, by - 10), (xx, by + 10)], fill=INK, width=2)
    _text(d, (rx + 50, by - 48), f"{L:g} M", 22, "SemiLight", INK, tracking=0.25)
    nx, ny = rx + right.shape[1] - 70, top + 60
    d.line([(nx, ny + 70), (nx, ny)], fill=INK, width=2)
    d.polygon([(nx, ny - 14), (nx - 9, ny + 6), (nx + 9, ny + 6)], fill=INK)
    _text(d, (nx - 9, ny + 82), "N", 24, "SemiLight", INK)

    # figures row
    y0 = top + panel_h + 90
    d.line([(margin, y0 - 40), (W - margin, y0 - 40)], fill=HAIR, width=2)
    med = float(summary["mean_speed"].median())
    figures = [("TRACKS", f"{len(summary)}"),
               ("MEDIAN SPEED", f"{med:.2f} m/s" + (f"  ·  {med * 3.6:.0f} km/h" if group == "vehicles" else ""))]
    lines = [cl for cl in stats.get("count_lines", [])
             if cl["crossings_left_to_right"] + cl["crossings_right_to_left"] > 0]
    if lines:
        figures.append(("FLOW", f"{lines[0]['flow_per_min']:.0f} / min"))
    else:
        figures.append(("STRAIGHTNESS", f"{summary['straightness'].median():.2f}"))
    figures.append(("DURATION", f"{stats['window_s'][1] - stats['window_s'][0]:.1f} s"))
    if cam:
        figures.append(("CAMERA", f"{cam['height_m']:.0f} m  ·  {cam['hfov_deg']:.0f}°"))

    # legend first (right-aligned), figures share the space to its left
    lw = 480
    lx, ly = W - margin - lw, y0 + 70
    avail = lx - 120 - margin
    widths = [max(d.textlength(v, font=_font(50)), d.textlength(lab, font=_font(22)) * 1.3)
              for lab, v in figures]
    gap = max(60, (avail - sum(widths)) / max(len(figures) - 1, 1))
    x = margin
    for (label, value), wv in zip(figures, widths):
        _text(d, (x, y0), label, 22, "SemiLight", GREY, tracking=0.3)
        _text(d, (x, y0 + 44), value, 50, "Light", INK, tracking=0.02)
        x += wv + gap

    # legend: hairline gradient
    grad = speed_rgb(np.linspace(*g.speed_range, lw), *g.speed_range).astype(np.uint8)
    canvas.paste(Image.fromarray(np.repeat(grad[None], 6, axis=0)), (lx, ly))
    _text(d, (lx, y0), "SPEED", 22, "SemiLight", GREY, tracking=0.3)
    _text(d, (lx, ly + 22), f"{g.speed_range[0]:g}", 24, "Light", GREY)
    hi_lbl = f"{g.speed_range[1]:g} m/s"
    hw = d.textlength(hi_lbl, font=_font(24))
    _text(d, (lx + lw - hw, ly + 22), hi_lbl, 24, "Light", GREY)
    _text(d, (margin, H - 95), "Trails: every tracked "
          + ("person" if group == "people" else "vehicle")
          + " of the clip, coloured by measured speed. People are blurred."
          + (f" Imagery: {credit.rstrip('.')}." if credit else ""), 22, "Light", (96, 96, 100),
          tracking=0.02)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, optimize=True)
    return out_png
