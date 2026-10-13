"""A picture of a clip's calibration, to show the process.

Three frameless panels of the same height, written into <run>/images/:
    calibration_video.png    the calibration frame (people and number plates blurred) with
                             the clicked ground points and a ground grid as the camera sees it
    calibration_aerial.png   the aerial with the same points, the same grid over the ground
                             the camera sees, and the camera
    calibration_warped.png   the video frame laid onto the aerial through the calibration
    calibration.png          the three side by side
and the caption as a label (labels/calibration_caption.png, + .txt).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image

from . import portfolio as pf
from .config import Config
from .geo import GeoRaster
from .homography import Homography

GRID_M = 2.0                 # ground grid spacing
POINT_RGB = (255, 40, 200)   # the clicked points: the web diagram's magenta
GRID_RGB = (255, 255, 255)
GUTTER_PX = 24
MARGIN_M = 6.0               # aerial panels: around the seen ground, the camera and the points


def seen_range(cfg: Config, h: Homography, default: float = 40.0) -> float:
    """How far from the camera the pictures show the ground: the precise range the tracks
    use (projection.max_range_m, a number or "auto"; the largest over the groups)."""
    from .trajectories import max_range_for

    r = [max_range_for(cfg, h, g) for g in cfg.groups]
    r = [v for v in r if v is not None and np.isfinite(v)]
    return float(max(r)) if r else default


def seen_ground(h: Homography, size, max_range_m: float, cam, step: int = 8) -> np.ndarray:
    """Outline (E, N) of the ground the camera sees within max_range_m: the convex hull of
    the ground under every step-th pixel."""
    W, H = size
    uu, vv = np.meshgrid(np.arange(0, W + 1, step, float), np.arange(0, H + 1, step, float))
    world = h.to_world(np.column_stack([uu.ravel(), vv.ravel()]))
    ok = np.isfinite(world).all(axis=1)
    if cam is not None:
        ok &= np.hypot(world[:, 0] - cam[0], world[:, 1] - cam[1]) <= max_range_m
    world = world[ok]
    if len(world) < 3:
        return world
    hull = cv2.convexHull(world.astype(np.float32)).reshape(-1, 2)
    return hull.astype(float)


def grid_lines(outline: np.ndarray, spacing: float = GRID_M, step: float = 0.1) -> list:
    """Grid lines (east-west and north-south, on the national grid) clipped to the outline:
    a list of (N, 2) world polylines."""
    poly = MplPath(outline)
    x0, y0 = np.floor(outline.min(axis=0) / spacing) * spacing
    x1, y1 = np.ceil(outline.max(axis=0) / spacing) * spacing
    lines = []
    for vertical in (True, False):
        for c in np.arange(x0 if vertical else y0, (x1 if vertical else y1) + 1e-6, spacing):
            t = np.arange(y0 if vertical else x0, (y1 if vertical else x1) + 1e-6, step)
            xy = np.column_stack([np.full_like(t, c), t] if vertical else [t, np.full_like(t, c)])
            inside = poly.contains_points(xy)
            # split into runs of consecutive inside samples
            edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a >= 2:
                    lines.append(xy[a:b])
    return lines


def _line_layer(shape, polylines, width: float, rgb, alpha: float) -> np.ndarray:
    """Polylines in pixels on transparency (straight-alpha RGBA), antialiased."""
    h, w = shape
    ss = 2
    cov = np.zeros((h * ss, w * ss), np.float32)
    for pl in polylines:
        pl = pl[np.isfinite(pl).all(axis=1)]
        if len(pl) >= 2:
            cv2.polylines(cov, [np.round(pl * ss).astype(np.int32)], False, 1.0,
                          max(1, int(round(width * ss))), cv2.LINE_AA)
    cov = cv2.resize(cov, (w, h), interpolation=cv2.INTER_AREA).clip(0, 1)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = (cov * alpha * 255).astype(np.uint8)
    return out


def _markers(img: np.ndarray, pts_px, labels, radius: float) -> np.ndarray:
    """Numbered dots (magenta, white number, thin white ring) on an RGB uint8 image."""
    from PIL import ImageDraw

    im = Image.fromarray(img.astype(np.uint8))
    d = ImageDraw.Draw(im)
    font = pf._font(int(radius * 1.25), "SemiBold")
    r = radius
    for (x, y), lab in zip(pts_px, labels):
        d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(255, 255, 255))
        d.ellipse([x - r, y - r, x + r, y + r], fill=POINT_RGB)
        tw = d.textlength(lab, font=font)
        bb = d.textbbox((0, 0), lab, font=font)
        d.text((x - tw / 2, y - (bb[1] + bb[3]) / 2), lab, font=font, fill=(255, 255, 255))
    return np.asarray(im)


def _camera(img: np.ndarray, xy, radius: float) -> np.ndarray:
    out = img.copy()
    c = (int(round(xy[0])), int(round(xy[1])))
    cv2.circle(out, c, int(radius * 1.5), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(out, c, int(radius), (29, 29, 31), -1, cv2.LINE_AA)
    return out


def still_people(cfg: Config, img: np.ndarray, conf: float = 0.08) -> np.ndarray:
    """Person boxes (x1, y1, x2, y2) on one still, found at full resolution and a low
    confidence, so small far people the tracking missed are blurred too."""
    from ultralytics import YOLO

    H, W = img.shape[:2]
    r = YOLO(cfg.model_path()).predict(img, imgsz=int(np.ceil(max(H, W) / 32) * 32), conf=conf,
                                       classes=[0], verbose=False)[0]
    b = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
    pad = 0.15 * np.column_stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]])
    return np.column_stack([b[:, :2] - pad, b[:, 2:] + pad])


def blurred_still(cfg: Config, run_dir: Path, frame: int, log=print):
    """A video frame (BGR) with people and number plates blurred: the tracked ones (held as in
    the videos) and anyone a full-resolution detection finds on it. Returns (image, the extra
    boxes found on the still)."""
    import json

    import pandas as pd

    from .debug_video import HOLD_S, blur_boxes, privacy_boxes, raw_all
    from .layout import open_run
    from .video import read_frame

    run = open_run(run_dir)
    img = read_frame(cfg.path("video"), frame)
    if run.raw_tracks.exists():
        fps = float(json.loads(run.meta.read_text()).get("fps", 25)) \
            if run.meta.exists() else 25.0
        dv = cfg.get("debug_video") or {}
        hide = privacy_boxes(raw_all(run_dir, pd.read_csv(run.raw_tracks)),
                             bool(dv.get("blur_people", True)), bool(dv.get("blur_plates", True)),
                             int(round(HOLD_S * fps)))
        blur_boxes(img, hide.loc[hide["frame"] == frame, ["x1", "y1", "x2", "y2"]].to_numpy())
    else:
        log("no tracked run: only the still's own detection blurs people - check the picture")
    extra = still_people(cfg, img)
    blur_boxes(img, extra)
    return img, extra


def calibration_figure(cfg: Config, run_dir: Path, raster: GeoRaster, h: Homography,
                       out_h: int = 1080, log=print) -> dict:
    """Write the calibration panels of a run (see the module docstring); returns their paths."""
    from .layout import open_run

    run = open_run(run_dir)
    frame = int(h.reference_frame or 0)
    img, _ = blurred_still(cfg, run_dir, frame, log)
    Hv, Wv = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cam_par = getattr(h, "camera_params", None) or {}
    cam = (float(cam_par["E"]), float(cam_par["N"])) if cam_par.get("E") is not None else None
    rng = seen_range(cfg, h)
    outline = seen_ground(h, (Wv, Hv), rng, cam)
    lines = grid_lines(outline)
    pts = [p for p in h.points if p.get("inlier", True)]
    uv = np.array([[p["u"], p["v"]] for p in pts], float).reshape(-1, 2)
    en = np.array([[p["E"], p["N"]] for p in pts], float).reshape(-1, 2)
    labels = [str(k + 1) for k in range(len(pts))]

    # ---- the video: grid on the ground as the camera sees it, the clicked points
    s = out_h / Hv
    vid = cv2.resize(rgb, (int(round(Wv * s)), out_h), interpolation=cv2.INTER_AREA)
    lw = max(1.2, 1.6 * out_h / 1080)
    grid_v = [h.to_pixel(l) * s for l in lines]
    vid = pf.over(vid, _line_layer(vid.shape[:2], grid_v, lw, GRID_RGB, 0.7))
    rad = 17 * out_h / 1080
    video_png = _markers(vid, uv * s, labels, rad)

    # ---- the aerial: the same grid over the seen ground, its outline, the camera, the points
    allp = np.vstack([outline, en] + ([np.array([cam])] if cam else []))
    x0, y0 = allp.min(axis=0) - MARGIN_M
    x1, y1 = allp.max(axis=0) + MARGIN_M
    ext = (x0, x1, y0, y1)
    W, H = pf.plan_size(ext, out_h)
    aer = pf.aerial_rgb(raster, ext, W, H)
    k = H / (y1 - y0)

    def to_px(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.column_stack([(xy[:, 0] - x0) * k, (y1 - xy[:, 1]) * k])

    soft = pf._soften(aer, 0.9, 0.0, 0.82)     # a pinch darker, so the white grid reads
    edge = [to_px(np.vstack([outline, outline[:1]]))]
    if cam:                                     # the camera's sight lines to the seen ground
        ang = np.unwrap(np.arctan2(outline[:, 1] - cam[1], outline[:, 0] - cam[0]))
        for v in (outline[np.argmin(ang)], outline[np.argmax(ang)]):
            edge.append(to_px([cam, v]))
    out_l = _line_layer((H, W), edge, lw * 1.6, GRID_RGB, 0.95)
    grid_a = _line_layer((H, W), [to_px(l) for l in lines], lw, GRID_RGB, 0.7)
    plan = pf.over(pf.over(soft, grid_a), out_l)
    if cam:
        plan = _camera(plan.clip(0, 255).astype(np.uint8), to_px([cam])[0], rad * 0.7)
    aerial_png = _markers(plan, to_px(en), labels, rad)

    # ---- the video laid onto the aerial (within the seen ground)
    cols, rows = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    world = np.column_stack([x0 + cols.ravel() / k, y1 - rows.ravel() / k])
    inside = MplPath(outline).contains_points(world)
    uvw = np.full((len(world), 2), -1.0)
    uvw[inside] = h.to_pixel(world[inside])
    mx = uvw[:, 0].reshape(H, W).astype(np.float32)
    my = uvw[:, 1].reshape(H, W).astype(np.float32)
    warped = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    m = inside.reshape(H, W) & (mx >= 0) & (my >= 0) & (mx < Wv) & (my < Hv)
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)[..., None]
    lay = soft * (1 - a) + warped.astype(np.float32) * a
    lay = pf.over(lay, out_l)
    if cam:
        lay = _camera(lay.clip(0, 255).astype(np.uint8), to_px([cam])[0], rad * 0.7)
    warped_png = lay.clip(0, 255).astype(np.uint8)

    # ---- write
    images, labels_dir = run.root / "images", run.root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, im in (("calibration_video", video_png), ("calibration_aerial", aerial_png),
                     ("calibration_warped", warped_png)):
        paths[name] = images / f"{name}.png"
        Image.fromarray(np.asarray(im).clip(0, 255).astype(np.uint8)).save(paths[name])
    panels = [np.asarray(video_png), np.asarray(aerial_png), warped_png]
    gap = np.full((out_h, GUTTER_PX, 3), 255, np.uint8)
    strip = np.hstack([panels[0], gap, panels[1], gap, panels[2]]).clip(0, 255).astype(np.uint8)
    paths["calibration"] = images / "calibration.png"
    Image.fromarray(strip).save(paths["calibration"])

    camtxt = f"Camera {cam_par['height_m']:.1f} m above the ground; the " \
        if cam_par.get("height_m") else "The "
    people = "people" in str(getattr(h, "method", ""))
    caption = (f"Calibration of {cfg.site}: {len(pts)} ground points clicked in the video and "
               f"on the aerial (magenta, same numbers)"
               + (", fitted together with the heights of the people walking through" if people
                  else "")
               + f"; white: a {GRID_M:g} m grid on the ground "
               f"the camera sees within {rng:g} m, drawn through the calibration in both views; "
               f"right: the video frame laid onto the aerial. {camtxt}"
               + (f"points fit to {h.rmse_m:.2f} m (RMSE)." if h.rmse_m is not None
                  else "points as clicked."))
    paths["caption_txt"] = labels_dir / "calibration_caption.txt"
    paths["caption_txt"].write_text(caption + "\n", encoding="utf-8")
    for variant in ("dark", "light"):
        paths[f"caption_{variant}"] = _caption_png(
            labels_dir / f"calibration_caption_{variant}.png", caption, variant)
    log(f"calibration picture: {paths['calibration']}")
    return paths


def _caption_png(path: Path, text: str, variant: str, width: int = 1600) -> Path:
    from .package import INK_VARIANTS, _canvas

    ink, _ = INK_VARIANTS[variant]
    font = pf._font(28)
    words, rows, cur = text.split(), [], ""
    _, d = _canvas(10, 10)
    for w in words:
        t = f"{cur} {w}".strip()
        if d.textlength(t, font=font) > width and cur:
            rows.append(cur)
            cur = w
        else:
            cur = t
    rows.append(cur)
    img, d = _canvas(width + 8, 12 + 40 * len(rows))
    for i, r in enumerate(rows):
        pf._text(d, (4, 4 + 40 * i), r, 28, "Light", ink)
    img.save(path)
    return path


# --------------------------------------------------------------------------- process sequence
SEQUENCE = (
    ("01_footage", "The footage: one frame of the clip (people blurred)."),
    ("02_calibration", ("Calibration: ground points clicked in the video (magenta) and a {grid} m "
                        "grid on the ground, drawn through the fitted camera.")),
    ("02b_calibration_plan", ("Calibration on the plan: the same points on the aerial, the "
                              "ground the camera sees (white grid) and the camera.")),
    ("03_tracked", ("Tracking: every person detected and followed from frame to frame, with "
                    "their trail so far.")),
    ("04_projection", ("Projection: the frame laid onto the plan through the calibration, with "
                       "the tracks so far in plan coordinates (dots: positions at this moment).")),
    ("05_tracks", "Analysis: all tracks of the clip on the plan, coloured by speed."),
    ("06_flowfield", ("The flow field: streamlines of the movement, colour = speed, width = how "
                      "much data supports it.")),
    ("07_density", "Occupancy: how long people were at each place (log scale)."),
)


def process_sequence(cfg: Config, run_dir: Path, raster: GeoRaster, h: Homography,
                     at_s: float = 2.0, out_h: int = 1080, log=print) -> dict:
    """The whole process of a clip, one image per step, at one moment (at_s seconds into the
    run), written to <run>/images/process/: 01 footage, 02 calibration (video), 02b
    calibration (plan), 03 tracked, 04 projection, 05 tracks, 06 flow field, 07 density, an
    overview of all, and the captions (labels/process_captions.txt). The video steps share the
    frame; the plan steps share the extent and size of the run's plan images (they stack)."""
    import json
    import shutil

    import pandas as pd

    from . import original as og
    from .layout import open_run
    from .registration import load_registration
    from .video import read_frame
    from .visuals import _extent

    run = open_run(run_dir)
    meta = json.loads(run.meta.read_text(encoding="utf-8"))
    fps = float(meta["fps"])
    start = int(meta.get("start_frame", 0))
    frame = start + int(round(at_s * fps))
    out = run.images / "process"
    out.mkdir(parents=True, exist_ok=True)
    paths: dict = {}

    def save(name, im):
        paths[name] = out / f"{name}.png"
        Image.fromarray(np.asarray(im).clip(0, 255).astype(np.uint8)).save(paths[name])

    img, extra = blurred_still(cfg, run_dir, frame, log)
    Hv, Wv = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    reg = load_registration(run.registration)[0] if run.registration.exists() else None
    to_raw = np.linalg.inv(reg[frame]) if reg is not None and frame in reg else np.eye(3)

    def raw_px(uv):         # calibration-frame pixels -> this frame's pixels (camera motion)
        uv = np.asarray(uv, float).reshape(-1, 2)
        res = np.full_like(uv, np.nan)
        ok = np.isfinite(uv).all(axis=1)
        if ok.any():
            res[ok] = cv2.perspectiveTransform(uv[ok][None], to_raw)[0]
        return res

    s = out_h / Hv
    lw = max(1.2, 1.6 * out_h / 1080)
    rad = 17 * out_h / 1080
    small = (int(round(Wv * s)), out_h)

    # 01 footage
    save("01_footage", cv2.resize(rgb, small, interpolation=cv2.INTER_AREA))

    # 02 calibration in the video
    cam_par = getattr(h, "camera_params", None) or {}
    cam = (float(cam_par["E"]), float(cam_par["N"])) if cam_par.get("E") is not None else None
    rng = seen_range(cfg, h)
    outline = seen_ground(h, (Wv, Hv), rng, cam)
    lines = grid_lines(outline)
    pts = [p for p in h.points if p.get("inlier", True)]
    uv = np.array([[p["u"], p["v"]] for p in pts], float).reshape(-1, 2)
    en = np.array([[p["E"], p["N"]] for p in pts], float).reshape(-1, 2)
    labels = [str(k + 1) for k in range(len(pts))]
    vid = cv2.resize(rgb, small, interpolation=cv2.INTER_AREA)
    vid = pf.over(vid, _line_layer(vid.shape[:2], [raw_px(h.to_pixel(ln)) * s for ln in lines],
                                   lw, GRID_RGB, 0.7))
    save("02_calibration", _markers(vid, raw_px(uv) * s, labels, rad))

    # 03 tracked: that frame of the overlay video (trails so far, boxes; people blurred)
    ov = run.videos / "overlay_clean.mp4"
    if ov.exists():
        from .debug_video import blur_boxes

        tr = read_frame(ov, frame - start)
        blur_boxes(tr, extra)                       # the still's extra people too
        save("03_tracked", cv2.resize(cv2.cvtColor(tr, cv2.COLOR_BGR2RGB), small,
                                      interpolation=cv2.INTER_AREA))
    else:
        log("no overlay video (render it with debug-video): step 03 left out")

    # the plan steps: the extent and size of the run's plan images
    points = pd.read_csv(run.points)
    ext = _extent(points, raster, cfg)
    pk = cfg.get("package") or {}
    W, H = pf.plan_size(ext, int(pk.get("plan_height_px", 2400)))
    k = H / (ext[3] - ext[2])
    base = og.grey_base(raster, ext, (W, H), float(cfg["visuals"]["basemap_brightness"]))
    prad, plw = rad * H / out_h * 0.8, lw * H / out_h

    def to_plan(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.column_stack([(xy[:, 0] - ext[0]) * k, (ext[3] - xy[:, 1]) * k])

    edge = [to_plan(np.vstack([outline, outline[:1]]))]
    if cam:
        ang = np.unwrap(np.arctan2(outline[:, 1] - cam[1], outline[:, 0] - cam[0]))
        for v in (outline[np.argmin(ang)], outline[np.argmax(ang)]):
            edge.append(to_plan([cam, v]))
    edge_l = _line_layer((H, W), edge, plw * 1.6, GRID_RGB, 0.95)

    # 02b calibration on the plan
    plan = pf.over(pf.over(base, _line_layer((H, W), [to_plan(ln) for ln in lines], plw,
                                             GRID_RGB, 0.6)), edge_l)
    if cam:
        plan = _camera(plan.clip(0, 255).astype(np.uint8), to_plan([cam])[0], prad * 0.7)
    save("02b_calibration_plan", _markers(plan, to_plan(en), labels, prad))

    # 04 projection: the frame on the plan, the tracks up to this moment, positions now
    cols, rows = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    world = np.column_stack([ext[0] + cols.ravel() / k, ext[3] - rows.ravel() / k])
    inside = MplPath(outline).contains_points(world)
    uvw = np.full((len(world), 2), -1.0)
    uvw[inside] = raw_px(h.to_pixel(world[inside]))
    uvw[~np.isfinite(uvw).all(axis=1)] = -1.0
    mx = uvw[:, 0].reshape(H, W).astype(np.float32)
    my = uvw[:, 1].reshape(H, W).astype(np.float32)
    warped = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    m = inside.reshape(H, W) & (mx >= 0) & (my >= 0) & (mx < Wv) & (my < Hv)
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)[..., None]
    proj = base * (1 - a) + warped.astype(np.float32) * a
    ranges = pf.group_ranges(cfg, points)
    t_now = float(points.loc[(points["frame"] - frame).abs().idxmin(), "time_s"]) \
        if len(points) else 0.0
    sofar = points[points["time_s"] <= t_now]
    proj = pf.over(proj, og.tracks_layer(sofar, ext, (W, H), ranges,
                                         float(cfg["visuals"]["line_width"]), False))
    proj = pf.over(proj, edge_l).clip(0, 255).astype(np.uint8)
    if cam:
        proj = _camera(proj, to_plan([cam])[0], prad * 0.7)
    now = sofar[sofar["time_s"] == t_now]
    for (x, y), sp, g in zip(to_plan(now[["x", "y"]].to_numpy()), now["speed"], now["group"]):
        lo, hi = ranges.get(g, (0.0, 2.5))
        c = tuple(int(v) for v in pf.speed_rgb(sp, lo, hi))
        cv2.circle(proj, (int(x), int(y)), int(prad * 0.55), (20, 20, 22), -1, cv2.LINE_AA)
        cv2.circle(proj, (int(x), int(y)), int(prad * 0.38), c, -1, cv2.LINE_AA)
    save("04_projection", proj)

    # 05-07: the run's plan images, as they are
    for name, src in (("05_tracks", "plan_tracks.png"), ("06_flowfield", "plan_flowfield.png"),
                      ("07_density", "plan_density.png")):
        if (run.images / src).exists():
            paths[name] = out / f"{name}.png"
            shutil.copy2(run.images / src, paths[name])

    # an overview: the video steps in one row, the plan steps in the next (no text)
    g = GUTTER_PX
    vids = [np.asarray(Image.open(paths[n]).convert("RGB")) for n in
            ("01_footage", "02_calibration", "03_tracked") if n in paths]
    plans = [np.asarray(Image.open(paths[n]).convert("RGB")) for n in
             ("02b_calibration_plan", "04_projection", "05_tracks", "06_flowfield", "07_density")
             if n in paths]

    def row(ims, height):
        gap = np.full((height, g, 3), 255, np.uint8)
        ims = [cv2.resize(i, (int(round(i.shape[1] * height / i.shape[0])), height),
                          interpolation=cv2.INTER_AREA) for i in ims]
        return np.hstack([x for i in ims for x in (i, gap)][:-1])

    top = row(vids, out_h)
    ph = int(round((top.shape[1] - g * (len(plans) - 1)) / sum(p.shape[1] / p.shape[0]
                                                               for p in plans)))
    bottom = row(plans, ph)
    wmax = max(top.shape[1], bottom.shape[1])
    top, bottom = (np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)), constant_values=255)
                   for r in (top, bottom))
    save("00_overview", np.vstack([top, np.full((g, wmax, 3), 255, np.uint8), bottom]))

    run.labels.mkdir(parents=True, exist_ok=True)
    cap = [f"{cfg.site}, {at_s:g} s into the clip (frame {frame}). People blurred."]
    cap += [f"{n}: " + t.format(grid=f"{GRID_M:g}") for n, t in SEQUENCE if n in paths]
    paths["captions"] = run.labels / "process_captions.txt"
    paths["captions"].write_text("\n".join(cap) + "\n", encoding="utf-8")
    log(f"process sequence: {out}")
    return paths
