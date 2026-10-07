"""Interactive calibration: pick matching ground points in a video frame and a GeoTIFF.

Controls (shown in the window title too):
  left click            add a point (video first, then the matching point on the map)
  right click / u       undo the last click
  scroll wheel          zoom around the cursor (toolbar zoom/pan also work)
  enter                 finish and save
  escape                quit without saving
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .geo import GeoRaster, load_geotiff
from .homography import Homography, fit_homography, report
from .lens import Lens
from .video import read_frame


def load_points_csv(path) -> tuple[np.ndarray, np.ndarray]:
    px, world = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            px.append([float(row["u"]), float(row["v"])])
            world.append([float(row["E"]), float(row["N"])])
    return np.array(px), np.array(world)


def save_points_csv(path, px, world) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "u", "v", "E", "N"])
        for i, ((u, v), (e, n)) in enumerate(zip(px, world), 1):
            w.writerow([i, f"{u:.2f}", f"{v:.2f}", f"{e:.3f}", f"{n:.3f}"])


def _scroll_zoom(ax, event, base=1.4):
    if event.inaxes is not ax or event.xdata is None:
        return
    scale = 1 / base if event.button == "up" else base
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx, cy = event.xdata, event.ydata
    ax.set_xlim(cx - (cx - x0) * scale, cx + (x1 - cx) * scale)
    ax.set_ylim(cy - (cy - y0) * scale, cy + (y1 - cy) * scale)
    ax.figure.canvas.draw_idle()


def pick_points(frame_rgb: np.ndarray, raster: GeoRaster, init_px=None, init_world=None,
                warn_m: float = 0.5):
    """Open the two-pane picker. Returns (px, world) arrays, or None if cancelled."""
    import matplotlib.pyplot as plt

    px: list[list[float]] = [list(p) for p in (init_px if init_px is not None else [])]
    world: list[list[float]] = [list(p) for p in (init_world if init_world is not None else [])]
    state = {"done": False, "cancel": False}

    fig, (axv, axm) = plt.subplots(1, 2, figsize=(17, 8.5))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.9, bottom=0.04, wspace=0.08)
    axv.imshow(frame_rgb)
    axv.set_title("VIDEO FRAME")
    axm.imshow(raster.image, extent=raster.extent, interpolation="bilinear")
    axm.set_title(f"MAP  (EPSG:{raster.epsg})")
    axm.ticklabel_format(useOffset=False, style="plain")
    artists: list = []

    def redraw():
        for a in artists:
            a.remove()
        artists.clear()
        for i, p in enumerate(px):
            artists.append(axv.plot(*p, "+", color="#ff2d55", ms=16, mew=1.6)[0])
            artists.append(axv.annotate(str(i + 1), p, xytext=(6, 6), textcoords="offset points",
                                        color="#ff2d55", fontsize=11, weight="bold"))
        for i, p in enumerate(world):
            artists.append(axm.plot(*p, "+", color="#00e5ff", ms=16, mew=1.6)[0])
            artists.append(axm.annotate(str(i + 1), p, xytext=(6, 6), textcoords="offset points",
                                        color="#00e5ff", fontsize=11, weight="bold"))
        n = min(len(px), len(world))
        status = f"{n} pairs"
        if len(px) > len(world):
            status += f" | now click point {len(px)} on the MAP"
        else:
            status += f" | now click point {len(px) + 1} on the VIDEO"
        if n >= 5:
            try:
                h = fit_homography(np.array(px[:n]), np.array(world[:n]))
                proj = h.to_world(np.array(px[:n]))
                for (e, nn), (pe, pn) in zip(world[:n], proj):
                    if np.isfinite(pe):
                        artists.append(axm.plot([e, pe], [nn, pn], "-", color="yellow", lw=1.2)[0])
                        artists.append(axm.plot(pe, pn, "x", color="yellow", ms=8)[0])
                flag = "  <-- TOO HIGH" if h.rmse_m > warn_m else ""
                status += f" | RMSE {h.rmse_m:.2f} m{flag}"
                worst = max(h.points, key=lambda p: p["error_m"])
                status += f" | worst #{worst['id']} {worst['error_m']:.2f} m"
            except Exception as e:  # noqa: BLE001
                status += f" | fit failed: {e}"
        fig.suptitle(status + "\nleft-click add · right-click/u undo · scroll zoom · "
                     "enter save · esc cancel", fontsize=11)
        fig.canvas.draw_idle()

    def on_click(ev):
        tb = getattr(fig.canvas.manager, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return  # zoom/pan tool active
        if ev.button == 3:
            undo()
            return
        if ev.button != 1 or ev.xdata is None:
            return
        if ev.inaxes is axv and len(px) == len(world):
            px.append([ev.xdata, ev.ydata])
        elif ev.inaxes is axm and len(px) == len(world) + 1:
            world.append([ev.xdata, ev.ydata])
        redraw()

    def undo():
        if len(px) > len(world):
            px.pop()
        elif world:
            world.pop()
        redraw()

    def on_key(ev):
        if ev.key in ("u", "backspace"):
            undo()
        elif ev.key == "enter":
            state["done"] = True
            plt.close(fig)
        elif ev.key == "escape":
            state["cancel"] = True
            plt.close(fig)

    def on_scroll(ev):
        _scroll_zoom(axv, ev)
        _scroll_zoom(axm, ev)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    redraw()
    plt.show()
    if state["cancel"]:
        return None
    n = min(len(px), len(world))
    return np.array(px[:n]), np.array(world[:n])


def warp_frame_to_map(frame_bgr: np.ndarray, h: Homography, raster: GeoRaster,
                      max_dim: int = 3000) -> np.ndarray:
    """Project the video frame onto the map raster grid (RGB, NaN-free, black where invalid)."""
    H_img, W_img = raster.image.shape[:2]
    s = min(1.0, max_dim / max(H_img, W_img))
    out_w, out_h = int(W_img * s), int(H_img * s)
    cols, rows = np.meshgrid(np.arange(out_w) + 0.5, np.arange(out_h) + 0.5)
    world = raster.pixel_to_world(np.column_stack([cols.ravel() / s, rows.ravel() / s]))
    local = world - np.asarray(h.origin)
    Hinv = np.linalg.inv(h.H)
    ph = np.column_stack([local, np.ones(len(local))]) @ Hinv.T
    valid = np.sign(ph[:, 2]) == np.sign(h.w_sign)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = ph[:, :2] / ph[:, 2:3]
    uv[~valid] = -1e6
    mapx = uv[:, 0].reshape(out_h, out_w).astype(np.float32)
    mapy = uv[:, 1].reshape(out_h, out_w).astype(np.float32)
    warped = cv2.remap(frame_bgr, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


def save_check_image(path, frame_bgr, h: Homography, raster: GeoRaster) -> None:
    """Side-by-side: map with the warped video blended on top, plus point errors."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    warped = warp_frame_to_map(frame_bgr, h, raster)
    base = cv2.resize(raster.image, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_AREA)
    mask = warped.sum(axis=2) > 0
    blend = base.copy()
    blend[mask] = (0.5 * base[mask] + 0.5 * warped[mask]).astype(np.uint8)

    pts = np.array([[p["E"], p["N"]] for p in h.points])
    proj = h.to_world(np.array([[p["u"], p["v"]] for p in h.points]))
    pad = max(20.0, 0.3 * np.ptp(pts, axis=0).max())
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, img, title in ((axes[0], blend, "video warped onto map (50% blend)"),
                           (axes[1], base, "calibration points: + clicked, x projected")):
        ax.imshow(img, extent=raster.extent)
        ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
        ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
        ax.set_title(title)
        ax.ticklabel_format(useOffset=False, style="plain")
    ax = axes[1]
    for p, (pe, pn) in zip(h.points, proj):
        c = "#00e5ff" if p["inlier"] else "#ff2d55"
        ax.plot(p["E"], p["N"], "+", color=c, ms=14, mew=2)
        if np.isfinite(pe):
            ax.plot([p["E"], pe], [p["N"], pn], "-", color="yellow", lw=1.5)
            ax.plot(pe, pn, "x", color="yellow", ms=9, mew=2)
        ax.annotate(f"{p['id']}: {p['error_m']:.2f} m", (p["E"], p["N"]), xytext=(8, 8),
                    textcoords="offset points", color=c, fontsize=10, weight="bold")
    fig.suptitle(f"RMSE {h.rmse_m:.3f} m · max {h.max_error_m:.3f} m"
                 + (f" · leave-one-out RMSE {h.loo_rmse_m:.3f} m" if h.loo_rmse_m else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_calibration(video: Path, geotiff: Path, out_json: Path, frame: int = 0,
                    lens: Lens | None = None, points_csv: Path | None = None,
                    interactive: bool = True, ransac_thresh_m: float = 1.0,
                    warn_m: float = 0.5, log=print) -> Homography:
    frame_bgr = read_frame(video, frame)
    if lens is not None:
        frame_bgr = lens.undistort_image(frame_bgr)
    raster = load_geotiff(geotiff, warn=log)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    default_csv = out_json.with_name(out_json.stem + "_points.csv")

    init_px = init_world = None
    src_csv = points_csv or (default_csv if default_csv.exists() else None)
    if src_csv:
        init_px, init_world = load_points_csv(src_csv)
        log(f"loaded {len(init_px)} point pairs from {src_csv}")

    if interactive:
        import matplotlib.pyplot as plt  # noqa: F401  (ensure an interactive backend loads)

        res = pick_points(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), raster, init_px,
                          init_world, warn_m)
        if res is None:
            raise SystemExit("Calibration cancelled; nothing saved.")
        px, world = res
    else:
        if init_px is None:
            raise ValueError("Non-interactive calibration needs --points-csv")
        px, world = init_px, init_world

    h = fit_homography(px, world, ransac_thresh_m=ransac_thresh_m,
                       image_size=(frame_bgr.shape[1], frame_bgr.shape[0]),
                       undistorted=lens is not None, crs=f"EPSG:{raster.epsg}")
    h.reference_frame = int(frame)
    save_points_csv(default_csv, px, world)
    h.save(out_json)
    check_png = out_json.with_name(out_json.stem + "_check.png")
    save_check_image(check_png, frame_bgr, h, raster)
    log(report(h, warn_m))
    log(f"saved {out_json}\n      {default_csv}\n      {check_png}  <- open this and check "
        "that kerbs and markings line up")
    return h


def pick_roi(video: Path, frame: int = 0, out_json: Path | None = None, log=print):
    """Click a polygon on a video frame; enter to finish. Saves [[x, y], ...] JSON."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    img = cv2.cvtColor(read_frame(video, frame), cv2.COLOR_BGR2RGB)
    pts: list[list[float]] = []
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img)
    patch = Polygon(np.zeros((1, 2)), closed=True, fill=True, alpha=0.25, color="#00e5ff")
    ax.add_patch(patch)
    (line,) = ax.plot([], [], "o-", color="#00e5ff")
    done = {"ok": False}

    def update():
        a = np.array(pts) if pts else np.zeros((0, 2))
        line.set_data(a[:, 0] if len(a) else [], a[:, 1] if len(a) else [])
        if len(a) >= 3:
            patch.set_xy(a)
        ax.set_title(f"ROI: {len(pts)} vertices. Left-click add, right-click undo, enter save.\n"
                     "Exclude sky, far/blurry areas and reflections.")
        fig.canvas.draw_idle()

    def on_click(ev):
        tb = getattr(fig.canvas.manager, "toolbar", None)
        if (tb is not None and getattr(tb, "mode", "")) or ev.inaxes is not ax:
            return
        if ev.button == 1:
            pts.append([round(ev.xdata, 1), round(ev.ydata, 1)])
        elif ev.button == 3 and pts:
            pts.pop()
        update()

    def on_key(ev):
        if ev.key == "enter":
            done["ok"] = True
            plt.close(fig)
        elif ev.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.show()
    if not done["ok"] or len(pts) < 3:
        raise SystemExit("ROI cancelled (need >= 3 vertices).")
    if out_json:
        Path(out_json).write_text(json.dumps(pts), encoding="utf-8")
        log(f"saved {out_json}. In the site config set:\n  detection:\n    roi: {out_json}")
    log("or paste inline:\n  detection:\n    roi: " + json.dumps(pts))
    return pts
