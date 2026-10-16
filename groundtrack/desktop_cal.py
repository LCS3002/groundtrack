"""Calibration and region of interest inside the desktop app.

The app shows a video frame and the aerial side by side; the user clicks pairs of ground
points. Each change is fitted here exactly as the `groundtrack calibrate` picker does (a
physical camera when the site knows where the camera stood, else a homography), and the
answer comes back as numbers plus the overlays to draw: projected points, a ground grid in
the video and the camera's view on the aerial. Saving writes the pairs to the usual
`<calibration>_points.csv` and runs `groundtrack calibrate --points-csv ... --no-gui`, so the
saved calibration, check images and report are the command line's own.

The ROI is saved where `groundtrack roi` saves it (`<site>_roi.json` next to the config, or
the file `detection.roi` names). Every file replaced is copied to a `backup` folder first.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import load_config

MAP_MAX_PX = 6000          # the aerial as loaded for the picker (same as load_geotiff)
LIVE_MIN_CAMERA = 3        # pairs needed for a live fit: with a known camera position ...
LIVE_MIN_PLANE = 5         # ... or a plain homography (the CLI picker's numbers)


class Site:
    """One site config's frame / aerial / terrain, loaded once."""

    def __init__(self, config: Path):
        self.config = Path(config)
        self.cfg = load_config(self.config)
        self.mtime = self.config.stat().st_mtime
        self._lock = threading.Lock()
        self._frames: dict[int, np.ndarray] = {}
        self._raster = None
        self._terrain = False                  # False: not resolved yet; None: flat
        self._lens = False
        self._info = None

    # ---- inputs
    @property
    def video(self) -> Path:
        v = self.cfg.path("video")
        if v is None or not v.exists():
            raise FileNotFoundError(f"video of {self.cfg.site} not found ({v})")
        return v

    @property
    def info(self):
        if self._info is None:
            from .video import video_info

            self._info = video_info(self.video, self.cfg.get("fps"))
        return self._info

    @property
    def lens(self):
        if self._lens is False:
            from .lens import Lens

            self._lens = Lens.load(self.cfg.path("lens")) if self.cfg.get("lens") else None
        return self._lens

    def frame(self, n: int) -> np.ndarray:
        """BGR frame n (lens-undistorted when the site has a lens: clicks are made on that)."""
        from .video import read_frame

        n = int(max(0, min(n, max(self.info.n_frames - 1, 0))))
        with self._lock:
            if n not in self._frames:
                img = read_frame(self.video, n)
                if self.lens is not None:
                    img = self.lens.undistort_image(img)
                if len(self._frames) > 6:
                    self._frames.clear()
                self._frames[n] = img
            return self._frames[n]

    @property
    def raster(self):
        if self._raster is None:
            from .geo import load_geotiff

            g = self.cfg.path("geotiff")
            if g is None or not g.exists():
                raise FileNotFoundError(f"aerial (GeoTIFF) of {self.cfg.site} not found ({g})")
            with self._lock:
                if self._raster is None:
                    self._raster = load_geotiff(g, max_dim=MAP_MAX_PX, warn=lambda *_: None)
        return self._raster

    @property
    def terrain(self):
        if self._terrain is False:
            from .terrain import resolve_terrain

            self._terrain = resolve_terrain(self.cfg, log=lambda *_: None)
        return self._terrain

    @property
    def size(self) -> tuple[int, int]:
        return (self.info.width, self.info.height)

    # ---- paths
    @property
    def homography(self) -> Path:
        hp = self.cfg.path("homography")
        if hp is None:
            raise ValueError("set `homography:` (where to save the calibration) in the config")
        return hp

    @property
    def points_csv(self) -> Path:
        hp = self.homography
        return hp.with_name(hp.stem + "_points.csv")

    def roi_target(self) -> Path:
        """Where the ROI lives: the file detection.roi names, else <site>_roi.json."""
        roi = self.cfg["detection"].get("roi")
        if isinstance(roi, list):
            raise ValueError("this config has the ROI written inline (detection.roi: [...]); "
                             "replace it with a file name to edit it here")
        if isinstance(roi, str):
            p = Path(roi)
            return p if p.is_absolute() else self.cfg.base_dir / p
        return self.cfg.base_dir / f"{self.cfg.site}_roi.json"


class Sites:
    """Site objects by config path, reloaded when the config file changes."""

    def __init__(self):
        self._sites: dict[str, Site] = {}

    def get(self, config: Path) -> Site:
        key = str(Path(config).resolve())
        s = self._sites.get(key)
        if s is None or s.mtime != Path(config).stat().st_mtime:
            old = s
            s = Site(config)
            if old is not None and old.cfg.path("geotiff") == s.cfg.path("geotiff"):
                s._raster = old._raster                 # a camera edit keeps the aerial
            self._sites[key] = s
        return s


# --------------------------------------------------------------------------- images
def jpeg(img_bgr: np.ndarray, w: int = 0, quality: int = 90) -> bytes:
    if w and img_bgr.shape[1] > w:
        h = int(round(img_bgr.shape[0] * w / img_bgr.shape[1]))
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode image")
    return buf.tobytes()


def frame_jpeg(site: Site, n: int, w: int = 0) -> bytes:
    return jpeg(site.frame(n), w)


def map_jpeg(site: Site, w: int = 0) -> bytes:
    return jpeg(cv2.cvtColor(site.raster.image, cv2.COLOR_RGB2BGR), w, 88)


# --------------------------------------------------------------------------- state
def state(site: Site) -> dict:
    """Everything the calibrator needs to open: video, aerial, saved pairs, camera, ROI."""
    from .calibrate import load_points_csv
    from .detect import reference_frame_for, site_roi

    cfg = site.cfg
    d: dict = {"site": cfg.site, "warn_m": 0.5}
    i = site.info
    d["video"] = {"width": i.width, "height": i.height, "fps": i.fps, "frames": i.n_frames,
                  "duration_s": round(i.duration_s, 2)}
    r = site.raster
    d["map"] = {"left": r.left, "right": r.right, "bottom": r.bottom, "top": r.top,
                "epsg": r.epsg, "width": int(r.image.shape[1]), "height": int(r.image.shape[0]),
                "native_res_m": r.native_res}
    d["camera_position"] = cfg.get("camera_position")
    d["terrain"] = cfg.get("terrain") not in (None, "", False)
    d["reference_frame"] = reference_frame_for(cfg)
    d["pairs"] = []
    if site.points_csv.exists():
        px, world = load_points_csv(site.points_csv)
        d["pairs"] = [{"u": float(u), "v": float(v), "E": float(e), "N": float(n)}
                      for (u, v), (e, n) in zip(px, world)]
    hp = site.homography
    d["saved"] = None
    if hp.exists():
        from .homography import Homography

        h = Homography.load(hp)
        d["saved"] = {"rmse_m": h.rmse_m, "loo_rmse_m": h.loo_rmse_m, "frame": h.reference_frame,
                      "n_points": len(h.points or []), "method": getattr(h, "method", "points"),
                      "camera": getattr(h, "camera_params", None),
                      "mtime": hp.stat().st_mtime}
    try:
        roi = site_roi(cfg)
        d["roi"] = roi.round(1).tolist() if roi is not None else None
        d["roi_file"] = str(site.roi_target())
        d["roi_inline"] = False
    except ValueError:
        d["roi"], d["roi_file"], d["roi_inline"] = None, None, True
    return d


# --------------------------------------------------------------------------- fitting
def _fit(site: Site, px: np.ndarray, world: np.ndarray, prior: dict | None):
    from .homography import fit_homography

    if prior:
        from .posefit import camera_calibration

        return camera_calibration(px, world, site.size, prior, terrain=site.terrain)
    return fit_homography(px, world, image_size=site.size)


def _finite(a) -> bool:
    return bool(np.all(np.isfinite(a)))


def _num(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 4)


def _camera(h) -> dict | None:
    cp = getattr(h, "camera_params", None)
    if cp:
        return {k: (round(float(v), 3) if isinstance(v, (int, float)) else v)
                for k, v in cp.items()}
    c = h.camera() if h.image_size else None
    if not c:
        return None
    return {k: round(float(v), 3) for k, v in c.items()
            if k in ("E", "N", "height_m", "focal_px", "hfov_deg")}


def _visible(h, uv: np.ndarray, world: np.ndarray, size) -> np.ndarray:
    """Pixels that really show these ground points: in front of the camera (the way back
    lands on the same spot) and inside the frame (with a margin)."""
    w, hh = size
    ok = np.isfinite(uv).all(axis=1)
    ok &= (uv[:, 0] > -0.25 * w) & (uv[:, 0] < 1.25 * w) & (uv[:, 1] > -0.25 * hh) \
        & (uv[:, 1] < 1.25 * hh)
    if ok.any():
        back = np.full_like(world, np.nan)
        back[ok] = h.to_world(uv[ok])
        err = np.linalg.norm(back - world, axis=1)
        ok &= np.nan_to_num(err, nan=1e9) < 0.05 * np.maximum(
            1.0, np.linalg.norm(world - world.mean(axis=0), axis=1))
    return ok


def ground_grid(h, size, centre, span_m: float) -> dict:
    """Ground grid lines around centre, in video pixels: {step_m, lines: [[[u, v], ...]]}."""
    step = 2.0 if span_m <= 50 else 5.0 if span_m <= 150 else 10.0
    half = min(max(span_m * 0.75, 10.0), 400.0)
    cx, cy = (np.round(np.asarray(centre) / step) * step).tolist()
    ticks = np.arange(-half, half + step / 2, step)[:81]
    samples = np.linspace(-half, half, 41)
    segs = [np.column_stack([np.full_like(samples, cx + t), cy + samples]) for t in ticks] + \
           [np.column_stack([cx + samples, np.full_like(samples, cy + t)]) for t in ticks]
    world = np.concatenate(segs)                  # all lines at once: one projection
    uv = h.to_pixel(world)
    ok = _visible(h, uv, world, size)
    lines = []
    k = len(samples)
    for i in range(len(segs)):
        run: list = []
        for p, good in zip(uv[i * k:(i + 1) * k], ok[i * k:(i + 1) * k]):
            if good:                              # split where the line leaves the view
                run.append([round(float(p[0]), 1), round(float(p[1]), 1)])
            elif len(run) > 1:
                lines.append(run)
                run = []
            else:
                run = []
        if len(run) > 1:
            lines.append(run)
    return {"step_m": step, "lines": lines[:600]}


def footprint(h, size, max_range_m: float) -> list:
    """The ground the camera sees, as a polygon on the aerial (world coordinates)."""
    w, hh = size
    us, vs = np.meshgrid(np.linspace(0, w, 25), np.linspace(0, hh, 17))
    uv = np.column_stack([us.ravel(), vs.ravel()])
    world = h.to_world(uv)
    ok = np.isfinite(world).all(axis=1)
    cam = _camera(h)
    if cam and cam.get("E") is not None:
        ok &= np.linalg.norm(world - [cam["E"], cam["N"]], axis=1) < max_range_m
    pts = world[ok]
    if len(pts) < 3:
        return []
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    return [[round(float(e), 2), round(float(n), 2)] for e, n in hull]


def fit(site: Site, pairs: list[dict], query: list | None = None,
        prior: dict | None = None) -> dict:
    """Live fit of the clicked pairs: errors, projected points, grid, view footprint and,
    with query [u, v], where that video point should be on the aerial."""
    from .homography import point_spread

    prior = prior if prior is not None else site.cfg.get("camera_position")
    px = np.array([[p["u"], p["v"]] for p in pairs], float).reshape(-1, 2)
    world = np.array([[p["E"], p["N"]] for p in pairs], float).reshape(-1, 2)
    need = LIVE_MIN_CAMERA if prior else LIVE_MIN_PLANE
    out: dict = {"n": len(px), "need": need, "method": "camera" if prior else "homography",
                 "ok": False}
    if len(px) >= 2:
        out["spread"] = round(float(point_spread(world)), 3)
    if len(px) < need:
        return out
    t0 = time.time()
    try:
        h = _fit(site, px, world, prior)
    except Exception as e:  # noqa: BLE001 (the message is shown as is)
        out["error"] = str(e)
        return out
    proj = h.to_world(px)
    out.update(ok=True, rmse_m=_num(h.rmse_m), max_error_m=_num(h.max_error_m),
               loo_rmse_m=_num(h.loo_rmse_m), camera=_camera(h),
               dropped=getattr(h, "consensus_dropped", None) or [])
    out["points"] = [{"id": p["id"], "error_m": _num(p.get("error_m")),
                      "loo_error_m": _num(p.get("loo_error_m")), "inlier": bool(p["inlier"]),
                      "proj": [round(float(pr[0]), 3), round(float(pr[1]), 3)]
                      if _finite(pr) else None}
                     for p, pr in zip(h.points, proj)]
    span = float(np.ptp(world, axis=0).max()) if len(world) > 1 else 20.0
    try:
        out["grid"] = ground_grid(h, site.size, world.mean(axis=0), span)
    except Exception:  # noqa: BLE001 (an overlay only)
        out["grid"] = None
    height = (out["camera"] or {}).get("height_m") or 10.0
    try:
        out["footprint"] = footprint(h, site.size, max(10 * abs(float(height)), 1.5 * span, 60))
    except Exception:  # noqa: BLE001
        out["footprint"] = []
    if query is not None and np.isfinite(h.rmse_m) and h.rmse_m <= 5:
        q = h.to_world(np.array([query], float))[0]
        if _finite(q):
            out["guess"] = {"E": round(float(q[0]), 3), "N": round(float(q[1]), 3),
                            "r_m": round(max(8.0, 4 * float(h.rmse_m) + 6), 2)}
    out["ms"] = int((time.time() - t0) * 1000)
    return out


# --------------------------------------------------------------------------- saving
def backup(p: Path) -> Path | None:
    """Copy p into <its folder>/backup/<name>.<timestamp><ext> before it is replaced."""
    if not p.exists():
        return None
    b = p.parent / "backup" / f"{p.stem}.{time.strftime('%Y%m%d-%H%M%S')}{p.suffix}"
    b.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, b)
    return b


def save_pairs(site: Site, pairs: list[dict]) -> Path:
    """Write the pairs where `groundtrack calibrate` keeps them (the old file backed up)."""
    from .calibrate import save_points_csv

    if len(pairs) < 3:
        raise ValueError("at least 3 point pairs are needed")
    for p in pairs:
        if not all(np.isfinite([p["u"], p["v"], p["E"], p["N"]])):
            raise ValueError("a point has no position")
    csv = site.points_csv
    csv.parent.mkdir(parents=True, exist_ok=True)
    backup(csv)
    for suffix in (".json", "_check.png", "_check_video.jpg"):
        backup(site.homography.with_name(site.homography.stem + suffix))
    save_points_csv(csv, [[p["u"], p["v"]] for p in pairs], [[p["E"], p["N"]] for p in pairs])
    return csv


def save_roi(site: Site, pts: list) -> Path:
    a = np.asarray(pts, float)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 3 or not np.isfinite(a).all():
        raise ValueError("the area needs at least 3 corners")
    target = site.roi_target()
    backup(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(a.round(1).tolist()), encoding="utf-8")
    return target


def clear_roi(site: Site) -> Path | None:
    """Remove the ROI file (moved to backup/): the whole frame is used again."""
    target = site.roi_target()
    if not target.exists():
        return None
    b = backup(target)
    target.unlink()
    return b
