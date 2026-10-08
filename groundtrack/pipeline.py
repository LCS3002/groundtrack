"""Orchestration: one output folder per run."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from . import exports, visuals
from .config import Config
from .geo import load_geotiff
from .homography import Homography
from .lens import Lens
from .trajectories import assign_track_classes, process_tracks
from .video import video_info


class RunLog:
    def __init__(self, path: Path | None = None):
        self.path = path

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


def new_run_dir(cfg: Config, name: str | None = None) -> Path:
    root = cfg.path("output_dir") or (cfg.base_dir / "runs")
    d = root / cfg.site / (name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _require(path: Path | None, what: str) -> Path:
    if path is None:
        raise ValueError(f"`{what}` is not set in the site config")
    if not path.exists():
        raise FileNotFoundError(f"{what}: {path} does not exist")
    return path


def load_calibration(cfg: Config, video: Path | None = None) -> tuple[Homography, Lens | None]:
    h = Homography.load(_require(cfg.path("homography"), "homography"))
    lens = Lens.load(cfg.path("lens")) if cfg.get("lens") else None
    if video is not None and h.image_size:
        info = video_info(video, cfg.get("fps"))
        if (info.width, info.height) != tuple(h.image_size):
            raise ValueError(
                f"Video is {info.width}x{info.height} but the homography was calibrated on a "
                f"{h.image_size[0]}x{h.image_size[1]} frame. Re-calibrate with this footage "
                "(same camera position, resolution and lens).")
    return h, lens


def load_run_registration(run_dir: Path, h: Homography, log=print) -> dict | None:
    """Per-frame camera-motion registration of a run, if it was tracked with stabilize."""
    p = Path(run_dir) / "registration.npz"
    if not p.exists():
        return None
    from .registration import load_registration

    reg, ref, _ = load_registration(p)
    if ref != h.reference_frame:
        raise ValueError(
            f"This run was stabilized to frame {ref} but the homography was clicked on frame "
            f"{h.reference_frame}. Re-run `track` (or `run`) after calibrating.")
    log(f"  camera-motion compensation: {len(reg)} frames aligned to frame {ref}")
    return reg


def stage_track(cfg: Config, run_dir: Path, device: str, log=print,
                video: Path | None = None, max_frames: int | None = None) -> pd.DataFrame:
    from .detect import run_tracking

    video = video or _require(cfg.path("video"), "video")
    return run_tracking(cfg, video, run_dir / "raw_tracks.csv", device, log=log,
                        max_frames=max_frames)


def stage_process(cfg: Config, run_dir: Path, log=print, raw: pd.DataFrame | None = None,
                  video: Path | None = None) -> dict:
    t0 = time.time()
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "raw_tracks_meta.json").read_text(encoding="utf-8"))
    if raw is None:
        raw = pd.read_csv(run_dir / "raw_tracks.csv")
    video = video or Path(meta["video"])
    h, lens = load_calibration(cfg, video if video.exists() else None)
    # keep the run's calibration snapshot in sync with the one these results are made with
    # (the overlay video projects the results back into the footage with it)
    shutil.copy2(cfg.path("homography"), run_dir / "homography_used.json")
    fps, stride = float(meta["fps"]), int(meta["vid_stride"])

    reg = load_run_registration(run_dir, h, log)
    log("projecting, cleaning and smoothing tracks ...")
    points, summary = process_tracks(raw, cfg, h, fps, stride, lens, log=log, registration=reg)
    log(f"  {summary.shape[0]} tracks, {len(points)} samples "
        f"({int(points['predicted'].sum()) if len(points) else 0} predicted/interpolated)")

    out = {}
    out["points"] = run_dir / "points.csv"
    exports.write_points_csv(points, out["points"])
    out["track_summary"] = run_dir / "track_summary.csv"
    summary.to_csv(out["track_summary"], index=False)
    out["tracks_geojson"] = run_dir / "tracks.geojson"
    exports.write_geojson(exports.tracks_geojson(points, summary), out["tracks_geojson"])
    out["gaps_geojson"] = run_dir / "predicted_gaps.geojson"
    exports.write_geojson(exports.predicted_segments_geojson(points), out["gaps_geojson"])

    gcfg = cfg["grid"]
    cell = float(gcfg["cell_size_m"])
    out["field_grid"] = run_dir / "field_grid.csv"
    exports.field_grid(points, cell, gcfg["include_predicted"]).to_csv(out["field_grid"],
                                                                       index=False)
    present = sorted(set(points["group"])) if len(points) else []
    if len(present) > 1:
        for gname in present:
            p = run_dir / f"field_grid_{gname}.csv"
            exports.field_grid(points[points["group"] == gname], cell,
                               gcfg["include_predicted"]).to_csv(p, index=False)

    out.update(_export_field(cfg, run_dir, points, h, stride / fps))

    window = (meta["start_frame"] / fps, meta["end_frame"] / fps)
    stats = exports.compute_stats(points, summary, cfg, window)
    out["stats_json"] = run_dir / "stats.json"
    out["stats_json"].write_text(json.dumps(stats, indent=2), encoding="utf-8")
    out["stats_csv"] = run_dir / "stats.csv"
    exports.stats_table(stats).to_csv(out["stats_csv"], index=False)

    out["houdini"] = run_dir / "houdini_import.py"
    exports.write_houdini_script(out["houdini"], out["points"], h.origin, cfg)

    log("rendering visuals ...")
    raster = None
    if cfg.get("geotiff"):
        raster = load_geotiff(_require(cfg.path("geotiff"), "geotiff"), max_dim=8000, warn=log)
    title = f"{cfg.site} · {Path(meta['video']).name}"
    out["topdown"] = run_dir / "topdown.png"
    visuals.plot_topdown(points, raster, cfg, out["topdown"], title)
    out["density"] = run_dir / "density.png"
    visuals.plot_density(points, raster, cfg, out["density"], dt=stride / fps)
    out["speed_histogram"] = run_dir / "speed_histogram.png"
    visuals.plot_speed_histogram(summary, cfg, out["speed_histogram"])
    for gname, (fpath, label, rng) in _field_plots(cfg, run_dir, points).items():
        field = pd.read_csv(fpath)
        sel = points if gname == "all" else points[points["group"] == gname]
        png = run_dir / ("flow_field.png" if gname == "all" else f"flow_field_{gname}.png")
        visuals.plot_flow_field(field, sel, raster, cfg, png, float(cfg["field"]["cell_size_m"]),
                                rng, label)
        out[png.stem] = png
    if cfg["visuals"].get("topdown_video", True) and len(points):
        from .animation import render_topdown_video

        out["topdown_video"] = run_dir / "topdown.mp4"
        render_topdown_video(points, raster, cfg, out["topdown_video"], fps / stride,
                             visuals._extent(points, raster, cfg),
                             speedup=float(cfg["visuals"].get("topdown_video_speedup", 1.0)),
                             log=log)
    if cfg["visuals"].get("flowfield_video", True) and len(points):
        from .animation import render_flowfield_video

        gname = sorted(set(points["group"]))[0]
        out["flowfield_video"] = run_dir / "flowfield.mp4"
        render_flowfield_video(pd.read_csv(run_dir / "vector_field.csv"), points, raster, cfg,
                               out["flowfield_video"], visuals._extent(points, raster, cfg),
                               float(cfg["field"]["cell_size_m"]),
                               cfg.groups[gname].speed_range, log=log)

    if cfg["debug_video"].get("enabled"):
        from .debug_video import render_debug_video
        from .detect import site_roi

        if video.exists():
            r = raw.copy()
            cls = assign_track_classes(r)
            r["class"] = r["track_id"].map(cls).fillna(r["class"])
            out["debug_video"] = run_dir / "debug.mp4"
            render_debug_video(video, r, cfg, out["debug_video"], fps,
                               site_roi(cfg), log=log, run_dir=run_dir)
        else:
            log(f"debug video skipped: {video} not found")

    _log_stats(stats, log)
    log(f"processing took {time.time() - t0:.1f} s. Outputs in {run_dir}")
    return out


def _field_groups(points: pd.DataFrame) -> list[str]:
    present = sorted(set(points["group"])) if len(points) else []
    return ["all"] + (present if len(present) > 1 else [])


def _export_field(cfg: Config, run_dir: Path, points: pd.DataFrame, h: Homography,
                  dt: float) -> dict:
    """vector_field*.csv (+ time slices) and houdini_field.py."""
    from .field import smooth_field, time_sliced_fields

    fc = cfg["field"]
    cell, smooth = float(fc["cell_size_m"]), float(fc["smooth_m"])
    kw = dict(include_predicted=bool(fc["include_predicted"]),
              confidence_samples=float(fc["confidence_samples"]))
    out, files, slices = {}, {}, {}
    for gname in _field_groups(points):
        sel = points if gname == "all" else points[points["group"] == gname]
        suffix = "" if gname == "all" else f"_{gname}"
        p = run_dir / f"vector_field{suffix}.csv"
        smooth_field(sel, cell, smooth, dt, **kw).to_csv(p, index=False)
        files[gname] = p
        out[p.stem] = p
        if fc.get("time_window_s"):
            ps = run_dir / f"vector_field_slices{suffix}.csv"
            time_sliced_fields(sel, float(fc["time_window_s"]), cell, smooth, dt, **kw)                 .to_csv(ps, index=False)
            slices[gname] = ps
            out[ps.stem] = ps
    out["houdini_field"] = run_dir / "houdini_field.py"
    exports.write_houdini_field_script(out["houdini_field"], files, slices,
                                       run_dir / "points.csv", h.origin, cell, cfg)
    return out


def _field_plots(cfg: Config, run_dir: Path, points: pd.DataFrame) -> dict:
    """group -> (field csv, label, speed range) for the flow-field PNGs."""
    from .visuals import GROUP_LABELS

    groups = _field_groups(points)
    res = {}
    for gname in groups:
        if gname == "all" and len(groups) > 1:
            continue  # mixed speeds (people + cycles) would share one colour scale: skip
        g = gname if gname != "all" else (sorted(set(points["group"]))[0] if len(points)
                                          else next(iter(cfg.groups)))
        suffix = "" if gname == "all" else f"_{gname}"
        res[gname] = (run_dir / f"vector_field{suffix}.csv", GROUP_LABELS.get(g, g),
                      cfg.groups[g].speed_range)
    return res


def _log_stats(stats: dict, log) -> None:
    for g, b in stats["per_group"].items():
        log(f"  {g:9s} {b['n_tracks']:5d} tracks  mean "
            f"{b['mean_speed'] or 0:6.2f} m/s  median {b['median_speed'] or 0:6.2f} m/s  "
            f"straightness {b['mean_straightness'] or 0:.2f}")
    for cl in stats["count_lines"]:
        log(f"  count line {cl['name']}: {cl['crossings_left_to_right']} L->R, "
            f"{cl['crossings_right_to_left']} R->L")


def snapshot_inputs(cfg: Config, run_dir: Path) -> None:
    (run_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg.data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    hp = cfg.path("homography")
    if hp and hp.exists():
        shutil.copy2(hp, run_dir / "homography_used.json")


def run_all(cfg: Config, device: str, run_name: str | None = None,
            max_frames: int | None = None) -> Path:
    run_dir = new_run_dir(cfg, run_name)
    log = RunLog(run_dir / "run_log.txt")
    log(f"run folder: {run_dir}")
    snapshot_inputs(cfg, run_dir)
    video = _require(cfg.path("video"), "video")
    load_calibration(cfg, video)  # fail fast before spending time on detection
    raw = stage_track(cfg, run_dir, device, log=log, video=video, max_frames=max_frames)
    stage_process(cfg, run_dir, log=log, raw=raw, video=video)
    return run_dir
