"""groundtrack command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cfg(args, overrides=None):
    from .config import load_config

    ov = dict(overrides or {})
    if getattr(args, "device", None):
        ov["device"] = args.device
    return load_config(args.config, ov)


def _device(cfg, log=print):
    from .device import describe_device, resolve_device

    d = resolve_device(cfg["device"])
    log(describe_device(d))
    return d


def _run_dir_arg(cfg, run: str | None) -> Path:
    """--run accepts a folder path, a run name, or 'latest'."""
    from .layout import Run, open_run

    root = (cfg.path("output_dir") or cfg.base_dir / "runs") / cfg.site
    if run in (None, "latest"):
        runs = sorted(p for p in root.glob("*") if Run(p).is_run())
        if not runs:
            raise SystemExit(f"No tracked runs under {root}")
        return open_run(runs[-1]).root
    p = Path(run)
    return open_run(p if p.exists() else root / run).root


# --------------------------------------------------------------------------- commands
def cmd_device(args):
    from .device import describe_device, resolve_device

    print(describe_device(resolve_device(args.device or "auto")))


def cmd_init(args):
    from .templates import TEMPLATES

    out = Path(args.out or f"sites/{args.site}.yaml")
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (use --force to overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATES[args.template].format(site=args.site, fname=out.as_posix()),
                   encoding="utf-8")
    for d in ("footage", "maps", "calibration", "runs"):
        (out.parent.parent / d).mkdir(exist_ok=True)
    print(f"wrote {out}. Put the video in footage/ and the GeoTIFF in maps/, then run:\n"
          f"  groundtrack calibrate --config {out.as_posix()}")


def cmd_demo(args):
    from .calibrate import run_calibration
    from .config import load_config
    from .demo import make_demo_site
    from .pipeline import run_all

    out = Path(args.out)
    site = make_demo_site(out)
    print(f"demo site written to {out}")
    cfg = load_config(site["config"], {"device": args.device} if args.device else None)
    run_calibration(site["video"], site["geotiff"], cfg.path("homography"),
                    points_csv=site["points_csv"], interactive=False)
    run_all(cfg, _device(cfg), "demo")


def cmd_lens(args):
    from .lens import calibrate_lens

    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    lens = calibrate_lens(args.source, (cols, rows), args.square_mm, args.every_s)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lens.save(args.out)
    print(f"saved {args.out}. Set `lens: {args.out}` in the site config, then re-calibrate.")


def cmd_calibrate(args):
    from .calibrate import run_calibration
    from .lens import Lens
    from .video import video_info

    cfg = _cfg(args)
    video = cfg.path("video")
    frame = args.frame
    if args.time is not None:
        frame = int(round(args.time * video_info(video, cfg.get("fps")).fps))
    lens = Lens.load(cfg.path("lens")) if cfg.get("lens") else None
    out = cfg.path("homography")
    if out is None:
        raise SystemExit("Set `homography:` (output path) in the site config")
    from .terrain import resolve_terrain

    h = run_calibration(video, cfg.path("geotiff"), out, frame=frame or 0, lens=lens,
                        points_csv=Path(args.points_csv) if args.points_csv else None,
                        interactive=not args.no_gui, ransac_thresh_m=args.ransac_m,
                        warn_m=args.warn_m, camera_prior=cfg.get("camera_position"),
                        terrain=resolve_terrain(cfg), hide=_privacy_boxes_of(cfg))
    if h is not None and getattr(h, "camera_prior_set", None):
        from .locate import write_camera_position

        write_camera_position(args.config, h.camera_prior_set, note="set on the map in the picker")
        print(f"camera position saved to {args.config}")


def _privacy_boxes_of(cfg):
    """People / number plates in the site's latest tracked run, to blur saved check images."""
    import pandas as pd

    from .autocal import _latest_run
    from .debug_video import privacy_boxes
    from .layout import open_run

    rd = _latest_run(cfg)
    if rd is None or not open_run(rd).raw_tracks.exists():
        return None
    dv = cfg.get("debug_video") or {}
    return privacy_boxes(pd.read_csv(open_run(rd).raw_tracks), bool(dv.get("blur_people", True)),
                         bool(dv.get("blur_plates", True)))


def cmd_autocalibrate(args):
    from .autocal import run_autocalibration

    cfg = _cfg(args)
    run_dir = _run_dir_arg(cfg, args.run) if args.run else None
    rep = run_autocalibration(cfg, run_dir=run_dir, device=cfg["device"], replace=args.replace,
                              track_frames=args.frames)
    if rep.get("result") is None:
        raise SystemExit(1)


def cmd_roi(args):
    from .calibrate import pick_roi

    cfg = _cfg(args)
    from .detect import reference_frame_for

    out = Path(args.out) if args.out else cfg.base_dir / f"{cfg.site}_roi.json"
    # draw on the calibration frame: with `stabilize` the ROI is applied in that frame
    frame = args.frame if args.frame is not None else reference_frame_for(cfg)
    pick_roi(cfg.path("video"), frame, out)
    if not args.out:
        print(f"{out.name} is picked up automatically for site '{cfg.site}'.")


def cmd_track(args):
    from .layout import Run
    from .pipeline import RunLog, load_calibration, new_run_dir, snapshot_inputs, stage_track

    cfg = _cfg(args)
    run_dir = new_run_dir(cfg, args.run_name)
    log = RunLog(Run(run_dir).log)
    snapshot_inputs(cfg, run_dir)
    if cfg.get("homography") and cfg.path("homography").exists():
        load_calibration(cfg, cfg.path("video"))
    stage_track(cfg, run_dir, _device(cfg, log), log=log, max_frames=args.max_frames)
    log(f"next: groundtrack process --config {args.config} --run {run_dir}")


def cmd_process(args):
    from .layout import Run
    from .pipeline import RunLog, stage_process

    ov = {"debug_video": {"enabled": True}} if args.debug_video else {}
    cfg = _cfg(args, ov)
    run_dir = _run_dir_arg(cfg, args.run)
    stage_process(cfg, run_dir, log=RunLog(Run(run_dir).log))


def cmd_package(args):
    from .geo import load_geotiff
    from .package import build_package

    cfg = _cfg(args)
    run_dir = _run_dir_arg(cfg, args.run)
    raster = load_geotiff(cfg.path("geotiff"), max_dim=8000) if cfg.get("geotiff") else None
    build_package(cfg, run_dir, raster, plate=not args.no_plate)


def cmd_locate(args):
    from .locate import camera_from_place, search, write_camera_position

    hits = search(args.place, limit=args.limit)
    if not hits:
        raise SystemExit(f"OpenStreetMap found nothing for {args.place!r}")
    for k, h in enumerate(hits, 1):
        print(f"{k}. {h['name']}\n   E {h['E']:.1f}  N {h['N']:.1f}  (~{h['size_m']:.0f} m across)")
    pick = hits[args.pick - 1]
    cam = camera_from_place(pick, args.floor, args.floor_height, ground_offset_m=args.ground_m)
    if args.height is not None:
        cam["height_m"] = args.height
        cam["height_tol_m"] = round(max(1.0, 0.06 * args.height), 1)
    print("\ncamera_position:\n" + "".join(f"  {k}: {v}\n" for k, v in cam.items()))
    if args.config and args.write:
        write_camera_position(args.config, cam, note=f"OSM: {pick['name'][:60]}")
        print(f"written to {args.config}")
    elif args.config:
        print("add --write to put this into the config")


def cmd_ui(args):
    from .ui import serve

    serve(Path(args.root), port=args.port, open_browser=not args.no_browser)


def cmd_run(args):
    from .pipeline import run_all

    ov = {"debug_video": {"enabled": True}} if args.debug_video else {}
    cfg = _cfg(args, ov)
    run_all(cfg, _device(cfg), args.run_name, max_frames=args.max_frames)


def cmd_debug_video(args):
    import json

    import pandas as pd

    from .debug_video import render_debug_video
    from .detect import site_roi
    from .layout import Run
    from .trajectories import assign_track_classes

    dv = {"blur_people": not args.no_blur}
    if args.boxes:
        dv["style"] = "boxes"
    if args.no_stabilize_output:
        dv["stabilize_output"] = False
    if args.trail_s is not None:
        dv["trail_s"] = args.trail_s
    cfg = _cfg(args, {"debug_video": dv})
    run = Run(_run_dir_arg(cfg, args.run)).make()
    meta = json.loads(run.meta.read_text(encoding="utf-8"))
    raw = pd.read_csv(run.raw_tracks)
    raw["class"] = raw["track_id"].map(assign_track_classes(raw)).fillna(raw["class"])
    clean = None
    if (cfg.get("package") or {}).get("enabled", True) and not args.boxes:
        clean = run.videos / "overlay_clean.mp4"                 # the same, without labels
    out = run.videos / ("overlay_boxes.mp4" if args.boxes else "overlay.mp4")
    render_debug_video(Path(meta["video"]), raw, cfg, out, meta["fps"], site_roi(cfg),
                       run_dir=run.root, clean_path=clean)


def cmd_groundtruth(args):
    import json

    from .config import load_config
    from .groundtruth import analyse_walk, pick_track, plot_walk, write_report
    from .layout import Run
    from .pipeline import RunLog, load_calibration, new_run_dir, stage_track
    from .trajectories import process_tracks

    base = load_config(args.config)
    people = base["groups"].get("people") or {"classes": ["person"]}
    ov = {"groups": {"people": people}, "detection": {"roi": None}, "site": base.site}
    if args.device:
        ov["device"] = args.device
    cfg = load_config(args.config, ov)
    video = Path(args.video)
    h, lens = load_calibration(cfg, video)
    run_dir = new_run_dir(cfg, args.run_name or "groundtruth-" + video.stem)
    run = Run(run_dir)
    log = RunLog(run.log)
    raw = stage_track(cfg, run_dir, _device(cfg, log), log=log, video=video)
    meta = json.loads(run.meta.read_text(encoding="utf-8"))
    from .pipeline import load_run_registration

    points, summary = process_tracks(raw, cfg, h, meta["fps"], meta["vid_stride"], lens, log=log,
                                     registration=load_run_registration(run_dir, h, log))
    points.to_csv(run.points, index=False)
    summary.to_csv(run.tracks, index=False)
    a = tuple(args.start) if args.start else None
    b = tuple(args.end) if args.end else None
    tr = pick_track(points, summary, args.track_id, a, b)
    rep = analyse_walk(tr, a, b, args.length, args.stopwatch)
    write_report(rep, run.extras, log)
    plot_walk(tr, rep, run.extras / "groundtruth.png", a, b)
    log(f"report: {run.extras / 'groundtruth_report.json'}\n"
        f"plot:   {run.extras / 'groundtruth.png'}")


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="groundtrack",
                                description="Oblique video -> top-down movement data (EPSG:27700)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, config=True):
        if config:
            sp.add_argument("--config", "-c", required=True, help="site YAML")
        sp.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None,
                        help="override the config's device")
        return sp

    s = common(sub.add_parser("device", help="show which PyTorch device will be used"), False)
    s.set_defaults(func=cmd_device)

    s = sub.add_parser("init", help="write a new site config from a template")
    s.add_argument("site")
    s.add_argument("--template", choices=["plaza", "motorway"], default="plaza")
    s.add_argument("--out")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_init)

    s = common(sub.add_parser("demo", help="build a synthetic site and run the whole pipeline"),
               False)
    s.add_argument("--out", default="demo_site")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("lens-calibrate", help="checkerboard lens calibration (optional)")
    s.add_argument("--source", required=True, help="folder, glob of images, or a video")
    s.add_argument("--pattern", default="9x6", help="inner corners COLSxROWS (default 9x6)")
    s.add_argument("--square-mm", type=float, default=25.0)
    s.add_argument("--every-s", type=float, default=1.0, help="video: sample a frame every N s")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_lens)

    s = common(sub.add_parser("calibrate", help="pick ground points, fit + save the homography"))
    s.add_argument("--frame", type=int, default=0)
    s.add_argument("--time", type=float, help="pick the frame at this many seconds")
    s.add_argument("--points-csv", help="start from / use these point pairs (u,v,E,N)")
    s.add_argument("--no-gui", action="store_true", help="fit from --points-csv without the picker")
    s.add_argument("--ransac-m", type=float, default=1.0,
                   help="RANSAC inlier threshold in m (far points are less precise; keep >= 1)")
    s.add_argument("--warn-m", type=float, default=0.5, help="warn if RMSE exceeds this (m)")
    s.set_defaults(func=cmd_calibrate)

    s = common(sub.add_parser("autocalibrate",
                              help="calibrate without clicking (same spot / vehicles on roads)"))
    s.add_argument("--run", help="use this tracked run (default: the latest, else track first)")
    s.add_argument("--frames", type=int, default=900, help="frames to track if there is no run")
    s.add_argument("--replace", action="store_true",
                   help="write over the existing calibration (a backup is kept)")
    s.set_defaults(func=cmd_autocalibrate)

    s = common(sub.add_parser("roi", help="draw a region-of-interest polygon on the video"))
    s.add_argument("--frame", type=int, default=None, help="default: the calibration frame")
    s.add_argument("--out")
    s.set_defaults(func=cmd_roi)

    s = common(sub.add_parser("track", help="stage 1 only: detect + track -> raw_tracks.csv"))
    s.add_argument("--run-name")
    s.add_argument("--max-frames", type=int, help="process only this many frames (testing)")
    s.set_defaults(func=cmd_track)

    s = common(sub.add_parser("process", help="stages 3-6 on an existing run (fast re-run)"))
    s.add_argument("--run", default="latest", help="run folder, run name, or 'latest'")
    s.add_argument("--debug-video", action="store_true")
    s.set_defaults(func=cmd_process)

    s = common(sub.add_parser("run", help="full pipeline: track + process + exports + visuals"))
    s.add_argument("--run-name")
    s.add_argument("--max-frames", type=int)
    s.add_argument("--debug-video", action="store_true", help="also write debug.mp4")
    s.set_defaults(func=cmd_run)

    s = common(sub.add_parser("package", help="frameless images + labels/metrics for a run"))
    s.add_argument("--run", default="latest", help="run folder, run name, or 'latest'")
    s.add_argument("--no-plate", action="store_true", help="skip the complete plate")
    s.set_defaults(func=cmd_package)

    s = sub.add_parser("locate", help="camera position from a place name (OpenStreetMap)")
    s.add_argument("place", help='e.g. "Urbanest Canary Wharf"')
    s.add_argument("--floor", type=float, help="floor you filmed from (height = floor x 3.1 m)")
    s.add_argument("--floor-height", type=float, default=3.1, help="m per floor")
    s.add_argument("--height", type=float, help="camera height above the ground in m instead")
    s.add_argument("--ground-m", type=float, default=0.0,
                   help="building ground level above the mapped ground (m)")
    s.add_argument("--pick", type=int, default=1, help="use the Nth search result")
    s.add_argument("--limit", type=int, default=5)
    s.add_argument("--config", "-c", help="site YAML to put camera_position into")
    s.add_argument("--write", action="store_true", help="write it into --config")
    s.set_defaults(func=cmd_locate)

    s = sub.add_parser("ui", help="open the local web interface")
    s.add_argument("--root", default=".", help="project folder (with sites/, runs/)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_ui)

    s = common(sub.add_parser("debug-video", help="render debug.mp4 for an existing run"))
    s.add_argument("--run", default="latest")
    s.add_argument("--no-blur", action="store_true", help="do NOT blur people (careful)")
    s.add_argument("--boxes", action="store_true",
                   help="debug style: boxes, IDs and rejected tracks (default: clean trails)")
    s.add_argument("--no-stabilize-output", action="store_true",
                   help="keep the original (moving) camera framing")
    s.add_argument("--trail-s", type=float, help="only show the last N seconds of each trail")
    s.set_defaults(func=cmd_debug_video)

    s = common(sub.add_parser("groundtruth", help="check accuracy on a walk along a known line"))
    s.add_argument("--video", required=True)
    s.add_argument("--start", type=float, nargs=2, metavar=("E", "N"), help="surveyed point A")
    s.add_argument("--end", type=float, nargs=2, metavar=("E", "N"), help="surveyed point B")
    s.add_argument("--length", type=float, help="known A-B length in m (if no coordinates)")
    s.add_argument("--stopwatch", type=float, help="walking time A->B you timed, in s (optional)")
    s.add_argument("--track-id", type=int, help="default: the longest person track")
    s.add_argument("--run-name")
    s.set_defaults(func=cmd_groundtruth)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
