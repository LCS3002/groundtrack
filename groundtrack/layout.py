"""Where every output of a run goes, sorted by what you use it for.

    runs/<site>/<run>/
      <site>_plate.png   the complete plate (in situ | plan, figures, legend)
      images/    frameless images: nothing around the data. <name>_layer.png = the drawing
                 alone on transparency. Every plan_* image shares one extent and size.
      labels/    legends, scale bar, north arrow (_light / _dark ink), title.txt
      videos/    overlay, topdown, flowfield (+ <name>_clean.mp4: the same without labels)
      data/      metrics.csv/json, points.csv, track_metrics.csv, tracks.geojson, grids, stats
      houdini/   houdini_import.py, houdini_field.py and the CSVs they read
      extras/    the maps with legends (topdown, flow field, density, speed histogram)
      _working/  tracker output, camera motion, config + calibration snapshot, log

All code finds files through `Run`, so the CLI, the UI and the scripts always agree. Runs made
before this layout are moved into it the first time they are opened (`open_run`).
"""

from __future__ import annotations

import shutil
from pathlib import Path

FOLDERS = ("images", "labels", "videos", "data", "houdini", "extras", "_working")


class Run:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    # folders
    images = property(lambda s: s.root / "images")
    labels = property(lambda s: s.root / "labels")
    videos = property(lambda s: s.root / "videos")
    data = property(lambda s: s.root / "data")
    houdini = property(lambda s: s.root / "houdini")
    extras = property(lambda s: s.root / "extras")
    work = property(lambda s: s.root / "_working")

    # tracker + provenance
    raw_tracks = property(lambda s: s.work / "raw_tracks.csv")
    meta = property(lambda s: s.work / "raw_tracks_meta.json")
    registration = property(lambda s: s.work / "registration.npz")
    config_used = property(lambda s: s.work / "config_used.yaml")
    homography_used = property(lambda s: s.work / "homography_used.json")
    log = property(lambda s: s.work / "run_log.txt")

    # data
    points = property(lambda s: s.data / "points.csv")
    tracks = property(lambda s: s.data / "track_metrics.csv")     # one row per track
    stats_json = property(lambda s: s.data / "stats.json")
    stats_csv = property(lambda s: s.data / "stats.csv")
    metrics_csv = property(lambda s: s.data / "metrics.csv")
    metrics_json = property(lambda s: s.data / "metrics.json")

    def field_csv(self, group: str = "all") -> Path:
        return self.houdini / ("vector_field.csv" if group == "all" else f"vector_field_{group}.csv")

    def plate(self, site: str | None = None) -> Path:
        return self.root / f"{site or self.root.parent.name}_plate.png"

    def make(self) -> Run:
        for f in FOLDERS:
            (self.root / f).mkdir(parents=True, exist_ok=True)
        return self

    def is_run(self) -> bool:
        return self.raw_tracks.exists() or (self.root / "raw_tracks.csv").exists()

    def processed(self) -> bool:
        return self.points.exists()


# --------------------------------------------------------------------------- migration
# old flat run folder -> new place (a folder keeps the name; a full path renames)
_MOVES = {
    "raw_tracks.csv": "_working", "raw_tracks_meta.json": "_working",
    "registration.npz": "_working", "config_used.yaml": "_working",
    "homography_used.json": "_working", "run_log.txt": "_working",
    "points.csv": "data", "track_summary.csv": "data/track_metrics.csv",
    "tracks.geojson": "data", "predicted_gaps.geojson": "data", "stats.json": "data",
    "stats.csv": "data", "groundtruth_report.json": "extras", "groundtruth.png": "extras",
    "houdini_import.py": "houdini", "houdini_field.py": "houdini",
    "topdown.png": "extras", "density.png": "extras", "speed_histogram.png": "extras",
    "topdown.mp4": "videos", "flowfield.mp4": "videos", "debug.mp4": "videos/overlay.mp4",
}
_GLOBS = {"field_grid*.csv": "data", "vector_field*.csv": "houdini", "flow_field*.png": "extras"}
_LAYER_NAMES = {"plan_trails": "plan_tracks", "insitu_trails": "insitu_tracks"}


def open_run(path: str | Path) -> Run:
    """A Run for this folder, moving an old flat run folder into the current layout."""
    run = Run(path)
    if (run.root / "raw_tracks.csv").exists() or (run.root / "package").exists():
        _migrate(run)
    return run


def _move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def _migrate(run: Run) -> None:
    r = run.root
    for name, target in _MOVES.items():
        if (r / name).exists():
            _move(r / name, r / target if "." in Path(target).name else r / target / name)
    for pattern, folder in _GLOBS.items():
        for f in r.glob(pattern):
            _move(f, r / folder / f.name)
    pkg = r / "package"
    if pkg.exists():
        img = pkg / "images"
        for f in sorted(img.glob("*.png")):
            if f.name == "00_complete.png":
                _move(f, run.plate())
            else:
                _move(f, run.images / f.name)
        for f in sorted((img / "layers").glob("*.png")):
            stem = _LAYER_NAMES.get(f.stem, f.stem)
            _move(f, run.images / f"{stem}_layer.png")
        for f in sorted(img.glob("*.mp4")):
            _move(f, run.videos / f"{f.stem}_clean.mp4")
        lab = pkg / "labels"
        for f in sorted(lab.glob("*")):
            if f.name == "00_complete.png":
                f.unlink()
            elif f.name in ("metrics.csv", "metrics.json", "track_metrics.csv"):
                if not (f.name == "track_metrics.csv" and run.tracks.exists()):
                    _move(f, run.data / f.name)
                else:
                    f.unlink()
            else:
                _move(f, run.labels / f.name)
        shutil.rmtree(pkg, ignore_errors=True)
    # the Houdini scripts embed absolute CSV paths: point them at the moved files
    if run.points.exists() and run.houdini.exists():
        shutil.copy2(run.points, run.houdini / "points.csv")
        for script in run.houdini.glob("houdini_*.py"):
            text = script.read_text(encoding="utf-8")
            old = r.resolve().as_posix()
            for name in ["points.csv", *[f.name for f in run.houdini.glob("vector_field*.csv")]]:
                text = text.replace(f"'{old}/{name}'", f"'{(run.houdini / name).resolve().as_posix()}'")
            script.write_text(text, encoding="utf-8")
    run.make()
