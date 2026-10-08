"""Run folder layout: everything sorted by use, and old flat runs are moved into it."""

from groundtrack.layout import FOLDERS, Run, open_run

OLD_FLAT = ["raw_tracks.csv", "raw_tracks_meta.json", "registration.npz", "config_used.yaml",
            "homography_used.json", "run_log.txt", "points.csv", "track_summary.csv",
            "tracks.geojson", "predicted_gaps.geojson", "field_grid.csv", "field_grid_people.csv",
            "stats.json", "stats.csv", "vector_field.csv", "vector_field_slices.csv",
            "topdown.png", "density.png", "speed_histogram.png", "flow_field.png",
            "topdown.mp4", "flowfield.mp4", "debug.mp4",
            "package/images/00_complete.png", "package/images/plan_tracks.png",
            "package/images/layers/plan_trails.png", "package/images/layers/insitu_trails.png",
            "package/images/topdown.mp4", "package/images/overlay.mp4",
            "package/labels/00_complete.png", "package/labels/legend_speed_light.png",
            "package/labels/metrics.csv", "package/labels/metrics.json",
            "package/labels/track_metrics.csv", "package/labels/title.txt"]


def test_old_flat_run_is_migrated(tmp_path):
    rd = tmp_path / "plaza3" / "fused"
    for name in OLD_FLAT:
        (rd / name).parent.mkdir(parents=True, exist_ok=True)
        (rd / name).write_text(name)
    old = rd.resolve().as_posix()
    (rd / "houdini_import.py").write_text(f"CSV_PATH = '{old}/points.csv'\n")
    (rd / "houdini_field.py").write_text(f"F = '{old}/vector_field.csv'\nP = '{old}/points.csv'\n")

    run = open_run(rd)
    for f in FOLDERS:
        assert (rd / f).is_dir()
    assert not (rd / "package").exists()
    assert [p.name for p in rd.iterdir() if p.is_file()] == ["plaza3_plate.png"]
    assert run.raw_tracks.read_text() == "raw_tracks.csv"
    assert run.registration.exists() and run.log.exists() and run.meta.exists()
    assert run.points.exists() and (run.houdini / "points.csv").exists()
    assert run.tracks.read_text() == "track_summary.csv"        # the run's own, not the copy
    assert run.metrics_csv.exists() and run.metrics_json.exists()
    assert (run.data / "field_grid_people.csv").exists()
    assert run.field_csv().exists() and (run.houdini / "vector_field_slices.csv").exists()
    assert (run.extras / "flow_field.png").exists()
    assert (run.videos / "overlay.mp4").read_text() == "debug.mp4"
    assert (run.videos / "overlay_clean.mp4").read_text() == "package/images/overlay.mp4"
    assert (run.videos / "topdown_clean.mp4").exists()
    assert (run.images / "plan_tracks_layer.png").exists()
    assert (run.images / "insitu_tracks_layer.png").exists()
    assert (run.labels / "legend_speed_light.png").exists()
    # the Houdini scripts now read the moved files
    imp = (run.houdini / "houdini_import.py").read_text()
    assert f"'{(run.houdini / 'points.csv').resolve().as_posix()}'" in imp
    fld = (run.houdini / "houdini_field.py").read_text()
    assert f"'{run.field_csv().resolve().as_posix()}'" in fld
    # opening again changes nothing
    before = sorted(p.relative_to(rd).as_posix() for p in rd.rglob("*"))
    open_run(rd)
    assert sorted(p.relative_to(rd).as_posix() for p in rd.rglob("*")) == before


def test_new_run_is_recognised(tmp_path):
    run = Run(tmp_path / "site" / "r1").make()
    assert not run.is_run()
    run.raw_tracks.write_text("x")
    assert run.is_run() and not run.processed()
    assert run.plate().name == "site_plate.png"
