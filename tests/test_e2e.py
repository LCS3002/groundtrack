"""End-to-end: synthetic video -> calibration -> YOLO + tracking -> all exports.

Uses the demo site: two people with known paths and speeds, one passing behind a pillar.
Runs on whatever device is available (a few seconds on a GPU, ~1 min on CPU).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.e2e
REPO_MODELS = Path(__file__).resolve().parents[1] / "models"

EXPECTED_FILES = ["raw_tracks.csv", "raw_tracks_meta.json", "points.csv", "track_summary.csv",
                  "tracks.geojson", "predicted_gaps.geojson", "field_grid.csv", "stats.json",
                  "stats.csv", "houdini_import.py", "topdown.png", "density.png",
                  "speed_histogram.png", "config_used.yaml", "homography_used.json",
                  "run_log.txt", "debug.mp4", "vector_field.csv", "flow_field.png",
                  "houdini_field.py", "topdown.mp4", "flowfield.mp4",
                  # package: frameless images + videos, labels + metrics, the complete plate
                  "package/images/00_complete.png", "package/images/insitu_frame.png",
                  "package/images/insitu_tracks.png", "package/images/plan_map.png",
                  "package/images/plan_tracks.png", "package/images/plan_flowfield.png",
                  "package/images/plan_density.png", "package/images/layers/plan_trails.png",
                  "package/images/topdown.mp4", "package/images/flowfield.mp4",
                  "package/images/overlay.mp4", "package/labels/00_complete.png",
                  "package/labels/legend_speed_light.png", "package/labels/scale_bar_dark.png",
                  "package/labels/north_arrow_light.png", "package/labels/metrics.csv",
                  "package/labels/metrics.json", "package/labels/track_metrics.csv"]


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    from groundtrack.calibrate import run_calibration
    from groundtrack.config import load_config
    from groundtrack.demo import make_demo_site
    from groundtrack.device import resolve_device
    from groundtrack.pipeline import run_all

    root = tmp_path_factory.mktemp("demo")
    site = make_demo_site(root)
    cfg = load_config(site["config"], {"models_dir": str(REPO_MODELS),
                                       "debug_video": {"enabled": True}})
    h = run_calibration(site["video"], site["geotiff"], cfg.path("homography"),
                        points_csv=site["points_csv"], interactive=False, log=lambda *_: None)
    run_dir = run_all(cfg, resolve_device("auto"), "e2e")
    return site, cfg, h, run_dir


def test_calibration(demo_run):
    site, _, h, _ = demo_run
    assert h.rmse_m < 0.3  # 0.7 px click noise, points up to 55 m away at a low angle
    assert all(p["inlier"] for p in h.points)
    cam = h.camera()
    assert cam["height_m"] == pytest.approx(10.0, abs=0.5)


def test_all_outputs_written(demo_run):
    *_, run_dir = demo_run
    missing = [f for f in EXPECTED_FILES if not (run_dir / f).exists()]
    assert not missing
    assert (run_dir / "topdown.png").stat().st_size > 50_000


def test_package_images_stack(demo_run):
    """Every plan image and layer shares one size, and the scale bar matches its scale."""
    from PIL import Image

    *_, run_dir = demo_run
    img = run_dir / "package" / "images"
    plan = sorted(img.glob("plan_*.png")) + sorted((img / "layers").glob("plan_*.png"))
    sizes = {Image.open(p).size for p in plan}
    assert len(plan) >= 6 and len(sizes) == 1
    info = json.loads((run_dir / "package" / "labels" / "metrics.json").read_text())
    assert tuple(info["plan"]["size_px"]) == sizes.pop()
    e = info["plan"]["extent_m"]
    assert info["plan"]["px_per_m"] == pytest.approx(info["plan"]["size_px"][0]
                                                     / (e["east"] - e["west"]), rel=1e-3)
    assert Image.open(img / "layers" / "plan_trails.png").mode == "RGBA"
    figures = pd.read_csv(run_dir / "package" / "labels" / "metrics.csv")
    assert {"tracks", "median_speed", "duration"} <= set(figures["metric"])


def test_tracks_match_ground_truth(demo_run):
    site, _, _, run_dir = demo_run
    truth = site["truth"]
    pts = pd.read_csv(run_dir / "points.csv")
    summ = pd.read_csv(run_dir / "track_summary.csv")
    # exactly the two people; the occlusion behind the pillar must not split person A
    assert len(summ) == 2, summ

    a_end = np.array(truth["a_end"])
    a = summ.loc[(summ["end_x"] - a_end[0]).abs().idxmin()]
    a_pts = pts[pts["track_id"] == a["track_id"]]
    # positions: start / end within 25 cm, path stays within 25 cm of the true line
    assert np.hypot(a["start_x"] - truth["a_start"][0], a["start_y"] - truth["a_start"][1]) < 0.25
    assert np.hypot(a["end_x"] - a_end[0], a["end_y"] - a_end[1]) < 0.25
    assert (a_pts["y"] - a_end[1]).abs().max() < 0.25
    # speed while walking (excluding the standing phases)
    walking = a_pts[a_pts["speed"] > 0.6]
    assert walking["speed"].median() == pytest.approx(truth["a_speed"], rel=0.06)
    assert walking["heading"].median() == pytest.approx(90, abs=4)
    # the gap behind the pillar is machine-made and flagged
    gap = a_pts[(a_pts["x"] > truth["pillar_x"][0]) & (a_pts["x"] < truth["pillar_x"][1])]
    assert gap["predicted"].mean() > 0.5
    assert set(a_pts.loc[a_pts["predicted"], "source"]) <= {"predicted", "stitched",
                                                             "interpolated"}

    b = summ.loc[(summ["start_x"] - truth["b_x"]).abs().idxmin()]
    b_pts = pts[pts["track_id"] == b["track_id"]]
    assert (b_pts["x"] - truth["b_x"]).abs().max() < 0.25
    assert b_pts["speed"].iloc[10:-10].median() == pytest.approx(truth["b_speed"], rel=0.06)
    assert b_pts["heading"].median() == pytest.approx(180, abs=4)
    assert b["straightness"] > 0.98


def test_geojson_and_stats(demo_run):
    *_, run_dir = demo_run
    gj = json.loads((run_dir / "tracks.geojson").read_text())
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) == 2
    assert all(f["geometry"]["type"] == "LineString" for f in gj["features"])
    stats = json.loads((run_dir / "stats.json").read_text())
    assert stats["per_class"]["person"]["n_tracks"] == 2
    line = stats["count_lines"][0]
    assert line["crossings_left_to_right"] + line["crossings_right_to_left"] == 1
    grid = pd.read_csv(run_dir / "field_grid.csv")
    assert grid["count"].sum() > 300


@pytest.fixture(scope="module")
def shaky_site(tmp_path_factory):
    """Same scene, but the camera sways ~2 deg like a handheld phone."""
    from groundtrack.calibrate import run_calibration
    from groundtrack.config import load_config
    from groundtrack.demo import make_demo_site

    root = tmp_path_factory.mktemp("shaky")
    site = make_demo_site(root, shake_deg=2.0)
    cfg = load_config(site["config"], {"models_dir": str(REPO_MODELS)})
    run_calibration(site["video"], site["geotiff"], cfg.path("homography"),
                    points_csv=site["points_csv"], interactive=False, log=lambda *_: None)
    return site


def _walker_error(site, stabilize: bool, name: str):
    from groundtrack.config import load_config
    from groundtrack.device import resolve_device
    from groundtrack.pipeline import run_all

    cfg = load_config(site["config"], {"models_dir": str(REPO_MODELS),
                                       "detection": {"stabilize": stabilize}})
    run_dir = run_all(cfg, resolve_device("auto"), name)
    pts = pd.read_csv(run_dir / "points.csv")
    det = pts[~pts["predicted"]]
    # person A walks east along N = 180005 (wide x range); B walks south along E = 529993
    errs = []
    for _, g in det.groupby("track_id"):
        if np.ptp(g["x"]) > 5:
            errs.append((g["y"] - 180005).abs().max())
        else:
            errs.append((g["x"] - 529993).abs().max())
    return float(max(errs)), run_dir


def test_stabilize_compensates_camera_motion(shaky_site):
    err_on, run_dir = _walker_error(shaky_site, True, "stab-on")
    err_off, _ = _walker_error(shaky_site, False, "stab-off")
    meta = json.loads((run_dir / "raw_tracks_meta.json").read_text())
    assert meta["max_drift_px"] > 20          # the camera really moved
    assert meta["registration_failures"] == 0
    assert err_on < 0.3, err_on               # compensated: still accurate
    assert err_off > 2 * err_on               # uncompensated: visibly worse


def test_groundtruth_command(demo_run, capsys):
    from groundtrack.cli import main

    site, cfg, _, _ = demo_run
    t = site["truth"]
    main(["groundtruth", "--config", str(site["config"]), "--video", str(site["video"]),
          "--start", *map(str, t["a_start"]), "--end", *map(str, t["a_end"]),
          "--run-name", "gt"])
    run_dir = cfg.path("output_dir") / "demo" / "gt"
    rep = json.loads((run_dir / "groundtruth_report.json").read_text())
    # picks person A (the longest path is B, so this also checks the A-B matching)
    assert rep["known_length_m"] == pytest.approx(9.0)
    assert abs(rep["length_error_pct"]) < 3
    assert abs(rep["speed_error_pct"]) < 3
    assert rep["lateral_deviation_m"]["rms"] < 0.15
    assert rep["start_error_m"] < 0.25 and rep["end_error_m"] < 0.25
    assert (run_dir / "groundtruth.png").exists()
