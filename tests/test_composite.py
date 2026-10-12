"""Two clips of one place on one plan: coverage-normalised density, separate layers."""
import json

import numpy as np
import pandas as pd

from groundtrack.composite import build_composite, load_clip
from groundtrack.config import default_config
from groundtrack.demo import ORIGIN
from groundtrack.layout import Run
from groundtrack.posefit import pose_homography

FPS = 25


def _clip(tmp_path, name, cam_xy, yaw, n_people, seconds=20.0):
    """A camera 8 m up looking at the plaza; n_people walk east through its view."""
    run = Run(tmp_path / name / "r1").make()
    h = pose_homography(cam_xy[0], cam_xy[1], 8.0, yaw, 25.0, 1400.0, (1920, 1080))
    h.camera_params = {"E": cam_xy[0], "N": cam_xy[1], "height_m": 8.0, "yaw_deg": yaw,
                       "tilt_deg": 25.0, "roll_deg": 0.0, "focal_px": 1400.0}
    h.save(run.homography_used)
    n = int(seconds * FPS)
    t = np.arange(n) / FPS
    rows = []
    for k in range(n_people):
        rows.append(pd.DataFrame({
            "track_id": k + 1, "class": "person", "group": "people", "frame": np.arange(n),
            "time_s": t, "x": ORIGIN[0] - 10 + 1.2 * t, "y": ORIGIN[1] + 20 + 2 * k, "vx": 1.2,
            "vy": 0.0, "speed": 1.2, "heading": 90.0, "predicted": False}))
    pd.concat(rows).to_csv(run.points, index=False)
    run.meta.write_text(json.dumps({"fps": FPS, "vid_stride": 1, "start_frame": 0,
                                    "end_frame": n, "video": "missing.mp4"}))
    return run


def test_composite_normalises_density_by_time_in_view(tmp_path):
    cfg = default_config()
    a = _clip(tmp_path, "a", (ORIGIN[0], ORIGIN[1]), 0.0, 3)
    b = _clip(tmp_path, "b", (ORIGIN[0] + 2, ORIGIN[1]), 0.0, 3)      # nearly the same view
    clips = [load_clip("a", cfg, a.root), load_clip("b", cfg, b.root)]
    out = build_composite(clips, None, tmp_path / "composite", "Test plaza",
                          plan_height_px=600, log=lambda *x: None)
    for f in ("images/plan_tracks.png", "images/plan_tracks_layer.png",
              "images/plan_density_people_layer.png", "images/plan_coverage_layer.png",
              "images/plan_flowfield_people.png", "labels/legend_density_people_light.png",
              "labels/camera_numbers.txt", "data/points_all.csv", "data/clips.csv",
              "houdini/houdini_import.py", "houdini/houdini_field.py",
              "houdini/vector_field_people.csv", "rhino/grasshopper_import.py"):
        assert (out / f).exists(), f
    pts = pd.read_csv(out / "data/points_all.csv")
    assert pts["track_id"].nunique() == 6                     # ids stay apart across clips
    grid = pd.read_csv(out / "data/coverage_density_grid.csv")
    assert grid["seconds_in_view"].max() == 40.0               # both clips: 2 x 20 s
    # the same people, seen twice for twice as long: the same density, not double
    single = build_composite(clips[:1], None, tmp_path / "single", "One clip",
                             plan_height_px=600, log=lambda *x: None)
    g1 = pd.read_csv(single / "data/coverage_density_grid.csv")
    lane = ORIGIN[1] + 22
    pick = lambda g: g[(abs(g["y"] - lane) < 0.6) & (abs(g["x"] - ORIGIN[0]) < 6)]  # noqa: E731
    d2, d1 = pick(grid)["people_per_m2"].median(), pick(g1)["people_per_m2"].median()
    assert d1 > 0 and abs(d2 / d1 - 1) < 0.2
    # both clips see the same eastward walkers: where they overlap, they agree
    pairs = pd.read_csv(out / "data/overlap_pairs.csv")
    assert pairs["shared_cells"].iloc[0] > 0 and pairs["agree_share"].iloc[0] == 1.0
    assert (out / "images/plan_agreement_people_layer.png").exists()


def test_master_field_puts_every_group_on_one_grid(tmp_path):
    cfg = default_config()
    a = _clip(tmp_path, "a", (ORIGIN[0], ORIGIN[1]), 0.0, 5)                # walkers, east
    b = _clip(tmp_path, "b", (ORIGIN[0] + 2, ORIGIN[1]), 0.0, 5)
    p = pd.read_csv(b.points)                                               # b: cars, west
    p["group"], p["class"], p["vx"], p["speed"] = "vehicles", "car", -8.0, 8.0
    p["x"] = ORIGIN[0] + 10 - 8.0 * p["time_s"] / 4
    p["y"] = p["y"] + 12
    p.to_csv(b.points, index=False)
    clips = [load_clip("a", cfg, a.root), load_clip("b", cfg, b.root)]
    out = build_composite(clips, None, tmp_path / "composite", "Two groups",
                          plan_height_px=600, log=lambda *x: None)
    m = pd.read_csv(out / "data/field_master.csv")
    for c in ("vx", "vy", "people_vx", "vehicles_vx", "dominant_group"):
        assert c in m
    assert set(m["dominant_group"].dropna()) >= {"people", "vehicles"}
    walk = m[m["dominant_group"] == "people"]
    assert (walk["people_vx"].dropna() > 0).mean() > 0.9                  # walkers go east
    cars = m[m["dominant_group"] == "vehicles"]
    assert (cars["vehicles_vx"].dropna() < 0).mean() > 0.9                # cars go west
    assert (out / "images/plan_flowfield_master_layer.png").exists()
    assert (out / "labels/legend_flow_master_light.png").exists()
    assert "vector_field_master.csv" in (out / "houdini/houdini_field.py").read_text()
