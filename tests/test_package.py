"""Images, labels and metrics for a clip with people AND vehicles (no YOLO needed)."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from groundtrack.config import default_config
from groundtrack.field import smooth_field
from groundtrack.layout import Run
from groundtrack.package import build_package
from groundtrack.portfolio import plan_extent, plan_size, speed_rgb

DT = 1 / 25


def _lane(track, group, cls, y, v, n=200, x0=530000.0):
    t = np.arange(n) * DT
    return pd.DataFrame({"track_id": track, "class": cls, "group": group, "frame": np.arange(n),
                         "time_s": t, "x": x0 + v * t, "y": y, "vx": v, "vy": 0.0,
                         "speed": abs(v), "heading": 90.0, "predicted": False})


@pytest.fixture()
def mixed_run(tmp_path):
    run = Run(tmp_path / "mixed" / "r1").make()
    pts = pd.concat([_lane(1, "people", "person", 180000.0, 1.3),
                     _lane(2, "people", "person", 180002.0, 1.2),
                     _lane(3, "vehicles", "car", 180030.0, 15.0, n=60),
                     _lane(4, "vehicles", "bus", 180036.0, 12.0, n=60)], ignore_index=True)
    pts.to_csv(run.points, index=False)
    summ = pts.groupby("track_id").agg(group=("group", "first"), mean_speed=("speed", "mean"),
                                       x0=("x", "min"), x1=("x", "max")).reset_index()
    summ["straightness"] = 1.0
    summ["path_length_m"] = summ["x1"] - summ["x0"]
    summ.to_csv(run.tracks, index=False)
    run.stats_json.write_text(json.dumps({"window_s": [0, 8], "count_lines": [],
                                          "predicted_share": 0.0}))
    run.meta.write_text(json.dumps({"fps": 25, "vid_stride": 1, "video": "missing.mp4",
                                    "width": 1280, "height": 720}))
    for g in ("people", "vehicles"):
        smooth_field(pts[pts["group"] == g], 1.0, 2.0, DT).to_csv(run.field_csv(g), index=False)
    smooth_field(pts, 1.0, 2.0, DT).to_csv(run.field_csv(), index=False)
    return run, pts


def test_each_group_on_its_own_scale(mixed_run):
    run, pts = mixed_run
    cfg = default_config()
    cfg.data["package"]["plan_height_px"] = 600
    build_package(cfg, run.root, raster=None, video=None, log=lambda *_: None)

    # one speed legend per group, in both inks
    for g in ("people", "vehicles"):
        for v in ("light", "dark"):
            assert (run.labels / f"legend_speed_{g}_{v}.png").exists()
    figs = pd.read_csv(run.metrics_csv).set_index("metric")["value"]
    assert float(figs["median_speed_people"]) == pytest.approx(1.25, abs=0.05)
    assert float(figs["median_speed_vehicles"]) == pytest.approx(13.5, abs=0.5)
    info = json.loads(run.metrics_json.read_text())
    assert info["speed_range_m_s"] == {"people": [0.0, 2.5], "vehicles": [0.0, 35.0]}

    # cars at 15 m/s on the vehicle scale (0-35) are cyan, not "fastest" amber as they would
    # be on the people scale (0-2.5)
    layer = np.asarray(Image.open(run.images / "plan_tracks_layer.png"), float)
    ext = plan_extent(pts)
    W, H = plan_size(ext, 600)
    row = int(round((ext[3] - 180030.0) / (ext[3] - ext[2]) * H))
    band = layer[row - 1:row + 2]
    lit = band[band[..., 3] > 128]
    assert len(lit) > 10
    r, b = lit[:, 0].mean(), lit[:, 2].mean()
    want = speed_rgb(15.0, 0, 35)
    assert b > r and want[2] > want[0]


def test_flowfield_video_with_two_fields(mixed_run, tmp_path):
    from groundtrack.animation import render_flowfield_video

    run, pts = mixed_run
    cfg = default_config()
    fields = [(pd.read_csv(run.field_csv("people")), (0.0, 2.5)),
              (pd.read_csv(run.field_csv("vehicles")), (0.0, 35.0))]
    out = tmp_path / "ff.mp4"
    clean = tmp_path / "ff_clean.mp4"
    render_flowfield_video(fields, pts, None, cfg, out, plan_extent(pts), 1.0, None,
                           seconds=1.0, fps=10, width=320, log=lambda *_: None, clean_path=clean)
    assert out.stat().st_size > 1000 and clean.stat().st_size > 1000
