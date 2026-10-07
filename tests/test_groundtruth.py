import numpy as np
import pandas as pd
import pytest

from groundtrack.groundtruth import analyse_walk, pick_track
from groundtrack.trajectories import smooth_and_differentiate

FPS = 25.0


def _walk(scale=1.0, offset=(0.0, 0.0), lateral_noise=0.0, seed=0):
    """Stand 2 s at A, walk 12 m east at 1.4 m/s, stand 2 s at B, seen through a calibration
    that is `scale` too large about A and shifted by `offset`."""
    rng = np.random.default_rng(seed)
    a = np.array([530000.0, 180000.0])
    t = np.arange(0, 2 + 12 / 1.4 + 2, 1 / FPS)
    s = np.clip((t - 2) * 1.4, 0, 12)
    x = a[0] + s * scale + offset[0] + rng.normal(0, 0.02, len(t))
    y = a[1] + offset[1] + rng.normal(0, lateral_noise + 1e-9, len(t))
    xs, ys, vx, vy = smooth_and_differentiate(x, y, 1 / FPS, 1.0)
    pts = pd.DataFrame({"track_id": 1, "class": "person", "time_s": t, "x": xs, "y": ys,
                        "speed": np.hypot(vx, vy), "predicted": False})
    return pts, tuple(a), tuple(a + [12.0, 0.0])


def test_perfect_walk():
    pts, a, b = _walk()
    r = analyse_walk(pts, a, b)
    assert abs(r["length_error_pct"]) < 0.5
    assert abs(r["speed_error_pct"]) < 1.0
    assert r["reference_speed"] == pytest.approx(1.4, rel=0.01)
    assert r["start_error_m"] < 0.05 and r["end_error_m"] < 0.05
    assert all(v.startswith("OK") for v in r["verdict"])


def test_scale_error_is_detected():
    pts, a, b = _walk(scale=1.05)
    r = analyse_walk(pts, a, b)
    assert r["length_error_pct"] == pytest.approx(5.0, abs=0.5)
    # the reference speed is independent of the calibration, so a 5 % scale error shows up
    assert r["reference_speed"] == pytest.approx(1.4, rel=0.01)
    assert r["speed_error_pct"] == pytest.approx(5.0, abs=1.0)
    assert any(v.startswith("WARN") and "length" in v for v in r["verdict"])


def test_offset_and_wobble_reported():
    pts, a, b = _walk(offset=(0.0, 0.6), lateral_noise=0.15)
    r = analyse_walk(pts, a, b)
    assert r["start_error_m"] == pytest.approx(0.6, abs=0.1)
    assert r["lateral_deviation_m"]["mean_signed"] == pytest.approx(0.6, abs=0.1)
    # without A/B only the shape is checked: the offset disappears, the wobble stays
    r2 = analyse_walk(pts, known_length=12.0)
    assert abs(r2["lateral_deviation_m"]["mean_signed"]) < 0.05
    assert "start_error_m" not in r2


def test_pick_track_prefers_the_ab_walker():
    walker, a, b = _walk()
    other = walker.copy()
    other["track_id"] = 2
    other["x"], other["y"] = other["y"] - 180000 + 530020, other["x"] - 530000 + 180000
    pts = pd.concat([walker, other])
    summ = pd.DataFrame({"track_id": [1, 2], "class": "person", "path_length_m": [12.0, 30.0]})
    assert pick_track(pts, summ, None, a, b)["track_id"].iloc[0] == 1
    assert pick_track(pts, summ)["track_id"].iloc[0] == 2  # no A/B: longest
