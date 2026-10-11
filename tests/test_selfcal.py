import numpy as np
import pandas as pd
import pytest

from groundtrack.config import default_config
from groundtrack.demo import ORIGIN, SyntheticCamera
from groundtrack.homography import fit_homography
from groundtrack.selfcal import implied_heights, people_check
from groundtrack.trajectories import max_range_for


def _scene(height_m=6.0, pitch=20.0, person=1.70, n=400, seed=0):
    cam = SyntheticCamera(height_m=height_m, pitch_deg=pitch, cam_local=(0.0, -20.0))
    rng = np.random.default_rng(seed)
    # calibration: a spread of ground points in view
    g = np.column_stack([rng.uniform(-12, 12, 40), rng.uniform(-8, 25, 40)]) + ORIGIN
    h = fit_homography(cam.ground_to_pixel(g), g, image_size=(cam.width, cam.height))
    # people standing on the ground, boxes from feet to the top of the head
    feet = np.column_stack([rng.uniform(-8, 8, n), rng.uniform(-5, 20, n)]) + ORIGIN
    foot_px = cam.ground_to_pixel(feet)
    top_px = cam.project(np.column_stack([feet, np.full(n, person)]))
    return cam, h, foot_px, top_px


def test_implied_heights_recover_person_height():
    _, h, foot, top = _scene()
    z = implied_heights(foot, top, h)
    assert np.nanmedian(z) == pytest.approx(1.70, abs=0.03)


def test_people_check_flags_a_wrong_scale():
    cam, h, foot, top = _scene(person=1.70)
    n = len(foot)
    raw = pd.DataFrame({"frame": np.arange(n), "track_id": np.arange(n), "class": "person",
                        "x1": foot[:, 0] - 10, "x2": foot[:, 0] + 10,
                        "y1": top[:, 1], "y2": foot[:, 1], "predicted": False})
    ok = people_check(raw, h)
    assert ok["implied_person_height_m"] == pytest.approx(1.70, abs=0.05)
    assert abs(ok["scale_error_pct"]) < 4
    # the same boxes, but people 25 % taller in the image: the check must notice
    raw2 = raw.assign(y1=raw["y2"] - 1.25 * (raw["y2"] - raw["y1"]))
    off = people_check(raw2, h)
    assert off["scale_error_pct"] > 15


def test_max_range_auto_per_group():
    _, h, _, _ = _scene(height_m=6.0)
    cfg = default_config()
    cam = h.camera()
    assert max_range_for(cfg, h, "people") is None                  # default: keep all
    cfg.data["projection"]["max_range_m"] = 25
    assert max_range_for(cfg, h, "people") == 25
    cfg.data["projection"]["max_range_m"] = "auto"
    f, hh = cam["focal_px"], cam["height_m"]
    assert max_range_for(cfg, h, "people") == pytest.approx(np.sqrt(0.25 * f * hh - hh * hh))
    # vehicles tolerate coarser depth resolution than people: they are kept farther out
    assert max_range_for(cfg, h, "vehicles") > max_range_for(cfg, h, "people")


def test_plane_homography_puts_raised_points_back():
    """A point 8 m up (a train on a viaduct) lands right on the 8 m plane, not on the ground."""
    from groundtrack.homography import plane_homography

    cam, h, _, _ = _scene(height_m=40.0, pitch=25.0)
    rng = np.random.default_rng(4)
    xy = np.column_stack([rng.uniform(-10, 10, 20), rng.uniform(30, 70, 20)]) + ORIGIN
    px = cam.project(np.column_stack([xy, np.full(len(xy), 8.0)]))
    on_plane = plane_homography(h, 8.0).to_world(px)
    on_ground = h.to_world(px)
    assert np.abs(on_plane - xy).max() < 0.05
    # treated as ground, the same pixels land several metres further from the camera
    assert np.median(np.linalg.norm(on_ground - xy, axis=1)) > 5


def test_people_by_distance_tell_a_wrong_tilt():
    """Right camera: people ~1.7 m in every distance band. Tilt 2 deg off: the bands drift."""
    from groundtrack.posefit import pose_homography

    cam = SyntheticCamera(height_m=1.6, pitch_deg=4.0, cam_local=(0.0, 0.0), f=1400)
    rng = np.random.default_rng(3)
    n = 3000
    feet = np.column_stack([rng.uniform(-8, 8, n), rng.uniform(5, 70, n)]) + ORIGIN
    foot_px = cam.ground_to_pixel(feet)
    top_px = cam.project(np.column_stack([feet, np.full(n, 1.70)]))
    raw = pd.DataFrame({"frame": np.arange(n), "track_id": np.arange(n), "class": "person",
                        "x1": foot_px[:, 0] - 10, "x2": foot_px[:, 0] + 10,
                        "y1": top_px[:, 1], "y2": foot_px[:, 1], "predicted": False})
    raw = raw[(raw.y2 < 1075) & (raw.y1 > 3)]

    def h_with(pitch):
        h = pose_homography(ORIGIN[0], ORIGIN[1], 1.6, 0.0, pitch, 1400.0, (1920, 1080))
        h.camera_params = {"E": ORIGIN[0], "N": ORIGIN[1], "height_m": 1.6, "yaw_deg": 0.0,
                           "tilt_deg": pitch, "roll_deg": 0.0, "focal_px": 1400.0}
        return h

    right = people_check(raw, h_with(4.0))
    wrong = people_check(raw, h_with(6.0))
    assert right["consistent_share"] == 1.0
    assert all(abs(v - 1.7) < 0.1 for v in right["by_distance_m"].values())
    assert wrong["consistent_share"] < 0.5
