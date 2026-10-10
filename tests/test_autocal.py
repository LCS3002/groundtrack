"""Calibration without clicks, on synthetic scenes with a known camera."""

import math

import cv2
import numpy as np
import pandas as pd
import pytest

from groundtrack.autocal import Road, fit_to_roads, transfer
from groundtrack.demo import ORIGIN, SyntheticCamera
from groundtrack.homography import fit_homography


def _texture(seed=0, n=1200):
    """A random 'aerial' texture: blobs and lines, so SIFT finds plenty to match."""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), 90, np.uint8)
    for _ in range(900):
        c = tuple(int(v) for v in rng.integers(0, n, 2))
        cv2.circle(img, c, int(rng.integers(3, 18)), int(rng.integers(0, 255)), -1)
    for _ in range(150):
        a, b = (tuple(int(v) for v in rng.integers(0, n, 2)) for _ in range(2))
        cv2.line(img, a, b, int(rng.integers(0, 255)), int(rng.integers(1, 4)))
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def _render(cam, tex, metres_per_px=0.1):
    """What `cam` sees of a ground texture centred on ORIGIN."""
    n = tex.shape[0]
    T = np.array([[metres_per_px, 0, ORIGIN[0] - n / 2 * metres_per_px],
                  [0, -metres_per_px, ORIGIN[1] + n / 2 * metres_per_px], [0, 0, 1]])
    Hg = cam.ground_homography() @ T                      # texture pixel -> image pixel
    out = cv2.warpPerspective(tex, Hg, (cam.width, cam.height), flags=cv2.INTER_LINEAR)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _calib(cam):
    g = np.column_stack([np.random.default_rng(1).uniform(-15, 15, (60, 2))]) + ORIGIN
    px = cam.ground_to_pixel(g)
    ok = (px[:, 0] > 0) & (px[:, 0] < cam.width) & (px[:, 1] > 0) & (px[:, 1] < cam.height)
    return fit_homography(px[ok], g[ok], image_size=(cam.width, cam.height))


def test_same_spot_transfer_is_exact():
    """Same position, other direction and zoom: the transfer must reproduce the truth."""
    tex = _texture()
    a = SyntheticCamera(f=1100, height_m=25, pitch_deg=50, yaw_deg=0, cam_local=(0, -20))
    b = SyntheticCamera(f=1500, height_m=25, pitch_deg=55, yaw_deg=12, cam_local=(0, -20))
    h, info = transfer(_calib(a), _render(a, tex), _render(b, tex), log=lambda *_: None)
    assert h is not None, info
    truth = _calib(b)
    gu, gv = np.meshgrid(np.linspace(100, 1180, 12), np.linspace(300, 700, 6))
    px = np.column_stack([gu.ravel(), gv.ravel()])
    err = np.linalg.norm(h.to_world(px) - truth.to_world(px), axis=1)
    assert np.nanmedian(err) < 0.15, info


def test_different_views_are_not_transferred():
    a = SyntheticCamera(f=1100, height_m=25, pitch_deg=50, yaw_deg=0, cam_local=(0, -20))
    h, info = transfer(_calib(a), _render(a, _texture(0)), _render(a, _texture(7)),
                       log=lambda *_: None)
    assert h is None and "match" in info["reason"]


def _vehicle_scene(cam, fps=25.0):
    """Roads (OSM-like lines) through the view and cars driving on them, as raw tracks."""
    centre = cam.C[:2] + cam.C[2] / math.tan(math.radians(cam.pitch)) * np.array(
        [math.sin(math.radians(cam.yaw)), math.cos(math.radians(cam.yaw))])

    def road(angle_off, shift, kind):
        a = math.radians(cam.yaw + angle_off)
        d = np.array([math.sin(a), math.cos(a)])
        n = np.array([-d[1], d[0]])
        c = centre + n * shift
        return Road(np.array([c - d * 500, c + d * 500]), kind)

    roads = [road(75, 0, "primary"), road(15, 20, "secondary"), road(100, -60, "residential"),
             road(40, 90, "service")]
    rows, tid = [], 0
    rng = np.random.default_rng(3)
    for k, rd in enumerate(roads[:2]):
        a, b = rd[0], rd[1]
        d = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-d[1], d[0]])
        for j in range(18):
            tid += 1
            lane = (1.8 if j % 2 else -1.8)
            v = rng.uniform(8, 16) * (1 if j % 2 else -1)
            s0 = rng.uniform(0.42, 0.58) * np.linalg.norm(b - a)
            for f in range(0, 60, 2):
                s = s0 + v * f / fps
                c = a + d * s + normal * lane
                head = d if v > 0 else -d
                side = np.array([-head[1], head[0]])
                corners = np.array([c + head * 2.25 + side * 0.9, c + head * 2.25 - side * 0.9,
                                    c - head * 2.25 + side * 0.9, c - head * 2.25 - side * 0.9])
                px = cam.ground_to_pixel(corners)
                if not np.isfinite(px).all():
                    continue
                x1, x2, y2 = px[:, 0].min(), px[:, 0].max(), px[:, 1].max()
                if x1 < 0 or x2 > cam.width or y2 < 0 or y2 > cam.height:
                    continue
                rows.append((tid, "car", 2, f, f / fps, x1, y2 - 30, x2, y2, 0.9, False))
    raw = pd.DataFrame(rows, columns=["track_id", "class", "class_id", "frame", "time_s", "x1",
                                      "y1", "x2", "y2", "confidence", "predicted"])
    return raw, roads


@pytest.mark.parametrize("yaw, pitch, f", [(35.0, 18.0, 1500.0), (290.0, 25.0, 900.0)])
def test_vehicles_on_roads_recover_the_camera(yaw, pitch, f):
    cam = SyntheticCamera(width=1280, height=720, f=f, height_m=80.0, pitch_deg=pitch,
                          yaw_deg=yaw, cam_local=(0.0, 0.0))
    raw, roads = _vehicle_scene(cam)
    prior = {"E": ORIGIN[0] + 4.0, "N": ORIGIN[1] - 3.0, "height_m": 82.0,
             "position_tol_m": 10.0, "height_tol_m": 5.0}
    h, info = fit_to_roads(raw, (1280, 720), prior, roads, log=lambda *_: None, radius_m=600)
    assert h is not None, info
    c = info["camera"]
    true_fov = math.degrees(2 * math.atan(640 / f))
    assert abs((c["yaw_deg"] - yaw + 180) % 360 - 180) < 3, c
    assert abs(c["tilt_deg"] - pitch) < 3, c
    assert abs(c["hfov_deg"] / true_fov - 1) < 0.15, c
    assert info["on_road_share"] > 0.9


def _people_scene(cam, fps=25.0, n_people=40, seed=5):
    """People (1.7 m +- 7 cm, walking ~1.3 m/s) in front of an eye-level camera, as raw tracks."""
    rng = np.random.default_rng(seed)
    fwd = np.array([math.sin(math.radians(cam.yaw)), math.cos(math.radians(cam.yaw))])
    side = np.array([fwd[1], -fwd[0]])
    rows = []
    for tid in range(1, n_people + 1):
        start = cam.C[:2] + fwd * rng.uniform(6, 25) + side * rng.uniform(-8, 8)
        ang = rng.uniform(0, 2 * math.pi)
        vel = rng.normal(1.3, 0.12) * np.array([math.cos(ang), math.sin(ang)])
        height = rng.normal(1.70, 0.07)
        for f in range(0, 100, 2):
            g = start + vel * f / fps
            foot = cam.ground_to_pixel(g[None])[0]
            top = cam.project(np.array([[g[0], g[1], height]]))[0]
            half = 0.25 * (foot[1] - top[1]) / height        # ~0.5 m wide
            x1, x2, y1, y2 = foot[0] - half, foot[0] + half, top[1], foot[1]
            if not np.isfinite([x1, x2, y1, y2]).all() or x1 < 0 or x2 > cam.width or \
                    y1 < 3 or y2 > cam.height or y2 - y1 < 45:
                continue
            rows.append((tid, "person", 0, f, f / fps, x1, y1, x2, y2, 0.9, False))
    return pd.DataFrame(rows, columns=["track_id", "class", "class_id", "frame", "time_s", "x1",
                                       "y1", "x2", "y2", "confidence", "predicted"])


def test_plaza_people_plus_two_clicks():
    """Eye-level camera: the people fix tilt, height and zoom; 2 clicks fix the direction."""
    from groundtrack.autocal import fit_to_people

    cam = SyntheticCamera(width=1280, height=720, f=1000.0, height_m=2.2, pitch_deg=4.0,
                          yaw_deg=30.0, cam_local=(0.0, 0.0))
    raw = _people_scene(cam)
    fwd = np.array([math.sin(math.radians(30)), math.cos(math.radians(30))])
    side = np.array([fwd[1], -fwd[0]])
    pts = np.array([cam.C[:2] + fwd * 12 + side * 4, cam.C[:2] + fwd * 20 - side * 5])
    clicks = (cam.ground_to_pixel(pts), pts)
    prior = {"E": ORIGIN[0] + 2.0, "N": ORIGIN[1] - 1.5, "height_m": 1.8,
             "position_tol_m": 6.0, "height_tol_m": 1.5}
    h, info = fit_to_people(raw, (1280, 720), prior, [], log=lambda *_: None, clicks=clicks,
                            radius_m=100)
    assert h is not None, info
    check = np.array([cam.C[:2] + fwd * d + side * s for d in (8, 15, 25) for s in (-6, 0, 6)])
    err = np.linalg.norm(h.to_world(cam.ground_to_pixel(check)) - check, axis=1)
    assert np.median(err) < 0.5, (err, info["camera"])
    assert abs(info["camera"]["height_m"] - 2.2) < 0.4


def test_picker_with_two_points_defers_to_autocalibrate(tmp_path):
    """With a camera position, fewer than 3 clicks are saved for `autocalibrate`."""
    from groundtrack.calibrate import load_points_csv, run_calibration, save_points_csv
    from groundtrack.demo import make_demo_site

    site = make_demo_site(tmp_path)
    px, world = load_points_csv(site["points_csv"])
    two = tmp_path / "two.csv"
    save_points_csv(two, px[:2], world[:2])
    out = tmp_path / "cal" / "x_homography.json"
    res = run_calibration(site["video"], site["geotiff"], out, points_csv=two, interactive=False,
                          camera_prior={"E": world[0][0], "N": world[0][1] - 30, "height_m": 10},
                          log=lambda *_: None)
    assert res is None and not out.exists()
    assert len(load_points_csv(out.with_name("x_homography_points.csv"))[0]) == 2


def test_cross_validation_picks_and_pools(tmp_path):
    """4+ clicks: methods are compared on held-out clicks; same-spot clicks only train."""
    from groundtrack.autocal import best_by_cross_validation

    cam = SyntheticCamera(width=1280, height=720, f=1400.0, height_m=30.0, pitch_deg=25.0,
                          yaw_deg=20.0, cam_local=(0.0, 0.0))
    rng = np.random.default_rng(2)
    fwd = np.array([math.sin(math.radians(20)), math.cos(math.radians(20))])
    side = np.array([fwd[1], -fwd[0]])
    ground = np.array([cam.C[:2] + fwd * d + side * s
                       for d, s in ((50, -15), (60, 10), (80, -5), (95, 20), (120, 0))])
    own = (cam.ground_to_pixel(ground) + rng.normal(0, 1.5, (5, 2)), ground)
    extra_g = np.array([cam.C[:2] + fwd * d + side * s for d, s in ((55, 25), (100, -20),
                                                                     (70, 0))])
    pooled = {"other": (cam.ground_to_pixel(extra_g), extra_g)}
    raw = pd.DataFrame(columns=["track_id", "class", "frame", "time_s", "x1", "y1", "x2", "y2",
                                "predicted"])
    prior = {"E": ORIGIN[0], "N": ORIGIN[1], "height_m": 30.0, "position_tol_m": 5,
             "height_tol_m": 3}
    h, scores = best_by_cross_validation(raw, (1280, 720), prior, own, None,
                                         tmp_path / "x_homography.json", log=lambda *_: None,
                                         pooled=pooled)
    assert "clicked points" in scores and any("same spot" in k for k in scores)
    assert all(np.isfinite(v) for v in scores.values())
    check = np.array([cam.C[:2] + fwd * d for d in (60, 90)])
    err = np.linalg.norm(h.to_world(cam.ground_to_pixel(check)) - check, axis=1)
    assert err.max() < 1.0


def test_lane_check_measures_a_known_rotation():
    """Painted lines at 30 deg, vehicles driving at 32 deg -> the check reports +2 deg."""
    from groundtrack.autocal import lane_check
    from groundtrack.geo import GeoRaster

    res, n = 0.1, 2000                                     # 200 m x 200 m at 10 cm
    img = np.full((n, n, 3), 70, np.uint8)
    a = math.radians(30)
    d = np.array([math.cos(a), -math.sin(a)])              # image coords (y down)
    nrm = np.array([-d[1], d[0]])
    for k in range(-60, 61):                               # lane lines every 3.5 m
        c = np.array([n / 2, n / 2]) + nrm * k * 3.5 / res
        p0, p1 = c - d * 3000, c + d * 3000
        cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), (230, 230, 230), 2)
    E0, N0 = ORIGIN
    raster = GeoRaster(image=img, left=E0 - 100, right=E0 + 100, bottom=N0 - 100,
                       top=N0 + 100, epsg=27700, native_res=res)
    rows = []
    b = math.radians(32)
    for tid in range(12):
        start = np.array([E0 - 60, N0 - 50 + tid * 6.0])
        for f in range(40):
            x, y = start + f * 2.0 * np.array([math.cos(b), math.sin(b)])
            rows.append((tid, "car", "vehicles", f, x, y, False))
    pts = pd.DataFrame(rows, columns=["track_id", "class", "group", "frame", "x", "y",
                                      "predicted"])
    chk = lane_check(pts, raster)
    assert chk is not None
    assert chk["rotation_vs_painted_lanes_deg"] == pytest.approx(2.0, abs=0.5)
    assert chk["sideways_scatter_cm"] < 1


def test_rotation_spread_tells_same_spot_from_a_moved_camera():
    """Two views from one spot differ by a rotation (and zoom); from two spots, a homography
    fitted to the ground is not a rotation - the transfer must refuse it."""
    import cv2

    from groundtrack.autocal import ROTATION_SPREAD_MAX, rotation_spread
    from groundtrack.demo import ORIGIN, SyntheticCamera

    rng = np.random.default_rng(2)
    pts = np.column_stack([rng.uniform(-30, 30, 300), rng.uniform(10, 120, 300),
                           rng.uniform(0, 15, 300)]) + np.r_[ORIGIN, 0]
    ground = pts.copy()
    ground[:, 2] = 0
    a = SyntheticCamera(height_m=8.0, pitch_deg=15, yaw_deg=0, f=1400, cam_local=(0, 0))
    turned = SyntheticCamera(height_m=8.0, pitch_deg=12, yaw_deg=9, f=1900, cam_local=(0, 0))
    moved = SyntheticCamera(height_m=11.0, pitch_deg=15, yaw_deg=4, f=1400, cam_local=(6, -4))
    H_same, _ = cv2.findHomography(turned.project(pts), a.project(pts))     # any depth
    H_moved, _ = cv2.findHomography(moved.project(ground), a.project(ground))  # ground only
    size = (1920, 1080)
    assert rotation_spread(H_same, size, size) < 1.02
    assert rotation_spread(H_moved, size, size) > ROTATION_SPREAD_MAX
