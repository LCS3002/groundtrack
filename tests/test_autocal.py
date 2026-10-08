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
