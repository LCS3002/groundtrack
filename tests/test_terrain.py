"""Ground on two levels (a terrace 3 m above a square, joined by steps): a terrain model."""
import numpy as np
import pytest

from groundtrack.demo import ORIGIN, SyntheticCamera
from groundtrack.posefit import camera_calibration
from groundtrack.selfcal import implied_heights
from groundtrack.terrain import Terrain

Z_SQUARE = 9.4              # absolute elevations, like Trafalgar Square
Z_TERRACE = 12.4


def _terrain():
    """North of local y = 20 m is the terrace, south of y = 14 m the square, steps between."""
    res = 1.0
    left, top = ORIGIN[0] - 100, ORIGIN[1] + 120
    yy = top - (np.arange(220) + 0.5) * res - ORIGIN[1]
    z = np.interp(yy, [14.0, 20.0], [Z_SQUARE, Z_TERRACE])[:, None] * np.ones((1, 200))
    return Terrain(z=z, left=left, top=top, res=res, path="")


def _scene():
    """Camera 2 m above the terrace (on the gallery steps), looking south over both levels."""
    cam_xy = np.array([0.0, 30.0])
    cam = SyntheticCamera(height_m=Z_TERRACE + 2.0, pitch_deg=12.0, yaw_deg=180.0,
                          cam_local=cam_xy, origin=ORIGIN + [0.0, 0.0])
    # (SyntheticCamera heights are absolute here: its ground z = 0 is sea level)
    t = _terrain()
    rng = np.random.default_rng(1)
    xy = np.column_stack([rng.uniform(-15, 15, 30), rng.uniform(-30, 26, 30)]) + ORIGIN
    z = t.height(xy)
    px = cam.project(np.column_stack([xy, z]))
    ok = (px[:, 0] > 0) & (px[:, 0] < cam.width) & (px[:, 1] > 0) & (px[:, 1] < cam.height)
    ok &= _visible(cam, t, xy, z)
    return cam, t, xy[ok], z[ok], px[ok]


def _visible(cam, t, xy, z):
    """Not hidden behind the terrace edge (no clicking what you cannot see)."""
    a = np.linspace(0.02, 0.98, 200)[:, None, None]
    P = cam.C + a * (np.column_stack([xy, z])[None] - cam.C)
    return (P[..., 2] > t.height(P[..., :2].reshape(-1, 2)).reshape(P.shape[:2]) - 1e-6).all(axis=0)


def test_terrain_heights_and_ramp():
    t = _terrain()
    assert t.height(ORIGIN[None] + [[0, 40]])[0] == pytest.approx(Z_TERRACE)
    assert t.height(ORIGIN[None] + [[0, 0]])[0] == pytest.approx(Z_SQUARE)
    assert t.height(ORIGIN[None] + [[0, 17]])[0] == pytest.approx((Z_SQUARE + Z_TERRACE) / 2, abs=0.1)
    assert np.isnan(t.height(ORIGIN[None] + [[500, 0]])[0])


def test_clicks_on_two_levels_fit_one_camera():
    cam, t, xy, z, px = _scene()
    prior = {"E": ORIGIN[0], "N": ORIGIN[1] + 30.0, "height_m": 2.0, "position_tol_m": 5,
             "height_tol_m": 1.0}
    # yaw is measured from the clicks' mean direction; the terrace prior gives the datum
    h = camera_calibration(px, xy, (cam.width, cam.height), prior, terrain=t)
    assert h.rmse_m < 0.05
    assert h.ground_z_m == pytest.approx(Z_TERRACE)
    assert h.camera_params["height_m"] == pytest.approx(2.0, abs=0.05)
    # every point lands where it is, terrace and square alike
    err = np.linalg.norm(h.to_world(px) - xy, axis=1)
    assert err.max() < 0.05
    back = h.to_pixel(xy)
    assert np.abs(back - px).max() < 0.5
    # a single plane cannot: the same clicks without the terrain model miss by metres
    flat = camera_calibration(px, xy, (cam.width, cam.height), prior)
    miss = np.nan_to_num(np.linalg.norm(flat.to_world(px) - xy, axis=1), nan=1e3)  # NaN: sky
    assert np.median(miss) > 0.5


def test_people_on_the_terrace_and_in_the_square_are_1_7_m():
    cam, t, _, _, px = _scene()
    rng = np.random.default_rng(3)
    feet = np.column_stack([rng.uniform(-10, 10, 200), rng.uniform(-25, 25, 200)]) + ORIGIN
    zf = t.height(feet)
    seen = _visible(cam, t, feet, zf)
    feet, zf = feet[seen], zf[seen]
    foot_px = cam.project(np.column_stack([feet, zf]))
    top_px = cam.project(np.column_stack([feet, zf + 1.70]))
    _, _, xy, _, cpx = _scene()
    prior = {"E": ORIGIN[0], "N": ORIGIN[1] + 30.0, "height_m": 2.0, "position_tol_m": 5,
             "height_tol_m": 1.0}
    h = camera_calibration(cpx, xy, (cam.width, cam.height), prior, terrain=t)
    hz = implied_heights(foot_px, top_px, h)
    assert np.nanmedian(hz) == pytest.approx(1.70, abs=0.03)
    assert np.nanmax(np.abs(hz - 1.70)) < 0.1


def test_terrain_survives_save_and_load(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from groundtrack.homography import Homography

    t0 = _terrain()
    tif = tmp_path / "dtm" / "site_dtm.tif"
    tif.parent.mkdir()
    with rasterio.open(tif, "w", driver="GTiff", height=t0.z.shape[0], width=t0.z.shape[1],
                       count=1, dtype="float32", crs="EPSG:27700",
                       transform=from_origin(t0.left, t0.top, 1.0, 1.0)) as dst:
        dst.write(t0.z.astype(np.float32), 1)
    from groundtrack.terrain import load_terrain

    t = load_terrain(tif)
    cam, _, xy, _, px = _scene()
    prior = {"E": ORIGIN[0], "N": ORIGIN[1] + 30.0, "height_m": 2.0, "position_tol_m": 5,
             "height_tol_m": 1.0}
    h = camera_calibration(px, xy, (cam.width, cam.height), prior, terrain=t)
    h.save(tmp_path / "cal.json")
    h2 = Homography.load(tmp_path / "cal.json")
    assert h2.ground_z_m == pytest.approx(h.ground_z_m)
    assert np.abs(h2.to_world(px) - h.to_world(px)).max() < 1e-6
