import numpy as np

from groundtrack.homography import Homography, fit_homography, report


def test_exact_recovery(cam, ground_points):
    px = cam.ground_to_pixel(ground_points)
    h = fit_homography(px, ground_points)
    assert h.rmse_m < 1e-4  # 0.1 mm
    # unseen points across the scene project to the right place
    rng = np.random.default_rng(0)
    test = ground_points.mean(0) + rng.uniform(-15, 15, (200, 2))
    err = np.linalg.norm(h.to_world(cam.ground_to_pixel(test)) - test, axis=1)
    assert err.max() < 1e-4
    # and back again
    assert np.abs(h.to_pixel(test) - cam.ground_to_pixel(test)).max() < 1e-3


def test_click_noise_gives_small_error(cam, ground_points):
    rng = np.random.default_rng(1)
    px = cam.ground_to_pixel(ground_points) + rng.normal(0, 1.0, (len(ground_points), 2))
    h = fit_homography(px, ground_points)
    assert h.rmse_m < 0.3
    assert h.loo_rmse_m is not None and h.loo_rmse_m >= h.rmse_m
    assert all(p["inlier"] for p in h.points)


def test_bad_click_is_rejected_and_reported(cam, ground_points):
    world = ground_points.copy()
    world[3] += [4.0, -3.0]  # one wrong map click, 5 m off
    h = fit_homography(cam.ground_to_pixel(ground_points), world)
    assert not h.points[3]["inlier"]
    assert sum(p["inlier"] for p in h.points) == 7
    assert h.rmse_m < 1e-3
    assert h.points[3]["error_m"] > 4.0
    assert "rejected as outliers" in report(h)


def test_high_rmse_warns(cam, ground_points):
    rng = np.random.default_rng(2)
    world = ground_points + rng.normal(0, 0.8, ground_points.shape)
    h = fit_homography(cam.ground_to_pixel(ground_points), world, ransac_thresh_m=5.0)
    assert h.rmse_m > 0.5
    assert "WARNING: RMSE" in report(h)


def test_above_horizon_is_nan(cam, ground_points):
    h = fit_homography(cam.ground_to_pixel(ground_points), ground_points)
    # pitch 30 deg, f=1400: horizon at v = 540 - 1400*tan(30deg) ~ -268, so v=-600 is sky
    out = h.to_world(np.array([[960.0, -600.0], [960.0, 900.0]]))
    assert np.isnan(out[0]).all()
    assert np.isfinite(out[1]).all()


def test_save_load_roundtrip(tmp_path, cam, ground_points):
    h = fit_homography(cam.ground_to_pixel(ground_points), ground_points, image_size=(1920, 1080))
    p = tmp_path / "h.json"
    h.save(p)
    h2 = Homography.load(p)
    px = cam.ground_to_pixel(ground_points)
    assert np.allclose(h.to_world(px), h2.to_world(px), atol=1e-9)
    assert h2.image_size == (1920, 1080)
    # H_world (for other tools) agrees with origin + H_local
    Hw = np.array(__import__("json").loads(p.read_text())["H_world"])
    ph = np.column_stack([px, np.ones(len(px))]) @ Hw.T
    assert np.allclose(ph[:, :2] / ph[:, 2:], ground_points, atol=1e-4)
