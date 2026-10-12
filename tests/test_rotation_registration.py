"""Handheld pans: the motion as a rotation (H = K R K^-1) survives a long pan; the quick
motion probe calls a pan that leaves the reference view 'moved', not 'static'."""
import cv2
import numpy as np

from groundtrack.registration import RotationRegistrar, measure_camera_motion

W, H, F = 640, 360, 500.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
OFF = np.array([[1, 0, 1800], [0, 1, 500], [0, 0, 1.0]])     # reference px -> texture px


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    t = np.full((1400, 4400, 3), 128, np.uint8)
    for _ in range(9000):
        x, y = int(rng.integers(0, 4400)), int(rng.integers(0, 1400))
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.circle(t, (x, y), int(rng.integers(3, 14)), c, -1)
        else:
            cv2.rectangle(t, (x, y), (x + int(rng.integers(4, 25)), y + int(rng.integers(4, 25))),
                          c, -1)
    return cv2.GaussianBlur(t, (3, 3), 0)


def _frame(tex, pan_deg):
    R = cv2.Rodrigues(np.array([0.0, np.radians(pan_deg), 0.0]))[0]   # ref <- cur
    M = OFF @ K @ R @ np.linalg.inv(K)
    return cv2.warpPerspective(tex, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def _pan(Hm):
    R = np.linalg.inv(K) @ Hm @ K
    R /= np.cbrt(np.linalg.det(R))
    return float(np.degrees(cv2.Rodrigues(R)[0].ravel()[1]))


def test_rotation_model_follows_a_long_pan():
    tex = _texture()
    pans = np.linspace(0, 40, 81)                     # 40 deg, far beyond the reference view
    reg = RotationRegistrar(_frame(tex, 0), F, abs_every=5)
    for p in pans:
        reg.register(_frame(tex, p))
    Hs = reg.finalize(30.0)
    assert abs(_pan(Hs[-1]) - 40) < 1.5
    assert max(abs(_pan(Hm) - p) for Hm, p in zip(Hs, pans)) < 2.0


def test_probe_calls_a_pan_moved(tmp_path):
    tex = _texture(1)
    for name, pans, moved in (("static.mp4", np.zeros(60), False),
                              ("pan.mp4", np.linspace(0, 60, 60), True)):
        path = tmp_path / name
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        for p in pans:
            out.write(_frame(tex, p))
        out.release()
        m = measure_camera_motion(path, 0)
        assert (m > 2.5) == moved, (name, m)
