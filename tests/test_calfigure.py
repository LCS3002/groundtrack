import numpy as np

from groundtrack.calfigure import grid_lines, seen_ground
from groundtrack.homography import Homography


def _top_down():
    # a camera looking straight down: 1 px = 1 cm, world = (u, v) / 100
    return Homography(H=np.diag([0.01, 0.01, 1.0]), origin=(0.0, 0.0), w_sign=1.0)


def test_seen_ground_is_the_frame_on_the_ground():
    out = seen_ground(_top_down(), (400, 200), max_range_m=100, cam=None)
    assert np.allclose(out.min(axis=0), [0, 0]) and np.allclose(out.max(axis=0), [4, 2])


def test_seen_ground_stops_at_the_range():
    out = seen_ground(_top_down(), (400, 200), max_range_m=1.5, cam=(0.0, 0.0))
    assert np.hypot(out[:, 0], out[:, 1]).max() <= 1.5 + 1e-6


def test_grid_lines_stay_on_the_seen_ground():
    out = seen_ground(_top_down(), (400, 200), max_range_m=100, cam=None)
    lines = grid_lines(out, spacing=1.0)
    pts = np.vstack(lines)
    assert pts[:, 0].min() >= -1e-6 and pts[:, 0].max() <= 4 + 1e-6
    assert pts[:, 1].min() >= -1e-6 and pts[:, 1].max() <= 2 + 1e-6
    xs = sorted({round(float(l[0, 0]), 3) for l in lines if np.ptp(l[:, 0]) == 0})
    assert {1.0, 2.0, 3.0} <= set(xs)     # every line inside (those on the edge may be cut)
