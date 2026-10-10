"""What gets blurred: every person, and the number-plate band of every near vehicle."""
import numpy as np
import pandas as pd

from groundtrack.debug_video import PLATE_BAND, PLATE_MIN_PX, blur_box, privacy_boxes


def _raw():
    return pd.DataFrame({
        "frame": [0, 0, 0, 0, 1],
        "class": ["person", "car", "car", "bicycle", "bus"],
        "x1": [10, 100, 300, 500, 600], "y1": [20, 200, 400, 50, 100],
        "x2": [40, 300, 320, 560, 900], "y2": [120, 320, 410, 150, 400],
    })


def test_people_whole_box_vehicles_lower_band():
    b = privacy_boxes(_raw())
    person = b[b["x1"] == 10].iloc[0]
    assert (person["y1"], person["y2"]) == (20, 120)              # the whole person
    car = b[b["x1"] == 100].iloc[0]
    assert car["y2"] == 320 and car["y1"] == 320 - PLATE_BAND * 120  # the plate band only
    bus = b[b["x1"] == 600].iloc[0]
    assert bus["y1"] == 400 - PLATE_BAND * 300
    # a far car (10 px tall) has no readable plate; bicycles have none
    assert 300 not in set(b["x1"]) and 500 not in set(b["x1"])
    assert PLATE_MIN_PX > 10


def test_switches():
    assert set(privacy_boxes(_raw(), people=True, plates=False)["x1"]) == {10}
    assert set(privacy_boxes(_raw(), people=False, plates=True)["x1"]) == {100, 600}
    assert privacy_boxes(_raw(), people=False, plates=False).empty


def test_blur_removes_detail_inside_the_box_only():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    out = img.copy()
    blur_box(out, 50, 50, 150, 150)
    inside, outside = (slice(60, 140), slice(60, 140)), (slice(0, 40), slice(0, 40))
    assert out[inside].std() < 0.3 * img[inside].std()
    assert np.array_equal(out[outside], img[outside])
