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


def test_untracked_detections_are_blurred_too(tmp_path):
    """A person cut off by the bottom of the frame is dropped from the tracks (no feet) but
    must still be blurred: privacy_boxes.csv joins the tracked boxes."""
    from groundtrack.debug_video import raw_all
    from groundtrack.layout import Run

    run = Run(tmp_path / "run").make()
    tracked = pd.DataFrame({"track_id": [1], "class": ["person"], "class_id": [0], "frame": [0],
                            "time_s": [0.0], "x1": [10], "y1": [20], "x2": [40], "y2": [120],
                            "confidence": [0.9], "predicted": [False]})
    tracked.to_csv(run.raw_tracks, index=False)
    pd.DataFrame({"frame": [0], "class": ["person"], "x1": [500], "y1": [700], "x2": [700],
                  "y2": [1080]}).to_csv(run.privacy, index=False)
    b = privacy_boxes(raw_all(run.root, tracked))
    assert set(b["x1"]) == {10, 500}


def test_hold_bridges_frames_where_the_detector_lost_someone():
    from groundtrack.debug_video import privacy_boxes

    r = pd.DataFrame({"frame": [10, 14], "class": ["person", "person"], "x1": [0, 0],
                      "y1": [0, 0], "x2": [50, 50], "y2": [100, 100]})
    frames = set(privacy_boxes(r, hold=2)["frame"])
    assert {11, 12, 13} <= frames and frames == set(range(8, 17))


def test_blur_boxes_blurs_overlapping_boxes_as_one_area():
    from groundtrack.debug_video import blur_boxes

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    out = img.copy()
    blur_boxes(out, [(20, 20, 100, 100), (60, 60, 140, 140)])
    assert out[30:90, 30:90].std() < 0.3 * img[30:90, 30:90].std()
    assert out[110:130, 110:130].std() < 0.3 * img[110:130, 110:130].std()
    assert np.array_equal(out[150:, :50], img[150:, :50])      # outside: untouched
