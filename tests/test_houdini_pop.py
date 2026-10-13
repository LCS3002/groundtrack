import json

import pandas as pd

from groundtrack.houdini_pop import prepare_kit

HEADER = """ORIGIN = (1000.0, 2000.0)
SPEED_RANGES = {'people': (0.0, 2.5), 'vehicles': (0.0, 35.0), 'all': (0.0, 35.0)}
RAMP = [(0.0, (0.1, 0.2, 0.9)), (1.0, (0.9, 0.1, 0.0))]
"""


def _field(path, extra=None):
    rows = []
    for i in range(10, 14):
        for j in range(20, 22):
            r = {"i": i, "j": j, "cell_x": (i + 0.5) * 100.0, "cell_y": (j + 0.5) * 100.0,
                 "vx": 1.0, "vy": 2.0, "speed": 5 ** 0.5, "heading": 26.6, "density": 0.5,
                 "weight": 1.0, "confidence": 0.9, "dominance": 1.0}
            r.update(extra or {})
            rows.append(r)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_kit_modules_in_the_sites_houdini_frame(tmp_path):
    comp, clip = tmp_path / "composite" / "houdini", tmp_path / "clipA" / "houdini"
    comp.mkdir(parents=True)
    clip.mkdir(parents=True)
    (comp / "houdini_field.py").write_text(HEADER)
    (clip / "houdini_field.py").write_text(HEADER.replace("1000.0", "1050.0"))
    _field(comp / "vector_field_master.csv", {"dominant_group": "vehicles"})
    _field(comp / "vector_field_people_continuous.csv")
    _field(clip / "vector_field_people.csv")
    out = prepare_kit("site", tmp_path / "composite", [tmp_path / "clipA"], tmp_path / "kit",
                      log=lambda *a: None)
    kit = json.loads((out / "modules.json").read_text())
    names = [m["name"] for m in kit["modules"]]
    assert names == ["site_whole", "clipA"]
    whole, cl = kit["modules"]
    assert whole["fields"].keys() == {"all", "people"} and whole["group"] == "all"
    assert whole["visible"] and not cl["visible"]
    f = pd.read_csv(out / whole["fields"]["all"])
    # Easting -> +X, Northing -> -Z around the SITE origin (the clip shares it)
    assert f["x"].min() == 1050.0 - 1000.0 and f["z"].max() == -(2050.0 - 2000.0)
    assert (f["vx"] == 1.0).all() and (f["vz"] == -2.0).all()
    assert (f["smax"] == 35.0).all()                       # the cell's dominant group
    assert (pd.read_csv(out / cl["fields"]["people"])["smax"] == 2.5).all()
    assert abs(f["cell"].iloc[0] - 100.0) < 1e-6
    assert (out / "hou_pop_build.py").exists() and (out / "README.txt").exists()
