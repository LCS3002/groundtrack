import json
import math
import py_compile
import sys
import types

import numpy as np
import pandas as pd
import pytest

from groundtrack import exports


def _points(rows):
    df = pd.DataFrame(rows, columns=["track_id", "class", "frame", "time_s", "x", "y", "vx", "vy",
                                     "speed", "predicted"])
    df["heading"] = (np.degrees(np.arctan2(df["vx"], df["vy"])) + 360) % 360
    df["source"] = np.where(df["predicted"], "predicted", "detected")
    df["group"] = np.where(df["class"] == "person", "people", "vehicles")
    return df


# --------------------------------------------------------------------------- grid
def test_field_grid_aggregation():
    pts = _points([
        (1, "person", 0, 0.0, 530000.2, 180000.5, 1.0, 0.0, 1.0, False),
        (1, "person", 1, 0.1, 530000.7, 180000.5, 1.0, 0.0, 1.0, False),
        (2, "person", 0, 0.0, 530000.5, 180000.2, -1.0, 0.0, 1.0, False),  # opposite flow
        (2, "person", 1, 0.1, 530001.5, 180000.2, 0.0, 2.0, 2.0, False),
        (3, "person", 0, 0.0, 530001.5, 180000.9, 9.0, 9.0, 12.7, True),   # predicted
    ])
    g = exports.field_grid(pts, cell=1.0, include_predicted=False)
    c0 = g[(g["i"] == 530000) & (g["j"] == 180000)].iloc[0]
    assert c0["count"] == 3 and c0["n_tracks"] == 2
    assert c0["mean_vx"] == pytest.approx(1 / 3, abs=1e-4)
    assert c0["mean_speed"] == pytest.approx(1.0)
    assert c0["cell_x"] == 530000.5 and c0["cell_y"] == 180000.5
    assert c0["coherence"] == pytest.approx(1 / 3, abs=1e-3)  # partly cancelling flows
    c1 = g[(g["i"] == 530001)].iloc[0]
    assert c1["count"] == 1 and c1["mean_vy"] == 2.0 and c1["heading"] == 0.0
    assert g["count"].sum() == 4
    g2 = exports.field_grid(pts, cell=1.0, include_predicted=True)
    assert g2["count"].sum() == 5


def test_grid_cells_are_world_aligned():
    pts = _points([(1, "person", 0, 0.0, 530003.9, 180007.1, 1, 0, 1, False)])
    g = exports.field_grid(pts, cell=2.0)
    assert g["cell_x"].iloc[0] == 530003.0 and g["cell_y"].iloc[0] == 180007.0


# --------------------------------------------------------------------------- geojson
def _track_points():
    rows = []
    for k in range(20):
        rows.append((1, "person", k, k * 0.1, 530000 + 0.14 * k, 180000.0, 1.4, 0.0, 1.4,
                     8 <= k < 12))
    for k in range(5):  # stationary at the start -> duplicate coordinates
        rows.append((2, "car", k, k * 0.1, 530010.0, 180010.0, 0.0, 0.0, 0.0, False))
    for k in range(5):
        rows.append((2, "car", 5 + k, 0.5 + k * 0.1, 530010.0 + 2.0 * k, 180010.0, 20, 0, 20,
                     False))
    return _points(rows)


def _summary(points):
    return pd.DataFrame([{"track_id": t, "class": g["class"].iloc[0], "path_length_m": 1.0,
                          "straightness": np.nan if t == 2 else 0.9, "mean_speed": 1.0}
                         for t, g in points.groupby("track_id")])


def _assert_valid_linestring_fc(gj):
    assert gj["type"] == "FeatureCollection"
    assert gj["crs"]["properties"]["name"].endswith("27700")
    for f in gj["features"]:
        assert f["type"] == "Feature"
        geom = f["geometry"]
        assert geom["type"] == "LineString"
        coords = geom["coordinates"]
        assert len(coords) >= 2
        for c in coords:
            assert len(c) == 2 and all(isinstance(v, float) and math.isfinite(v) for v in c)
        # no zero-length consecutive duplicates
        assert all(a != b for a, b in zip(coords[:-1], coords[1:]))
        assert isinstance(f["properties"], dict)
    # strict JSON (no NaN/Infinity tokens), as QGIS/GDAL expects
    json.loads(json.dumps(gj, allow_nan=False))


def test_tracks_geojson_valid(tmp_path):
    pts = _track_points()
    gj = exports.tracks_geojson(pts, _summary(pts))
    _assert_valid_linestring_fc(gj)
    assert [f["properties"]["track_id"] for f in gj["features"]] == [1, 2]
    assert gj["features"][1]["properties"]["straightness"] is None  # NaN -> null
    p = tmp_path / "t.geojson"
    exports.write_geojson(gj, p)
    assert json.loads(p.read_text())["features"][0]["geometry"]["coordinates"][0] == \
        [530000.0, 180000.0]


def test_tracks_geojson_readable_by_gdal(tmp_path):
    pyogrio = pytest.importorskip("pyogrio")
    pts = _track_points()
    p = tmp_path / "t.geojson"
    exports.write_geojson(exports.tracks_geojson(pts, _summary(pts)), p)
    info = pyogrio.read_info(p)
    assert info["features"] == 2 and "27700" in info["crs"]


def test_predicted_gaps_geojson():
    gj = exports.predicted_segments_geojson(_track_points())
    _assert_valid_linestring_fc(gj)
    assert len(gj["features"]) == 1
    f = gj["features"][0]
    assert f["properties"]["n_samples"] == 4
    assert len(f["geometry"]["coordinates"]) == 6  # joined to the real path on both sides


# --------------------------------------------------------------------------- stats
def test_count_line_directions(cfg):
    pts = _points([
        (1, "person", 0, 0.0, 0.0, -1.0, 0, 1, 1, False),
        (1, "person", 1, 1.0, 0.0, 1.0, 0, 1, 1, False),     # crosses northbound
        (2, "person", 0, 0.0, 0.5, 1.0, 0, -1, 1, False),
        (2, "person", 1, 1.0, 0.5, -1.0, 0, -1, 1, False),   # crosses southbound
        (3, "person", 0, 0.0, 5.0, -1.0, 0, 1, 1, False),
        (3, "person", 1, 1.0, 5.0, 1.0, 0, 1, 1, False),     # passes beyond the line's end
    ])
    cfg.data["stats"]["count_lines"] = [{"name": "gate", "a": [-2, 0], "b": [2, 0]}]
    summ = pd.DataFrame([{"track_id": t, "class": "person", "group": "people", "start_s": 0.0,
                          "mean_speed": 1.0, "straightness": 1.0, "duration_s": 1.0,
                          "path_length_m": 2.0} for t in (1, 2, 3)])
    st = exports.compute_stats(pts, summ, cfg, (0.0, 60.0))
    cl = st["count_lines"][0]
    # standing at a=(-2,0) looking at b=(2,0) (east): left is north
    assert cl["crossings_right_to_left"] == 1 and cl["crossings_left_to_right"] == 1
    assert st["per_class"]["person"]["n_tracks"] == 3
    assert st["per_class"]["person"]["flow_per_minute"] == [3]
    hist = st["per_group"]["people"]["speed_histogram"]
    assert sum(hist["counts"]) == 3
    table = exports.stats_table(st)
    assert {"scope", "name", "metric", "value"} <= set(table.columns)
    assert (table["metric"] == "speed_hist").sum() == 2 * cfg["stats"]["histogram_bins"]


# --------------------------------------------------------------------------- houdini
class _FakeGeo:
    def __init__(self):
        self.attribs, self.values, self.points, self.polys = {}, {}, [], []

    def addAttrib(self, kind, name, default):
        self.attribs[(kind, name)] = default

    def createPoints(self, positions):
        self.points = list(range(len(positions)))
        self.positions = positions
        return self.points

    def _set(self, name, vals):
        self.values[name] = list(vals)

    setPointIntAttribValues = setPointFloatAttribValues = setPointStringAttribValues = _set

    def createPolygon(self, is_closed=True):
        geo = self

        class Poly:
            def __init__(self):
                self.verts, self.attrs = [], {}
                geo.polys.append(self)

            def addVertex(self, p):
                self.verts.append(p)

            def setAttribValue(self, k, v):
                self.attrs[k] = v

        return Poly()


def _fake_hou(geo):
    hou = types.ModuleType("hou")
    hou.attribType = types.SimpleNamespace(Point="pt", Prim="prim", Global="detail")
    hou.Vector3 = lambda *a: tuple(a)
    node = types.SimpleNamespace(geometry=lambda: geo, parm=lambda name: None)
    hou.pwd = lambda: node
    hou.time = lambda: 0.05
    return hou


def test_houdini_script_runs(tmp_path, cfg, monkeypatch):
    pts = _track_points()
    csv = tmp_path / "points.csv"
    pts.to_csv(csv, index=False)
    script = tmp_path / "houdini_import.py"
    exports.write_houdini_script(script, csv, (530000.0, 180000.0), cfg)
    py_compile.compile(str(script), doraise=True)

    geo = _FakeGeo()
    monkeypatch.setitem(sys.modules, "hou", _fake_hou(geo))
    exec(compile(script.read_text(), str(script), "exec"), {"__name__": "__main__"})
    assert len(geo.points) == len(pts)
    assert len(geo.polys) == 2
    assert len(geo.polys[0].verts) == 20
    # Y-up mapping: E -> X, N -> -Z, relative to origin
    assert geo.positions[1] == pytest.approx((0.14, 0.0, -0.0))
    assert geo.positions[-1][2] == pytest.approx(-10.0)
    cd = np.array(geo.values["Cd"]).reshape(-1, 3)
    # person @1.4 m/s in a 0..2.5 range sits mid-ramp; stopped car is pure 'slow' blue
    car_stopped = cd[20]
    assert car_stopped[2] > car_stopped[0]
    assert geo.values["predicted"][8] == 1 and geo.values["predicted"][0] == 0
    assert geo.values["v"][:3] == [1.4, 0.0, -0.0]

    # 'current' mode: one point per track alive at t=0.05 s
    src = script.read_text().replace('MODE = "tracks"', 'MODE = "current"')
    geo2 = _FakeGeo()
    monkeypatch.setitem(sys.modules, "hou", _fake_hou(geo2))
    exec(compile(src, "houdini_current", "exec"), {"__name__": "__main__"})
    assert len(geo2.points) == 2 and not geo2.polys
