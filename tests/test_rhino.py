"""Rhino / Grasshopper export: the .3dm reads back, the Grasshopper script runs (Rhino stubbed)."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from groundtrack.config import default_config
from groundtrack.field import smooth_field
from groundtrack.rhino import write_3dm, write_grasshopper_script

DT = 0.2
E0, N0 = 530000.0, 180000.0


def _points():
    rows = []
    for k, (group, cls, v) in enumerate((("people", "person", 1.3), ("people", "person", 1.1),
                                         ("vehicles", "car", 9.0))):
        t = np.arange(40) * DT
        rows.append(pd.DataFrame({"track_id": k + 1, "class": cls, "group": group,
                                  "frame": np.arange(40), "time_s": t, "x": E0 + v * t,
                                  "y": N0 + 3 * k, "z": 12.0 + 0.1 * k, "vx": v, "vy": 0.0,
                                  "speed": v, "heading": 90.0, "predicted": False}))
    return pd.concat(rows, ignore_index=True)


def _rhino_stubs():
    class P:
        def __init__(self, *a):
            self.xyz = a

    class Polyline:
        def __init__(self):
            self.pts = []

        def Add(self, x, y, z):  # noqa: N802  (RhinoCommon name)
            self.pts.append(P(x, y, z))

    class Curve:
        def __init__(self, polyline):
            assert isinstance(polyline, Polyline)      # RhinoCommon takes no Python list
            self.pts = list(polyline.pts)

    class Color:
        @staticmethod
        def FromArgb(r, g, b):
            return (r, g, b)

    rhino = types.ModuleType("Rhino")
    geom = types.ModuleType("Rhino.Geometry")
    geom.Point3d, geom.Vector3d, geom.PolylineCurve, geom.Polyline = P, P, Curve, Polyline
    rhino.Geometry = geom
    sc = types.ModuleType("scriptcontext")
    sc.sticky = {}
    system = types.ModuleType("System")
    drawing = types.ModuleType("System.Drawing")
    drawing.Color = Color
    system.Drawing = drawing
    return {"Rhino": rhino, "Rhino.Geometry": geom, "scriptcontext": sc, "System": system,
            "System.Drawing": drawing}


def test_grasshopper_script_runs(tmp_path, monkeypatch):
    pts = _points()
    pts.to_csv(tmp_path / "points.csv", index=False)
    smooth_field(pts[pts["group"] == "people"], 1.0, 2.0, DT).to_csv(
        tmp_path / "vector_field_people.csv", index=False)
    script = write_grasshopper_script(tmp_path / "gh.py", tmp_path / "points.csv",
                                      {"people": tmp_path / "vector_field_people.csv"},
                                      (E0, N0), 10.0, default_config(), 1.0, 7.8)
    for k, v in _rhino_stubs().items():
        monkeypatch.setitem(sys.modules, k, v)
    code = compile(script.read_text(encoding="utf-8"), str(script), "exec")

    g = {"__name__": "__main__"}
    exec(code, g)                                       # nothing connected: everything
    assert len(g["tracks"]) == 3 and len(g["field_points"]) > 0
    assert g["track_info"][2].startswith("3 vehicles car 9.00")
    first = g["tracks"][0].pts[0].xyz
    assert first == pytest.approx((0.0, 0.0, 2.0))     # metres from ORIGIN, z above Z0
    assert g["field_points"][0].xyz[2] == pytest.approx(2.0, abs=0.2)   # field on the ground

    g = {"__name__": "__main__", "t": 4.0, "tail": 1.0, "group": "people"}
    exec(code, g)                                       # a moment: one point per person
    assert len(g["points"]) == 2 and len(g["tracks"]) == 2
    assert g["vectors"][0].xyz == pytest.approx((1.3, 0.0, 0.0))
    assert g["points"][0].xyz[0] == pytest.approx(1.3 * 4.0, abs=0.3)
    assert g["colours"][0] != g["colours"][1]           # 1.3 vs 1.1 m/s


def test_3dm_layers_and_attributes(tmp_path):
    r3 = pytest.importorskip("rhino3dm")
    pts = _points()
    fld = smooth_field(pts[pts["group"] == "people"], 1.0, 2.0, DT)
    out = write_3dm(tmp_path / "x.3dm", pts, {"people": fld}, (E0, N0), 10.0, default_config(),
                    1.0, cameras=[("cam", E0, N0 - 10, 5.0)])
    m = r3.File3dm.Read(str(out))
    names = {L.Name for L in m.Layers}
    assert {"groundtrack", "tracks_people", "tracks_vehicles", "field_people", "cameras"} <= names
    tracks = [o for o in m.Objects if o.Attributes.Name.startswith("track ")]
    assert len(tracks) == 3
    car = next(o for o in tracks if o.Attributes.GetUserString("group") == "vehicles")
    assert float(car.Attributes.GetUserString("mean_speed")) == pytest.approx(9.0)
    bb = car.Geometry.GetBoundingBox()
    assert bb.Min.Y == pytest.approx(6.0) and bb.Min.Z == pytest.approx(2.2)


def test_grasshopper_unknown_field_falls_back_to_the_default(tmp_path, monkeypatch):
    pts = _points()
    pts.to_csv(tmp_path / "points.csv", index=False)
    smooth_field(pts[pts["group"] == "people"], 1.0, 2.0, DT).to_csv(
        tmp_path / "vector_field_people.csv", index=False)
    script = write_grasshopper_script(tmp_path / "gh.py", tmp_path / "points.csv",
                                      {"people": tmp_path / "vector_field_people.csv"},
                                      (E0, N0), 10.0, default_config(), 1.0, 7.8)
    for k, v in _rhino_stubs().items():
        monkeypatch.setitem(sys.modules, k, v)
    g = {"__name__": "__main__", "field": "people_continuous"}       # only composites have it
    exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), g)
    assert len(g["field_points"]) > 0
