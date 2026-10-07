import sys
import types

import numpy as np
import pandas as pd
import pytest

from groundtrack import exports
from groundtrack.field import smooth_field, time_sliced_fields

DT = 1 / 30


def _lane(y, vx, n=300, x0=530000.0, track=1, t0=0.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = t0 + np.arange(n) * DT
    return pd.DataFrame({"track_id": track, "class": "person", "frame": np.arange(n),
                         "time_s": t, "x": x0 + vx * (t - t0), "y": y,
                         "vx": vx + rng.normal(0, noise, n), "vy": 0.0,
                         "speed": abs(vx), "predicted": False, "group": "people"})


def test_uniform_flow_is_preserved():
    pts = pd.concat([_lane(180000.0 + k, 1.3, track=k, noise=0.2, seed=k) for k in range(6)])
    f = smooth_field(pts, cell=0.5, smooth_m=1.0, dt=DT)
    core = f[(f["confidence"] > 0.9)]
    assert len(core) > 50
    assert core["vx"].mean() == pytest.approx(1.3, abs=0.02)   # averages, not sums
    assert core["vy"].abs().max() < 1e-9
    assert core["heading"].round().eq(90).all()


def test_opposite_lanes_stay_separate():
    """Two carriageways 10 m apart: a 2 m kernel must not cancel them out."""
    pts = pd.concat([_lane(180000.0, 25.0, n=60, track=1), _lane(180010.0, -25.0, n=60, track=2)])
    f = smooth_field(pts, cell=1.0, smooth_m=2.0, dt=DT)
    north = f[(f["cell_y"] - 180010).abs() < 1]
    south = f[(f["cell_y"] - 180000).abs() < 1]
    assert north["vx"].median() == pytest.approx(-25, abs=0.5)
    assert south["vx"].median() == pytest.approx(25, abs=0.5)


def test_density_units_and_fade():
    # one person standing still in one 1 m cell for 10 s -> 10 object-seconds in ~ that area
    n = 300
    pts = pd.DataFrame({"track_id": 1, "class": "person", "time_s": np.arange(n) * DT,
                        "x": 530000.5, "y": 180000.5, "vx": 0.0, "vy": 0.0, "speed": 0.0,
                        "predicted": False, "group": "people"})
    f = smooth_field(pts, cell=1.0, smooth_m=1.0, dt=DT)
    assert (f["density"] * 1.0).sum() == pytest.approx(10.0, rel=0.02)  # integrates to 10 s
    # confidence falls off away from the data
    centre = f.loc[f["density"].idxmax()]
    far = f.loc[((f["cell_x"] - centre["cell_x"]) ** 2 + (f["cell_y"] - centre["cell_y"]) ** 2)
                .idxmax()]
    assert centre["confidence"] > 0.99 and far["confidence"] < centre["confidence"]


def test_predicted_samples_excluded_by_default():
    pts = _lane(180000.0, 1.0, n=60)
    pts.loc[:, "predicted"] = True
    assert smooth_field(pts, 1.0, 1.0, DT).empty
    assert len(smooth_field(pts, 1.0, 1.0, DT, include_predicted=True)) > 0


def test_time_slices_share_grid():
    pts = pd.concat([_lane(180000.0, 1.0, n=300, track=1, t0=0.0),
                     _lane(180005.0, -1.0, n=300, track=2, t0=10.0)])
    s = time_sliced_fields(pts, 10.0, 1.0, 1.0, DT)
    assert sorted(s["t_start"].unique()) == [0.0, 10.0]
    first = s[s["t_start"] == 0.0]
    second = s[s["t_start"] == 10.0]
    assert first.loc[first["weight"].idxmax(), "vx"] > 0.9
    assert second.loc[second["weight"].idxmax(), "vx"] < -0.9


# --------------------------------------------------------------------------- houdini
class _Vol:
    def __init__(self, res, bbox):
        self.res, self.bbox, self.attrs, self.voxels = res, bbox, {}, None

    def setAttribValue(self, k, v):
        self.attrs[k] = v

    def setAllVoxels(self, vals):
        assert len(vals) == self.res[0] * self.res[1] * self.res[2]
        self.voxels = list(vals)


class _Pt:
    def __init__(self):
        self.pos, self.attrs = None, {}

    def setPosition(self, p):
        self.pos = p

    def setAttribValue(self, k, v):
        self.attrs[k] = v


class _Grp:
    def __init__(self):
        self.items = []

    def add(self, p):
        self.items.append(p)


class _Geo:
    def __init__(self):
        self.vols, self.pts, self.groups, self.values, self.attribs = [], [], {}, {}, {}

    def addAttrib(self, kind, name, default):
        self.attribs[(kind, name)] = default

    def createVolume(self, x, y, z, bbox):
        v = _Vol((x, y, z), bbox)
        self.vols.append(v)
        return v

    def createPoints(self, positions):
        self.positions = list(positions)
        return list(range(len(positions)))

    def createPoint(self):
        p = _Pt()
        self.pts.append(p)
        return p

    def createPointGroup(self, name):
        self.groups[name] = _Grp()
        return self.groups[name]

    def setPointFloatAttribValues(self, name, vals):
        self.values[name] = list(vals)


def _hou(geo, t=0.0):
    hou = types.ModuleType("hou")
    hou.attribType = types.SimpleNamespace(Point="pt", Prim="prim", Global="detail")
    hou.Vector3 = lambda *a: tuple(a)
    hou.BoundingBox = lambda *a: tuple(a)
    hou.pwd = lambda: types.SimpleNamespace(geometry=lambda: geo)
    hou.time = lambda: t
    return hou


def _run(script_text, geo, monkeypatch, t=0.0, **overrides):
    for k, v in overrides.items():
        line = {"MODE": 'MODE = "volume"', "USE_TIME_SLICES": "USE_TIME_SLICES = False"}[k]
        script_text = script_text.replace(line, f"{k} = {v!r}")
    monkeypatch.setitem(sys.modules, "hou", _hou(geo, t))
    exec(compile(script_text, "houdini_field", "exec"), {"__name__": "__main__"})


def test_houdini_field_script(tmp_path, cfg, monkeypatch):
    # northbound flow (vy = +1.2) in the west, eastbound (vx = +1.2) further east
    rows = []
    for k in range(200):
        rows.append((1, "person", k, k * DT, 530000.25, 180000.0 + 1.2 * k * DT, 0.0, 1.2))
        rows.append((2, "person", k, k * DT, 530010.0 + 1.2 * k * DT, 180002.25, 1.2, 0.0))
    pts = pd.DataFrame(rows, columns=["track_id", "class", "frame", "time_s", "x", "y", "vx", "vy"])
    pts["speed"] = np.hypot(pts["vx"], pts["vy"])
    pts["heading"] = 0.0
    pts["predicted"] = False
    pts["source"] = "detected"
    pts["group"] = "people"
    pcsv = tmp_path / "points.csv"
    pts.to_csv(pcsv, index=False)
    fcsv = tmp_path / "vector_field.csv"
    field = smooth_field(pts, 0.5, 0.5, DT)
    field.to_csv(fcsv, index=False)
    scsv = tmp_path / "vector_field_slices.csv"
    time_sliced_fields(pts, 3.0, 0.5, 0.5, DT).to_csv(scsv, index=False)
    script = tmp_path / "houdini_field.py"
    exports.write_houdini_field_script(script, {"all": fcsv}, {"all": scsv}, pcsv,
                                       (530000.0, 180000.0), 0.5, cfg)
    src = script.read_text()

    # volume mode: 5 volumes, vector stored in Houdini axes, rows flipped to north = -Z
    geo = _Geo()
    _run(src, geo, monkeypatch)
    names = [v.attrs["name"] for v in geo.vols]
    assert names == ["vel.x", "vel.y", "vel.z", "density", "confidence"]
    vx, vy, vz = (geo.vols[k] for k in range(3))
    nx, _, nz = vx.res
    i0 = int(field["i"].min())
    j1 = int(field["j"].max())
    r = field.loc[field["confidence"].idxmax()]
    idx = (int(r["i"]) - i0) + (j1 - int(r["j"])) * nx
    assert vx.voxels[idx] == pytest.approx(r["vx"] * r["confidence"], abs=1e-6)
    assert vz.voxels[idx] == pytest.approx(-r["vy"] * r["confidence"], abs=1e-6)
    assert all(v == 0 for v in vy.voxels)
    bb = vx.bbox  # (xmin, ymin, zmin, xmax, ymax, zmax)
    assert bb[3] - bb[0] == pytest.approx(nx * 0.5) and bb[5] - bb[2] == pytest.approx(nz * 0.5)
    # the northbound lane sits at local x ~ 0.25 and runs to N+4.8 m -> z ~ -4.8
    assert bb[2] < -4.0 and bb[0] < 0.25

    # points mode
    geo = _Geo()
    _run(src, geo, monkeypatch, MODE="points")
    assert len(geo.positions) == len(field)
    assert len(geo.values["v"]) == 3 * len(field) and len(geo.values["Cd"]) == 3 * len(field)

    # sources mode: one source + one sink per track, at the real start / end
    geo = _Geo()
    _run(src, geo, monkeypatch, MODE="sources")
    assert len(geo.groups["sources"].items) == 2 and len(geo.groups["sinks"].items) == 2
    s1 = [p for p in geo.groups["sources"].items if p.attrs["track_id"] == 1][0]
    assert s1.pos == pytest.approx((0.25, 0.0, -0.0))
    assert s1.attrs["v"] == pytest.approx((0.0, 0.0, -1.2))

    # time slices: at t = 4 s only the second window's field is used
    geo = _Geo()
    _run(src, geo, monkeypatch, t=4.0, USE_TIME_SLICES=True)
    assert len(geo.vols) == 5
