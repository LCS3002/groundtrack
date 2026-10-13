"""A Houdini POP kit of one site: the whole site's flow field and every clip's field as
modules, each with a ready particle simulation (see hou_pop_build.py).

    <out>/
      <site>_pop.hipnc     the scene (.hip / .hiplc with a commercial / Indie licence)
      modules.json         what the scene is built from
      fields/<module>_<group>.csv   x, z (metres, Houdini axes around the site's origin:
                           Easting -> +X, Northing -> -Z), vx, vz (m/s), speed, density,
                           confidence, smax (top of the group's speed scale), cell
      hou_pop_build.py     rebuilds the scene: hython hou_pop_build.py <out>
      README.txt

Every module of a site shares the site's origin (the combined plan's), so the clips sit on
the whole exactly; move them apart with each module's "place" node.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ("people", "cycles", "vehicles", "buses", "trains")
SKIP = ("_continuous", "_slices", "_master")


def _script_values(script: Path) -> dict:
    """ORIGIN, SPEED_RANGES and RAMP written into a groundtrack houdini_field.py."""
    txt = script.read_text(encoding="utf-8")
    out = {}
    for key in ("ORIGIN", "SPEED_RANGES", "RAMP"):
        m = re.search(rf"^{key} = (.+?)(?:\s+#.*)?$", txt, re.M)
        if m:
            out[key] = ast.literal_eval(m.group(1))
    return out


def _convert(src: Path, dst: Path, origin, smax) -> dict:
    """A groundtrack vector field CSV -> module CSV in the site's Houdini frame; smax: a
    number, or {group: top speed} looked up by the row's dominant_group."""
    df = pd.read_csv(src)
    df = df[np.isfinite(df["vx"]) & np.isfinite(df["vy"])]
    cell = float(np.median(df["cell_x"] / (df["i"] + 0.5)))
    if isinstance(smax, dict):
        top = max(smax.values())
        sm = df["dominant_group"].map(smax).fillna(top) if "dominant_group" in df else top
    else:
        sm = smax
    out = pd.DataFrame({
        "x": (df["cell_x"] - origin[0]).round(3), "z": (-(df["cell_y"] - origin[1])).round(3),
        "vx": df["vx"].round(4), "vz": (-df["vy"]).round(4), "speed": df["speed"].round(4),
        "density": df["density"], "confidence": df["confidence"].round(4), "smax": sm,
        "cell": round(cell, 4)})
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    w = out["density"].clip(lower=0).to_numpy() + 1e-12
    span = float(max(np.ptp(out["x"]), np.ptp(out["z"]))) + cell if len(out) else cell
    med = float(np.average(out["speed"], weights=w)) if len(out) else 1.0
    return {"cell": cell, "centre": [float(np.average(out["x"], weights=w)),
                                     float(np.average(out["z"], weights=w))],
            "span": span, "speed": med, "n": len(out)}


def _module(name, files: dict, default: str, origin, ranges, fields_dir: Path,
            visible: bool, rate: float) -> dict | None:
    """Convert a module's field files ({group: csv}) and set its simulation defaults."""
    conv, info = {}, None
    for g, src in files.items():
        dst = fields_dir / f"{name}_{g}.csv"
        smax = {k: v[1] for k, v in ranges.items() if k != "all"} if g == "all" \
            else ranges.get(g, ranges.get("all", (0, 2.5)))[1]
        st = _convert(src, dst, origin, smax)
        if st["n"] == 0:
            continue
        conv[g] = f"fields/{dst.name}"
        if g == default or info is None:
            info = st
    if not conv or info is None:
        return None
    default = default if default in conv else next(iter(conv))
    speed = max(info["speed"], 0.2)
    return {
        "name": name, "fields": conv, "group": default, "visible": visible,
        "cell": info["cell"], "centre": info["centre"],
        "rate": rate, "life": round(float(np.clip(1.2 * info["span"] / speed, 4, 30)), 1),
        "noise": round(0.3 * speed, 2), "min_speed": round(0.04 * speed, 3),
        "trail_frames": 24, "seeds": 20000, "min_confidence": 0.2,
    }


def prepare_kit(site: str, composite: Path, clips: list[Path], out: Path,
                combined: list[str] | None = None, log=print) -> Path:
    """Write the kit (fields, modules.json, builder, README) of one site into out."""
    hou_dir = Path(composite) / "houdini"
    vals = _script_values(hou_dir / "houdini_field.py")
    origin, ranges = vals["ORIGIN"], dict(vals.get("SPEED_RANGES") or {})
    fields_dir = Path(out) / "fields"
    if fields_dir.exists():
        shutil.rmtree(fields_dir)
    modules = []
    # the whole site: the combined plan's fields (master = all groups, dominant per cell)
    files = {}
    if (hou_dir / "vector_field_master.csv").exists():
        files["all"] = hou_dir / "vector_field_master.csv"
    for g in GROUPS:
        f = hou_dir / f"vector_field_{g}_continuous.csv"
        if f.exists():
            files[g] = f
    m = _module(f"{site}_whole", files, "all" if "all" in files else next(iter(files)),
                origin, ranges, fields_dir, True, 400.0)
    if m:
        modules.append(m)
    for clip in clips:
        cdir = Path(clip) / "houdini"
        cvals = _script_values(cdir / "houdini_field.py") if (cdir / "houdini_field.py").exists() \
            else {}
        cr = {**ranges, **(cvals.get("SPEED_RANGES") or {})}
        files = {g: cdir / f"vector_field_{g}.csv" for g in GROUPS
                 if (cdir / f"vector_field_{g}.csv").exists()}
        if not files and (cdir / "vector_field.csv").exists():   # one group: its field
            gs = [g for g in (cvals.get("SPEED_RANGES") or {}) if g != "all"] or ["people"]
            files = {gs[0]: cdir / "vector_field.csv"}
        if not files:
            continue
        m = _module(Path(clip).name, files, "people" if "people" in files else next(iter(files)),
                    origin, cr, fields_dir, False, 120.0)
        if m:
            modules.append(m)
    clip_mods = [m for m in modules if not m["name"].endswith("_whole")]
    if combined is None:     # by default the two clips with the most field cells
        sizes = {m["name"]: sum(1 for _ in open(Path(out) / next(iter(m["fields"].values())),
                                                encoding="utf-8")) for m in clip_mods}
        combined = sorted(sizes, key=sizes.get, reverse=True)[:2] or \
            [m["name"] for m in modules[:1]]
    kit = {
        "site": site, "origin": list(origin), "fps": 24, "frames": 480,
        "ramp": vals.get("RAMP") or [], "combined": combined, "modules": modules,
        "note": NOTE.format(site=site, n=len(clip_mods), e=origin[0], n_=origin[1],
                            combined=", ".join(combined)),
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "modules.json").write_text(json.dumps(kit, indent=1), encoding="utf-8")
    shutil.copy2(Path(__file__).with_name("hou_pop_build.py"), Path(out) / "hou_pop_build.py")
    (Path(out) / "README.txt").write_text(README.format(site=site, e=origin[0], n=origin[1]),
                                          encoding="utf-8")
    log(f"houdini kit {site}: {len(modules)} modules ({len(clip_mods)} clips) -> {out}")
    return Path(out)


def find_hython() -> Path | None:
    import os

    if os.environ.get("HFS"):
        p = Path(os.environ["HFS"]) / "bin" / "hython.exe"
        if p.exists():
            return p
    found = sorted(Path(r"C:\Program Files\Side Effects Software").glob("Houdini */bin/hython.exe"))
    return found[-1] if found else None


def build_scene(kit: Path, hython: Path | None = None, log=print) -> Path | None:
    """Run the builder in Houdini (hython): writes <site>_pop.hip* into the kit."""
    hython = hython or find_hython()
    if hython is None:
        log("Houdini not found: open Houdini and run hou_pop_build.py from the kit "
            "(File > Run Script), or set HFS")
        return None
    r = subprocess.run([str(hython), str(Path(kit) / "hou_pop_build.py"), str(kit)],
                       capture_output=True, text=True, timeout=1800)
    saved = [ln.split("saved ", 1)[1].strip() for ln in r.stdout.splitlines()
             if ln.startswith("saved ")]
    if r.returncode != 0 or not saved:
        raise RuntimeError(f"Houdini build failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    log(f"scene: {saved[0]}")
    return Path(saved[0])


NOTE = """groundtrack - {site}: particles following the measured movement

Each object is one module: {site}_whole = the whole site (all clips combined), and one per
clip ({n}). Only the whole is shown; switch others on with their display flag.
Metres, Easting -> +X, Northing -> -Z, origin E {e} N {n_} (all modules line up).

In each module:
  field   group menu (all / people / vehicles ...), min confidence, emitter seeds
  place   move / rotate / scale the module (the particles follow its field)
  sim     source: birth rate, life | follow_field: how strictly they follow
          turbulence (bypassed): switch on for noise | kill_outside: min speed
  trails  trail length (frames)       particles: dot size

combined = the modules listed in its object_merge ({combined}): their fields
blend where they overlap (density-weighted), one simulation through all. Move modules on top
of each other with their place nodes, then add them here to see how the flows meet.
Play from frame 1 (the simulation builds up over the first seconds)."""

README = """groundtrack Houdini POP kit - {site}

Open {site}_pop.hipnc (Apprentice; .hip / .hiplc with other licences) and press play.

Modules: {site}_whole (the whole site) and one object per clip, each with its own particle
simulation (POP Source at the places people / vehicles were -> POP Advect by Volumes through
the measured flow field -> optional POP Wind turbulence -> POP Kill outside the field),
coloured by speed on each group's own scale, with trails. The object "combined" merges
chosen modules: their fields blend where they overlap.

The sticky note in /obj explains the knobs. Coordinates: metres around E {e} N {n}
(British National Grid), Easting -> +X, Northing -> -Z, ground at y = 0.

fields/*.csv are plain CSVs (x, z, vx, vz, speed, density, confidence, smax, cell), read by
each module's "field" node relative to the scene ($HIP): keep the folder together.
Rebuild the scene: hython hou_pop_build.py <this folder>  (or run it from Houdini).
"""
