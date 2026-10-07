"""Ground-truth check: film yourself walking a known straight line, measure the error.

Protocol (see README): stand still on point A for ~3 s, walk at a steady pace in a straight
line to point B, stand still on B for ~3 s. A and B are surveyed points (e.g. read from the
GeoTIFF in QGIS, or a tape-measured length).

What we can then measure independently of the homography:
  * the known length |AB| (tape / map) and the walking time from video frames, so the
    reference speed |AB| / T is a ground truth the tool's measured speed is compared with;
  * the known positions of A and B, compared with where the tool puts you while standing;
  * the lateral deviation of the projected path from the straight line.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MOVING_SPEED = 0.4  # m/s


def _first_crossing(t: np.ndarray, u: np.ndarray, level: float) -> float | None:
    """Interpolated time at which u first reaches `level`."""
    k = np.flatnonzero(u >= level)
    if len(k) == 0:
        return None
    k = k[0]
    if k == 0:
        return float(t[0])
    f = (level - u[k - 1]) / max(u[k] - u[k - 1], 1e-12)
    return float(t[k - 1] + f * (t[k] - t[k - 1]))


def _longest_run(mask: np.ndarray) -> tuple[int, int] | None:
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    if len(edges) == 0:
        return None
    runs = list(zip(edges[::2], edges[1::2]))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return int(a), int(b)  # [a, b)


def analyse_walk(track: pd.DataFrame, a=None, b=None, known_length: float | None = None,
                 stopwatch_s: float | None = None) -> dict:
    """track: points of a single person track (x, y, speed, time_s, predicted)."""
    tr = track.sort_values("time_s")
    xy = tr[["x", "y"]].to_numpy(float)
    t = tr["time_s"].to_numpy(float)
    spd = tr["speed"].to_numpy(float)
    moving = spd > MOVING_SPEED
    run = _longest_run(moving)
    if run is None:
        raise ValueError("The track never moves faster than 0.4 m/s; is this the right track?")
    i0, i1 = run
    walk_t = float(t[i1 - 1] - t[i0])
    pre, post = slice(0, i0), slice(i1, len(t))
    stand_a = xy[pre].mean(axis=0) if i0 >= 3 else xy[i0]
    stand_b = xy[post].mean(axis=0) if len(t) - i1 >= 3 else xy[i1 - 1]
    walk = xy[i0:i1]

    rep: dict = {
        "track_id": int(tr["track_id"].iloc[0]),
        "n_samples": int(len(tr)),
        "predicted_share": round(float(tr["predicted"].astype(bool).mean()), 3),
        "walking_time_s": round(walk_t, 3),
        "standing_at_start_s": round(float(t[i0] - t[0]), 2),
        "standing_at_end_s": round(float(t[-1] - t[i1 - 1]), 2),
        "measured_start": np.round(stand_a, 3).tolist(),
        "measured_end": np.round(stand_b, 3).tolist(),
    }
    measured_len = float(np.linalg.norm(stand_b - stand_a))
    rep["measured_length_m"] = round(measured_len, 3)
    rep["measured_mean_walking_speed"] = round(float(spd[i0:i1].mean()), 3)

    # reference line: surveyed A-B if given, otherwise the best-fit line through the walk
    if a is not None and b is not None:
        a, b = np.asarray(a, float), np.asarray(b, float)
        ref = "surveyed A-B"
        known_length = known_length or float(np.linalg.norm(b - a))
        rep["start_error_m"] = round(float(np.linalg.norm(stand_a - a)), 3)
        rep["end_error_m"] = round(float(np.linalg.norm(stand_b - b)), 3)
    else:
        c = walk.mean(axis=0)
        _, _, vt = np.linalg.svd(walk - c)
        a, b = c - vt[0] * 1e3, c + vt[0] * 1e3
        ref = "best-fit line through the walk (no A/B given: absolute position not checked)"
    d = (b - a) / np.linalg.norm(b - a)
    nrm = np.array([-d[1], d[0]])
    lateral = (walk - a) @ nrm
    rep["reference_line"] = ref
    rep["lateral_deviation_m"] = {
        "mean_signed": round(float(lateral.mean()), 3),
        "rms": round(float(np.sqrt(np.mean(lateral ** 2))), 3),
        "max_abs": round(float(np.abs(lateral).max()), 3),
        "rms_about_own_mean": round(float(lateral.std()), 3),
    }
    if known_length:
        rep["known_length_m"] = round(float(known_length), 3)
        rep["length_error_m"] = round(measured_len - known_length, 3)
        rep["length_error_pct"] = round(100 * (measured_len - known_length) / known_length, 2)
        # Time the middle 80 % of the walk: when the walker passes 10 % and 90 % of the way
        # from A to B. Fractions of the walk do not depend on the calibration's scale, and the
        # middle avoids the smoothed start/stop, so 0.8 * known length / that time is a true
        # reference speed. The tool's mean speed over the same window is compared with it.
        u = (xy - stand_a) @ (stand_b - stand_a) / max(measured_len ** 2, 1e-9)
        t10, t90 = _first_crossing(t, u, 0.1), _first_crossing(t, u, 0.9)
        if stopwatch_s:
            ref_speed, win = known_length / stopwatch_s, (t[i0], t[i1 - 1])
            rep["reference_speed_source"] = "stopwatch"
        elif t10 is not None and t90 is not None and t90 > t10:
            ref_speed, win = 0.8 * known_length / (t90 - t10), (t10, t90)
            rep["reference_speed_source"] = "known length / video time between 10 % and 90 %"
        else:
            ref_speed, win = known_length / walk_t, (t[i0], t[i1 - 1])
            rep["reference_speed_source"] = "known length / walking time (video frames)"
        in_win = (t >= win[0]) & (t <= win[1])
        rep["reference_speed"] = round(ref_speed, 3)
        rep["measured_speed"] = round(float(spd[in_win].mean()), 3)
        rep["speed_error_pct"] = round(100 * (rep["measured_speed"] - ref_speed) / ref_speed, 2)
    rep["verdict"] = _verdict(rep)
    return rep


def _verdict(rep: dict) -> list[str]:
    out = []
    lat = rep["lateral_deviation_m"]["rms"]
    out.append(f"{'OK  ' if lat < 0.3 else 'WARN'} lateral RMS {lat:.2f} m (target < 0.30 m)")
    if "length_error_pct" in rep:
        e = rep["length_error_pct"]
        out.append(f"{'OK  ' if abs(e) < 3 else 'WARN'} length error {e:+.1f}% (target within 3%)")
        s = rep["speed_error_pct"]
        out.append(f"{'OK  ' if abs(s) < 5 else 'WARN'} speed error {s:+.1f}% (target within 5%)")
    for k in ("start_error_m", "end_error_m"):
        if k in rep:
            out.append(f"{'OK  ' if rep[k] < 0.5 else 'WARN'} {k.replace('_', ' ')} "
                       f"{rep[k]:.2f} m (target < 0.5 m)")
    return out


def plot_walk(track: pd.DataFrame, rep: dict, path: Path, a=None, b=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = track.sort_values("time_s")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.plot(tr["x"], tr["y"], "-", color="#2a6fdb", lw=1.5, label="projected path")
    pr = tr[tr["predicted"].astype(bool)]
    if len(pr):
        ax1.plot(pr["x"], pr["y"], ".", color="#999", ms=3, label="predicted")
    if a is not None:
        ax1.plot([a[0], b[0]], [a[1], b[1]], "--", color="#d62728", lw=1.2, label="surveyed A-B")
        ax1.plot(*a, "o", color="#d62728")
        ax1.plot(*b, "s", color="#d62728")
    ax1.plot(*rep["measured_start"], "o", mfc="none", mec="k", ms=10, label="measured standing")
    ax1.plot(*rep["measured_end"], "s", mfc="none", mec="k", ms=10)
    ax1.set_aspect("equal")
    ax1.ticklabel_format(useOffset=False, style="plain")
    ax1.legend(fontsize=8)
    ax1.set_title("path (EPSG:27700)")
    ax2.plot(tr["time_s"], tr["speed"], color="#2a6fdb")
    if "reference_speed" in rep:
        ax2.axhline(rep["reference_speed"], color="#d62728", ls="--",
                    label=f"reference {rep['reference_speed']:.2f} m/s")
        ax2.legend(fontsize=8)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("speed (m/s)")
    ax2.set_title("speed")
    fig.suptitle("  |  ".join(rep["verdict"]), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _dist_to_segment(xy: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    s = np.clip(((xy - a) @ d) / (d @ d), 0, 1)
    return np.linalg.norm(xy - (a + s[:, None] * d), axis=1)


def pick_track(points: pd.DataFrame, summary: pd.DataFrame, track_id: int | None = None,
               a=None, b=None):
    """The walker's track: given id; else the person track that best follows A-B (if given)
    while covering most of it; else the longest person track."""
    if track_id is not None:
        tr = points[points["track_id"] == track_id]
        if tr.empty:
            raise ValueError(f"track {track_id} not found; ids: {sorted(points['track_id'].unique())}")
        return tr
    people = summary[summary["class"] == "person"]
    if people.empty:
        raise ValueError("No person tracks found in the ground-truth video.")
    if a is not None and b is not None:
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = np.linalg.norm(b - a)
        scores = {}
        for tid in people["track_id"]:
            xy = points.loc[points["track_id"] == tid, ["x", "y"]].to_numpy()
            along = (xy - a) @ (b - a) / L
            coverage = (np.ptp(along) / L) if len(xy) > 1 else 0.0
            if coverage > 0.5:
                scores[tid] = np.median(_dist_to_segment(xy, a, b))
        if scores:
            best = min(scores, key=scores.get)
            return points[points["track_id"] == best]
    best = people.sort_values("path_length_m", ascending=False).iloc[0]
    return points[points["track_id"] == best["track_id"]]


def write_report(rep: dict, run_dir: Path, log=print) -> None:
    (run_dir / "groundtruth_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    log("\nGROUND-TRUTH CHECK")
    for k in ("track_id", "walking_time_s", "known_length_m", "measured_length_m",
              "length_error_pct", "reference_speed", "measured_speed",
              "speed_error_pct", "start_error_m", "end_error_m"):
        if k in rep:
            log(f"  {k:30s} {rep[k]}")
    log(f"  {'lateral_deviation_m':30s} {rep['lateral_deviation_m']}")
    for v in rep["verdict"]:
        log("  " + v)
