"""Optional debug.mp4: boxes, IDs and speed-coloured trails over the original video.

Off by default; people are blurred. Trails are drawn in the calibration frame's pixel space
and warped onto every video frame, so with `stabilize` they stay attached to the ground while
the camera moves. Colour = speed on the same blue -> red ramp as topdown.png:
  * calibrated runs (points.csv present): the smoothed ground path and its speed in m/s;
  * before calibration: smoothed image path; people's speed ~m/s from their box height,
    other classes relative on-screen speed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .colors import ramp_rgb
from .config import Config

GROUP_BGR = {"people": (255, 200, 0), "cycles": (0, 220, 120), "vehicles": (60, 80, 255)}


def _dashed_rect(img, p1, p2, color, thickness=1, dash=6):
    x1, y1 = p1
    x2, y2 = p2
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)),
                 ((x1, y2), (x1, y1))):
        length = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        for s in range(0, length, dash * 2):
            e = min(s + dash, length)
            pa = (int(a[0] + (b[0] - a[0]) * s / max(length, 1)),
                  int(a[1] + (b[1] - a[1]) * s / max(length, 1)))
            pb = (int(a[0] + (b[0] - a[0]) * e / max(length, 1)),
                  int(a[1] + (b[1] - a[1]) * e / max(length, 1)))
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)


def blur_box(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(x2), w), min(int(y2), h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    k = max(15, (max(x2 - x1, y2 - y1) // 3) | 1)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


PERSON_HEIGHT_M = 1.7


def trail_samples(raw: pd.DataFrame, cfg: Config, run_dir: Path | None, fps: float,
                  uv: np.ndarray):
    """Smoothed trail position (reference-frame px) + speed colour range for every raw row.

    Returns (pos, speed, lo, hi, mode):
      "calibrated": smoothed ground path from points.csv projected back into the image, and
                    its speed in m/s (rows removed by cleaning keep their raw foot point and
                    get NaN speed -> drawn grey);
      "approx":     not calibrated yet; people's on-screen speed is converted to ~m/s using
                    their own box height (~1.7 m tall), so near and far people compare.
                    Motion towards / away from the camera reads slower;
      "relative":   other classes before calibration: px/s scaled to the 95th percentile.
    """
    from scipy.signal import savgol_filter

    n = len(raw)
    pos = uv.astype(float).copy()
    rd = Path(run_dir) if run_dir else None
    if rd is not None and (rd / "points.csv").exists() and (rd / "track_summary.csv").exists():
        from .homography import Homography

        hp = rd / "homography_used.json"
        h = Homography.load(hp if hp.exists() else cfg.path("homography"))
        pts = pd.read_csv(rd / "points.csv")
        summ = pd.read_csv(rd / "track_summary.csv")
        px = h.to_pixel(pts[["x", "y"]].to_numpy())
        pts["pu"], pts["pv"] = px[:, 0], px[:, 1]
        lookup = {}
        for row in summ.itertuples():
            tp = pts[pts["track_id"] == row.track_id]
            vals = list(zip(tp["frame"].tolist(), tp["speed"].tolist(), tp["pu"].tolist(),
                            tp["pv"].tolist()))
            for orig in str(row.tracker_ids).split("+"):
                for f, v, pu, pv in vals:
                    lookup[(int(orig), int(f))] = (v, pu, pv)
        speed = np.full(n, np.nan)
        for k, (t, f) in enumerate(zip(raw["track_id"].tolist(), raw["frame"].tolist())):
            hit = lookup.get((int(t), int(f)))
            if hit is not None:
                speed[k], pos[k, 0], pos[k, 1] = hit
        groups = [cfg.group_of(c) for c in raw["class"]]
        lo = np.array([g.speed_range[0] if g else 0.0 for g in groups])
        hi = np.array([g.speed_range[1] if g else 1.0 for g in groups])
        return pos, speed, lo, hi, "calibrated"

    speed = np.full(n, np.nan)
    tid_all = raw["track_id"].to_numpy()
    fr_all = raw["frame"].to_numpy()
    box_h = (raw["y2"] - raw["y1"]).to_numpy(float)
    is_person = (raw["class"] == "person").to_numpy()
    order = np.lexsort((fr_all, tid_all))
    for t in np.unique(tid_all):
        m = order[tid_all[order] == t]
        if len(m) < 5 or not np.isfinite(uv[m]).all():
            continue
        w = min(int(round(fps * 0.5)) | 1, len(m) if len(m) % 2 else len(m) - 1)
        if w < 3:
            continue
        dt = max(float(np.median(np.diff(fr_all[m]))), 1.0) / fps
        for i in (0, 1):
            pos[m, i] = savgol_filter(uv[m, i], w, 2)
        vx = savgol_filter(uv[m, 0], w, 2, deriv=1, delta=dt)
        vy = savgol_filter(uv[m, 1], w, 2, deriv=1, delta=dt)
        spd = np.hypot(vx, vy)
        if is_person[m].all():
            # body heights per second -> ~m/s (median box height: robust to partial boxes)
            spd = spd / max(float(np.median(box_h[m])), 1.0) * PERSON_HEIGHT_M
        speed[m] = spd
    if is_person.all():
        g = cfg.group_of("person")
        rng = g.speed_range if g else (0.0, 2.5)
        return pos, speed, np.full(n, rng[0]), np.full(n, rng[1]), "approx"
    ok = np.isfinite(speed) & ~raw["predicted"].to_numpy(bool)
    top = float(np.nanpercentile(speed[ok], 95)) if ok.any() else 1.0
    return pos, speed, np.zeros(n), np.full(n, max(top, 1e-6)), "relative"


def _bgr(speed, lo, hi):
    if not np.isfinite(speed):
        return (150, 150, 150)  # sample dropped by cleaning (short / parked / outside ROI)
    r, g, b = ramp_rgb(np.array([speed]), lo, hi)[0]
    return (int(b * 255), int(g * 255), int(r * 255))


def _legend(img, mode: str, cfg: Config, th: int):
    h, w = img.shape[:2]
    bw, bh = int(w * 0.32), max(8, 6 * th)
    x0, y0 = 12 * th, h - 24 * th
    cv2.rectangle(img, (x0 - 6 * th, y0 - 16 * th), (x0 + bw + 6 * th, y0 + bh + 12 * th),
                  (30, 30, 30), -1)
    bar = (ramp_rgb(np.linspace(0, 1, bw), 0, 1)[:, ::-1] * 255).astype(np.uint8)
    img[y0:y0 + bh, x0:x0 + bw] = bar[None, :, :]
    if mode == "relative":
        title, left, right = "speed (relative; calibrate for m/s)", "slow", "fast"
    else:
        g = cfg.group_of("person") if mode == "approx" else next(iter(cfg.groups.values()))
        title = "speed, m/s" + (" (approx. from body height)" if mode == "approx" else "")
        left, right = f"{g.speed_range[0]:g}", f"{g.speed_range[1]:g}"
    font, fs, tt = cv2.FONT_HERSHEY_SIMPLEX, 0.4 * th, max(1, th // 2)
    white = (255, 255, 255)
    cv2.putText(img, title, (x0, y0 - 5 * th), font, fs, white, tt, cv2.LINE_AA)
    cv2.putText(img, left, (x0, y0 + bh + 8 * th), font, fs, white, tt, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(right, font, fs, tt)
    cv2.putText(img, right, (x0 + bw - tw, y0 + bh + 8 * th), font, fs, white, tt, cv2.LINE_AA)


def _draw(canvas, alpha, a, b, col, t):
    pa = (int(round(a[0])), int(round(a[1])))
    pb = (int(round(b[0])), int(round(b[1])))
    cv2.line(canvas, pa, pb, col, t, cv2.LINE_AA)
    cv2.line(alpha, pa, pb, 255, t, cv2.LINE_AA)


def _good_tracks(raw: pd.DataFrame, speed: np.ndarray, mode: str, fps: float,
                 min_track_s: float) -> set:
    """Tracks worth drawing in the clean style: survived cleaning (calibrated) or long enough."""
    if mode == "calibrated":
        return set(raw.loc[np.isfinite(speed), "track_id"].unique())
    span = raw.groupby("track_id")["frame"].agg(lambda f: (f.max() - f.min()) / fps)
    return set(span[span >= min_track_s].index)


def _box_in_ref(r, H):
    """Axis-aligned box of a frame-pixel bbox mapped into the reference frame."""
    c = np.array([[r.x1, r.y1], [r.x2, r.y1], [r.x2, r.y2], [r.x1, r.y2]], np.float64)
    if H is not None:
        c = cv2.perspectiveTransform(c.reshape(-1, 1, 2), H).reshape(-1, 2)
    return (int(c[:, 0].min()), int(c[:, 1].min())), (int(c[:, 0].max()), int(c[:, 1].max()))


def render_debug_video(video: Path, raw: pd.DataFrame, cfg: Config, out_path: Path,
                       fps: float, roi=None, log=print, max_width: int = 1920,
                       run_dir: Path | None = None) -> None:
    """Render the overlay video.

    style "clean" (default): only the tracks that survive cleaning, as speed-coloured trails
    with a dot at each current position, over a slightly dimmed video.
    style "boxes": everything the tracker saw, with boxes, IDs and rejected tracks in grey.
    With a camera-motion registration the output itself is aligned to the calibration
    frame (`stabilize_output`), so the picture and the trails stay still.
    """
    from tqdm import tqdm

    from .registration import apply_registration, load_registration
    from .trajectories import foot_points, recover_feet

    dv = cfg["debug_video"]
    blur = bool(dv.get("blur_people", True))
    trail_s = dv.get("trail_s")              # None = keep the whole path
    style = dv.get("style", "clean")
    raw = raw.copy()
    raw["predicted"] = raw["predicted"].astype(str).str.lower().isin(["true", "1"])
    if raw.empty:
        log("debug video: no tracks to draw")
        return
    raw = raw.sort_values(["frame", "track_id"]).reset_index(drop=True)

    reg = None
    if run_dir is not None and (Path(run_dir) / "registration.npz").exists():
        reg, _, _ = load_registration(Path(run_dir) / "registration.npz")
    stable = reg is not None and bool(dv.get("stabilize_output", True))
    clean = cfg["cleaning"]
    if clean.get("recover_occluded_feet", True):
        foot, _ = recover_feet(raw, int(float(clean.get("feet_window_s", 2.0)) * fps),
                               float(clean.get("squish_ratio", 0.8)))
    else:
        foot = foot_points(raw)
    ref_uv = apply_registration(raw["frame"].to_numpy(), foot, reg) if reg else foot
    pos, speed, lo, hi, mode = trail_samples(raw, cfg, run_dir, fps, ref_uv)
    raw["ref_u"], raw["ref_v"], raw["spd"], raw["spd_lo"], raw["spd_hi"] = \
        pos[:, 0], pos[:, 1], speed, lo, hi
    if style == "clean":
        keep = _good_tracks(raw, speed, mode, fps, float(clean.get("min_track_s", 1.0)))
        raw = raw[raw["track_id"].isin(keep)]
        if mode == "calibrated":
            raw = raw[np.isfinite(raw["spd"])]
    draw_rows = {f: g for f, g in raw.groupby("frame")}

    cap = cv2.VideoCapture(str(video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = min(draw_rows) if draw_rows else 0
    f1 = max(draw_rows) if draw_rows else n_frames - 1
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / max(W, H))
    size = (int(W * scale), int(H * scale))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    th = max(1, int(round(2 * max(W, H) / 1920)))
    # every blur needs the unfiltered detections, even in the clean style
    blur_rows = {f: g for f, g in raw_all_people(run_dir, raw).groupby("frame")} \
        if blur else {}

    canvas = np.zeros((H, W, 3), np.uint8)   # trail layer, calibration-frame pixels
    alpha = np.zeros((H, W), np.uint8)
    last: dict[int, tuple] = {}               # track -> (frame, point)
    segments: list[tuple] = []                # (frame, p0, p1, colour, thickness), windowed mode
    max_gap = max(1, int(fps * 0.5))
    dim = float(dv.get("dim", 0.75 if style == "clean" else 1.0))

    for f in tqdm(range(f0, f1 + 1), desc="overlay video", unit="frame"):
        ok, img = cap.read()
        if not ok:
            break
        rows = draw_rows.get(f)
        new, heads = [], []
        if rows is not None:
            for r in rows.itertuples():
                pt = (float(r.ref_u), float(r.ref_v))
                if not (np.isfinite(pt[0]) and np.isfinite(pt[1])):
                    continue
                col = _bgr(r.spd, r.spd_lo, r.spd_hi)
                prev = last.get(r.track_id)
                if prev is not None and f - prev[0] <= max_gap:
                    if r.predicted:  # machine-made gap: dotted white
                        if f % 4 < 2:
                            new.append((f, prev[1], pt, (235, 235, 235), max(1, th - 1)))
                    else:
                        new.append((f, prev[1], pt, col, max(2, th)))
                last[r.track_id] = (f, pt)
                heads.append((pt, col, r))
        if trail_s is None:
            for _, a, b, c, t in new:
                _draw(canvas, alpha, a, b, c, t)
        else:
            segments.extend(new)
            segments = [sg for sg in segments if sg[0] >= f - float(trail_s) * fps]
            canvas[:] = 0
            alpha[:] = 0
            for _, a, b, c, t in segments:
                _draw(canvas, alpha, a, b, c, t)

        for r in (blur_rows.get(f).itertuples() if f in blur_rows else ()):
            blur_box(img, r.x1, r.y1, r.x2, r.y2)
        H_f = reg.get(f) if reg is not None else None
        if stable and H_f is not None:
            # output in calibration-frame pixels: the picture stops moving
            img = cv2.warpPerspective(img, H_f, (W, H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
            lay, a = canvas, alpha
            to_out = None
        elif H_f is not None:
            Hinv = np.linalg.inv(H_f)
            lay = cv2.warpPerspective(canvas, Hinv, (W, H), flags=cv2.INTER_LINEAR)
            a = cv2.warpPerspective(alpha, Hinv, (W, H), flags=cv2.INTER_LINEAR)
            to_out = Hinv
        else:
            lay, a, to_out = canvas, alpha, None
        if dim < 1:
            img = (img * dim).astype(np.uint8)
        a = (a.astype(np.float32) * (0.92 / 255.0))[..., None]
        img = (img * (1 - a) + lay * a).astype(np.uint8)

        if roi is not None and style != "clean":
            poly = np.asarray(roi, np.float64).reshape(-1, 1, 2)
            if to_out is not None:
                poly = cv2.perspectiveTransform(poly, to_out)
            cv2.polylines(img, [poly.astype(np.int32)], True, (255, 255, 255), th, cv2.LINE_AA)
        for pt, col, r in heads:
            q = np.array([[pt]], np.float64)
            if to_out is not None:
                q = cv2.perspectiveTransform(q, to_out)
            c = (int(q[0, 0, 0]), int(q[0, 0, 1]))
            if style == "clean":
                cv2.circle(img, c, 3 * th, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(img, c, 2 * th, col, -1, cv2.LINE_AA)
            else:
                box_col = GROUP_BGR.get(cfg.class_to_group.get(r[2], ""), (200, 200, 200))
                p1, p2 = _box_in_ref(r, H_f if stable else None)
                if r.predicted:
                    _dashed_rect(img, p1, p2, (180, 180, 180), max(1, th - 1))
                else:
                    cv2.rectangle(img, p1, p2, box_col, max(1, th - 1), cv2.LINE_AA)
                cv2.putText(img, f"{int(r.track_id)}" + (" pred" if r.predicted else ""),
                            (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4 * th, box_col, max(1, th // 2), cv2.LINE_AA)
        cv2.putText(img, f"t = {f / fps:6.2f} s", (10 * th, 18 * th), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * th, (255, 255, 255), max(1, th // 2), cv2.LINE_AA)
        _legend(img, mode, cfg, th)
        if scale < 1:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        writer.write(img)
    writer.release()
    cap.release()
    log(f"overlay video -> {out_path} (style {style}"
        + (", people blurred" if blur else "") + (", stabilized" if stable else "")
        + f"; trail colour = speed, {mode})")


def raw_all_people(run_dir, fallback: pd.DataFrame) -> pd.DataFrame:
    """Every person box of the run (incl. those the clean style hides), for blurring."""
    if run_dir is not None and (Path(run_dir) / "raw_tracks.csv").exists():
        r = pd.read_csv(Path(run_dir) / "raw_tracks.csv")
    else:
        r = fallback
    return r[r["class"] == "person"]
