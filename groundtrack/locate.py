"""Camera position from a place name (OpenStreetMap) + floor, for `camera_position`.

Only the search text is sent to OpenStreetMap's Nominatim service; no footage, images or
results leave the computer. Results are converted to British National Grid (EPSG:27700).
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

NOMINATIM = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "groundtrack (architecture research tool)"


def search(query: str, limit: int = 5, timeout: float = 15.0) -> list[dict]:
    """Places matching `query`: [{name, lat, lon, E, N, bbox_m}], best first."""
    from rasterio.warp import transform

    url = NOMINATIM + "?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "limit": limit, "countrycodes": "gb"})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        hits = json.loads(r.read().decode("utf-8"))
    out = []
    for h in hits:
        lat, lon = float(h["lat"]), float(h["lon"])
        E, N = transform("EPSG:4326", "EPSG:27700", [lon], [lat])
        s, n_, w, e = (float(v) for v in h.get("boundingbox", [lat, lat, lon, lon]))
        bx, by = transform("EPSG:4326", "EPSG:27700", [w, e], [s, n_])
        out.append({"name": h.get("display_name", ""), "lat": lat, "lon": lon,
                    "E": round(E[0], 1), "N": round(N[0], 1),
                    "size_m": round(max(abs(bx[1] - bx[0]), abs(by[1] - by[0])), 1)})
    return out


def camera_from_place(hit: dict, floor: float | None = None, floor_height_m: float = 3.1,
                      eye_height_m: float = 1.5, ground_offset_m: float = 0.0) -> dict:
    """camera_position block: building centre, height from the floor number.

    position_tol_m is half the place's size (you may stand anywhere in the building)."""
    height = (floor or 0) * floor_height_m + eye_height_m + ground_offset_m
    return {"E": hit["E"], "N": hit["N"], "height_m": round(height, 1),
            "position_tol_m": round(max(5.0, hit.get("size_m", 20.0) / 2), 1),
            "height_tol_m": round(max(1.0, 0.06 * height), 1)}


def write_camera_position(config_path: str | Path, cam: dict, note: str = "") -> None:
    """Replace (or add) the camera_position block in a site YAML, keeping everything else."""
    p = Path(config_path)
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    block = "camera_position:" + (f"            # {note}" if note else "") + "\n" + "".join(
        f"  {k}: {v}\n" for k, v in cam.items())
    # drop an existing block (the key line and its indented children)
    text = re.sub(r"(?m)^camera_position:.*\n(?:[ \t]+.*\n?)*", "", text)
    p.write_text(text.rstrip() + "\n" + block, encoding="utf-8")
