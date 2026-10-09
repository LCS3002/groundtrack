"""Ground that is not one flat plane: terraces, steps, ramps (from a LiDAR terrain model).

One plane puts everybody on one level. Where the walking surface changes level (Trafalgar
Square's north terrace is 3 m above the square), a person on the upper level, projected onto
the lower plane, lands tens of metres too far from the camera. With a terrain model (DTM)
the camera ray through each foot pixel is followed until it meets the ground.

Heights are handled relative to the calibration's datum (``ground_z_m``, the absolute
elevation of local z = 0), so on flat ground the result is the plane result.

England: the Environment Agency's 1 m LiDAR composite DTM is free (Open Government Licence)
and is fetched automatically with ``terrain: auto``.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

EA_WCS = ("https://environment.data.gov.uk/spatialdata/"
          "lidar-composite-digital-terrain-model-dtm-1m/wcs")
EA_COVERAGE = "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"
CREDIT = ("terrain: Environment Agency LiDAR DTM 1 m, \u00a9 Environment Agency, "
          "Open Government Licence v3.0")


@dataclass
class Terrain:
    z: np.ndarray            # rows x cols, absolute elevation (m), gaps filled
    left: float              # world E of the left edge
    top: float               # world N of the top edge
    res: float               # metres per cell
    path: str = ""

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        rows, cols = self.z.shape
        return (self.left, self.top - rows * self.res, self.left + cols * self.res, self.top)

    def height(self, xy: np.ndarray) -> np.ndarray:
        """Absolute ground elevation at world (E, N), bilinear. NaN outside the model."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        rows, cols = self.z.shape
        c = (xy[:, 0] - self.left) / self.res - 0.5
        r = (self.top - xy[:, 1]) / self.res - 0.5
        ok = np.isfinite(c) & np.isfinite(r) & (c >= 0) & (r >= 0) & (c <= cols - 1) & (r <= rows - 1)
        out = np.full(len(xy), np.nan)
        if not ok.any():
            return out
        c, r = c[ok], r[ok]
        c0 = np.minimum(np.floor(c).astype(int), cols - 2)
        r0 = np.minimum(np.floor(r).astype(int), rows - 2)
        fc, fr = c - c0, r - r0
        z = self.z
        out[ok] = ((z[r0, c0] * (1 - fc) + z[r0, c0 + 1] * fc) * (1 - fr)
                   + (z[r0 + 1, c0] * (1 - fc) + z[r0 + 1, c0 + 1] * fc) * fr)
        return out


def load_terrain(path: str | Path) -> Terrain:
    """Read a DTM GeoTIFF in British National Grid; no-data cells take the nearest value."""
    return _load_terrain(str(Path(path).resolve()))


@lru_cache(maxsize=8)
def _load_terrain(path: str) -> Terrain:
    import rasterio
    from scipy.ndimage import distance_transform_edt

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"terrain model {path} is missing; `terrain: auto` in the site "
                                "config fetches it again")
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 27700:
            raise ValueError(f"{path}: terrain must be in EPSG:27700 (British National Grid)")
        t = src.transform
        if abs(t.a + t.e) > 1e-9 or t.b or t.d:
            raise ValueError(f"{path}: terrain must be a north-up grid with square cells")
        z = src.read(1).astype(np.float64)
        if src.nodata is not None:
            z[z == src.nodata] = np.nan
    z[(z < -100) | (z > 2000)] = np.nan
    gap = ~np.isfinite(z)
    if gap.all():
        raise ValueError(f"{path}: the terrain model has no data here")
    if gap.any():
        idx = distance_transform_edt(gap, return_distances=False, return_indices=True)
        z = z[tuple(idx)]
    return Terrain(z=z, left=t.c, top=t.f, res=t.a, path=str(path.resolve()))


def fetch_ea_dtm(bounds: tuple[float, float, float, float], out: Path, tile_m: int = 1000,
                 log=print) -> Path:
    """Download the Environment Agency 1 m LiDAR DTM for (left, bottom, right, top) in BNG."""
    import rasterio
    from rasterio.transform import from_origin

    x_lo, y_lo = int(np.floor(bounds[0])), int(np.floor(bounds[1]))
    x_hi, y_hi = int(np.ceil(bounds[2])), int(np.ceil(bounds[3]))
    z = np.full((y_hi - y_lo, x_hi - x_lo), np.nan, np.float32)
    for x0 in range(x_lo, x_hi, tile_m):
        for y0 in range(y_lo, y_hi, tile_m):
            x1, y1 = min(x0 + tile_m, x_hi), min(y0 + tile_m, y_hi)
            url = (f"{EA_WCS}?service=WCS&version=2.0.1&request=GetCoverage"
                   f"&CoverageId={EA_COVERAGE}&format=image/tiff"
                   f"&subset=E({x0},{x1})&subset=N({y0},{y1})")
            req = urllib.request.Request(url, headers={"User-Agent": "groundtrack"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if data[:4] not in (b"II*\x00", b"MM\x00*"):
                raise OSError("Environment Agency terrain service: "
                              + data[:300].decode("utf-8", "replace"))
            with rasterio.MemoryFile(data) as mem, mem.open() as src:
                a = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    a[a == src.nodata] = np.nan
                rr = int(round((y_hi - src.transform.f) / 1.0))
                cc = int(round((src.transform.c - x_lo) / 1.0))
                z[rr:rr + a.shape[0], cc:cc + a.shape[1]] = a[:z.shape[0] - rr, :z.shape[1] - cc]
    if not np.isfinite(z).any():
        raise OSError("no terrain data here (the Environment Agency DTM covers England)")
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1,
                       dtype="float32", crs="EPSG:27700", nodata=np.nan, compress="deflate",
                       transform=from_origin(x_lo, y_hi, 1.0, 1.0)) as dst:
        dst.write(z, 1)
    log(f"terrain: Environment Agency LiDAR DTM 1 m, {z.shape[1]} x {z.shape[0]} m -> {out}")
    return out


def resolve_terrain(cfg, log=print) -> Terrain | None:
    """The site's terrain (config key ``terrain``: null, a DTM GeoTIFF, or ``auto``)."""
    v = cfg.get("terrain")
    if v in (None, "", False):
        return None
    if str(v).lower() == "auto":
        geo = cfg.path("geotiff")
        if geo is None:
            raise ValueError("terrain: auto needs the site's geotiff (its extent is fetched)")
        out = geo.parent / "dtm" / f"{geo.stem}_dtm.tif"
        if not out.exists():
            import rasterio

            with rasterio.open(geo) as src:
                bb = src.bounds
            m = 50.0      # a margin: cameras and far walkers just outside the map
            fetch_ea_dtm((bb.left - m, bb.bottom - m, bb.right + m, bb.top + m), out, log=log)
        return load_terrain(str(out))
    return load_terrain(str(cfg.path("terrain")))


# --------------------------------------------------------------------------- ray casting
def raycast(C: np.ndarray, D: np.ndarray, terrain: Terrain, z0: float, origin,
            max_range_m: float = 1500.0, step_m: float = 0.5) -> np.ndarray:
    """First ground hit of rays C + t D (local metres, z relative to the datum z0).

    Returns world (E, N) per ray; NaN where a ray never meets the ground (sky, horizon).
    Beyond the terrain model the ground is taken as the datum plane."""
    origin = np.asarray(origin, float)
    D = np.asarray(D, float).reshape(-1, 3)
    n = len(D)
    horiz = np.maximum(np.hypot(D[:, 0], D[:, 1]), 1e-9)
    dx, dy, m = D[:, 0] / horiz, D[:, 1] / horiz, D[:, 2] / horiz   # per metre across ground
    cx, cy, cz = (float(v) for v in C)
    zl_min = float(np.nanmin(terrain.z)) - z0
    zl_max = float(np.nanmax(terrain.z)) - z0
    zl_min, zl_max = min(zl_min, 0.0), max(zl_max, 0.0)

    def ground(s, i):
        h = terrain.height(np.column_stack([cx + dx[i] * s + origin[0],
                                            cy + dy[i] * s + origin[1]])) - z0
        return np.where(np.isfinite(h), h, 0.0)

    # the hit lies between where the ray passes the highest and the lowest ground
    # (level or rising rays only meet ground higher than the camera)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_lo = np.where(m < 0, (cz - zl_max) / -m - step_m, 0.0)
        s_hi = np.where(m < 0, (cz - zl_min) / -m + step_m,
                        max_range_m if cz < zl_max else 0.0)
    s_lo = np.clip(np.nan_to_num(s_lo), 0.0, max_range_m)
    s_hi = np.clip(np.nan_to_num(s_hi, nan=max_range_m), 0.0, max_range_m)

    hit = np.full(n, np.nan)
    act = np.flatnonzero(s_hi > s_lo)
    s = s_lo[act]
    above = cz + m[act] * s >= ground(s, act) - 1e-6
    while act.size:
        s_new = np.minimum(s + step_m, s_hi[act])
        below = cz + m[act] * s_new <= ground(s_new, act)
        found = above & below
        if found.any():                         # refine the crossing by bisection
            i, a, b = act[found], s[found], s_new[found]
            for _ in range(8):
                mid = (a + b) / 2
                up = cz + m[i] * mid > ground(mid, i)
                a, b = np.where(up, mid, a), np.where(up, b, mid)
            hit[i] = (a + b) / 2
        done = found | (s_new >= s_hi[act])
        above = ~below                           # a ray starting underground must surface first
        act, s, above = act[~done], s_new[~done], above[~done]
    out = np.column_stack([cx + dx * hit, cy + dy * hit]) + origin
    return out
