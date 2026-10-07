"""Reading georeferenced top-down images (GeoTIFF exported from QGIS / Digimap)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_EPSG = 27700


@dataclass
class GeoRaster:
    image: np.ndarray            # H x W x 3 uint8 (possibly downsampled for display)
    left: float
    right: float
    bottom: float
    top: float
    epsg: int | None
    native_res: float            # metres per pixel of the source file
    path: str = ""

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """matplotlib imshow extent so that axes are in world coordinates."""
        return (self.left, self.right, self.bottom, self.top)

    @property
    def res(self) -> float:
        return (self.right - self.left) / self.image.shape[1]

    def world_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        """World (E, N) -> continuous pixel (col, row) of self.image, corner convention."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        sx = self.image.shape[1] / (self.right - self.left)
        sy = self.image.shape[0] / (self.top - self.bottom)
        return np.column_stack([(xy[:, 0] - self.left) * sx, (self.top - xy[:, 1]) * sy])

    def pixel_to_world(self, cr: np.ndarray) -> np.ndarray:
        cr = np.asarray(cr, float).reshape(-1, 2)
        sx = (self.right - self.left) / self.image.shape[1]
        sy = (self.top - self.bottom) / self.image.shape[0]
        return np.column_stack([self.left + cr[:, 0] * sx, self.top - cr[:, 1] * sy])


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(arr.shape, np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def load_geotiff(path: str | Path, max_dim: int = 6000, require_epsg: int | None = TARGET_EPSG,
                 warn=print) -> GeoRaster:
    """Load a north-up GeoTIFF as RGB uint8, downsampled so the long side <= max_dim."""
    import rasterio
    from rasterio.enums import Resampling

    path = Path(path)
    with rasterio.open(path) as ds:
        t = ds.transform
        if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
            raise ValueError(
                f"{path.name} has a rotated geotransform. Export a north-up image from QGIS "
                "(Project > Import/Export > Export Map to Image, with 'Append georeference')."
            )
        epsg = ds.crs.to_epsg() if ds.crs else None
        if epsg is None:
            warn(f"WARNING: {path.name} has no CRS; assuming EPSG:{require_epsg}.")
        elif require_epsg and epsg != require_epsg:
            raise ValueError(
                f"{path.name} is in EPSG:{epsg}, expected EPSG:{require_epsg} (British National "
                "Grid). Reproject it in QGIS (Raster > Projections > Warp) first."
            )
        scale = max(ds.width, ds.height) / max_dim
        if scale > 1:
            out_h, out_w = int(round(ds.height / scale)), int(round(ds.width / scale))
        else:
            out_h, out_w = ds.height, ds.width
        n = ds.count
        if n >= 3:
            bands = [1, 2, 3]
        else:
            bands = [1]
        data = ds.read(bands, out_shape=(len(bands), out_h, out_w), resampling=Resampling.average)
        if n == 1 and ds.colorinterp and ds.colorinterp[0].name == "palette":
            cmap = ds.colormap(1)
            lut = np.zeros((256, 3), np.uint8)
            for k, rgba in cmap.items():
                if 0 <= k < 256:
                    lut[k] = rgba[:3]
            rgb = lut[data[0].astype(np.uint8)]
        else:
            data = _to_uint8(data)
            rgb = np.repeat(data, 3, axis=0) if len(bands) == 1 else data
            rgb = np.transpose(rgb, (1, 2, 0))
        b = ds.bounds
        return GeoRaster(
            image=np.ascontiguousarray(rgb),
            left=b.left, right=b.right, bottom=b.bottom, top=b.top,
            epsg=epsg or require_epsg, native_res=abs(t.a), path=str(path),
        )


def write_geotiff(path: str | Path, rgb: np.ndarray, left: float, top: float, res: float,
                  epsg: int = TARGET_EPSG) -> None:
    """Write an RGB uint8 array as a north-up GeoTIFF (used by tests and the demo)."""
    import rasterio
    from rasterio.transform import from_origin

    h, w = rgb.shape[:2]
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=3, dtype="uint8",
        crs=f"EPSG:{epsg}", transform=from_origin(left, top, res, res),
    ) as ds:
        for i in range(3):
            ds.write(rgb[:, :, i], i + 1)
