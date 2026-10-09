"""Project the map into the video frame with a calibration: the most readable check for
oblique views (roads, kerbs and markings must line up with the video)."""

from __future__ import annotations

import cv2
import numpy as np

from .geo import GeoRaster
from .homography import Homography


def map_in_video(frame_bgr: np.ndarray, h: Homography, raster: GeoRaster) -> tuple[np.ndarray, np.ndarray]:
    """Map image as the camera would see it (BGR) + mask of pixels that hit the map."""
    Hh, W = frame_bgr.shape[:2]
    # a terrain model is ray-cast: every 4th pixel, interpolated in between
    k = 4 if h.terrain is not None else 1
    gw, gh = (-(-W // k) + 1, -(-Hh // k) + 1) if k > 1 else (W, Hh)
    uu, vv = np.meshgrid(np.arange(gw, dtype=np.float64) * k, np.arange(gh, dtype=np.float64) * k)
    world = h.to_world(np.column_stack([uu.ravel(), vv.ravel()]))
    cr = raster.world_to_pixel(np.nan_to_num(world, nan=-1e9))
    mx = cr[:, 0].reshape(gh, gw).astype(np.float32)
    my = cr[:, 1].reshape(gh, gw).astype(np.float32)
    ok = np.isfinite(world).all(axis=1).reshape(gh, gw).astype(np.uint8)
    if k > 1:
        size = (gw * k, gh * k)
        mx, my = (cv2.resize(a, size, interpolation=cv2.INTER_LINEAR)[:Hh, :W] for a in (mx, my))
        ok = cv2.resize(ok, size, interpolation=cv2.INTER_NEAREST)[:Hh, :W]
    ok = ok.astype(bool)
    mx[~ok] = -1
    img = cv2.remap(cv2.cvtColor(raster.image, cv2.COLOR_RGB2BGR), mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    inside = ok & (mx >= 0) & (my >= 0) & (mx < raster.image.shape[1]) & (my < raster.image.shape[0])
    return img, inside


def edge_overlay(frame_bgr, h: Homography, raster: GeoRaster, color=(0, 255, 255)) -> np.ndarray:
    """Video frame with the map's edges (kerbs, markings, buildings) drawn on top."""
    mp, inside = map_in_video(frame_bgr, h, raster)
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(mp, cv2.COLOR_BGR2GRAY), (3, 3), 0), 60, 140)
    edges[~inside] = 0
    out = frame_bgr.copy()
    out[edges > 0] = color
    return out
