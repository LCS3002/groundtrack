"""One blue (slow) -> red (fast) ramp, shared by the PNGs, the Houdini script and debug video.

Turbo, trimmed at both ends (no near-black purple / maroon): blue -> cyan -> green -> yellow
-> red. Unlike diverging ramps it has no pale middle, so mid speeds stay visible on both
white backgrounds and dimmed aerial imagery.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

RAMP_NAME = "groundtrack_speed"
_LO, _HI = 0.10, 0.95


@lru_cache(maxsize=1)
def cmap():
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    base = matplotlib.colormaps["turbo"]
    return LinearSegmentedColormap.from_list(RAMP_NAME, base(np.linspace(_LO, _HI, 256)))


def ramp_rgb(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Values -> (N, 3) RGB floats in 0..1 (sRGB) using the shared ramp."""
    v = np.clip((np.asarray(values, float) - lo) / max(hi - lo, 1e-9), 0, 1)
    return cmap()(v)[:, :3]


def ramp_stops(n: int = 11) -> list[tuple[float, tuple[float, float, float]]]:
    c = cmap()
    return [(round(i / (n - 1), 4), tuple(round(float(x), 4) for x in c(i / (n - 1))[:3]))
            for i in range(n)]
