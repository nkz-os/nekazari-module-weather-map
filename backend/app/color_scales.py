"""Color scale palettes and lookup table generator for weather metrics."""

from __future__ import annotations


import numpy as np


# Each palette entry: (value, R, G, B, A) where value is the data value
# and R, G, B, A are 0-255 integer color components.
PALETTES: dict[str, list[tuple[float, int, int, int, int]]] = {
    "temperature_avg": [
        (-5,   20,  20,  120, 255),   # Deep blue
        (0,    50,  80,  200, 255),   # Blue
        (10,   100, 180, 230, 255),   # Light blue
        (20,   200, 220, 100, 255),   # Yellow-green
        (30,   240, 150, 50,  255),   # Orange
        (40,   220, 60,  30,  255),   # Red
        (45,   150, 0,   0,   255),   # Dark red
    ],
    "temperature_min": [
        (-10,  50,  0,   100, 255),   # Violet
        (0,    100, 80,  180, 255),
        (10,   180, 200, 100, 255),
        (20,   240, 180, 50,  255),   # Orange
        (30,   220, 80,  20,  255),
    ],
    "solar_radiation": [
        (0,    80,  80,  80,  255),   # Grey
        (200,  140, 140, 100, 255),
        (500,  220, 220, 80,  255),   # Yellow
        (800,  255, 240, 60,  255),
        (1000, 255, 255, 200, 255),   # White-bright
    ],
    "eto": [
        (0,    50,  180, 50,  255),   # Green
        (3,    100, 200, 80,  255),
        (6,    200, 200, 50,  255),   # Yellow
        (10,   220, 120, 40,  255),   # Orange
        (15,   200, 40,  30,  255),   # Red
    ],
    "water_balance": [
        (-30,  180, 0,   0,   255),   # Red (deficit)
        (-15,  220, 80,  30,  255),
        (-5,   240, 200, 80,  255),   # Yellow
        (5,    200, 220, 180, 255),   # Pale
        (15,   100, 200, 180, 255),   # Cyan (surplus)
        (30,   0,   100, 200, 255),   # Blue
    ],
    "frost_risk": [
        (0,    50,  180, 50,  255),   # Green (safe)
        (25,   180, 200, 50,  255),
        (50,   220, 180, 40,  255),   # Orange
        (75,   240, 100, 30,  255),
        (100,  180, 0,   0,   255),   # Dark red (risk)
    ],
    "soil_moisture": [
        (0,    120, 60,  20,  255),   # Brown (dry)
        (15,   180, 140, 60,  255),
        (30,   100, 180, 100, 255),   # Green (moist)
        (40,   60,  160, 200, 255),   # Blue-green
        (50,   20,  80,  200, 255),   # Blue (wet)
    ],
}


def _get_stops(metric: str) -> list[tuple[float, int, int, int, int]]:
    """Look up palette stops for a given metric.

    Raises ValueError if metric is unknown.
    """
    if metric not in PALETTES:
        raise ValueError(f"Unknown metric: {metric}")
    return PALETTES[metric]


def build_lut(metric: str, n_colors: int = 256) -> np.ndarray:
    """Build a lookup table of RGBA colors for a given metric.

    Linearly interpolates between palette stops to produce an array of
    ``n_colors`` evenly-spaced RGBA entries covering the full data range.

    Args:
        metric: Name of the weather metric.
        n_colors: Number of LUT entries (default 256).

    Returns:
        ``(n_colors, 4)`` uint8 array.
    """
    stops = _get_stops(metric)
    values = np.array([s[0] for s in stops], dtype=np.float64)
    colors = np.array([list(s[1:5]) for s in stops], dtype=np.float64)

    vmin = values[0]
    vmax = values[-1]

    # Evenly-spaced query points from vmin to vmax
    query = np.linspace(vmin, vmax, n_colors, dtype=np.float64)

    # Interpolate each RGBA channel independently
    lut = np.column_stack([
        np.interp(query, values, colors[:, c])
        for c in range(4)
    ])

    return np.clip(np.round(lut), 0, 255).astype(np.uint8)


def apply_color_scale(data: np.ndarray, metric: str) -> np.ndarray:
    """Apply a color scale to a 2D float32 array.

    Each pixel's value is mapped through the metric's lookup table to
    produce an RGBA color. NaN/Inf pixels are set to transparent.

    Args:
        data: ``(H, W)`` float32 array of metric values.
        metric: Name of the weather metric.

    Returns:
        ``(H, W, 4)`` uint8 RGBA array.
    """
    stops = _get_stops(metric)
    vmin = stops[0][0]
    vmax = stops[-1][0]

    lut = build_lut(metric)

    height, width = data.shape
    result = np.zeros((height, width, 4), dtype=np.uint8)

    # Only process finite pixels to avoid NaN-cast warnings
    valid = np.isfinite(data)
    if np.any(valid):
        normalized = (data[valid] - vmin) / (vmax - vmin) * 255.0
        normalized = np.clip(np.round(normalized).astype(np.int32), 0, 255)
        result[valid] = lut[normalized]

    # Invalid pixels remain (0, 0, 0, 0) — transparent

    return result
