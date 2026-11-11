"""Georeferencing of eu-elevation ``/api/elevation/raster`` grids.

The grid comes straight from a north-up GeoTIFF: ``elevations[0]`` is the
NORTH edge of the bbox and column 0 the west edge. ``origin_lat`` is the
south edge (kept by eu-elevation for compatibility), so the grid is placed
from its north edge.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from rasterio.transform import Affine


def north_edge(dem: dict[str, Any]) -> float:
    """North edge latitude: the declared ``north_lat``, else origin + rows."""
    declared = dem.get("north_lat")
    if declared is not None:
        return float(declared)
    return float(dem["origin_lat"]) + float(dem["pixel_size_deg"]) * int(dem["rows"])


def dem_transform(dem: dict[str, Any]) -> Affine:
    """North-up affine transform: pixel (col, row) -> (lon, lat)."""
    size = float(dem["pixel_size_deg"])
    return Affine(size, 0.0, float(dem["origin_lon"]), 0.0, -size, north_edge(dem))


def pixel_center_grids(dem: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """``(lon_g, lat_g)`` of pixel centres, shape (rows, cols); row 0 is north."""
    size = float(dem["pixel_size_deg"])
    rows, cols = int(dem["rows"]), int(dem["cols"])
    lons = float(dem["origin_lon"]) + size * (np.arange(cols) + 0.5)
    lats = north_edge(dem) - size * (np.arange(rows) + 0.5)
    return np.meshgrid(lons, lats)
