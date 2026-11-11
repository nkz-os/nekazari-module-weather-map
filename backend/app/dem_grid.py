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


def row_height(dem: dict[str, Any]) -> float:
    """Row height in degrees.

    eu-elevation's ``pixel_size_deg`` is the column width (lon span / cols);
    its row count is rounded separately, so when the north edge is declared
    the rows span exactly ``north_lat - origin_lat``.
    """
    if dem.get("north_lat") is not None:
        return (float(dem["north_lat"]) - float(dem["origin_lat"])) / int(dem["rows"])
    return float(dem["pixel_size_deg"])


def dem_transform(dem: dict[str, Any]) -> Affine:
    """North-up affine transform: pixel (col, row) -> (lon, lat)."""
    width = float(dem["pixel_size_deg"])
    return Affine(width, 0.0, float(dem["origin_lon"]), 0.0, -row_height(dem), north_edge(dem))


def pixel_center_grids(dem: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """``(lon_g, lat_g)`` of pixel centres, shape (rows, cols); row 0 is north."""
    width = float(dem["pixel_size_deg"])
    rows, cols = int(dem["rows"]), int(dem["cols"])
    lons = float(dem["origin_lon"]) + width * (np.arange(cols) + 0.5)
    lats = north_edge(dem) - row_height(dem) * (np.arange(rows) + 0.5)
    return np.meshgrid(lons, lats)
