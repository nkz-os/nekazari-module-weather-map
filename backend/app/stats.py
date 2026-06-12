"""
Zonal statistics — extract per-parcel weather metrics from COG rasters.

Computes statistics (mean, min, max, std and metric-specific aggregates)
for a given parcel polygon by intersecting its geometry with precomputed
COG tiles stored in MinIO.
"""

from __future__ import annotations

import io
import logging
import math
from typing import Any

import numpy as np
import rasterio
from rasterio import mask as rio_mask

from app.config import settings
from app.minio_io import download_cog, get_latest_date

logger = logging.getLogger(__name__)

# ── Water-balance thresholds ──────────────────────────────────────
# Used by the ``water_balance`` metric family only.
DEFICIT_THRESHOLD_MM = -5.0       # pixels <= this are "in deficit"
SEVERE_DEFICIT_THRESHOLD_MM = -15.0  # pixels <= this are "severe deficit"


def _collect_pixels(
    geometry: dict[str, Any],
    metric: str,
    date: str,
    zoom: int = 14,
) -> np.ndarray | None:
    """Mask the COG tile(s) intersecting *geometry* and return pixel values.

    Returns a 1-D float32 array of pixel values that fall inside the
    parcel polygon, or ``None`` when no data is available.
    """
    # Compute the TMS tiles that cover the geometry bbox
    coords = _geometry_bbox(geometry)
    if coords is None:
        return None
    min_lon, min_lat, max_lon, max_lat = coords

    tiles = _bbox_to_tiles(min_lon, min_lat, max_lon, max_lat, zoom)
    if not tiles:
        logger.warning("No tiles intersect parcel bbox")
        return None

    all_pixels: list[np.ndarray] = []

    for z, tx, ty in tiles:
        cog_bytes = download_cog("default", metric, date, z, tx, ty)
        if cog_bytes is None:
            continue

        try:
            with rasterio.open(io.BytesIO(cog_bytes)) as src:
                # Mask the raster with the geometry
                out_image, _ = rio_mask.mask(
                    src, [geometry], crop=True, all_touched=True,
                    nodata=src.nodata or np.nan,
                )
                band = out_image[0]  # single-band COG
                valid = band[
                    ~np.isnan(band) & ~np.isinf(band) &
                    (band != (src.nodata or -9999))
                ]
                if len(valid) > 0:
                    all_pixels.append(valid.astype(np.float64))
        except Exception:
            logger.exception(
                "Failed to mask COG: metric=%s, tile=%d/%d/%d",
                metric, z, tx, ty,
            )
            continue

    if not all_pixels:
        return None
    return np.concatenate(all_pixels)


def _geometry_bbox(
    geometry: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    """Return (min_lon, min_lat, max_lon, max_lat) from a GeoJSON geometry."""
    try:
        coords = geometry["coordinates"]
        if geometry["type"] == "Polygon":
            flat = [c for ring in coords for point in ring for c in point]
        elif geometry["type"] == "MultiPolygon":
            flat = [c for poly in coords for ring in poly for point in ring for c in point]
        else:
            return None
        lons = flat[0::2]
        lats = flat[1::2]
        return min(lons), min(lats), max(lons), max(lats)
    except (KeyError, IndexError, TypeError):
        logger.exception("Cannot compute bbox from geometry")
        return None


def _bbox_to_tiles(
    min_lon: float, min_lat: float,
    max_lon: float, max_lat: float,
    zoom: int = 14,
) -> list[tuple[int, int, int]]:
    """Convert a geographic bbox to a list of (z, x, y) TMS tile coords."""
    n = 2.0 ** zoom
    x_min = int((min_lon + 180.0) / 360.0 * n)
    x_max = int((max_lon + 180.0) / 360.0 * n)
    y_min = int(
        (1.0 - math.asinh(math.tan(math.radians(max_lat))) / math.pi)
        / 2.0 * n
    )
    y_max = int(
        (1.0 - math.asinh(math.tan(math.radians(min_lat))) / math.pi)
        / 2.0 * n
    )
    tiles = []
    for x in range(max(0, x_min), min(int(n), x_max + 1)):
        for y in range(max(0, y_min), min(int(n), y_max + 1)):
            tiles.append((zoom, x, y))
    return tiles


def _compute_histogram(
    values: np.ndarray, bins: int = 10,
) -> list[float]:
    """Compute a histogram with *bins* equal-width buckets."""
    if len(values) == 0:
        return [0.0] * bins
    hist, _ = np.histogram(values, bins=bins)
    return hist.tolist()


# ── Public API ────────────────────────────────────────────────────


def compute_zonal_stats(
    geometry: dict[str, Any],
    metrics: list[str],
    date: str | None = None,
) -> dict[str, Any]:
    """Compute per-metric statistics for a parcel polygon.

    Parameters
    ----------
    geometry : dict
        GeoJSON geometry dict (Polygon or MultiPolygon), e.g. from an
        AgriParcel ``location`` property.
    metrics : list of str
        Weather metric names (e.g. ``temperature_avg``, ``water_balance``).
    date : str or None
        COG date in ``YYYY-MM-DD`` format.  ``None`` → latest available.

    Returns
    -------
    dict
        ``{"parcel_geojson": …, "date": …, "metrics": {metric: {…}}}``.
    """
    resolved_date = date or get_latest_date("default", metrics[0] if metrics else "temperature_avg")
    if not resolved_date:
        return {"error": "No COG data available", "metrics": {}}

    result: dict[str, Any] = {
        "parcel_geojson": geometry,
        "date": resolved_date,
        "metrics": {},
    }

    for metric in metrics:
        if metric not in settings.metrics:
            result["metrics"][metric] = {"error": f"Unknown metric: {metric}"}
            continue

        pixels = _collect_pixels(geometry, metric, resolved_date)
        if pixels is None or len(pixels) == 0:
            result["metrics"][metric] = {
                "error": "No data for parcel at this metric/date",
            }
            continue

        stats: dict[str, Any] = {
            "mean": float(np.mean(pixels)),
            "min": float(np.min(pixels)),
            "max": float(np.max(pixels)),
            "std": float(np.std(pixels)),
            "p25": float(np.percentile(pixels, 25)),
            "p50": float(np.percentile(pixels, 50)),
            "p75": float(np.percentile(pixels, 75)),
            "pixel_count": int(len(pixels)),
            "histogram": _compute_histogram(pixels, 10),
        }

        # Metric-specific aggregates
        if metric == "water_balance":
            deficit = np.sum(pixels <= DEFICIT_THRESHOLD_MM)
            severe = np.sum(pixels <= SEVERE_DEFICIT_THRESHOLD_MM)
            stats["deficit_area_pct"] = round(float(deficit / len(pixels) * 100), 2)
            stats["severe_deficit_pct"] = round(float(severe / len(pixels) * 100), 2)

        elif metric == "frost_risk":
            high_risk = np.sum(pixels >= 75.0)
            moderate_risk = np.sum((pixels >= 50.0) & (pixels < 75.0))
            stats["high_risk_pct"] = round(float(high_risk / len(pixels) * 100), 2)
            stats["moderate_risk_pct"] = round(float(moderate_risk / len(pixels) * 100), 2)

        elif metric == "soil_moisture":
            dry = np.sum(pixels <= 15.0)
            saturated = np.sum(pixels >= 40.0)
            stats["dry_pct"] = round(float(dry / len(pixels) * 100), 2)
            stats["saturated_pct"] = round(float(saturated / len(pixels) * 100), 2)

        elif metric in ("temperature_avg", "temperature_min"):
            heat_stress = np.sum(pixels >= 35.0)
            frost = np.sum(pixels <= 0.0)
            stats["heat_stress_pct"] = round(float(heat_stress / len(pixels) * 100), 2)
            stats["frost_pct"] = round(float(frost / len(pixels) * 100), 2)

        result["metrics"][metric] = stats

    return result
