"""COG generator — weather raster computation and MinIO upload.

Runs daily to produce per-pixel weather rasters
for each tenant's parcels.  Each raster is written as a Cloud-Optimized
GeoTIFF (COG) and uploaded to MinIO under
``cogs/{tenant}/{metric}/{date}/{z}/{x}/{y}.tif``.
"""

from __future__ import annotations

import io
import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import rasterio
from nkz_platform_sdk.gis.terrain import aspect_degrees, slope_degrees
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds

from app.config import settings
from app.downscaler import (
    correct_temperature,
    correct_solar_radiation,
    compute_eto,
    clamp_ea_to_es,
    compute_water_balance,
    compute_frost_risk,
    compute_soil_moisture,
)
from app.minio_io import (
    clear_latest_pointer, set_latest_date, set_window_meta, upload_cog,
)
from app.records import build_agri_parcel_record, guard_frozen_metrics
from app.sources import (
    MissingWeatherInput,
    fetch_parcel_hydraulics,
    fetch_dem_tile,
    fetch_parcel_daily,
    fetch_parcel_weather,
    require,
    upsert_record,
)
from app.stats import UNAVAILABLE_METRICS, compute_zonal_stats

logger = logging.getLogger(__name__)

# Parcel clip margin: the raster should cover the whole parcel with a little
# slack outside the boundary (like the other raster modules). ~20 m in degrees.
_CLIP_MARGIN_M = 20.0
_CLIP_MARGIN_DEG = _CLIP_MARGIN_M / 111_320.0  # ~0.00018 degrees of latitude


# ---------------------------------------------------------------------------
# Tile / bbox helpers
# ---------------------------------------------------------------------------


def _tile_to_bbox(
    z: int, x: int, y: int
) -> tuple[float, float, float, float]:
    """Convert TMS tile coordinates to a geographic bounding box.

    Parameters
    ----------
    z : int
        Zoom level.
    x : int
        Tile column (0 … 2**z − 1).
    y : int
        Tile row (TMS convention, origin at 85°N).

    Returns
    -------
    tuple[float, float, float, float]
        ``(min_lon, min_lat, max_lon, max_lat)`` in decimal degrees.
    """
    n = 2.0 ** z
    min_lon = x / n * 360.0 - 180.0
    max_lon = (x + 1) / n * 360.0 - 180.0
    min_lat = math.degrees(
        math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    )
    max_lat = math.degrees(
        math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    )
    return (min_lon, min_lat, max_lon, max_lat)


def _bbox_to_tiles(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    zoom: int = 14,
) -> list[tuple[int, int, int]]:
    """Convert a geographic bounding box to a list of TMS tiles.

    Parameters
    ----------
    min_lon, min_lat, max_lon, max_lat : float
        Bounding box in decimal degrees.
    zoom : int
        Zoom level (default 14).

    Returns
    -------
    list[tuple[int, int, int]]
        List of ``(z, x, y)`` tile coordinates.
    """
    n = 2.0 ** zoom
    x_min = int((min_lon + 180.0) / 360.0 * n)
    x_max = int((max_lon + 180.0) / 360.0 * n)
    y_min = int(
        (1.0 - math.asinh(math.tan(math.radians(max_lat))) / math.pi)
        / 2.0
        * n
    )
    y_max = int(
        (1.0 - math.asinh(math.tan(math.radians(min_lat))) / math.pi)
        / 2.0
        * n
    )
    tiles: list[tuple[int, int, int]] = []
    for x in range(max(0, x_min), min(int(n), x_max + 1)):
        for y in range(max(0, y_min), min(int(n), y_max + 1)):
            tiles.append((zoom, x, y))
    return tiles


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _date_to_doy(date_str: str) -> int:
    """Convert a ``YYYY-MM-DD`` date string to day-of-year (1–366)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.timetuple().tm_yday


def _parcel_geometry(parcel: dict[str, Any]) -> dict[str, Any] | None:
    """Extract or build a GeoJSON geometry from a parcel dict.

    Tries ``location`` first (Polygon/MultiPolygon from Orion-LD), then
    falls back to building a small 0.001° bbox polygon from ``lon``/``lat``
    (or ``longitude``/``latitude``) if only a point is available.

    Returns ``None`` when no usable coordinates are found.
    """
    location = parcel.get("location")
    if isinstance(location, dict) and location.get("type") in ("Polygon", "MultiPolygon"):
        return location

    lon = parcel.get("lon", parcel.get("longitude"))
    lat = parcel.get("lat", parcel.get("latitude"))
    if lon is None or lat is None:
        return None

    lon = float(lon)
    lat = float(lat)
    delta = 0.0005  # ~55 m at mid-latitudes
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - delta, lat - delta],
            [lon + delta, lat - delta],
            [lon + delta, lat + delta],
            [lon - delta, lat + delta],
            [lon - delta, lat - delta],
        ]],
    }


def _parcel_lonlat(parcel: dict[str, Any]) -> tuple[float, float] | None:
    """Return a representative (lon, lat) centroid for a parcel.

    Derives it from the parcel's GeoJSON geometry (averaging the coordinates)
    so the COG bbox covers the parcel's real location. Parcels from
    ``fetch_tenant_parcels`` carry a ``location`` GeoJSON dict, NOT ``lon``/``lat``
    keys, so reading ``lon``/``lat`` directly would default to (0, 0).
    """
    geom = _parcel_geometry(parcel)
    if not geom:
        return None
    coords: list[float] = []

    def _walk(node: Any) -> None:
        if (
            isinstance(node, (list, tuple))
            and len(node) == 2
            and all(isinstance(c, (int, float)) for c in node)
        ):
            coords.append((float(node[0]), float(node[1])))
        elif isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)

    _walk(geom.get("coordinates"))
    if not coords:
        return None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (sum(lons) / len(lons), sum(lats) / len(lats))


def _nearest_parcel(
    parcels: list[dict[str, Any]], lat: float, lon: float,
) -> dict[str, Any] | None:
    """Return the parcel whose centroid is closest to (lat, lon).

    The weather base for a tile is the NEAREST parcel to its centre —
    ``parcels[0]`` was rejected: with parcels spread far apart it would use
    one parcel's weather for another parcel's tile. Squared-degree distance
    is enough for this ranking; parcels are compared at tile scale, so
    geodesic correction changes nothing here.
    """
    best: dict[str, Any] | None = None
    best_dist = float("inf")
    for parcel in parcels:
        lonlat = _parcel_lonlat(parcel)
        if lonlat is None:
            continue
        p_lon, p_lat = lonlat
        dist = (p_lon - lon) ** 2 + (p_lat - lat) ** 2
        if dist < best_dist:
            best_dist = dist
            best = parcel
    return best


# eto and water_balance read /daily and give every pixel the weather of the
# parcel that contains it. Pixels in no parcel are nodata.
_DAILY_METRICS = frozenset({"eto", "water_balance"})


def _parcel_pixel_labels(
    parcels: list[dict[str, Any]], transform: Any, rows: int, cols: int,
) -> np.ndarray:
    """Index of the parcel each pixel belongs to, -1 where there is none.

    A pixel centre inside a parcel polygon takes that parcel. Pixels left over
    inside the ~20 m clip margin of a parcel take it too, so a small parcel is
    never left without a pixel; the exact polygon always wins over a margin.
    """
    from shapely.geometry import shape

    polys: list[Any] = []
    for parcel in parcels:
        geom = _parcel_geometry(parcel)
        poly = None
        if geom is not None:
            try:
                poly = shape(geom)
                if not poly.is_valid:
                    poly = poly.buffer(0)
            except Exception:
                poly = None
        polys.append(poly)

    labels = np.full((rows, cols), -1, dtype=int)
    for pass_margin in (0.0, _CLIP_MARGIN_DEG):
        for idx, poly in enumerate(polys):
            if poly is None or poly.is_empty:
                continue
            shape_i = poly.buffer(pass_margin) if pass_margin else poly
            inside = geometry_mask(
                [shape_i], out_shape=(rows, cols), transform=transform, invert=True,
            )
            labels[(labels < 0) & inside] = idx
    return labels


async def _daily_metric_raster(
    tenant_id: str,
    metric: str,
    parcels: list[dict[str, Any]],
    labels: np.ndarray,
    weather_cache: dict[str, dict | None],
    elevations: np.ndarray,
    pixel_lats: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    z: int, x: int, y: int,
) -> np.ndarray | None:
    """eto / water_balance raster, one parcel at a time; None if no pixel is valid."""
    result = np.full(elevations.shape, np.nan, dtype=float)
    for idx in sorted(int(i) for i in np.unique(labels) if i >= 0):
        parcel_id = parcels[idx]["id"]
        key = f"daily:{parcel_id}"
        if key in weather_cache:
            daily = weather_cache[key]
        else:
            daily = await fetch_parcel_daily(tenant_id, parcel_id)
            weather_cache[key] = daily
        if daily is None:
            logger.warning(
                "No /daily weather for parcel %s (tile %d/%d/%d, tenant=%s, metric=%s)",
                parcel_id, z, x, y, tenant_id, metric,
            )
            continue
        mask = labels == idx

        if metric == "water_balance":
            closed5 = daily.get("closed5")
            if not closed5:
                logger.warning(
                    "Skipping water_balance for parcel %s (tenant=%s): fewer than "
                    "5 closed days",
                    parcel_id, tenant_id,
                )
                continue
            result[mask] = compute_water_balance(
                float(closed5["precip_mm"]), float(closed5["et0_mm"]),
            )
            continue

        try:
            t_min = require(daily, "t_min")
            t_max = require(daily, "t_max")
            ea_kpa = require(daily, "ea_kpa")
            u2_ms = require(daily, "u2_ms")
            rs_mj = require(daily, "rs_mj")
            # The altitude the running-day temperatures refer to; the
            # per-pixel correction is RELATIVE to it, never to the DEM grid.
            ref_elevation = require(daily, "elevation_ref_m")
            doy = int(require(daily, "doy"))
        except MissingWeatherInput as exc:
            logger.error(
                "Skipping eto for parcel %s (tile %d/%d/%d, tenant=%s): missing "
                "weather input, tried %s",
                parcel_id, z, x, y, tenant_id, exc.keys,
            )
            continue

        t_min_px = correct_temperature(t_min, ref_elevation, elevations)
        t_max_px = correct_temperature(t_max, ref_elevation, elevations)
        # Horizontal Rs for net shortwave and cloudiness, as zone_accumulator
        # does: the slope/aspect correction is not validated for daily ET0
        # (P5 brings a validated model), so it is not applied.
        eto = compute_eto(
            t_min_px, t_max_px, clamp_ea_to_es(ea_kpa, t_min_px, t_max_px), u2_ms,
            rs_mj, np.full(elevations.shape, rs_mj, dtype=float),
            elevations, pixel_lats, doy,
        )
        result[mask] = eto[mask]

    if not np.any(np.isfinite(result)):
        return None
    return result


# ---------------------------------------------------------------------------
# Per-tile COG computation
# ---------------------------------------------------------------------------


async def generate_cog_for_tile(
    tenant_id: str,
    metric: str,
    z: int,
    x: int,
    y: int,
    date_from: str,
    date_to: str,
    tile_center_lat: float,
    tile_center_lon: float,
    parcels: list[dict[str, Any]],
    weather_cache: dict[str, dict | None] | None = None,
) -> bytes | None:
    """Compute a weather raster for a single TMS tile and return COG bytes.

    Parameters
    ----------
    tenant_id : str
        Tenant identifier.
    metric : str
        Weather metric name (e.g. ``"temperature_avg"``, ``"eto"``).
    z, x, y : int
        TMS tile coordinates.
    date_from, date_to : str
        Date range (``"YYYY-MM-DD"``).
    tile_center_lat, tile_center_lon : float
        Centre of the tile — used to pick the nearest parcel (weather base)
        and, for ``soil_moisture``, the matching soil hydraulics. Not used
        for ``eto`` / ``water_balance``, which read the containing parcel.
    parcels : list[dict]
        Tenant's parcels. For the metrics other than ``eto`` / ``water_balance``
        the tile's weather (and soil) comes from the parcel NEAREST to the tile
        centre, never ``parcels[0]``; ``eto`` / ``water_balance`` use, per
        pixel, the parcel that contains it.
    weather_cache : dict or None
        Per-run cache, keyed by parcel id. The weather is identical for all
        seven metrics and for every tile sharing the same nearest parcel, so
        without it one run issues two Orion GETs per (metric × tile) — each
        on a fresh ``AsyncClient`` — against a broker that deadlocked on
        connection exhaustion. ``None`` means "no sharing": a private cache
        is used, so a lone call still behaves identically.

    Returns
    -------
    bytes or None
        COG GeoTIFF bytes, or ``None`` if the tile could not be computed
        (missing DEM, no nearby parcel, missing weather, or unsupported
        metric).
    """
    # ------------------------------------------------------------------
    # 1. Fetch DEM tile
    # ------------------------------------------------------------------
    dem_data = await fetch_dem_tile(z, x, y)
    if dem_data is None:
        logger.warning("No DEM data for tile %d/%d/%d", z, x, y)
        return None

    elevations = np.array(dem_data.get("elevations", []), dtype=float)
    origin_lon = float(dem_data.get("origin_lon", 0.0))
    origin_lat = float(dem_data.get("origin_lat", 0.0))
    pixel_size_deg = float(
        dem_data.get("pixel_size_deg", dem_data.get("pixel_size", 0.0001))
    )
    cols = int(
        dem_data.get(
            "cols",
            elevations.shape[1] if elevations.ndim == 2 else 0,
        )
    )
    rows = int(
        dem_data.get(
            "rows",
            elevations.shape[0] if elevations.ndim == 2 else 0,
        )
    )

    if elevations.size == 0 or rows < 2 or cols < 2:
        logger.warning(
            "Elevations too small for tile %d/%d/%d (%dx%d)",
            z, x, y, rows, cols,
        )
        return None

    # ------------------------------------------------------------------
    # 2. Slope & aspect (Horn 1981 — nkz_platform_sdk.gis.terrain)
    # ------------------------------------------------------------------
    slope = slope_degrees(elevations, pixel_size_deg)
    aspect = aspect_degrees(elevations, pixel_size_deg)

    # ------------------------------------------------------------------
    # 3. Pixel coordinate grids and raster transform
    # ------------------------------------------------------------------
    lon_grid = np.linspace(
        origin_lon, origin_lon + pixel_size_deg * cols, cols,
    )
    lat_grid = np.linspace(
        origin_lat, origin_lat + pixel_size_deg * rows, rows,
    )
    pixel_lons, pixel_lats = np.meshgrid(lon_grid, lat_grid)
    transform = from_bounds(
        origin_lon, origin_lat,
        origin_lon + pixel_size_deg * cols,
        origin_lat + pixel_size_deg * rows,
        cols, rows,
    )

    if weather_cache is None:
        weather_cache = {}

    parcel_geom = None
    if metric in _DAILY_METRICS:
        # ET0 and the 5-day balance: every pixel takes the /daily weather of
        # the parcel that contains it (no nearest-parcel guess).
        labels = _parcel_pixel_labels(parcels, transform, rows, cols)
        result = await _daily_metric_raster(
            tenant_id, metric, parcels, labels, weather_cache,
            elevations, pixel_lats, slope, aspect, z, x, y,
        )
        if result is None:
            logger.warning(
                "No valid pixel for %s in tile %d/%d/%d (tenant=%s)",
                metric, z, x, y, tenant_id,
            )
            return None
    else:
        # --------------------------------------------------------------
        # 4. Nearest parcel -> its weather from the broker (no defaults)
        # --------------------------------------------------------------
        nearest_parcel = _nearest_parcel(parcels, tile_center_lat, tile_center_lon)
        if nearest_parcel is None:
            logger.warning(
                "No parcel with usable coordinates for tile %d/%d/%d (tenant=%s)",
                z, x, y, tenant_id,
            )
            return None
        parcel_id = nearest_parcel["id"]
        parcel_geom = _parcel_geometry(nearest_parcel)

        if parcel_id in weather_cache:
            weather = weather_cache[parcel_id]
        else:
            # A missing observation is cached too — re-asking Orion once per
            # tile for a parcel that has no weather is the expensive half of
            # the bug.
            weather = await fetch_parcel_weather(tenant_id, parcel_id)
            weather_cache[parcel_id] = weather
        if weather is None:
            logger.warning(
                "No weather observation for parcel %s (tile %d/%d/%d, tenant=%s)",
                parcel_id, z, x, y, tenant_id,
            )
            return None

        try:
            t_avg = require(weather, "t_avg", "temperature_avg")
            t_min = require(weather, "t_min", "temperature_min")
            solar_rad_w_m2 = require(weather, "solar_rad_w_m2", "solar_radiation")
            # The altitude the readings are already downscaled to — the
            # per-pixel correction below is RELATIVE to this, never to the DEM
            # tile's own grid.
            station_elevation = require(weather, "reference_altitude_m")
        except MissingWeatherInput as exc:
            logger.error(
                "Skipping tile %d/%d/%d (tenant=%s, parcel=%s): missing weather input, tried %s",
                z, x, y, tenant_id, parcel_id, exc.keys,
            )
            return None

        # The run's own day (UTC), not the start of a look-back window.
        doy = _date_to_doy(date_to)

        # --------------------------------------------------------------
        # 5. Dispatch to downscaler
        # --------------------------------------------------------------
        if metric == "temperature_avg":
            result = correct_temperature(t_avg, station_elevation, elevations)

        elif metric == "temperature_min":
            result = correct_temperature(t_min, station_elevation, elevations)

        elif metric == "solar_radiation":
            result = correct_solar_radiation(
                solar_rad_w_m2, pixel_lats, aspect, slope, doy,
            )

        elif metric == "frost_risk":
            t_min_corrected = correct_temperature(
                t_min, station_elevation, elevations,
            )
            result = compute_frost_risk(t_min_corrected, elevations)

        elif metric == "soil_moisture":
            ptf = await fetch_parcel_hydraulics(tenant_id, parcel_id)
            if ptf is None:
                logger.warning(
                    "No soil hydraulics for parcel %s (tile %d/%d/%d, tenant=%s): "
                    "soil_moisture unavailable",
                    parcel_id, z, x, y, tenant_id,
                )
                return None
            # The 5-day input comes from the same closed days as the balance.
            key = f"daily:{parcel_id}"
            if key not in weather_cache:
                weather_cache[key] = await fetch_parcel_daily(tenant_id, parcel_id)
            daily = weather_cache[key]
            closed5 = (daily or {}).get("closed5")
            if not closed5:
                logger.warning(
                    "No 5 closed days for parcel %s (tile %d/%d/%d, tenant=%s): "
                    "soil_moisture unavailable",
                    parcel_id, z, x, y, tenant_id,
                )
                return None
            soil_moisture = compute_soil_moisture(
                ptf["awc"], ptf["field_capacity"], ptf["wilting_point"],
                float(closed5["precip_mm"]), float(closed5["et0_mm"]),
            )
            # Broadcast the scalar moisture to match the tile array shape
            result = np.full_like(elevations, soil_moisture)

        else:
            logger.warning(
                "Unknown metric '%s' for tile %d/%d/%d",
                metric, z, x, y,
            )
            return None

    # ------------------------------------------------------------------
    # 6. Write COG to in-memory bytes buffer
    # ------------------------------------------------------------------
    # Clip the raster to the parcel polygon plus a small margin: a hard clip
    # on the exact boundary leaves edge pixels out and the raster does not
    # cover the whole parcel. ~20 m of slack (as the other raster modules do)
    # lets it cover the parcel with room to spare.
    if parcel_geom is not None:
        from shapely.geometry import shape
        poly = shape(parcel_geom)
        if not poly.is_valid:
            poly = poly.buffer(0)
        buffered = poly.buffer(_CLIP_MARGIN_DEG)
        inside = geometry_mask(
            [buffered],
            out_shape=(rows, cols),
            transform=transform,
            invert=True,
        )
        result = np.where(inside, result, np.nan)

    buffer = io.BytesIO()
    with rasterio.open(
        buffer,
        "w",
        driver="COG",
        height=rows,
        width=cols,
        count=1,
        dtype=rasterio.float32,
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=np.nan,
        compress="DEFLATE",
        predictor=3,
    ) as dst:
        dst.write(result.astype(np.float32), 1)

    return buffer.getvalue()


def _clear_all_pointers(tenant_id: str) -> None:
    """A skipped run publishes nothing, so no metric may keep an older raster
    as current."""
    for metric in settings.metrics:
        clear_latest_pointer(tenant_id, metric)


# ---------------------------------------------------------------------------
# Orchestrator helpers
# ---------------------------------------------------------------------------


def _record_balance_window(
    tenant_id: str, today: str, weather_cache: dict[str, dict | None],
) -> None:
    """Record the closed-day window the water_balance raster was built from.

    Read from the same /daily responses the raster used. Parcels normally share
    one window; if they differ the sidecar spans them (earliest start, latest
    end). No parcel with a window: nothing is written, so /stats leaves the
    dates null instead of guessing.
    """
    windows = [
        d["closed5"] for k, d in weather_cache.items()
        if k.startswith("daily:") and d and d.get("closed5")
    ]
    if not windows:
        return
    set_window_meta(tenant_id, "water_balance", {
        "date": today,
        "window_start": min(w["start"] for w in windows),
        "window_end": max(w["end"] for w in windows),
        "days": windows[0]["days"],
    })


async def _generate_and_upload_cogs(
    tenant_id: str,
    parcels: list[dict[str, Any]],
    date_from: str,
    date_to: str,
    zoom: int,
    today: str,
) -> None:
    """Generate COGs for all metrics/tiles covering the parcels' aggregate bbox.

    Extracted from ``run_for_tenant`` to allow independent testing/patching.
    """
    # Enumerate tiles from the parcel POLYGONS, not their centroids: a parcel
    # that spans a tile boundary was only partly covered when the aggregate
    # bbox was built from centroids — the tile covering the other half of the
    # parcel was never generated.
    from shapely.geometry import shape as _shape
    bounds_list = []
    for p in parcels:
        geom = _parcel_geometry(p)
        if geom is None:
            continue
        try:
            b = _shape(geom).bounds  # (min_lon, min_lat, max_lon, max_lat)
        except Exception:
            continue
        if b:
            bounds_list.append(b)
    if not bounds_list:
        logger.warning(
            "Tenant '%s': no usable parcel geometry, skipping COG generation; "
            "clearing raster pointers",
            tenant_id,
        )
        _clear_all_pointers(tenant_id)
        return
    min_lon = min(b[0] for b in bounds_list)
    min_lat = min(b[1] for b in bounds_list)
    max_lon = max(b[2] for b in bounds_list)
    max_lat = max(b[3] for b in bounds_list)
    logger.info(
        "Parcel bbox for tenant '%s': %.4f, %.4f, %.4f, %.4f",
        tenant_id, min_lon, min_lat, max_lon, max_lat,
    )

    tiles = _bbox_to_tiles(min_lon, min_lat, max_lon, max_lat, zoom=zoom)
    logger.info(
        "Tenant '%s': %d tiles at zoom %d",
        tenant_id, len(tiles), zoom,
    )

    # One broker read per parcel for the whole run, shared across every
    # metric and every tile that resolves to the same nearest parcel.
    weather_cache: dict[str, dict | None] = {}

    for metric in settings.metrics:
        if metric in UNAVAILABLE_METRICS:
            # Not published: clear any old pointer so it is not served as
            # current. fetch_parcel_hydraulics and the PTF are kept for reuse
            # but this job must not call them.
            clear_latest_pointer(tenant_id, metric)
            continue
        logger.info(
            "Generating COGs for metric '%s' / tenant '%s'",
            metric, tenant_id,
        )

        success_count = 0
        total = len(tiles)

        for idx, (z_tile, x_tile, y_tile) in enumerate(tiles):
            min_lon_t, min_lat_t, max_lon_t, max_lat_t = _tile_to_bbox(
                z_tile, x_tile, y_tile,
            )
            tile_center_lon = (min_lon_t + max_lon_t) / 2.0
            tile_center_lat = (min_lat_t + max_lat_t) / 2.0

            cog_bytes = await generate_cog_for_tile(
                tenant_id, metric, z_tile, x_tile, y_tile,
                date_from, date_to, tile_center_lat, tile_center_lon,
                parcels=parcels,
                weather_cache=weather_cache,
            )

            if cog_bytes is not None:
                ok = upload_cog(
                    cog_bytes, tenant_id, metric, today,
                    z_tile, x_tile, y_tile,
                )
                if ok:
                    success_count += 1

            if (idx + 1) % 10 == 0:
                logger.info(
                    "  [%s] tile %d/%d (%d/%d/%d) — %d/%d succeeded",
                    metric, idx + 1, total, z_tile, x_tile, y_tile,
                    success_count, idx + 1,
                )

        logger.info(
            "Metric '%s': %d/%d tiles succeeded for tenant '%s'",
            metric, success_count, total, tenant_id,
        )

        if success_count > 0:
            set_latest_date(tenant_id, metric, today)
            if metric == "water_balance":
                _record_balance_window(tenant_id, today, weather_cache)
        else:
            # No tiles today: do not keep serving an older raster as current.
            clear_latest_pointer(tenant_id, metric)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_for_tenant(
    tenant_id: str,
    parcels: list[dict[str, Any]],
    date_from: str,
    date_to: str,
    zoom: int = 14,
) -> None:
    """Run COG generation for all metrics and all tiles covering a tenant's parcels,
    then persist one AgriParcelRecord per parcel with zonal stats.

    Parameters
    ----------
    tenant_id : str
        Tenant identifier.
    parcels : list[dict]
        List of parcel dicts, each with ``id``, ``lon``, ``lat`` keys
        (or ``longitude`` / ``latitude``).
    date_from, date_to : str
        Date range (``"YYYY-MM-DD"``).
    zoom : int
        TMS zoom level (default 14).
    """
    # 1. MinIO bucket — assume it exists (created by admin).
    # skip ensure_bucket() to avoid spurious AccessDenied.

    # 2. Today's date string (used for the COG path)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 3. Guard: no parcels
    if not parcels:
        logger.warning(
            "No parcels for tenant '%s' — skipping COG generation; "
            "clearing raster pointers",
            tenant_id,
        )
        _clear_all_pointers(tenant_id)
        return

    # 4. Generate and upload COGs for all metrics / tiles
    await _generate_and_upload_cogs(
        tenant_id, parcels, date_from, date_to, zoom, today,
    )

    # 5. Persist one AgriParcelRecord per parcel with zonal stats
    observed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for parcel in parcels:
        parcel_id = parcel["id"]
        geometry = _parcel_geometry(parcel)
        if geometry is None:
            logger.warning(
                "No usable geometry for parcel %s — skipping record", parcel_id,
            )
            continue

        stats = compute_zonal_stats(tenant_id, geometry, settings.metrics)

        # A metric with no tile today has its pointer cleared, so zonal stats
        # carry no date (or an older one if a clear failed). Never publish a
        # record whose stats are not from today's raster: stale data under a
        # fresh `observedAt` is worse than a gap, which is visible downstream.
        # Spec §5.3: a record with an incomplete input set is not published.
        stats_date = stats.get("date")
        if stats_date != today:
            logger.error(
                "Not publishing AgriParcelRecord for parcel %s (tenant=%s): "
                "zonal stats resolved to date=%r, not today's %s — the record "
                "would republish stale pixels under a fresh observedAt",
                parcel_id, tenant_id, stats_date, today,
            )
            continue

        flat_metrics: dict[str, float] = {}
        for metric_name, metric_stats in stats.get("metrics", {}).items():
            if "error" in metric_stats or "mean" not in metric_stats:
                continue
            flat_metrics[metric_name] = metric_stats["mean"]

        record = build_agri_parcel_record(
            tenant_id=tenant_id,
            parcel_id=parcel_id,
            geometry=geometry,
            metrics=flat_metrics,
            observed_at=observed_at,
        )
        await guard_frozen_metrics(tenant_id, parcel_id, flat_metrics)
        await upsert_record(tenant_id, record)
        logger.info(
            "Persisted AgriParcelRecord for parcel %s / tenant %s",
            parcel_id, tenant_id,
        )
