"""COG generator — weather raster computation and MinIO upload.

Runs periodically (every 5 days) to produce per-pixel weather rasters
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
    compute_water_balance,
    compute_frost_risk,
    compute_soil_moisture,
    saxton_rawls_ptf,
    get_texture_defaults,
)
from app.minio_io import set_latest_date, upload_cog
from app.records import build_agri_parcel_record, guard_frozen_metrics
from app.sources import (
    MissingWeatherInput,
    fetch_agri_soil,
    fetch_dem_tile,
    fetch_parcel_weather,
    require,
    upsert_record,
)
from app.stats import compute_zonal_stats

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
        and, for ``soil_moisture``, the matching AgriSoil entity.
    parcels : list[dict]
        Tenant's parcels. The tile's weather (and soil) comes from the
        parcel NEAREST to the tile centre, never ``parcels[0]``.
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
    # 3. Nearest parcel → its weather from the broker (no defaults)
    # ------------------------------------------------------------------
    nearest_parcel = _nearest_parcel(parcels, tile_center_lat, tile_center_lon)
    if nearest_parcel is None:
        logger.warning(
            "No parcel with usable coordinates for tile %d/%d/%d (tenant=%s)",
            z, x, y, tenant_id,
        )
        return None
    parcel_id = nearest_parcel["id"]
    parcel_geom = _parcel_geometry(nearest_parcel)

    if weather_cache is None:
        weather_cache = {}
    if parcel_id in weather_cache:
        weather = weather_cache[parcel_id]
    else:
        # A missing observation is cached too — re-asking Orion once per tile
        # for a parcel that has no weather is the expensive half of the bug.
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
        t_max = require(weather, "t_max", "temperature_max")
        rh_avg = require(weather, "rh_avg", "humidity")
        wind_speed_ms = require(weather, "wind_speed_ms", "wind_speed")
        solar_rad_w_m2 = require(weather, "solar_rad_w_m2", "solar_radiation")
        precip_mm = require(weather, "precip_mm", "precipitation")
        # The altitude the readings are already downscaled to — the per-pixel
        # correction below is RELATIVE to this, never to the DEM tile's own grid.
        station_elevation = require(weather, "reference_altitude_m")
    except MissingWeatherInput as exc:
        logger.error(
            "Skipping tile %d/%d/%d (tenant=%s, parcel=%s): missing weather input, tried %s",
            z, x, y, tenant_id, parcel_id, exc.keys,
        )
        return None

    doy = (
        int(weather["doy"])
        if "doy" in weather
        else _date_to_doy(date_from)
    )

    # ------------------------------------------------------------------
    # 4. Pixel coordinate grids
    # ------------------------------------------------------------------
    lon_grid = np.linspace(
        origin_lon, origin_lon + pixel_size_deg * cols, cols,
    )
    lat_grid = np.linspace(
        origin_lat, origin_lat + pixel_size_deg * rows, rows,
    )
    pixel_lons, pixel_lats = np.meshgrid(lon_grid, lat_grid)

    # ------------------------------------------------------------------
    # 5. Dispatch to downscaler
    # ------------------------------------------------------------------
    if metric == "temperature_avg":
        result = correct_temperature(t_avg, station_elevation, elevations)

    elif metric == "temperature_min":
        result = correct_temperature(t_min, station_elevation, elevations)

    elif metric == "solar_radiation":
        result = correct_solar_radiation(
            solar_rad_w_m2, pixel_lats, aspect, slope, doy,
        )

    elif metric == "eto":
        t_avg_corrected = correct_temperature(
            t_avg, station_elevation, elevations,
        )
        t_min_corrected = correct_temperature(
            t_min, station_elevation, elevations,
        )
        t_max_corrected = correct_temperature(
            t_max, station_elevation, elevations,
        )
        rad_corrected = correct_solar_radiation(
            solar_rad_w_m2, pixel_lats, aspect, slope, doy,
        )
        result = compute_eto(
            t_avg_corrected, t_min_corrected, t_max_corrected,
            rad_corrected, wind_speed_ms, rh_avg, elevations,
        )

    elif metric == "water_balance":
        t_avg_corrected = correct_temperature(
            t_avg, station_elevation, elevations,
        )
        t_min_corrected = correct_temperature(
            t_min, station_elevation, elevations,
        )
        t_max_corrected = correct_temperature(
            t_max, station_elevation, elevations,
        )
        rad_corrected = correct_solar_radiation(
            solar_rad_w_m2, pixel_lats, aspect, slope, doy,
        )
        eto_daily = compute_eto(
            t_avg_corrected, t_min_corrected, t_max_corrected,
            rad_corrected, wind_speed_ms, rh_avg, elevations,
        )
        # Aggregate ET₀ to a single mean scalar × 5 days
        eto_5d = float(np.nanmean(eto_daily)) * 5
        result = compute_water_balance(
            precip_mm * 5, np.full_like(elevations, eto_5d),
        )

    elif metric == "frost_risk":
        t_min_corrected = correct_temperature(
            t_min, station_elevation, elevations,
        )
        result = compute_frost_risk(t_min_corrected, elevations)

    elif metric == "soil_moisture":
        soil = await fetch_agri_soil(tenant_id, parcel_id)
        if soil is not None and soil.get("sand_pct") is not None:
            ptf = saxton_rawls_ptf(soil["sand_pct"], soil["clay_pct"])
        else:
            logger.warning(
                "No AgriSoil for %s; using texture defaults", parcel_id
            )
            defaults = get_texture_defaults()
            ptf = saxton_rawls_ptf(
                defaults["sand_pct"], defaults["clay_pct"],
            )

        t_avg_corrected = correct_temperature(
            t_avg, station_elevation, elevations,
        )
        t_min_corrected = correct_temperature(
            t_min, station_elevation, elevations,
        )
        t_max_corrected = correct_temperature(
            t_max, station_elevation, elevations,
        )
        rad_corrected = correct_solar_radiation(
            solar_rad_w_m2, pixel_lats, aspect, slope, doy,
        )
        eto_daily = compute_eto(
            t_avg_corrected, t_min_corrected, t_max_corrected,
            rad_corrected, wind_speed_ms, rh_avg, elevations,
        )
        eto_5d = float(np.nanmean(eto_daily)) * 5

        soil_moisture = compute_soil_moisture(
            ptf["awc"], ptf["field_capacity"], ptf["wilting_point"],
            precip_mm * 5, eto_5d,
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
    transform = from_bounds(
        origin_lon, origin_lat,
        origin_lon + pixel_size_deg * cols,
        origin_lat + pixel_size_deg * rows,
        cols, rows,
    )

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


# ---------------------------------------------------------------------------
# Orchestrator helpers
# ---------------------------------------------------------------------------


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
            "Tenant '%s': no usable parcel geometry, skipping COG generation",
            tenant_id,
        )
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
            "No parcels for tenant '%s' — skipping COG generation",
            tenant_id,
        )
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

        # Weather can now legitimately be absent: when every tile of a metric
        # returns None, `set_latest_date` never advances and this call — which
        # passes no date — resolves to the PREVIOUS pointer. Publishing that
        # would restate yesterday's pixels under a fresh `observedAt`. Stale
        # data presented as current is worse than a gap: the gap is visible
        # downstream, this is not. Spec §5.3: a record with an incomplete
        # input set is not published.
        stats_date = stats.get("date")
        if stats_date != today:
            logger.error(
                "Not publishing AgriParcelRecord for parcel %s (tenant=%s): "
                "zonal stats resolved to date=%r, not today's %s — the COG "
                "pointer did not advance, so the record would republish stale "
                "pixels under a fresh observedAt",
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

    # ── Generate PMTiles from the newly computed COGs ────────────────
    try:
        from app.pmtiles_generator import generate_all_pmtiles as _gen_pmtiles
        pmtiles_urls = _gen_pmtiles(tenant_id, today, zoom)
        logger.info("PMTiles generated for tenant '%s': %s", tenant_id, pmtiles_urls)
    except Exception as exc:
        logger.error("PMTiles generation failed for tenant '%s': %s", tenant_id, exc)
