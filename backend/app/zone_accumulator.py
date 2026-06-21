"""Daily zone accumulator — generates AgriParcelZone entities per parcel.

Runs as a K8s CronJob: fetches all parcels for a tenant, computes
zones from DEM, downscales Tmin/Tmax per zone, and upserts
AgriParcelZone entities to Orion-LD.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import geohash2
import numpy as np

from app.config import settings
from app.downscaler import compute_zones, correct_temperature, compute_eto
from app.records import build_agri_parcel_zone
from app.sources import (
    fetch_dem_tile,
    fetch_station_weather,
    fetch_tenant_parcels,
    find_nearby_sensors,
    upsert_agri_parcel_zones,
)
from app.cog_generator import _bbox_to_tiles, _parcel_geometry, _parcel_lonlat
from app.tenants import discover_tenants

logger = logging.getLogger(__name__)

PIXEL_AREA_HA = 0.01  # ~10m x 10m per pixel at zoom 14

_SECTOR_NAMES = ["flat", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _compute_area_ha(pixel_count: int) -> float:
    """Convert pixel count to hectares (10m resolution → 0.01 ha/pixel)."""
    return pixel_count * PIXEL_AREA_HA


def _gradient(elev: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute dz/dx and dz/dy (central differences) of a 2-D elevation array."""
    dzdx = np.zeros_like(elev, dtype=float)
    dzdy = np.zeros_like(elev, dtype=float)
    if elev.shape[0] < 3 or elev.shape[1] < 3:
        return dzdx, dzdy
    scale = 2.0 * resolution
    dzdx[1:-1, 1:-1] = (elev[1:-1, 2:] - elev[1:-1, :-2]) / scale
    dzdy[1:-1, 1:-1] = (elev[2:, 1:-1] - elev[:-2, 1:-1]) / scale
    return dzdx, dzdy


def _geom_geojson_bbox(geometry: dict) -> tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) from a GeoJSON geometry."""
    try:
        coords = geometry["coordinates"]
        if geometry["type"] == "Polygon":
            flat = [c for ring in coords for point in ring for c in point]
        elif geometry["type"] == "MultiPolygon":
            flat = [c for poly in coords for ring in poly for point in ring for c in point]
        else:
            return (0.0, 0.0, 0.0, 0.0)
        lons = flat[0::2]
        lats = flat[1::2]
        return min(lons), min(lats), max(lons), max(lats)
    except (KeyError, IndexError, TypeError):
        return (0.0, 0.0, 0.0, 0.0)


async def process_parcel(
    tenant_id: str,
    parcel: dict[str, Any],
    date_from: str,
) -> list[dict]:
    """Compute zones for a single parcel and return AgriParcelZone entities."""
    parcel_id = parcel["id"]
    geometry = _parcel_geometry(parcel)
    if geometry is None:
        logger.warning("No geometry for parcel %s, skipping", parcel_id)
        return []

    lonlat = _parcel_lonlat(parcel)
    if lonlat is None:
        return []
    centroid_lon, centroid_lat = lonlat

    # 1. Tiles covering the parcel
    min_lon, min_lat, max_lon, max_lat = _geom_geojson_bbox(geometry)
    tiles = _bbox_to_tiles(min_lon, min_lat, max_lon, max_lat, zoom=14)

    # 2. Fetch DEM per tile, compute slope/aspect
    tile_data: list[dict] = []
    for z, x, y in tiles:
        dem = await fetch_dem_tile(z, x, y)
        if dem is None:
            continue
        elev = np.array(dem.get("elevations", []), dtype=float)
        if elev.size == 0:
            continue
        pixel_size = float(dem.get("pixel_size_deg", 0.0001))
        dzdx, dzdy = _gradient(elev, pixel_size)
        slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
        aspect = np.degrees(np.arctan2(-dzdx, dzdy))
        aspect = np.where(aspect < 0, aspect + 360, aspect)
        np.nan_to_num(slope, nan=0.0, copy=False)
        np.nan_to_num(aspect, nan=0.0, copy=False)
        tile_data.append({
            "elev": elev,
            "slope": slope,
            "aspect": aspect,
            "origin_lon": float(dem.get("origin_lon", 0)),
            "origin_lat": float(dem.get("origin_lat", 0)),
            "pixel_size": pixel_size,
            "rows": int(dem.get("rows", elev.shape[0])),
            "cols": int(dem.get("cols", elev.shape[1])),
        })

    if not tile_data:
        logger.warning("No DEM data for parcel %s", parcel_id)
        return []

    # 3. Zones per tile
    zones_all: list[dict] = []
    for td in tile_data:
        rows, cols = td["rows"], td["cols"]
        lon = np.linspace(td["origin_lon"], td["origin_lon"] + td["pixel_size"] * cols, cols)
        lat = np.linspace(td["origin_lat"], td["origin_lat"] + td["pixel_size"] * rows, rows)
        lon_g, lat_g = np.meshgrid(lon, lat)

        tile_zones, tile_labels = compute_zones(
            td["elev"], td["aspect"], td["slope"],
            min_pixels=settings.zones_min_pixels,
            elevation_band_m=settings.zones_elevation_band_m,
        )

        for label in range(1, tile_labels.max() + 1):
            mask = (tile_labels == label)
            if np.sum(mask) == 0:
                continue
            zone_lon = float(np.mean(lon_g[mask]))
            zone_lat = float(np.mean(lat_g[mask]))
            elev_mean = float(np.mean(td["elev"][mask]))
            elev_band = int(elev_mean / settings.zones_elevation_band_m)
            sector_idx = int(np.round(np.mean(td["aspect"][mask]) / 45)) % 8 + 1
            if sector_idx < 0 or sector_idx >= len(_SECTOR_NAMES):
                sector_idx = 0
            gh = geohash2.encode(zone_lat, zone_lon, precision=7)
            zone_id = f"z{gh}-e{elev_band}-{_SECTOR_NAMES[sector_idx]}"

            zones_all.append({
                "id": zone_id,
                "elevationMean": round(elev_mean, 1),
                "elevationMin": round(float(np.min(td["elev"][mask])), 1),
                "elevationMax": round(float(np.max(td["elev"][mask])), 1),
                "aspectSector": _SECTOR_NAMES[sector_idx],
                "pixelCount": int(np.sum(mask)),
                "centroid": [round(zone_lon, 6), round(zone_lat, 6)],
            })

    if not zones_all:
        logger.warning("No zones computed for parcel %s", parcel_id)
        return []

    # Cap zones
    if len(zones_all) > settings.zones_max_count:
        zones_all.sort(key=lambda z: z["pixelCount"], reverse=True)
        zones_all = zones_all[:settings.zones_max_count]

    # 4. Fetch weather
    weather = await fetch_station_weather(
        tenant_id, centroid_lat, centroid_lon, date_from, date_from,
    )
    if weather is None:
        logger.warning("No weather data for parcel %s", parcel_id)
        return []

    t_min = float(weather.get("t_min", weather.get("temperature_min", 10.0)))
    t_max = float(weather.get("t_max", weather.get("temperature_max", 20.0)))
    station_elev = float(weather.get("elevation_m", weather.get("elevation", 0.0)))
    rh = float(weather.get("rh_avg", weather.get("humidity", 60.0)))
    wind = float(weather.get("wind_speed_ms", weather.get("wind_speed", 2.0)))
    solar = float(weather.get("solar_rad_w_m2", weather.get("solar_radiation", 200.0)))

    # 5. Compute metrics per zone, build entities
    observed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entity_list: list[dict] = []
    for zd in zones_all:
        elev_mean = zd["elevationMean"]
        t_min_z = float(correct_temperature(t_min, station_elev, np.array([elev_mean]))[0])
        t_max_z = float(correct_temperature(t_max, station_elev, np.array([elev_mean]))[0])
        t_avg_z = (t_min_z + t_max_z) / 2.0
        eto = compute_eto(
            np.array([t_avg_z]), np.array([t_min_z]), np.array([t_max_z]),
            np.array([solar]), wind, rh, np.array([elev_mean]),
        )
        metrics = {
            "tMin": round(t_min_z, 1),
            "tMax": round(t_max_z, 1),
            "eto": round(float(eto[0]), 2),
        }
        area_ha = _compute_area_ha(zd["pixelCount"])
        entity = build_agri_parcel_zone(
            tenant_id=tenant_id,
            parcel_id=parcel_id,
            zone=zd,
            geometry=geometry,
            metrics=metrics,
            observed_at=observed_at,
            area_ha=area_ha,
            sensor_nearby=zd.get("sensor_nearby"),
            sensor_distance_m=zd.get("sensor_distance_m"),
        )
        entity_list.append(entity)

    # 6. Enrich with sensors
    if entity_list:
        entity_list = await find_nearby_sensors(tenant_id, parcel_id, entity_list)

    return entity_list


async def run_for_tenant(tenant_id: str) -> None:
    """Run the full zone pipeline for a single tenant."""
    parcels = await fetch_tenant_parcels(tenant_id)
    if not parcels:
        logger.info("No parcels for tenant %s, skipping", tenant_id)
        return

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    all_entities: list[dict] = []

    for parcel in parcels:
        entities = await process_parcel(tenant_id, parcel, today)
        if entities:
            all_entities.extend(entities)

    if all_entities:
        await upsert_agri_parcel_zones(tenant_id, all_entities)
        logger.info(
            "Upserted %d zone entities for tenant %s",
            len(all_entities), tenant_id,
        )


async def main() -> None:
    """CLI entry point — called by the CronJob."""
    logging.basicConfig(level=logging.INFO)
    tenants = discover_tenants("MONITORED_TENANTS")
    if not tenants:
        logger.error("zone-accumulator: no tenants to process")
        return
    for tenant_id in tenants:
        try:
            await run_for_tenant(tenant_id)
        except Exception:
            logger.exception("Zone generation failed for tenant %s", tenant_id)


if __name__ == "__main__":
    asyncio.run(main())
