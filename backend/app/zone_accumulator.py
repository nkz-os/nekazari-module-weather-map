"""Daily zone accumulator — generates AgriParcelZone entities per parcel.

Runs as a K8s CronJob: fetches all parcels for a tenant, computes
zones from DEM, downscales Tmin/Tmax per zone, and upserts
AgriParcelZone entities to Orion-LD.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Any

import geohash2
import numpy as np

from app.config import settings
from app.dem_grid import dem_transform, north_edge, pixel_center_grids, row_height
from nkz_platform_sdk.gis.terrain import aspect_degrees, slope_degrees
from rasterio.features import geometry_mask
from app.downscaler import compute_zones, discretize_aspect, correct_temperature, compute_eto, clamp_ea_to_es
from app.records import build_agri_parcel_zone
from app.slope_radiation import slope_radiation_wm2
from nkz_platform_sdk import OrionClient
from app.sources import (
    MissingWeatherInput,
    fetch_dem_tile,
    fetch_parcel_daily,
    fetch_tenant_parcels,
    find_nearby_sensors,
    require,
    upsert_agri_parcel_zones,
    COG_CHECKED_KEY,
)
from app.cog_generator import _bbox_to_tiles, _parcel_geometry, _parcel_lonlat
from app.tenants import discover_tenants

logger = logging.getLogger(__name__)

_M_PER_DEG = 111_320.0  # same constant as _geometry_area_ha and the SDK terrain maths

_TERRAIN_KEY = "_terrain_px"

_SECTOR_NAMES = ["flat", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _dedupe_zone_ids(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return zones with unique ids, deterministically.

    Distinct zones can share geohash, elevation band and aspect sector. Within a
    group of equal ids, members are ordered north first, ties west to east; the
    first keeps the bare id and the rest get ``-2``, ``-3``...
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for z in zones:
        groups.setdefault(z["id"], []).append(z)
    renamed: dict[int, str] = {}
    for zid, members in groups.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda z: (-z["centroid"][1], z["centroid"][0]))
        for n, z in enumerate(members[1:], start=2):
            renamed[id(z)] = f"{zid}-{n}"
    if not renamed:
        return zones
    return [{**z, "id": renamed[id(z)]} if id(z) in renamed else z for z in zones]


def _assign_zone_ranks(zones: list[dict[str, Any]]) -> dict[str, int]:
    """1-based display rank by zone id: north first, ties west to east."""
    deduped = _dedupe_zone_ids(zones)
    if deduped is not zones:
        logger.error("Duplicate zone ids reached rank assignment; de-duplicating")
    ordered = sorted(deduped, key=lambda z: (-z["centroid"][1], z["centroid"][0]))
    return {z["id"]: i for i, z in enumerate(ordered, start=1)}


def _compute_area_ha(pixel_lats: np.ndarray, width_deg: float, height_deg: float) -> float:
    """Hectares covered by pixels centred at *pixel_lats* (degrees).

    A degree of longitude shrinks with cos(latitude), so a pixel is
    ``width * m_per_deg * cos(lat)`` by ``height * m_per_deg`` metres.
    """
    lats = np.asarray(pixel_lats, dtype=float)
    dx_m = width_deg * _M_PER_DEG * np.cos(np.radians(lats))
    dy_m = height_deg * _M_PER_DEG
    return float(np.sum(dx_m * dy_m) / 10_000.0)


# Smoothing window (pixels) for zoning terrain. IGN serves whole metres, so a
# gentle ramp is a staircase of 1 m steps; Horn on it gives > 2 deg at every
# step and the zones become stripes. A 7x7 mean spreads a step over ~70 m and
# leaves a true linear slope unchanged.
_TERRAIN_SMOOTH_PX = 7


def _terrain(
    elev: np.ndarray, pixel_size_deg: float, latitude_deg: float, pixel_height_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Slope and aspect (degrees, Horn 1981 in metres) of the smoothed DEM; rows north first.

    ``latitude_deg`` (grid centre) scales the east-west cell size;
    ``pixel_height_deg`` is the row height (it differs from the column width).
    """
    from scipy.ndimage import uniform_filter
    smooth = uniform_filter(elev.astype(float), size=_TERRAIN_SMOOTH_PX, mode="nearest")
    return (
        slope_degrees(smooth, pixel_size_deg, latitude_deg, pixel_height_deg),
        aspect_degrees(smooth, pixel_size_deg, latitude_deg, pixel_height_deg),
    )


def _repair_geometry(geometry: dict[str, Any]) -> dict[str, Any] | None:
    """Valid polygonal parcel geometry, or None when nothing polygonal is left."""
    from shapely.geometry import mapping, shape
    from shapely.validation import make_valid
    try:
        geom = shape(geometry)
        if not geom.is_valid:
            geom = make_valid(geom)
            if geom.geom_type == "GeometryCollection":
                from shapely.ops import unary_union
                geom = unary_union([g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")])
    except Exception:
        logger.exception("Parcel geometry could not be read")
        return None
    if geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    return mapping(geom)


def _geometry_area_ha(geometry: dict[str, Any]) -> float:
    """Area in hectares of a lon/lat GeoJSON polygon (local equirectangular, parcel scale)."""
    from shapely.geometry import shape
    geom = shape(geometry)
    if geom.is_empty:
        return 0.0
    m_per_deg = 111_320.0
    lat = geom.centroid.y
    return geom.area * m_per_deg * m_per_deg * math.cos(math.radians(lat)) / 10_000.0


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


def _zone_geometry(
    mask: np.ndarray,
    transform,
    parcel_geom: dict[str, Any] | None,
    simplify_tol: float,
) -> dict[str, Any]:
    """Polygonize a zone pixel mask, clip to the parcel, return GeoJSON.

    A raw per-pixel polygonization emits thousands of tiny rings; unary_union +
    simplify collapses them (same pattern as hydrology ``_compute_zones``). The
    result is intersected with the parcel polygon so zones never bleed outside
    the parcel boundary.
    """
    from rasterio.features import shapes as _rio_shapes
    from shapely.geometry import shape as _shp_shape, mapping as _shp_mapping
    from shapely.ops import unary_union as _union

    polys = [
        _shp_shape(g)
        for g, v in _rio_shapes(mask.astype("uint8"), mask=mask, transform=transform)
        if v == 1
    ]
    if not polys:
        return {}

    merged = _union(polys)
    if merged.is_empty:
        return {}

    # Smooth the pixel staircase first (topology kept: no spikes), then clip so
    # the zone follows the parcel boundary exactly.
    merged = merged.simplify(tolerance=simplify_tol / 2.0, preserve_topology=True)

    if parcel_geom and parcel_geom.get("type") in ("Polygon", "MultiPolygon"):
        try:
            merged = merged.intersection(_shp_shape(parcel_geom))
        except Exception:
            logger.exception("Zone clip to parcel failed")
            return {}
        if merged.is_empty:
            return {}

    if merged.geom_type == "GeometryCollection":
        parts = [g for g in merged.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        merged = _union(parts) if parts else merged
    if merged.is_empty or merged.geom_type not in ("Polygon", "MultiPolygon"):
        return {}
    return _shp_mapping(merged)


async def process_parcel(
    tenant_id: str,
    parcel: dict[str, Any],
    date_from: str,
) -> list[dict]:
    """Compute zones for a single parcel and return AgriParcelZone entities.

    ``[]`` = computed, the parcel has no zones (its old zones go).
    ``None`` = undecided (no geometry, DEM or weather): its zones are kept.
    """
    parcel_id = parcel["id"]
    geometry = _parcel_geometry(parcel)
    if geometry is not None:
        geometry = _repair_geometry(geometry)
    if geometry is None:
        logger.warning("No geometry for parcel %s, skipping", parcel_id)
        return None

    lonlat = _parcel_lonlat(parcel)
    if lonlat is None:
        return None

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
        rows_n = int(dem.get("rows", elev.shape[0]))
        grid_meta = {
            "origin_lon": float(dem.get("origin_lon", 0)),
            "origin_lat": float(dem.get("origin_lat", 0)),
            "north_lat": dem.get("north_lat"), "pixel_size_deg": pixel_size,
            "rows": rows_n, "cols": int(dem.get("cols", elev.shape[1])),
        }
        height_deg = row_height(grid_meta)
        centre_lat = north_edge(grid_meta) - height_deg * rows_n / 2.0
        slope, aspect = _terrain(elev, pixel_size, centre_lat, height_deg)
        tile_data.append({
            "elev": elev,
            "slope": slope,
            "aspect": aspect,
            "origin_lon": float(dem.get("origin_lon", 0)),
            "origin_lat": float(dem.get("origin_lat", 0)),
            "north_lat": dem.get("north_lat"),
            "pixel_size": pixel_size,
            "pixel_height": height_deg,
            "rows": rows_n,
            "cols": int(dem.get("cols", elev.shape[1])),
        })

    if not tile_data:
        logger.warning("No DEM data for parcel %s", parcel_id)
        return None

    # 3. Zones per tile
    zones_all: list[dict] = []
    for td in tile_data:
        grid = {
            "origin_lon": td["origin_lon"], "origin_lat": td["origin_lat"],
            "north_lat": td["north_lat"], "pixel_size_deg": td["pixel_size"],
            "rows": td["rows"], "cols": td["cols"],
        }
        lon_g, lat_g = pixel_center_grids(grid)

        _, tile_labels = compute_zones(
            td["elev"], td["aspect"], td["slope"],
            min_pixels=settings.zones_min_pixels,
            elevation_band_m=settings.zones_elevation_band_m,
        )

        tile_transform = dem_transform(grid)
        # Zone statistics describe the part inside the parcel, not the tile-wide zone.
        in_parcel = geometry_mask(
            [geometry], out_shape=tile_labels.shape, transform=tile_transform,
            invert=True, all_touched=True,
        )

        for label in range(1, tile_labels.max() + 1):
            tile_mask = (tile_labels == label)
            mask = tile_mask & in_parcel
            if not mask.any():
                continue
            zone_geom = _zone_geometry(tile_mask, tile_transform, geometry, td["pixel_size"])
            if not zone_geom or _geometry_area_ha(zone_geom) < settings.zones_min_parcel_ha:
                continue
            zone_lon = float(np.mean(lon_g[mask]))
            zone_lat = float(np.mean(lat_g[mask]))
            elev_mean = float(np.mean(td["elev"][mask]))
            elev_band = int(elev_mean / settings.zones_elevation_band_m)
            # Dominant discretised sector (flat below 2 deg), not the mean angle:
            # a flat zone has no aspect and averaging angles wraps at north.
            sectors = discretize_aspect(td["aspect"][mask], td["slope"][mask])
            sector_idx = int(np.bincount(sectors.astype(int), minlength=len(_SECTOR_NAMES)).argmax())
            gh = geohash2.encode(zone_lat, zone_lon, precision=7)
            zone_id = f"z{gh}-e{elev_band}-{_SECTOR_NAMES[sector_idx]}"

            zones_all.append({
                "id": zone_id,
                "elevationMean": round(elev_mean, 1),
                "elevationMin": round(float(np.min(td["elev"][mask])), 1),
                "elevationMax": round(float(np.max(td["elev"][mask])), 1),
                "aspectSector": _SECTOR_NAMES[sector_idx],
                "pixelCount": int(np.sum(mask)),
                "areaHa": round(_compute_area_ha(lat_g[mask], td["pixel_size"], td["pixel_height"]), 2),
                "centroid": [round(zone_lon, 6), round(zone_lat, 6)],
                "geometry": zone_geom,
                # Private (never published): the zone's own pixels for the slope
                # radiation model; see _zone_radiation.
                _TERRAIN_KEY: (td["slope"][mask].copy(), td["aspect"][mask].copy()),
            })

    if not zones_all:
        logger.warning("No zones computed for parcel %s", parcel_id)
        return []

    # Cap zones
    if len(zones_all) > settings.zones_max_count:
        zones_all.sort(key=lambda z: z["pixelCount"], reverse=True)
        zones_all = zones_all[:settings.zones_max_count]

    zones_all = _dedupe_zone_ids(zones_all)
    zone_ranks = _assign_zone_ranks(zones_all)

    # 4. Fetch the running day's weather from weather-api /daily (no defaults —
    # see app.sources.require)
    weather = await fetch_parcel_daily(tenant_id, parcel_id)
    if weather is None:
        logger.warning("No /daily weather for parcel %s", parcel_id)
        return None

    try:
        t_min = require(weather, "t_min")
        t_max = require(weather, "t_max")
        ea_kpa = require(weather, "ea_kpa")
        u2_ms = require(weather, "u2_ms")
        rs_mj = require(weather, "rs_mj")
        doy = int(require(weather, "doy"))
        # The altitude the readings are already downscaled to — the per-zone
        # correction below is RELATIVE to this, never to the DEM tile's own grid.
        station_elev = require(weather, "elevation_ref_m")
    except MissingWeatherInput as exc:
        logger.error(
            "Skipping parcel %s (tenant=%s): missing weather input, tried %s",
            parcel_id, tenant_id, exc.keys,
        )
        return None

    # 5. Compute metrics per zone, build entities
    observed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entity_list: list[dict] = []
    for zd in zones_all:
        elev_mean = zd["elevationMean"]
        # Altitude rule: Tmax follows the lapse rate, Tmin does not (cold air
        # pools in hollows, so a lower zone is not warmer at night).
        t_min_z = float(t_min)
        t_max_z = float(correct_temperature(t_max, station_elev, np.array([elev_mean]))[0])
        eto = compute_eto(
            np.array([t_min_z]), np.array([t_max_z]),
            clamp_ea_to_es(ea_kpa, np.array([t_min_z]), np.array([t_max_z])), u2_ms,
            rs_mj, np.array([rs_mj]),
            np.array([elev_mean]), np.array([zd["centroid"][1]]),
            doy,
        )
        metrics = {
            "tMin": round(t_min_z, 1),
            "tMax": round(t_max_z, 1),
            "eto": round(float(eto[0]), 2),
        }
        cog_values, cog_checked = _zone_radiation(
            zd, doy=doy, rs_mj=float(rs_mj), ea_kpa=float(ea_kpa),
        )
        metrics.update(cog_values)
        area_ha = zd["areaHa"]
        entity = build_agri_parcel_zone(
            tenant_id=tenant_id,
            parcel_id=parcel_id,
            zone=zd,
            geometry=zd.get("geometry") or geometry,
            metrics=metrics,
            observed_at=observed_at,
            area_ha=area_ha,
            zone_rank=zone_ranks[zd["id"]],
        )
        # Private marker (stripped before send): attributes whose COG lookup
        # succeeded, so the upsert may clear stale values for them.
        entity[COG_CHECKED_KEY] = cog_checked
        entity_list.append(entity)

    # 6. Enrich with sensors
    if entity_list:
        entity_list = await find_nearby_sensors(tenant_id, parcel_id, entity_list)

    return entity_list


# Retired metric: still reported as "checked" so a stale value is cleared from
# zones that carry it (contract step of the expand-contract retirement).
_RETIRED_ZONE_ATTRS = ("frostRisk",)
_MJ_DAY_TO_WM2 = 1e6 / 86_400.0


def _zone_radiation(
    zone: dict, *, doy: int, rs_mj: float, ea_kpa: float,
) -> tuple[dict[str, float], set[str]]:
    """Modelled daily radiation on the zone's slopes, from the horizontal parcel value.

    Allen, Trezza & Tasumi (2006) translation of the running day's horizontal
    ``rs_mj`` to each zone pixel's slope and aspect, averaged over the zone.
    Internal to the zone attribute ``nkz:solarRadiationWm2`` ("modelled from
    terrain"); the public raster and ``/stats`` radiation stay horizontal.

    Returns ``(values, checked)``: ``values`` holds only what was computed
    (nothing is fabricated); ``checked`` names the attributes a stale value may
    be cleared for. A zone without terrain pixels or a model failure leaves the
    radiation unchecked, so an earlier value is kept rather than erased.
    """
    checked: set[str] = set(_RETIRED_ZONE_ATTRS)
    px = zone.get(_TERRAIN_KEY)
    if px is None or len(px[0]) == 0:
        logger.debug("Zone %s has no terrain pixels; skipping radiation", zone.get("id"))
        return {}, checked
    slopes, aspects = px
    rs_wm2 = rs_mj * _MJ_DAY_TO_WM2
    lat = float(zone["centroid"][1])
    elev = float(zone.get("elevationMean", 0.0))
    cache: dict[tuple[float, float], float] = {}
    try:
        total = 0.0
        for s, a in zip(slopes.tolist(), aspects.tolist()):
            # 0.5 deg bins: the model is smooth, thousands of pixels share a bin.
            key = (round(min(max(s, 0.0), 90.0) * 2) / 2, round((a % 360.0) * 2) / 2)
            if key not in cache:
                cache[key] = slope_radiation_wm2(
                    rs_wm2, lat, doy, key[0], key[1], elev, ea_kpa,
                )
            total += cache[key]
        mean = total / len(slopes)
    except Exception as exc:
        logger.warning("Slope radiation failed for zone %s: %s", zone.get("id"), exc)
        return {}, checked
    checked.add("solarRadiationWm2")
    return {"solarRadiationWm2": round(mean, 1)}, checked


async def _delete_stale_zones(
    tenant_id: str,
    parcel_id: str,
    keep_ids: set[str],
) -> None:
    """Delete weather-map zones for a parcel whose id is no longer produced.

    Only zones with the weather-map id scheme (``z<geohash>-e<band>-<sector>``)
    are touched — hydrology TWI zones (``twi-*``) are left alone. Keeps the
    broker free of orphaned zones when the DEM tile set or zoning parameters
    change and a zone id disappears between runs.
    """
    orion = OrionClient(tenant_id)
    try:
        rows = await orion.query_entities(
            type="AgriParcelZone",
            q=f'(hasAgriParcel=="{parcel_id}")',
            options="keyValues",
            limit=200,
        ) or []
        deleted = 0
        for r in rows:
            if not isinstance(r, dict):
                continue
            eid = r.get("id") or ""
            zid = str(r.get("nkz:zoneId") or r.get("zoneId") or eid.split(":")[-1])
            if not (zid.startswith("z") and "-e" in zid):
                continue  # not a weather-map zone
            if zid in keep_ids:
                continue
            await orion.delete_entity(eid)
            deleted += 1
        if deleted:
            logger.info("Deleted %d stale zones for parcel %s", deleted, parcel_id)
    except Exception:
        logger.exception("Stale-zone reconciliation failed for %s", parcel_id)
    finally:
        await orion.close()


async def run_for_tenant(tenant_id: str) -> None:
    """Run the full zone pipeline for a single tenant."""
    parcels = await fetch_tenant_parcels(tenant_id)
    if not parcels:
        logger.info("No parcels for tenant %s, skipping", tenant_id)
        return

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    all_entities: list[dict] = []
    parcel_keep_ids: dict[str, set[str]] = {}

    for parcel in parcels:
        entities = await process_parcel(tenant_id, parcel, today)
        if entities is None:
            continue  # undecided: keep whatever the parcel has
        all_entities.extend(entities)
        keep_ids = set()
        for e in entities:
            zid = (e.get("nkz:zoneId") or {}).get("value")
            if zid:
                keep_ids.add(str(zid))
        parcel_keep_ids[parcel["id"]] = keep_ids

    if all_entities:
        await upsert_agri_parcel_zones(tenant_id, all_entities)
        logger.info(
            "Upserted %d zone entities for tenant %s",
            len(all_entities), tenant_id,
        )

    for parcel_id, keep_ids in parcel_keep_ids.items():
        await _delete_stale_zones(tenant_id, parcel_id, keep_ids)


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
