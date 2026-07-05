"""HTTP clients for external services: eu-elevation, weather-api, Orion-LD.

Provides async functions that fetch DEM tiles, station weather, AgriSoil
entities from Orion-LD, and a stub for tenant parcels.  Every function
returns ``None`` on error — callers check for ``None`` rather than catching.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import httpx
from nkz_platform_sdk import OrionClient

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tms_tile_to_bbox(z: int, x: int, y: int) -> dict[str, float]:
    """Convert TMS tile coordinates to a geographic bounding box (EPSG:4326).

    Parameters
    ----------
    z : int
        Zoom level.
    x : int
        Tile column (0 … 2**z − 1).
    y : int
        Tile row (0 … 2**z − 1), origin at 85°N.

    Returns
    -------
    dict[str, float]
        ``{min_lon, min_lat, max_lon, max_lat}`` in decimal degrees.
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

    return {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_dem_tile(z: int, x: int, y: int) -> dict[str, Any] | None:
    """Fetch a DEM elevation grid for a TMS tile from the eu-elevation service.

    Returns the JSON response (e.g. a 2-D height array / GeoTIFF metadata)
    or ``None`` on any error (network, HTTP error, parse issue).
    """
    bbox = tms_tile_to_bbox(z, x, y)
    params: dict[str, Any] = {
        "min_lon": bbox["min_lon"],
        "min_lat": bbox["min_lat"],
        "max_lon": bbox["max_lon"],
        "max_lat": bbox["max_lat"],
        "resolution_m": 10,
        "purpose": "weather",
    }
    try:
        headers: dict[str, str] = {}
        if settings.internal_service_secret:
            headers["X-Internal-Service-Secret"] = settings.internal_service_secret
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{settings.elevation_service_url}/api/elevation/raster",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception:
        logger.exception(
            "fetch_dem_tile(z=%d, x=%d, y=%d) failed", z, x, y
        )
        return None


async def fetch_station_weather(
    tenant_id: str,
    lat: float,
    lon: float,
    date_from: str,
    date_to: str,
) -> dict[str, Any] | None:
    """Fetch weather data from the nearest station to (lat, lon).

    Returns
    -------
    dict or None
        JSON payload from the weather API, or ``None`` on failure.
    """
    params: dict[str, Any] = {
        "lat": lat,
        "lon": lon,
        "date_from": date_from,
        "date_to": date_to,
    }
    headers = {"X-Tenant-ID": tenant_id}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{settings.weather_api_url}/api/weather/coordinates",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception:
        logger.exception(
            "fetch_station_weather(tenant=%s, lat=%f, lon=%f) failed",
            tenant_id,
            lat,
            lon,
        )
        return None


async def fetch_agri_soil(
    tenant_id: str, parcel_id: str
) -> dict[str, float | None] | None:
    """Fetch the ``AgriSoil`` NGSI-LD entity from Orion-LD via the SDK.

    Returns a dict with ``sand_pct``, ``silt_pct``, ``clay_pct`` keys,
    or ``None`` if the entity does not exist or the request fails.
    """
    entity_id = f"urn:ngsi-ld:AgriSoil:{parcel_id}"
    orion = OrionClient(tenant_id)
    try:
        data = await orion.get_entity(entity_id)
        return {
            "sand_pct": data.get("sandPct") or data.get("sand_pct"),
            "silt_pct": data.get("siltPct") or data.get("silt_pct"),
            "clay_pct": data.get("clayPct") or data.get("clay_pct"),
        }
    except Exception:
        logger.exception(
            "fetch_agri_soil(tenant=%s, parcel=%s) failed",
            tenant_id,
            parcel_id,
        )
        return None
    finally:
        await orion.close()


async def upsert_record(tenant_id: str, entity: dict) -> None:
    """Create an NGSI-LD entity (AgriParcelRecord) via the SDK.

    Same-second id collisions are tolerated: a 409/duplicate is logged and
    ignored (append-only timeseries).
    """
    orion = OrionClient(tenant_id)
    try:
        await orion.create_entity(entity)
    except Exception as exc:  # noqa: BLE001
        if "already exists" in str(exc).lower() or "409" in str(exc):
            logger.info("AgriParcelRecord %s already exists, skipping", entity.get("id"))
            return
        logger.warning("upsert_record(tenant=%s) failed: %s", tenant_id, exc)
    finally:
        await orion.close()


async def fetch_entity_attr(
    tenant_id: str,
    entity_id: str,
    attr_name: str,
) -> Any | None:
    """Fetch a single NGSI-LD attribute from an entity in Orion-LD via the SDK.

    Returns the attribute's ``value`` (with keyValues simplification) or
    ``None`` if the entity/attribute does not exist or on error.
    """
    orion = OrionClient(tenant_id)
    try:
        data = await orion.get_entity(entity_id)
        return data.get(attr_name)
    except Exception:
        logger.exception(
            "fetch_entity_attr(tenant=%s, entity=%s, attr=%s) failed",
            tenant_id, entity_id, attr_name,
        )
        return None
    finally:
        await orion.close()


async def fetch_tenant_parcels(tenant_id: str) -> list[dict[str, Any]]:
    """Fetch all AgriParcel entities for a tenant from Orion-LD.

    Uses ``POST /ngsi-ld/v1/entityOperations/query`` to retrieve
    entities of type ``AgriParcel`` with their ``location`` geometry.

    Returns a list of dicts with ``id``, ``location`` (GeoJSON geometry),
    and optionally ``name`` / ``description``.

    Returns an empty list on error or if no parcels exist.
    """
    headers = {
        "NGSILD-Tenant": tenant_id,
        "Fiware-Service": tenant_id,
        "Fiware-ServicePath": "/",
        "Content-Type": "application/json",
        "Link": f'<{settings.context_url}>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
    }
    # NGSI-LD POST /entityOperations/query body MUST be a Query object with the
    # type selector under `entities` — a bare {"type": "AgriParcel"} is a 400.
    payload: dict[str, Any] = {
        "type": "Query",
        "entities": [{"type": "AgriParcel"}],
        "attrs": ["location", "name", "description"],
    }
    params = {"options": "keyValues"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{settings.orion_url}/ngsi-ld/v1/entityOperations/query",
                params=params,
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                logger.warning(
                    "fetch_tenant_parcels(tenant=%s): unexpected response type %s",
                    tenant_id, type(data).__name__,
                )
                return []

            parcels = []
            for entity in data[:100]:  # limit to first 100
                location = entity.get("location")
                if location is None:
                    continue
                parcel: dict[str, Any] = {
                    "id": entity.get("id"),
                    "location": location,
                }
                if entity.get("name") is not None:
                    parcel["name"] = entity["name"]
                if entity.get("description") is not None:
                    parcel["description"] = entity["description"]
                parcels.append(parcel)

            return parcels
    except Exception:
        logger.exception(
            "fetch_tenant_parcels(tenant=%s) failed", tenant_id,
        )
        return []


async def fetch_agri_parcel(
    tenant_id: str, parcel_id: str
) -> dict[str, Any] | None:
    """Fetch an ``AgriParcel`` NGSI-LD entity from Orion-LD for geometry via SDK.

    Returns a dict with the parcel's GeoJSON ``location`` geometry and
    ``agriParcelOf`` (tenant/enterprise), or ``None`` if not found.
    """
    # Normalize: strip prefix if already has it
    eid = parcel_id if parcel_id.startswith("urn:ngsi-ld:AgriParcel:") else \
        f"urn:ngsi-ld:AgriParcel:{parcel_id}"
    orion = OrionClient(tenant_id)
    try:
        data = await orion.get_entity(eid)
        # get_entity returns NORMALIZED NGSI-LD: location is
        # {"type": "GeoProperty", "value": <GeoJSON>}. Unwrap to the raw GeoJSON.
        raw_loc = data.get("location") or data.get("geometry")
        if isinstance(raw_loc, dict) and raw_loc.get("type") == "GeoProperty":
            geometry = raw_loc.get("value")
        else:
            geometry = raw_loc
        if geometry is None:
            logger.warning(
                "fetch_agri_parcel(tenant=%s, parcel=%s): no location geometry",
                tenant_id, parcel_id,
            )
            return None
        return {
            "id": data.get("id", eid),
            "geometry": geometry,
            "name": data.get("name"),
            "description": data.get("description"),
        }
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.warning(
                "fetch_agri_parcel(tenant=%s, parcel=%s): entity not found (404)",
                tenant_id, parcel_id,
            )
        else:
            logger.exception(
                "fetch_agri_parcel(tenant=%s, parcel=%s) HTTP error",
                tenant_id, parcel_id,
            )
        return None
    except Exception:
        logger.exception(
            "fetch_agri_parcel(tenant=%s, parcel=%s) failed",
            tenant_id, parcel_id,
        )
        return None
    finally:
        await orion.close()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute great-circle distance in metres between two WGS-84 points.

    Uses the standard haversine formula with Earth radius R = 6,371,000 m.
    """
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return float(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def _unwrap_location(location: Any) -> dict | None:
    """Unwrap an NGSI-LD GeoProperty to a raw GeoJSON geometry dict.

    Handles both the normalized format
    ``{"type": "GeoProperty", "value": <GeoJSON>}`` and the simplified
    ``keyValues`` format where the value *is* the GeoJSON geometry.

    Returns ``None`` if *location* cannot be interpreted as a GeoJSON
    geometry (e.g. it is ``None`` or not a dict).
    """
    if isinstance(location, dict) and location.get("type") == "GeoProperty":
        return location.get("value")
    if isinstance(location, dict) and "type" in location and "coordinates" in location:
        return location  # already raw GeoJSON
    return None


def _centroid(geometry: dict) -> tuple[float, float] | None:
    """Return ``(lon, lat)`` centroid from a GeoJSON geometry dict.

    Supports ``Point``, ``Polygon`` and ``MultiPolygon``.
    For ``Polygon`` / ``MultiPolygon`` the centroid is the arithmetic
    mean of the exterior-ring vertex coordinates.
    """
    if not isinstance(geometry, dict):
        return None
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return None
    if geom_type == "Point":
        return (float(coords[0]), float(coords[1]))
    if geom_type == "Polygon":
        ring = coords[0]
        lon = sum(float(c[0]) for c in ring) / len(ring)
        lat = sum(float(c[1]) for c in ring) / len(ring)
        return (lon, lat)
    if geom_type == "MultiPolygon":
        lons, lats = [], []
        for polygon in coords:
            ring = polygon[0]
            lons.extend(float(c[0]) for c in ring)
            lats.extend(float(c[1]) for c in ring)
        return (sum(lons) / len(lons), sum(lats) / len(lats))
    return None


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------


async def upsert_agri_parcel_zones(tenant_id: str, zones: list[dict]) -> None:
    """Batch upsert ``AgriParcelZone`` entities via Orion-LD SDK.

    Uses ``OrionClient.upsert_entities_batch()`` to create or replace all
    zone entities in a single request.  Attributes on existing entities are
    updated; new entities are created.

    Parameters
    ----------
    tenant_id : str
        The tenant / FIWARE service.
    zones : list[dict]
        NGSI-LD entity dicts (from ``build_agri_parcel_zone``).
    """
    if not zones:
        logger.info(
            "upsert_agri_parcel_zones(tenant=%s): no zones to upsert", tenant_id,
        )
        return

    orion = OrionClient(tenant_id)
    try:
        result = await orion.upsert_entities_batch(zones)
        logger.info(
            "upsert_agri_parcel_zones(tenant=%s): upserted %d zone(s)",
            tenant_id, result.get("upserted", 0),
        )
        if result.get("errors"):
            logger.warning(
                "upsert_agri_parcel_zones(tenant=%s): %d error(s)",
                tenant_id, len(result["errors"]),
            )
    except Exception:
        logger.exception(
            "upsert_agri_parcel_zones(tenant=%s) failed for %d zone(s)",
            tenant_id, len(zones),
        )
    finally:
        await orion.close()


async def find_nearby_sensors(
    tenant_id: str,
    parcel_id: str,
    zones: list[dict],
) -> list[dict]:
    """Enrich zone descriptors with the nearest IoT sensor.

    Queries Orion-LD for ``Device`` entities associated with the tenant,
    then for each zone finds the closest device by haversine distance.
    Adds ``sensor_nearby`` (device URN) and ``sensor_distance_m``
    (distance in metres, rounded to 1 decimal) to each zone dict.

    Parameters
    ----------
    tenant_id : str
        The tenant / FIWARE service.
    parcel_id : str
        The parcel identifier (for logging).
    zones : list[dict]
        Zone descriptors (NGSI-LD entity dicts with a ``location``
        attribute).

    Returns
    -------
    list[dict]
        Enriched zone dicts; original zones if no devices found.
    """
    if not zones:
        return []

    # Fetch all Device entities for this tenant via SDK
    orion = OrionClient(tenant_id)
    devices: list[dict[str, Any]] = []
    try:
        data = await orion.query_entities(
            type="Device",
            attrs="location,name",
            options="keyValues",
        )
        if isinstance(data, list):
            for entity in data:
                raw_loc = entity.get("location")
                loc = _unwrap_location(raw_loc)
                if loc is None:
                    continue
                devices.append({
                    "id": entity.get("id"),
                    "location": loc,
                })
    except Exception:
        logger.exception(
            "find_nearby_sensors(tenant=%s, parcel=%s): failed to fetch devices",
            tenant_id, parcel_id,
        )
        return list(zones)  # return unenriched
    finally:
        await orion.close()

    if not devices:
        logger.info(
            "find_nearby_sensors(tenant=%s, parcel=%s): no devices found",
            tenant_id, parcel_id,
        )
        return list(zones)

    # Enrich each zone with the nearest device
    enriched: list[dict[str, Any]] = []
    for zone in zones:
        zone_copy = dict(zone)
        raw_loc = zone_copy.get("location") or zone_copy.get("geometry")
        loc = _unwrap_location(raw_loc)
        if loc is None:
            enriched.append(zone_copy)
            continue

        zone_centroid = _centroid(loc)
        if zone_centroid is None:
            enriched.append(zone_copy)
            continue

        zone_lon, zone_lat = zone_centroid
        best_dist = float("inf")
        best_sensor: str | None = None

        for device in devices:
            dev_centroid = _centroid(device["location"])
            if dev_centroid is None:
                continue
            d = _haversine(zone_lon, zone_lat, dev_centroid[0], dev_centroid[1])
            if d < best_dist:
                best_dist = d
                best_sensor = device["id"]

        if best_sensor is not None:
            zone_copy["sensor_nearby"] = best_sensor
            zone_copy["sensor_distance_m"] = round(best_dist, 1)

        enriched.append(zone_copy)

    logger.info(
        "find_nearby_sensors(tenant=%s, parcel=%s): enriched %d zone(s) with %d device(s)",
        tenant_id, parcel_id, len(enriched), len(devices),
    )
    return enriched
