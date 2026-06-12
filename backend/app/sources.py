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
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{settings.elevation_service_url}/api/elevation/raster",
                params=params,
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
    """Fetch the ``AgriSoil`` NGSI-LD entity from Orion-LD.

    Returns a dict with ``sand_pct``, ``silt_pct``, ``clay_pct`` keys,
    or ``None`` if the entity does not exist or the request fails.
    """
    entity_id = f"urn:ngsi-ld:AgriSoil:{parcel_id}"
    params = {"options": "keyValues"}
    headers = {
        "NGSILD-Tenant": tenant_id,
        "Fiware-Service": tenant_id,
        "Fiware-ServicePath": "/",
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{settings.orion_url}/ngsi-ld/v1/entities/{entity_id}",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
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


async def write_entity_attrs(
    tenant_id: str,
    entity_id: str,
    attrs: dict[str, Any],
) -> bool:
    """Write NGSI-LD attributes to an entity in Orion-LD via POST /attrs.

    Uses ``?options=append`` which creates the attribute if it does not
    exist or updates it if it does.  Each key in *attrs* becomes a
    top-level NGSI-LD attribute:

    .. code-block:: json

        {
          "weatherStats": {
            "type": "Property",
            "value": {…},
            "observedAt": "2026-06-12T00:00:00Z"
          }
        }

    Returns ``True`` on success, ``False`` on failure.
    """
    headers = {
        "NGSILD-Tenant": tenant_id,
        "Fiware-Service": tenant_id,
        "Fiware-ServicePath": "/",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{settings.orion_url}/ngsi-ld/v1/entities/{entity_id}/attrs",
                params={"options": "append"},
                headers=headers,
                json=attrs,
            )
            resp.raise_for_status()
            return True
    except Exception:
        logger.exception(
            "write_entity_attrs(tenant=%s, entity=%s) failed",
            tenant_id, entity_id,
        )
        return False


async def fetch_entity_attr(
    tenant_id: str,
    entity_id: str,
    attr_name: str,
) -> Any | None:
    """Fetch a single NGSI-LD attribute from an entity in Orion-LD.

    Returns the attribute's ``value`` (with keyValues simplification) or
    ``None`` if the entity/attribute does not exist or on error.
    """
    headers = {
        "NGSILD-Tenant": tenant_id,
        "Fiware-Service": tenant_id,
        "Fiware-ServicePath": "/",
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{settings.orion_url}/ngsi-ld/v1/entities/{entity_id}",
                params={"options": "keyValues"},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get(attr_name)
    except Exception:
        logger.exception(
            "fetch_entity_attr(tenant=%s, entity=%s, attr=%s) failed",
            tenant_id, entity_id, attr_name,
        )
        return None


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
    }
    payload: dict[str, Any] = {
        "type": "AgriParcel",
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
    """Fetch an ``AgriParcel`` NGSI-LD entity from Orion-LD for geometry.

    Returns a dict with the parcel's GeoJSON ``location`` geometry and
    ``agriParcelOf`` (tenant/enterprise), or ``None`` if not found.
    """
    # Normalize: strip prefix if already has it
    eid = parcel_id if parcel_id.startswith("urn:ngsi-ld:AgriParcel:") else \
        f"urn:ngsi-ld:AgriParcel:{parcel_id}"
    params = {"options": "keyValues"}
    headers = {
        "NGSILD-Tenant": tenant_id,
        "Fiware-Service": tenant_id,
        "Fiware-ServicePath": "/",
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{settings.orion_url}/ngsi-ld/v1/entities/{eid}",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            geometry = data.get("location") or data.get("geometry")
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
