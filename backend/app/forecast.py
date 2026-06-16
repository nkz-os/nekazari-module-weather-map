"""ET0 forecast endpoint — proxy to weather-api Open-Meteo forecast.

Returns a 7-day ET0 and precipitation forecast for a given parcel,
resolving coordinates from Orion-LD. Used by the soil module's water
budget worker to project soil moisture deficit.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import require_tenant
from app.config import settings
from app.sources import fetch_agri_parcel

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/forecast/et0")
async def forecast_et0(
    parcel_id: str = Query(..., description="AgriParcel URN or short ID"),
    days: int = Query(7, ge=1, le=14, description="Forecast days (max 14)"),
    tenant_id: str = Depends(require_tenant),
):
    """Return 7-day ET0 + precipitation forecast for a parcel.

    Resolves the parcel centroid from Orion-LD, then fetches the Open-Meteo
    forecast via the weather-api service. Returns daily ET0 (mm) and
    precipitation (mm) for the requested period.

    The response format matches what the soil module's water budget worker
    expects: ``{"forecast": [{"day": …, "et0": …, "precip": …, "deficitAfter": 0.0}]}``.
    """
    # 1. Resolve parcel coordinates from Orion-LD
    parcel = await fetch_agri_parcel(tenant_id, parcel_id)
    if parcel is None:
        raise HTTPException(
            status_code=404,
            detail=f"Parcel not found: {parcel_id}",
        )

    centroid = _parcel_centroid(parcel.get("geometry", {}))
    if centroid is None:
        raise HTTPException(
            status_code=400,
            detail="Parcel has no usable geometry for centroid extraction",
        )

    lat, lon = centroid

    # 2. Fetch forecast from weather-api (which proxies Open-Meteo)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end = (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d")

    params = {
        "lat": lat,
        "lon": lon,
        "days": days,
    }
    headers = {
        "X-Tenant-ID": tenant_id,
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{settings.weather_api_url}/api/weather/coordinates",
                params=params,
                headers=headers,
            )
            if resp.status_code != 200:
                logger.warning(
                    "weather-api returned %d for parcel=%s: %s",
                    resp.status_code, parcel_id, resp.text[:200],
                )
                raise HTTPException(
                    status_code=502,
                    detail=f"Weather forecast service returned {resp.status_code}",
                )
            weather_data = resp.json()
    except httpx.TimeoutException:
        logger.error("weather-api timeout for parcel=%s", parcel_id)
        raise HTTPException(status_code=504, detail="Weather forecast service timed out")
    except httpx.RequestError as exc:
        logger.error("weather-api request failed for parcel=%s: %s", parcel_id, exc)
        raise HTTPException(status_code=502, detail="Weather forecast service unavailable")

    # 3. Transform to the format expected by the water budget worker
    raw_forecast = weather_data.get("forecast", [])
    forecast = []
    for entry in raw_forecast:
        forecast.append({
            "day": entry.get("date", ""),
            "et0": entry.get("eto_mm") or 0.0,
            "precip": entry.get("precip_mm") or 0.0,
            "deficitAfter": 0.0,  # filled by the worker's _compute_projection
        })

    return {"forecast": forecast}


def _parcel_centroid(geometry: dict) -> Optional[tuple[float, float]]:
    """Return (lat, lon) centroid of a GeoJSON geometry."""
    if not geometry or "coordinates" not in geometry:
        return None

    coords: list[tuple[float, float]] = []

    def _walk(node):
        if (
            isinstance(node, (list, tuple))
            and len(node) == 2
            and all(isinstance(c, (int, float)) for c in node)
        ):
            coords.append((float(node[1]), float(node[0])))  # (lat, lon)
        elif isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)

    _walk(geometry.get("coordinates"))
    if not coords:
        return None
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    return (sum(lats) / len(lats), sum(lons) / len(lons))
