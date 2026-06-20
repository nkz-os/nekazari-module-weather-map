"""Zone query and GDD endpoints — reads AgriParcelZone data from Orion-LD and TimescaleDB."""
from __future__ import annotations
import logging
import re
from datetime import datetime, timezone
from typing import Any

import asyncpg
from fastapi import APIRouter, HTTPException, Query
from nkz_platform_sdk import OrionClient

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/weather-map", tags=["zones"])

_SAFE_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@router.get("/zones/{parcel_id:path}")
async def get_zones(
    parcel_id: str,
    tenant_id: str = Query(None, alias="tenant_id"),
    limit: int = Query(20, ge=1, le=50),
) -> dict[str, Any]:
    """Return the latest AgriParcelZone entities for a parcel."""
    pid = parcel_id if parcel_id.startswith("urn:") else f"urn:ngsi-ld:AgriParcel:{parcel_id}"

    # Use SDK for auto-header injection
    import httpx
    orion = OrionClient(tenant_id or "")
    try:
        headers = await orion._get_headers()  # reuse SDK's header builder
        headers["Accept"] = "application/ld+json"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{settings.orion_url}/ngsi-ld/v1/entities",
                params={
                    "type": "AgriParcelZone",
                    "q": f'hasAgriParcel=="{pid}"',
                    "options": "keyValues",
                    "orderBy": "observedAt:desc",
                    "limit": limit,
                },
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return {"parcel_id": pid, "zones": []}
        logger.exception("Zone query failed for %s", pid)
        raise HTTPException(status_code=502, detail="Orion-LD query failed") from exc
    except Exception:
        logger.exception("Zone query failed for %s", pid)
        raise HTTPException(status_code=502, detail="Orion-LD query failed")
    finally:
        await orion.close()

    zones = list(data) if isinstance(data, list) else []
    return {"parcel_id": pid, "zones": zones}


@router.get("/gdd")
async def get_gdd(
    parcel_id: str = Query(..., description="Parcel URN"),
    season_start: str = Query(..., description="Season start date (YYYY-MM-DD)"),
    base_temp: float = Query(10.0, description="Base temperature (°C)"),
    upper_cutoff: float = Query(30.0, description="Upper cutoff temperature (°C)"),
    tenant_id: str = Query(None, alias="tenant_id"),
) -> dict[str, Any]:
    """Accumulated GDD per zone from agriparcelzone.

    Takes the same contract as SP1 (timeseries-reader ``GET /api/weather/gdd``)
    but reads from the per-zone hypertable. Returns 404 with ``source: "regional_knn"``
    when no zone data is available — caller should fallback to timeseries-reader.
    """
    if not _SAFE_DATE_RE.match(season_start):
        raise HTTPException(status_code=400, detail="season_start must be YYYY-MM-DD")

    if not settings.postgres_url:
        raise HTTPException(status_code=503, detail="POSTGRES_URL not configured")

    pid = parcel_id if parcel_id.startswith("urn:") else f"urn:ngsi-ld:AgriParcel:{parcel_id}"
    start_dt = datetime.strptime(season_start, "%Y-%m-%d")
    end_dt = datetime.now(timezone.utc)

    if start_dt >= end_dt:
        raise HTTPException(status_code=400, detail="season_start must be in the past")

    try:
        conn = await asyncpg.connect(settings.postgres_url)
        try:
            rows = await conn.fetch(
                """
                SELECT
                    nkz_zone_id,
                    SUM(GREATEST(0, (LEAST(t_max, $4::double precision) + t_min) / 2.0 - $5::double precision)) AS gdd_accumulated,
                    COUNT(DISTINCT DATE(observed_at)) AS days_with_data
                FROM agriparcelzone
                WHERE has_agri_parcel = $1
                  AND observed_at >= $2::date
                  AND observed_at < $3
                GROUP BY nkz_zone_id
                """,
                pid, start_dt, end_dt, upper_cutoff, base_temp,
            )
        finally:
            await conn.close()
    except Exception as exc:
        logger.exception("GDD query failed for %s", pid)
        raise HTTPException(status_code=502, detail=f"Database query failed: {exc}") from exc

    if not rows:
        return {
            "detail": "No zone data available, fallback to regional KNN",
            "source": "regional_knn",
        }

    zones_list = []
    total_gdd = 0.0
    max_days = 0
    for row in rows:
        gdd = float(row["gdd_accumulated"] or 0.0)
        days = int(row["days_with_data"] or 0)
        total_gdd += gdd
        max_days = max(max_days, days)
        zones_list.append({"id": row["nkz_zone_id"], "gdd_total": round(gdd, 1)})

    mean_daily = round(total_gdd / max_days, 2) if max_days > 0 else 0.0

    return {
        "gdd_total": round(total_gdd, 1),
        "mean_daily_gdd": mean_daily,
        "days_count": max_days,
        "season_start": season_start,
        "season_end": end_dt.strftime("%Y-%m-%d"),
        "base_temp": base_temp,
        "upper_cutoff": upper_cutoff,
        "gdd_method": "simple_avg_capped",
        "source": "zonal",
        "zones": zones_list,
    }
