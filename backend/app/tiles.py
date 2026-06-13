"""Tile server endpoint — reads precomputed COGs from MinIO and serves PNG tiles."""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
import rasterio
from rasterio.windows import Window

from app.auth import require_tenant
from app.config import settings
from app.minio_io import download_cog, get_latest_date
from app.color_scales import apply_color_scale
from app.records import build_agri_parcel_record
from app.stats import compute_zonal_stats
from app.sources import fetch_agri_parcel, fetch_entity_attr, upsert_record

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/tiles/{metric}/{z}/{x}/{y}.png")
async def serve_tile(
    metric: str,
    z: int,
    x: int,
    y: int,
    tenant_id: str = Depends(require_tenant),
    date: Optional[str] = Query(None),
):
    """Serve a PNG map tile for a given weather metric and zoom/tile coordinates.

    The tile is read from a precomputed COG stored in MinIO, colorised with the
    metric's palette, and returned as a 256×256 PNG image.

    Parameters
    ----------
    metric : str
        Weather metric name (e.g. ``temperature_avg``, ``eto``). Must be in
        ``settings.metrics``.
    z : int
        Zoom level.
    x : int
        Tile column index (Spherical Mercator / Web Mercator).
    y : int
        Tile row index (Spherical Mercator / Web Mercator, TMS-flipped).
    tenant_id : str, optional
        Tenant namespace for multi-tenancy. Defaults to ``"default"``.
    date : str, optional
        COG date string (``YYYY-MM-DD``). If omitted the latest available date
        for this tenant/metric is used.

    Returns
    -------
    Response
        256×256 PNG image with ``Cache-Control`` header set to 5 days.
    """
    # -------------------------------------------------------------------
    # 1. Validate metric
    # -------------------------------------------------------------------
    if metric not in settings.metrics:
        raise HTTPException(
            status_code=404, detail=f"Unknown metric: {metric}"
        )

    # -------------------------------------------------------------------
    # 2. Resolve date (latest if not specified)
    # -------------------------------------------------------------------
    if date is None:
        date = get_latest_date(tenant_id, metric)
        if date is None:
            raise HTTPException(
                status_code=404,
                detail="No COG data available for this metric",
            )

    # -------------------------------------------------------------------
    # 3. Download COG bytes
    # -------------------------------------------------------------------
    cog_bytes = download_cog(tenant_id, metric, date, z, x, y)
    if cog_bytes is None:
        raise HTTPException(
            status_code=404, detail="Tile not found"
        )

    # -------------------------------------------------------------------
    # 4. Read COG with rasterio
    # -------------------------------------------------------------------
    try:
        with rasterio.open(io.BytesIO(cog_bytes)) as src:
            window = Window(0, 0, min(src.width, 256), min(src.height, 256))
            band = src.read(1, window=window)
            if band.shape != (256, 256):
                padded = np.full((256, 256), np.nan, dtype=np.float32)
                h, w = band.shape
                padded[:h, :w] = band
                band = padded
    except Exception:
        logger.exception("Failed to read COG tile %s/%s/%s/%s", metric, z, x, y)
        raise HTTPException(status_code=500, detail="Failed to read raster data")

    # -------------------------------------------------------------------
    # 5. Apply color scale
    # -------------------------------------------------------------------
    try:
        rgba = apply_color_scale(band, metric)
    except Exception:
        logger.exception("Failed to apply color scale for metric=%s", metric)
        raise HTTPException(status_code=500, detail="Failed to apply color scale")

    # -------------------------------------------------------------------
    # 6. Encode to PNG
    # -------------------------------------------------------------------
    try:
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        success, png_bytes = cv2.imencode(".png", bgr)
        if not success:
            raise RuntimeError("cv2.imencode returned False")
    except Exception:
        logger.exception("Failed to encode PNG for tile %s/%s/%s/%s", metric, z, x, y)
        raise HTTPException(status_code=500, detail="Failed to encode PNG")

    return Response(
        content=png_bytes.tobytes(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=432000"},
    )


@router.get("/stats/{parcel_id}")
async def parcel_zonal_stats(
    parcel_id: str,
    metrics: str = Query(
        ...,
        description="Comma-separated metric names (e.g. temperature_avg,water_balance)",
    ),
    date: Optional[str] = Query(
        None, description="COG date (YYYY-MM-DD). Defaults to latest."
    ),
    tenant_id: str = Depends(require_tenant),
    geometry: Optional[str] = Query(
        None,
        description=(
            "Optional GeoJSON geometry override as JSON string. When provided, "
            "bypasses the Orion-LD parcel lookup. Useful for testing or when "
            "the caller already has the parcel geometry."
        ),
    ),
    crop: Optional[str] = Query(
        None,
        description=(
            "Crop species name (e.g. 'wheat', 'maize') to fetch phenology "
            "params from Orion-LD AgriCrop entities. When provided together "
            "with *stage*, reads Kc/Ky and includes them in the response."
        ),
    ),
    stage: Optional[str] = Query(
        None,
        description=(
            "Phenological stage (e.g. 'mid-season', 'initial'). Required if "
            "*crop* is provided to look up the correct Kc/Ky values."
        ),
    ),
):
    """Compute zonal statistics for a parcel across one or more metrics.

    Fetches the parcel geometry from Orion-LD (or uses an explicit geometry
    override), intersects it with the precomputed COG tiles, and returns
    per-metric aggregates (mean, min, max, std, percentile, histogram, and
    metric-specific indicators such as deficit percentage for water balance).

    Parameters
    ----------
    parcel_id : str
        AgriParcel ID (``urn:ngsi-ld:AgriParcel:XXX`` or just ``XXX``).
    metrics : str
        Comma-separated list of metric names (e.g. ``temperature_avg,
        water_balance``).
    date : str, optional
        COG date.  Omit to use the latest available.
    tenant_id : str, optional
        Tenant namespace.
    geometry : str, optional
        **Experimental.**  Direct GeoJSON geometry as a JSON string (e.g.
        ``{\"type\": \"Polygon\", ...}``).  When provided the Orion-LD
        lookup is skipped and this geometry is used as-is.

    Returns
    -------
    dict
        Zonal statistics per metric, plus the parcel GeoJSON geometry.
    """
    import json

    # 1. Parse metrics
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    invalid = [m for m in metric_list if m not in settings.metrics]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metrics: {', '.join(invalid)}",
        )

    # 2a. Resolve geometry: explicit override vs Orion-LD lookup
    if geometry:
        try:
            geom = json.loads(geometry)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid GeoJSON geometry: {exc}",
            )
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            raise HTTPException(
                status_code=400,
                detail="geometry must be a Polygon or MultiPolygon",
            )
    else:
        parcel = await fetch_agri_parcel(tenant_id, parcel_id)
        if parcel is None or parcel.get("geometry") is None:
            raise HTTPException(
                status_code=404,
                detail=f"Parcel not found or has no geometry: {parcel_id}",
            )
        geom = parcel["geometry"]

    # 3. Compute zonal stats
    try:
        stats = compute_zonal_stats(tenant_id, geom, metric_list, date)
    except Exception:
        logger.exception("Failed to compute zonal stats for parcel=%s", parcel_id)
        raise HTTPException(
            status_code=500, detail="Failed to compute zonal statistics"
        )

    # 4. Read phenology params from Orion-LD (if crop+stage provided)
    phenology = None
    if crop and stage:
        crop_eid = f"urn:ngsi-ld:AgriCrop:{crop}_{stage}".replace(" ", "_")
        attrs = await fetch_entity_attr(tenant_id, crop_eid, "kc")
        if isinstance(attrs, (int, float)):
            ky_val = await fetch_entity_attr(tenant_id, crop_eid, "ky")
            phenology = {
                "crop": crop,
                "stage": stage,
                "kc": float(attrs),
                "ky": float(ky_val) if isinstance(ky_val, (int, float)) else None,
            }
        else:
            # Fallback: try reading the AgriCrop from Orion-LD via options
            pass

    # 5. Add parcel metadata + optional phenology
    stats["parcel_id"] = parcel_id
    if phenology:
        stats["phenology"] = phenology
    if not geometry:
        stats["parcel_name"] = parcel.get("name")

    return stats


@router.post("/stats/{parcel_id}")
async def persist_zonal_stats(
    parcel_id: str,
    metrics: str = Query("eto"),
    tenant_id: str = Depends(require_tenant),
):
    """Compute zonal stats and persist them as an AgriParcelRecord timeseries point.

    Fetches the parcel geometry from Orion-LD, computes zonal statistics for
    the requested metrics, builds an AgriParcelRecord entity, and upserts it
    into Orion-LD via the SDK.

    Parameters
    ----------
    parcel_id : str
        AgriParcel ID (``urn:ngsi-ld:AgriParcel:XXX`` or just ``XXX``).
    metrics : str
        Comma-separated metric names. Defaults to ``eto``.
    tenant_id : str
        Resolved from the ``X-Tenant-ID`` header by ``require_tenant``.

    Returns
    -------
    dict
        ``{"status": "persisted", "id": <record_id>, "stats": <zonal_stats>}``
    """
    parcel = await fetch_agri_parcel(tenant_id, parcel_id)
    if not parcel:
        raise HTTPException(status_code=404, detail="Parcel not found")

    geom = parcel.get("geometry")
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]

    try:
        stats = compute_zonal_stats(tenant_id, geom, metric_list)
    except Exception:
        logger.exception("Failed to compute zonal stats for parcel=%s", parcel_id)
        raise HTTPException(status_code=500, detail="Failed to compute zonal statistics")

    # Extract a flat scalar (mean) per metric for the AgriParcelRecord.
    # compute_zonal_stats returns {"metrics": {metric: {"mean": float, ...}}}.
    # Metrics with errors (dict with "error" key) are skipped — value stays None.
    flat_metrics: dict[str, float] = {}
    for metric_name, metric_data in stats.get("metrics", {}).items():
        if isinstance(metric_data, dict) and "mean" in metric_data:
            flat_metrics[metric_name] = metric_data["mean"]

    observed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = build_agri_parcel_record(
        tenant_id=tenant_id,
        parcel_id=parcel_id,
        geometry=geom,
        metrics=flat_metrics,
        observed_at=observed_at,
    )

    await upsert_record(tenant_id, record)
    return {"status": "persisted", "id": record["id"], "stats": stats}
