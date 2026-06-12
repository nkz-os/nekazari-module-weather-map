"""Tile server endpoint — reads precomputed COGs from MinIO and serves PNG tiles."""

from __future__ import annotations

import io
import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
import rasterio
from rasterio.windows import Window

from app.config import settings
from app.minio_io import download_cog, get_latest_date
from app.color_scales import apply_color_scale

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/tiles/{metric}/{z}/{x}/{y}.png")
async def serve_tile(
    metric: str,
    z: int,
    x: int,
    y: int,
    tenant_id: str = Query("default"),
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
