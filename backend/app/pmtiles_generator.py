"""PMTiles generation — merges per-tile COGs and produces a single PMTiles archive.

Runs after COG generation as a post-processing step.
Each metric+date gets one .pmtiles file uploaded to the frontend-accessible bucket.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import geolibre_wasm as gl
import rasterio
from rasterio.merge import merge as merge_tools

from app.config import settings
from app.minio_io import _get_client

logger = logging.getLogger(__name__)

FRONTEND_BUCKET = "nekazari-frontend"
PMTILES_PREFIX = "modules/weather-map/pmtiles"


def generate_pmtiles_for_metric(
    tenant_id: str,
    metric: str,
    date_str: str,
    zoom: int = 14,
) -> str | None:
    """Merge all per-tile COGs for a metric+date into a single PMTiles archive.

    Args:
        tenant_id: Tenant identifier.
        metric: Weather metric name.
        date_str: ISO date string (YYYY-MM-DD).
        zoom: Tile zoom level used during COG generation.

    Returns:
        Public URL to the PMTiles file, or None if no COGs were found.
    """
    client = _get_client()

    # 1. List all COG objects for this tenant/metric/date
    cog_prefix = f"cogs/{tenant_id}/{metric}/{date_str}/"
    try:
        objects = list(
            client.list_objects(
                settings.minio_bucket, prefix=cog_prefix, recursive=True
            )
        )
    except Exception as exc:
        logger.error("Failed to list COGs for %s: %s", cog_prefix, exc)
        return None

    if not objects:
        logger.warning("No COGs found for %s", cog_prefix)
        return None

    # 2. Download COGs to temp directory
    with tempfile.TemporaryDirectory(prefix="pmtiles_") as tmpdir:
        cog_paths: list[str] = []
        for obj in objects:
            if not obj.object_name.endswith(".tif"):
                continue
            local_path = os.path.join(tmpdir, os.path.basename(obj.object_name))
            try:
                client.fget_object(
                    settings.minio_bucket, obj.object_name, local_path
                )
                cog_paths.append(local_path)
            except Exception as exc:
                logger.warning("Failed to download %s: %s", obj.object_name, exc)

        if not cog_paths:
            logger.warning("No valid COGs downloaded for %s", cog_prefix)
            return None

        logger.info(
            "Downloaded %d COGs for %s/%s/%s",
            len(cog_paths), tenant_id, metric, date_str,
        )

        # 3. Merge COGs using rasterio merge
        src_files = [rasterio.open(p) for p in cog_paths]
        try:
            merged_data, merged_transform = merge_tools(src_files)
            merged_data = merged_data[0]  # single band
            merged_crs = src_files[0].crs
        finally:
            for src in src_files:
                src.close()

        # 4. Write merged raster to temp GeoTIFF
        merged_path = os.path.join(tmpdir, f"{metric}_{date_str}.tif")
        with rasterio.open(
            merged_path,
            "w",
            driver="GTiff",
            height=merged_data.shape[0],
            width=merged_data.shape[1],
            count=1,
            dtype=merged_data.dtype,
            crs=merged_crs,
            transform=merged_transform,
            nodata=float('nan'),
            compress="DEFLATE",
            predictor=3,
        ) as dst:
            dst.write(merged_data, 1)

        logger.info(
            "Merged COG: %s (%d x %d)",
            merged_path, merged_data.shape[1], merged_data.shape[0],
        )

        # 5. Convert merged COG to PMTiles via geolibre-wasm
        with open(merged_path, "rb") as f:
            cog_bytes = f.read()

        try:
            result = gl.run_tool(
                "write_pmtiles",
                args=[
                    "--input=/work/input.tif",
                    "--output=/work/output.pmtiles",
                    "--colormap=viridis",
                    f"--min_zoom={max(zoom - 2, 0)}",
                    f"--max_zoom={zoom + 1}",
                ],
                input={"input.tif": cog_bytes},
            )
        except Exception as exc:
            logger.error(
                "write_pmtiles failed for %s/%s: %s", metric, date_str, exc
            )
            return None

        if result.exit_code != 0:
            logger.error(
                "write_pmtiles non-zero exit %d: %s",
                result.exit_code, result.stdout,
            )
            return None

        pmtiles_bytes = result.files.get("output.pmtiles")
        if not pmtiles_bytes:
            logger.error("write_pmtiles produced no output")
            return None

        # 6. Upload PMTiles to frontend bucket
        pmtiles_key = f"{PMTILES_PREFIX}/{tenant_id}/{metric}/{date_str}.pmtiles"
        try:
            client.put_object(
                FRONTEND_BUCKET,
                pmtiles_key,
                io.BytesIO(pmtiles_bytes),
                length=len(pmtiles_bytes),
                content_type="application/vnd.pmtiles",
            )
        except Exception as exc:
            logger.error(
                "Failed to upload PMTiles to %s: %s", pmtiles_key, exc
            )
            return None

        logger.info(
            "PMTiles uploaded: %s/%s (%d bytes)",
            FRONTEND_BUCKET, pmtiles_key, len(pmtiles_bytes),
        )

    # Return the public URL for the frontend
    return f"/{FRONTEND_BUCKET}/{pmtiles_key}"


def generate_all_pmtiles(
    tenant_id: str, date_str: str, zoom: int = 14
) -> dict[str, str | None]:
    """Generate PMTiles for all configured metrics.

    Args:
        tenant_id: Tenant identifier.
        date_str: ISO date string.
        zoom: Tile zoom level.

    Returns:
        Dict mapping metric name to PMTiles URL (or None on failure).
    """
    results: dict[str, str | None] = {}
    for metric in settings.metrics:
        url = generate_pmtiles_for_metric(tenant_id, metric, date_str, zoom)
        results[metric] = url
        if url is None:
            logger.warning("PMTiles generation failed for %s/%s", metric, date_str)
    return results
