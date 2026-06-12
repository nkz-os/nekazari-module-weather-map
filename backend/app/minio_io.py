"""MinIO S3 read/write helpers for COG tile storage.

All public functions catch exceptions internally so callers do not need
to wrap every call in a try/except.  Functions that return a value will
return ``None`` (or ``False``) on error.
"""

from __future__ import annotations

import io
import logging
from typing import Any

from minio import Minio

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_client() -> Minio:
    """Create a :class:`Minio` client from application settings."""
    return Minio(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


# ---------------------------------------------------------------------------
# Bucket management
# ---------------------------------------------------------------------------


def ensure_bucket() -> bool:
    """Create the configured bucket if it does not exist.

    Returns ``True`` if the bucket is ready, ``False`` on any error.
    """
    try:
        client = _get_client()
        if not client.bucket_exists(settings.minio_bucket):
            client.make_bucket(settings.minio_bucket)
            logger.info("Created bucket '%s'", settings.minio_bucket)
        return True
    except Exception:
        logger.exception("ensure_bucket() failed")
        return False


# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------


def cog_path(
    tenant_id: str, metric: str, date: str, z: int, x: int, y: int
) -> str:
    """Return the MinIO object path for a single COG tile.

    Pattern: ``cogs/{tenant}/{metric}/{date}/{z}/{x}/{y}.tif``
    """
    return f"cogs/{tenant_id}/{metric}/{date}/{z}/{x}/{y}.tif"


def latest_pointer_path(tenant_id: str, metric: str) -> str:
    """Return the MinIO object path for the latest-date pointer file.

    Pattern: ``cogs/{tenant}/{metric}/latest.txt``
    """
    return f"cogs/{tenant_id}/{metric}/latest.txt"


# ---------------------------------------------------------------------------
# COG upload / download
# ---------------------------------------------------------------------------


def upload_cog(
    data: bytes,
    tenant_id: str,
    metric: str,
    date: str,
    z: int,
    x: int,
    y: int,
) -> bool:
    """Upload a COG tile to MinIO.

    Parameters
    ----------
    data : bytes
        Raw GeoTIFF bytes.
    tenant_id, metric, date, z, x, y : str or int
        Path components (see :func:`cog_path`).

    Returns
    -------
    bool
        ``True`` on success, ``False`` on error.
    """
    try:
        client = _get_client()
        obj_path = cog_path(tenant_id, metric, date, z, x, y)
        client.put_object(
            bucket_name=settings.minio_bucket,
            object_name=obj_path,
            data=io.BytesIO(data),
            length=len(data),
            content_type="image/tiff",
        )
        return True
    except Exception:
        logger.exception(
            "upload_cog(tenant=%s, metric=%s) failed", tenant_id, metric
        )
        return False


def download_cog(
    tenant_id: str,
    metric: str,
    date: str,
    z: int,
    x: int,
    y: int,
) -> bytes | None:
    """Download a COG tile from MinIO.

    Returns raw bytes or ``None`` if the object is missing or an error
    occurs.
    """
    try:
        client = _get_client()
        obj_path = cog_path(tenant_id, metric, date, z, x, y)
        response = client.get_object(
            bucket_name=settings.minio_bucket, object_name=obj_path
        )
        data = response.read()
        response.close()
        response.release_conn()
        return data
    except Exception:
        logger.exception(
            "download_cog(tenant=%s, metric=%s) failed", tenant_id, metric
        )
        return None


# ---------------------------------------------------------------------------
# Latest-date pointer
# ---------------------------------------------------------------------------


def get_latest_date(tenant_id: str, metric: str) -> str | None:
    """Read the latest-date pointer (``latest.txt``) for a tenant/metric.

    Returns the date string (e.g. ``"2026-06-10"``) or ``None`` if the
    pointer file does not exist or cannot be read.
    """
    try:
        client = _get_client()
        obj_path = latest_pointer_path(tenant_id, metric)
        response = client.get_object(
            bucket_name=settings.minio_bucket, object_name=obj_path
        )
        raw = response.read().decode("utf-8").strip()
        response.close()
        response.release_conn()
        return raw if raw else None
    except Exception:
        logger.exception(
            "get_latest_date(tenant=%s, metric=%s) failed",
            tenant_id,
            metric,
        )
        return None


def set_latest_date(tenant_id: str, metric: str, date: str) -> bool:
    """Write the latest-date pointer (``latest.txt``) for a tenant/metric.

    Returns ``True`` on success, ``False`` on error.
    """
    try:
        client = _get_client()
        obj_path = latest_pointer_path(tenant_id, metric)
        data_bytes = date.encode("utf-8")
        client.put_object(
            bucket_name=settings.minio_bucket,
            object_name=obj_path,
            data=io.BytesIO(data_bytes),
            length=len(data_bytes),
            content_type="text/plain",
        )
        return True
    except Exception:
        logger.exception(
            "set_latest_date(tenant=%s, metric=%s) failed",
            tenant_id,
            metric,
        )
        return False
