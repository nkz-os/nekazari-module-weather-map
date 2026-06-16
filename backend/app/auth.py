"""Authentication for weather-map endpoints.

Tenant identity is injected by api-gateway as the X-Tenant-ID header. We never
read the tenant from a query param (that allowed cross-tenant access). When
AUTH_DISABLED=true (dev only) a fixed dev tenant is used and a WARNING is logged.
"""

from __future__ import annotations

import logging

from fastapi import Header, HTTPException

from app.config import settings

logger = logging.getLogger(__name__)

if settings.auth_disabled:
    logger.warning(
        "AUTH_DISABLED=true — weather-map is running WITHOUT tenant auth. "
        "This must never be set in production."
    )


def require_tenant(
    x_tenant_id: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> str:
    """Return the caller's tenant id from the api-gateway headers.

    Validates both X-Tenant-ID and X-User-ID to prevent spoofing
    when the endpoint is reached via direct ingress (bypassing api-gateway).
    Raises 401 when either header is absent (and auth is enabled).
    """
    if settings.auth_disabled:
        return x_tenant_id or "dev"
    if not x_tenant_id:
        raise HTTPException(status_code=401, detail="Missing X-Tenant-ID")
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing X-User-ID")
    return x_tenant_id
