"""Authentication for weather-map endpoints.

Tenant identity comes from the X-Tenant-ID header, never from a query param
(that allowed cross-tenant access). X-Tenant-ID is only trustworthy with a
valid X-Auth-Signature: any pod in the namespace can set it. The signature is
the platform HMAC over the bearer token and the tenant; the api-gateway signs
over the user's token, and module backends, which call without a user token,
sign over an empty token.

REQUIRE_HMAC_SIGNATURE=true (the default) rejects unsigned requests. While
in-cluster callers are migrated to sign, a deployment can set it to false:
unsigned requests are then served as before and each caller is logged once,
so the remaining ones can be found before enforcing. A signature that is
present but invalid is always rejected. Without HMAC_SECRET and with
enforcement on, every request answers 503.

When AUTH_DISABLED=true (dev only) a fixed dev tenant is used and a WARNING is
logged.
"""

from __future__ import annotations

import logging
import os

from fastapi import Header, HTTPException, Request
from nkz_platform_sdk.crypto import verify_hmac_signature

from app.config import settings

logger = logging.getLogger(__name__)

if settings.auth_disabled:
    logger.warning(
        "AUTH_DISABLED=true — weather-map is running WITHOUT tenant auth. "
        "This must never be set in production."
    )

_UNSIGNED_SEEN: set = set()
_UNSIGNED_SEEN_MAX = 1000


def _require_signature() -> bool:
    return os.getenv("REQUIRE_HMAC_SIGNATURE", "true").strip().lower() == "true"


def _hmac_secret() -> str:
    return os.getenv("HMAC_SECRET", "")


def check_auth_config() -> None:
    """Log at startup when the configuration cannot verify signatures."""
    if settings.auth_disabled or _hmac_secret():
        return
    if _require_signature():
        logger.critical(
            "HMAC_SECRET is not set and REQUIRE_HMAC_SIGNATURE=true: every "
            "authenticated request will be rejected with 503."
        )
    else:
        logger.critical(
            "HMAC_SECRET is not set: X-Auth-Signature is not verified and "
            "X-Tenant-ID is trusted as sent."
        )


def _log_unsigned(request: Request, caller: str) -> None:
    """Log each unsigned caller once per route, to find who still has to sign."""
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    client = request.client.host if request.client else ""
    key = (caller, client, path)
    if key in _UNSIGNED_SEEN or len(_UNSIGNED_SEEN) >= _UNSIGNED_SEEN_MAX:
        return
    _UNSIGNED_SEEN.add(key)
    logger.warning(
        "Unsigned request: path=%s caller=%s client=%s", path, caller, client
    )


def require_tenant(
    request: Request,
    authorization: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_auth_signature: str | None = Header(default=None),
) -> str:
    """Return the caller's tenant id once its signature verifies.

    Raises 401 when X-Tenant-ID or X-User-ID is absent, or when the signature
    is invalid, expired or (with enforcement on) missing; 503 when enforcement
    is on and HMAC_SECRET is not configured.
    """
    if settings.auth_disabled:
        return x_tenant_id or "dev"
    tenant = (x_tenant_id or "").strip()
    caller = (x_user_id or "").strip()
    if not tenant:
        raise HTTPException(status_code=401, detail="Missing X-Tenant-ID")
    if not caller:
        raise HTTPException(status_code=401, detail="Missing X-User-ID")

    token = authorization.removeprefix("Bearer ").strip() if authorization else ""
    signature = (x_auth_signature or "").strip()
    enforce = _require_signature()

    if not _hmac_secret():
        if enforce:
            raise HTTPException(
                status_code=503, detail="Signature verification unavailable"
            )
        _log_unsigned(request, caller)
        return tenant

    if signature:
        if not verify_hmac_signature(_hmac_secret(), signature, token, tenant):
            logger.warning(
                "Invalid signature: path=%s caller=%s tenant=%s",
                request.url.path,
                caller,
                tenant,
            )
            raise HTTPException(
                status_code=401, detail="Invalid or expired signature"
            )
        return tenant

    if enforce:
        raise HTTPException(status_code=401, detail="Missing X-Auth-Signature")
    _log_unsigned(request, caller)
    return tenant
