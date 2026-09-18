"""Signed URL tokens for Cesium tile requests.

Cesium's ``UrlTemplateImageryProvider`` issues plain image requests with no
headers and no credentials, so the JWT/header auth used by the rest of the
module cannot apply to tile fetches. Instead an authenticated endpoint mints a
short-lived HMAC token and the tile route validates it from the query string.

The token follows the platform's single canonical HMAC format
(``services/common/keycloak_auth.py``): full SHA-256 hex digest (no truncation),
signature first, colon-separated expiry second; payload is
``{metric}|{tenant_id}|{expiry}`` (pipe-separated).
"""

from __future__ import annotations

import hashlib
import hmac
import time

from app.config import settings

TOKEN_TTL_SECONDS = 3600  # the layer is rebuilt on every metric/date change


def make_tile_token(tenant_id: str, metric: str, *, ttl: int = TOKEN_TTL_SECONDS) -> tuple[str, int]:
    """Return ``(token, expiry)`` where token is ``{hexdigest}:{expiry}``."""
    expiry = int(time.time()) + ttl
    payload = f"{metric}|{tenant_id}|{expiry}"
    digest = hmac.new(
        settings.tile_token_secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{digest}:{expiry}", expiry


def verify_tile_token(token: str | None, tenant_id: str | None, metric: str) -> bool:
    """Constant-time validation of a token for this tenant/metric pair."""
    if not token or ":" not in token:
        return False
    digest, _, expiry_s = token.partition(":")
    try:
        expiry = int(expiry_s)
    except ValueError:
        return False
    if expiry < int(time.time()):
        return False
    payload = f"{metric}|{tenant_id}|{expiry}"
    expected = hmac.new(
        settings.tile_token_secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(digest, expected)
