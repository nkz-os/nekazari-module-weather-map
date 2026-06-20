"""Canonical NGSI-LD header injection — single source of truth for FIWARE compliance.

ETSI NGSI-LD spec rule (mutual exclusivity):
  - @context in body  → Content-Type: application/ld+json, NO Link header
  - @context NOT in body → Content-Type: application/json, Link header with @context URL

NEVER set both Content-Type: application/ld+json AND a Link header simultaneously.
"""

import logging
import os
import re
import sys
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Import the canonical tenant-id normalizer (single source of truth).
# When running inside a container, /common is on PYTHONPATH.
try:
    from tenant_utils import normalize_tenant_id as _canonical_normalize
except ImportError:
    # Fallback: try relative path (local dev)
    _common_dir = os.path.join(os.path.dirname(__file__), "..", "common")
    if _common_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(_common_dir))
    try:
        from tenant_utils import normalize_tenant_id as _canonical_normalize
    except ImportError:
        logger.warning(
            "tenant_utils not available — ngsi_headers will fall back to basic normalization"
        )

        def _canonical_normalize(t: str) -> str:
            """Minimal fallback when tenant_utils is unavailable."""
            n = t.lower().strip().replace(" ", "-")
            n = re.sub(r"[^a-z0-9-]", "", n)
            return n.strip("-") or t


def inject_fiware_headers(
    headers: Dict[str, str],
    tenant: Optional[str] = None,
    has_context_in_body: bool = False,
) -> Dict[str, str]:
    """Inject NGSI-LD + FIWARE tenant headers for Orion-LD multitenancy.

    Args:
        headers: Existing headers dict. Modified in-place AND returned.
        tenant: Raw tenant ID (will be normalized).
        has_context_in_body: True if the JSON body already contains an @context key.
            Determines Content-Type and whether Link header is added.

    Returns:
        The same dict (modified in-place) for chaining convenience.
    """
    # ── Tenant headers (both NGSI-LD standard + legacy FIWARE v2) ──
    if tenant:
        headers["NGSILD-Tenant"] = _normalize_tenant(tenant)
        headers["Fiware-Service"] = _normalize_tenant(tenant)
        headers["Fiware-ServicePath"] = "/"

    # ── @context delivery (MUTUALLY EXCLUSIVE per ETSI spec) ──
    context_url = os.getenv("CONTEXT_URL", "")

    if has_context_in_body:
        headers["Content-Type"] = "application/ld+json"
    else:
        headers["Content-Type"] = "application/json"
        if context_url:
            headers["Link"] = (
                f"<{context_url}>; "
                f'rel="http://www.w3.org/ns/json-ld#context"; '
                f'type="application/ld+json"'
            )

    headers.setdefault("Accept", "application/ld+json")
    return headers


def _normalize_tenant(tenant: str) -> str:
    """Normalize tenant ID for FIWARE headers using the canonical platform normalizer.

    Delegates to tenant_utils.normalize_tenant_id() — the single source of truth
    across the entire platform. This ensures the tenant ID used as Fiware-Service /
    NGSILD-Tenant matches the value stored in PostgreSQL and embedded in JWTs.

    Falls back to basic cleaning if tenant_utils is unavailable (should never happen
    in production).
    """
    try:
        return _canonical_normalize(tenant)
    except ValueError:
        # If the canonical normalizer rejects the input (e.g. empty, too long),
        # fall back to basic cleaning so the request still reaches Orion-LD.
        logger.warning(
            "Canonical tenant normalization rejected %r, using basic fallback",
            tenant,
        )
        n = tenant.lower().strip().replace(" ", "-")
        n = re.sub(r"[^a-z0-9-]", "", n)
        return n.strip("-")[:47] or tenant
