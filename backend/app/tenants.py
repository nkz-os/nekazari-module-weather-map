"""Dynamic tenant discovery for weather-map scheduled jobs.

Resolution order: env override → tenant_installed_modules (active-tenant
registry; tenant_limits was retired) → empty (NEVER 'default'). Keeps
weather-map multitenant and hermetic: callers loop per tenant with
NGSILD-Tenant scoping.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _postgres_url() -> str:
    url = os.getenv("POSTGRES_URL", "").strip()
    if url:
        return url
    host = os.getenv("POSTGRES_HOST", "postgresql-service")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "nekazari")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    if not password:
        return ""
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _discover_from_db() -> list[str]:
    url = _postgres_url()
    if not url:
        logger.warning("tenants: no POSTGRES credentials — cannot discover tenants")
        return []
    try:
        import psycopg2
        conn = psycopg2.connect(url)
        try:
            cur = conn.cursor()
            # Canonical active-tenant registry (tenant_limits was retired).
            # 'platform' is a system pseudo-tenant with no agronomic parcels.
            cur.execute(
                "SELECT DISTINCT tenant_id FROM tenant_installed_modules "
                "WHERE tenant_id IS NOT NULL AND tenant_id != '' "
                "AND tenant_id <> 'platform' "
                "ORDER BY tenant_id"
            )
            rows = [r[0] for r in cur.fetchall()]
            cur.close()
            return rows
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("tenants: DB discovery failed: %s", exc)
        return []


def discover_tenants(env_override_var: str) -> list[str]:
    """Active tenants for a scheduled job.

    1. env override (``env_override_var``, comma-separated) — escape hatch.
    2. tenant_installed_modules (active-tenant registry).
    3. [] — skip and warn (never 'default').
    """
    override = os.getenv(env_override_var, "").strip()
    if override:
        return [t.strip() for t in override.split(",") if t.strip()]
    found = _discover_from_db()
    if found:
        logger.info("tenants: discovered %d from tenant_installed_modules: %s", len(found), found)
        return found
    logger.warning("tenants: none discovered (no override, empty DB) — skipping run")
    return []
