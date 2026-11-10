#!/usr/bin/env python3
"""CronJob entrypoint for the COG generator worker.

Usage::

    COG_TENANTS="tenant1,tenant2" python3 scripts/run_cog_generator.py

Runs ``run_for_tenant`` for each tenant ID listed in the ``COG_TENANTS``
environment variable (comma-separated), reads each parcel's weather from weather-api /daily
(needs ``HMAC_SECRET``) and uploads COGs to MinIO.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

# Ensure /app is on the path (scripts/ runs from /app root in Docker)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cog_generator import run_for_tenant
from app.sources import fetch_tenant_parcels
from app.tenants import discover_tenants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """Fetch tenant parcels and run COG generation for all configured tenants.

    Reads tenant IDs from the ``COG_TENANTS`` environment variable
    (comma-separated).  If empty, logs a warning and exits.
    """
    tenant_ids = discover_tenants("COG_TENANTS")
    if not tenant_ids:
        logger.warning("cog-generator: no tenants to process")
        return

    # The run's own day. The 5-day look-back is no longer a window of this
    # script: weather-api /daily supplies the running day and the last closed
    # days, dated.
    date_to = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_from = date_to

    for tenant_id in tenant_ids:
        logger.info(
            "Starting COG generation for tenant '%s' [%s → %s]",
            tenant_id, date_from, date_to,
        )

        parcels = await fetch_tenant_parcels(tenant_id)

        await run_for_tenant(tenant_id, parcels, date_from, date_to)

        logger.info("COG generation complete for tenant '%s'", tenant_id)


if __name__ == "__main__":
    asyncio.run(main())
