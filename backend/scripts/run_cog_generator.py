#!/usr/bin/env python3
"""CronJob entrypoint for the COG generator worker.

Usage::

    python3 scripts/run_cog_generator.py [tenant_id]

Runs ``run_for_tenant`` for the given tenant (default: ``"default"``)
over the last 5 days and uploads COGs to MinIO.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

# Ensure /app is on the path (scripts/ runs from /app root in Docker)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cog_generator import run_for_tenant
from app.sources import fetch_tenant_parcels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """Fetch tenant parcels and run COG generation."""
    tenant_id = sys.argv[1] if len(sys.argv) > 1 else "default"

    date_to = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_from = (
        datetime.now(timezone.utc) - timedelta(days=5)
    ).strftime("%Y-%m-%d")

    logger.info(
        "Starting COG generation for tenant '%s' [%s → %s]",
        tenant_id, date_from, date_to,
    )

    parcels = await fetch_tenant_parcels(tenant_id)

    await run_for_tenant(tenant_id, parcels, date_from, date_to)

    logger.info("COG generation complete for tenant '%s'", tenant_id)


if __name__ == "__main__":
    asyncio.run(main())
