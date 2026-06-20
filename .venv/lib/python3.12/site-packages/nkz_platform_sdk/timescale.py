"""
TimescaleClient — typed read-only client for the platform timeseries store.

Design constraint (defense in depth): tenant isolation lives in the
`timeseries-reader-service`, which owns Postgres RLS and is the only service
holding DB credentials. Module backends MUST NOT open direct DB connections —
they go through HTTP to that service, and the SDK auto-injects the canonical
tenant headers so a module cannot accidentally read another tenant's data.

Usage:
    from nkz_platform_sdk import ModuleApp, AuthContext

    app = ModuleApp(id="soil-health")

    @app.get("/parcels/{pid}/moisture")
    async def moisture(pid: str, ctx: AuthContext = app.auth()):
        ts = app.timescale(ctx)
        return await ts.query(
            entity_id=f"urn:ngsi-ld:AgriParcel:{pid}",
            attribute="soilMoisture",
            since="2026-05-01T00:00:00Z",
        )
"""

import os
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

import httpx


def _iso(value: str | datetime) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


class TimescaleClient:
    """Read-only timeseries client scoped to a single tenant.

    All calls go through `timeseries-reader-service`, which enforces
    Postgres row-level security based on the injected tenant headers.
    """

    def __init__(
        self,
        tenant_id: str,
        base_url: str | None = None,
    ):
        self.tenant_id = tenant_id
        self.base_url = base_url or os.getenv(
            "TIMESERIES_READER_URL",
            "http://timeseries-reader-service:5000",
        )
        self._client = httpx.AsyncClient(timeout=30.0)

    def _headers(self) -> dict[str, str]:
        return {
            "X-Tenant-ID": self.tenant_id,
            "NGSILD-Tenant": self.tenant_id,
            "Fiware-Service": self.tenant_id,
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        return urljoin(f"{self.base_url}/", path.lstrip("/"))

    async def query(
        self,
        entity_id: str,
        attribute: str,
        since: str | datetime,
        until: str | datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch a single attribute timeseries for a single entity.

        Returns a list of `{"ts": iso8601, "value": float}` points, oldest first.
        Non-numeric points are silently filtered out.
        """
        params: dict[str, Any] = {
            "entityId": entity_id,
            "attribute": attribute,
            "since": _iso(since),
            "limit": limit,
        }
        if until is not None:
            params["until"] = _iso(until)
        resp = await self._client.get(
            self._url("/api/timeseries"),
            params=params,
            headers=self._headers(),
        )
        resp.raise_for_status()
        raw = resp.json()
        points = raw.get("points", raw) if isinstance(raw, dict) else raw
        normalised: list[dict[str, Any]] = []
        for p in points or []:
            ts = p.get("ts") or p.get("timestamp") or p.get("time")
            val = p.get("value") if "value" in p else p.get("v")
            if ts is None or val is None:
                continue
            try:
                normalised.append({"ts": ts, "value": float(val)})
            except (TypeError, ValueError):
                continue
        return normalised

    async def query_aggregate(
        self,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a multi-series aggregation against `/api/v2/query`.

        The body is forwarded verbatim — see timeseries-reader-service for the
        accepted shape (entityIds, attributes, aggregation window, etc.).
        """
        resp = await self._client.post(
            self._url("/api/v2/query"),
            json=body,
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._client.aclose()
