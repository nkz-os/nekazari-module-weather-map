"""
OrionClient — typed NGSI-LD client with automatic FIWARE header injection.

Rules enforced at library level (impossible to forget):
- Every request sends NGSILD-Tenant AND Fiware-Service headers
- Content-Type: application/ld+json with @context in body
- Or Content-Type: application/json with Link header
"""

import os
from typing import Any
from urllib.parse import urljoin

import httpx

CONTEXT_URL = os.getenv(
    "CONTEXT_URL",
    "http://api-gateway-service:5000/ngsi-ld-context.json",
)


class OrionClient:
    """NGSI-LD client scoped to a single tenant."""

    def __init__(
        self,
        tenant_id: str,
        base_url: str | None = None,
        context_url: str | None = None,
    ):
        self.tenant_id = tenant_id
        self.base_url = base_url or os.getenv(
            "ORION_LD_URL", "http://orion-ld-service:1026"
        )
        self.context_url = context_url or CONTEXT_URL
        self._client = httpx.AsyncClient(timeout=30.0)

    def _headers(self, content_type: str = "application/ld+json") -> dict[str, str]:
        headers = {
            "NGSILD-Tenant": self.tenant_id,
            "Fiware-Service": self.tenant_id,
            "Fiware-ServicePath": "/",
        }
        if content_type == "application/ld+json":
            headers["Content-Type"] = "application/ld+json"
        elif content_type == "application/json":
            headers["Content-Type"] = "application/json"
            headers["Link"] = (
                f'<{self.context_url}>; rel="http://www.w3.org/ns/json-ld#context";'
                ' type="application/ld+json"'
            )
        return headers

    def _url(self, path: str) -> str:
        return urljoin(f"{self.base_url}/", path.lstrip("/"))

    async def query_entities(
        self,
        type: str | None = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
        attrs: str | None = None,
        options: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if type:
            params["type"] = type
        if q:
            params["q"] = q
        if attrs:
            params["attrs"] = attrs
        if options:
            params["options"] = options
        resp = await self._client.get(
            self._url("/ngsi-ld/v1/entities"),
            params=params,
            headers=self._headers("application/json"),
        )
        resp.raise_for_status()
        return resp.json()

    async def get_entity(
        self, entity_id: str, options: str | None = None
    ) -> dict[str, Any]:
        params = {"options": options} if options else None
        resp = await self._client.get(
            self._url(f"/ngsi-ld/v1/entities/{entity_id}"),
            params=params,
            headers=self._headers("application/json"),
        )
        resp.raise_for_status()
        return resp.json()

    def _ensure_context(self, entity: dict[str, Any]) -> dict[str, Any]:
        if "@context" not in entity:
            return {"@context": [self.context_url], **entity}
        return entity

    async def create_entity(self, entity: dict[str, Any]) -> dict[str, Any]:
        entity = self._ensure_context(entity)
        resp = await self._client.post(
            self._url("/ngsi-ld/v1/entities"),
            json=entity,
            headers=self._headers("application/ld+json"),
        )
        resp.raise_for_status()
        location = resp.headers.get("Location", "")
        return {"id": location.split("/")[-1] if location else "", "status": "created"}

    async def create_entities_batch(
        self,
        entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create multiple entities via POST /ngsi-ld/v1/entityOperations/create.

        Returns:
            dict with keys ``created``, ``errors``, ``entity_ids``.
        Raises:
            httpx.HTTPStatusError: on non-batchable failure (caller may fall back).
        """
        prepared = [self._ensure_context(e) for e in entities]
        if not prepared:
            return {"created": 0, "errors": [], "entity_ids": []}

        resp = await self._client.post(
            self._url("/ngsi-ld/v1/entityOperations/create"),
            json=prepared,
            headers=self._headers("application/ld+json"),
        )

        entity_ids = [e["id"] for e in prepared if e.get("id")]

        if resp.status_code in (200, 201, 204):
            return {
                "created": len(prepared),
                "errors": [],
                "entity_ids": entity_ids,
            }

        if resp.status_code == 207:
            body = resp.json() if resp.content else {}
            success = body.get("success", entity_ids)
            errors = body.get("errors", [])
            if isinstance(success, list) and success and isinstance(success[0], dict):
                success_ids = [s.get("id", "") for s in success if s.get("id")]
            else:
                success_ids = success if isinstance(success, list) else entity_ids
            return {
                "created": len(success_ids),
                "errors": errors,
                "entity_ids": success_ids,
            }

        resp.raise_for_status()
        return {"created": 0, "errors": [], "entity_ids": []}

    async def upsert_entities_batch(
        self,
        entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create-or-update entities via POST /entityOperations/upsert?options=update.

        Unlike create_entities_batch (create semantics — existing entities land
        in 207/errors), upsert overwrites existing attributes. This is the
        idempotent primitive for catalog (re-)ingestion.

        Returns:
            dict with keys ``upserted``, ``errors``, ``entity_ids``.
        Raises:
            httpx.HTTPStatusError: on non-batchable failure (caller may fall back).
        """
        prepared = [self._ensure_context(e) for e in entities]
        if not prepared:
            return {"upserted": 0, "errors": [], "entity_ids": []}

        resp = await self._client.post(
            self._url("/ngsi-ld/v1/entityOperations/upsert"),
            params={"options": "update"},
            json=prepared,
            headers=self._headers("application/ld+json"),
        )

        entity_ids = [e["id"] for e in prepared if e.get("id")]

        if resp.status_code in (200, 201, 204):
            return {"upserted": len(prepared), "errors": [], "entity_ids": entity_ids}

        if resp.status_code == 207:
            body = resp.json() if resp.content else {}
            success = body.get("success", entity_ids)
            errors = body.get("errors", [])
            if isinstance(success, list) and success and isinstance(success[0], dict):
                success_ids = [s.get("id", "") for s in success if s.get("id")]
            else:
                success_ids = success if isinstance(success, list) else entity_ids
            return {"upserted": len(success_ids), "errors": errors, "entity_ids": success_ids}

        resp.raise_for_status()
        return {"upserted": 0, "errors": [], "entity_ids": []}

    async def update_entity_attrs(self, entity_id: str, attrs: dict[str, Any]) -> None:
        # attrs fragments carry no @context, so the legal NGSI-LD combination
        # is application/json + Link header (ld+json without @context is a 400).
        resp = await self._client.patch(
            self._url(f"/ngsi-ld/v1/entities/{entity_id}/attrs"),
            json=attrs,
            headers=self._headers("application/json"),
        )
        resp.raise_for_status()

    async def append_entity_attrs(
        self,
        entity_id: str,
        attrs: dict[str, Any],
        overwrite: bool = True,
    ) -> None:
        # PATCH /attrs only updates EXISTING attributes (new ones land in
        # notUpdated). POST /attrs appends new attributes and overwrites
        # existing ones unless overwrite=False (options=noOverwrite).
        # Fragments carry no @context -> application/json + Link.
        params = {} if overwrite else {"options": "noOverwrite"}
        resp = await self._client.post(
            self._url(f"/ngsi-ld/v1/entities/{entity_id}/attrs"),
            params=params,
            json=attrs,
            headers=self._headers("application/json"),
        )
        if resp.status_code == 207:
            # Partial append: some attrs rejected (raise_for_status is a no-op
            # on 207 since it's < 400 — fail loudly per platform fail-safe rule)
            raise httpx.HTTPStatusError(
                f"Partial append on {entity_id}: {resp.text[:200]}",
                request=resp.request,
                response=resp,
            )
        resp.raise_for_status()

    async def delete_entity(self, entity_id: str) -> None:
        resp = await self._client.delete(
            self._url(f"/ngsi-ld/v1/entities/{entity_id}"),
            headers=self._headers(),
        )
        resp.raise_for_status()

    async def query_subscriptions(self, limit: int = 100) -> list[dict[str, Any]]:
        """List NGSI-LD subscriptions for this tenant."""
        resp = await self._client.get(
            self._url("/ngsi-ld/v1/subscriptions"),
            params={"limit": limit},
            headers=self._headers("application/json"),
        )
        resp.raise_for_status()
        return resp.json()

    async def create_subscription(self, subscription: dict[str, Any]) -> str:
        """Create an NGSI-LD subscription. Returns the subscription Location."""
        resp = await self._client.post(
            self._url("/ngsi-ld/v1/subscriptions"),
            json=subscription,
            headers=self._headers("application/json"),
        )
        resp.raise_for_status()
        return resp.headers.get("Location", "")

    async def close(self) -> None:
        await self._client.aclose()


class SyncOrionClient:
    """Synchronous NGSI-LD client with automatic FIWARE header injection.

    For use in existing synchronous codebases (Flask, sync FastAPI endpoints).
    Wraps requests.Session instead of httpx.AsyncClient.

    Rules enforced at library level (impossible to forget):
    - Every request sends NGSILD-Tenant AND Fiware-Service headers
    - Content-Type: application/ld+json with @context in body
    - Or Content-Type: application/json with Link header
    """

    def __init__(
        self,
        tenant_id: str,
        base_url: str | None = None,
        context_url: str | None = None,
        timeout: float = 30.0,
    ):
        import requests as sync_requests

        self.tenant_id = tenant_id
        self.base_url = base_url or os.getenv(
            "ORION_LD_URL", "http://orion-ld-service:1026"
        )
        self.context_url = context_url or CONTEXT_URL
        self.timeout = timeout
        self._session = sync_requests.Session()
        self._session.headers.update({"User-Agent": "NKZ-SyncOrionClient/1.0"})

    def _headers(self, content_type: str = "application/ld+json") -> dict[str, str]:
        headers = {
            "NGSILD-Tenant": self.tenant_id,
            "Fiware-Service": self.tenant_id,
            "Fiware-ServicePath": "/",
        }
        if content_type == "application/ld+json":
            headers["Content-Type"] = "application/ld+json"
        elif content_type == "application/json":
            headers["Content-Type"] = "application/json"
            headers["Link"] = (
                f'<{self.context_url}>; rel="http://www.w3.org/ns/json-ld#context";'
                ' type="application/ld+json"'
            )
        return headers

    def _url(self, path: str) -> str:
        return urljoin(f"{self.base_url}/", path.lstrip("/"))

    def query_entities(
        self,
        type: str | None = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
        attrs: str | None = None,
        options: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if type:
            params["type"] = type
        if q:
            params["q"] = q
        if attrs:
            params["attrs"] = attrs
        if options:
            params["options"] = options
        resp = self._session.get(
            self._url("/ngsi-ld/v1/entities"),
            params=params,
            headers=self._headers("application/json"),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_entity(self, entity_id: str, options: str | None = None) -> dict[str, Any]:
        params = {"options": options} if options else None
        resp = self._session.get(
            self._url(f"/ngsi-ld/v1/entities/{entity_id}"),
            params=params,
            headers=self._headers("application/json"),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def create_entity(self, entity: dict[str, Any]) -> None:
        resp = self._session.post(
            self._url("/ngsi-ld/v1/entities"),
            json=entity,
            headers=self._headers("application/ld+json"),
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def delete_entity(self, entity_id: str) -> None:
        resp = self._session.delete(
            self._url(f"/ngsi-ld/v1/entities/{entity_id}"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def query_types(self) -> list[str]:
        resp = self._session.get(
            self._url("/ngsi-ld/v1/types"),
            headers=self._headers("application/json"),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data.get("typeList", [])
        return data if isinstance(data, list) else []

    def query_subscriptions(self, limit: int = 100) -> list[dict[str, Any]]:
        resp = self._session.get(
            self._url("/ngsi-ld/v1/subscriptions"),
            params={"limit": limit},
            headers=self._headers("application/json"),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def delete_subscription(self, subscription_id: str) -> None:
        resp = self._session.delete(
            self._url(f"/ngsi-ld/v1/subscriptions/{subscription_id}"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "SyncOrionClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
