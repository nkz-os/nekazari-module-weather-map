"""
ModuleConfig — per-tenant encrypted configuration storage.

Values are encrypted at rest using Fernet symmetric encryption.
The encryption key is derived from MODULE_CONFIG_SECRET env var.
"""

import os
from base64 import urlsafe_b64encode
from hashlib import sha256

from cryptography.fernet import Fernet
import httpx


class ModuleConfig:
    """Per-tenant encrypted configuration backed by platform API."""

    def __init__(
        self,
        module_id: str,
        tenant_id: str,
        api_url: str | None = None,
    ):
        self.module_id = module_id
        self.tenant_id = tenant_id
        self._api_url = api_url or os.getenv(
            "MODULE_CONFIG_API",
            "http://entity-manager-service:5000/api/internal/module-config",
        )
        self._secret = os.getenv("MODULE_CONFIG_SECRET")
        self._fernet: Fernet | None = None
        if self._secret:
            key = urlsafe_b64encode(sha256(self._secret.encode()).digest())
            self._fernet = Fernet(key)

    async def get(self, key: str) -> str | None:
        resp = await self._request("GET", key)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        value = resp.json().get("value")
        if value and self._fernet:
            value = self._fernet.decrypt(value.encode()).decode()
        return value

    async def set(self, key: str, value: str) -> None:
        if self._fernet:
            value = self._fernet.encrypt(value.encode()).decode()
        await self._request("POST", key, {"value": value})

    async def delete(self, key: str) -> None:
        await self._request("DELETE", key)

    async def list_keys(self) -> list[str]:
        resp = await self._request("GET", "")
        resp.raise_for_status()
        return resp.json().get("keys", [])

    async def _request(
        self, method: str, key: str, data: dict | None = None
    ) -> httpx.Response:
        url = f"{self._api_url}/{self.module_id}/{self.tenant_id}"
        if key:
            url = f"{url}/{key}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {
                "X-Internal-Service": "nkz-platform-sdk",
                "Content-Type": "application/json",
            }
            if method == "GET":
                return await client.get(url, headers=headers)
            elif method == "POST":
                return await client.post(url, json=data, headers=headers)
            elif method == "DELETE":
                return await client.delete(url, headers=headers)
            raise ValueError(f"Unknown method: {method}")
