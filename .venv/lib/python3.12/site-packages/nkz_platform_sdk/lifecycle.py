"""
ModuleLifecycle — base class for module lifecycle hooks.

The platform calls lifecycle webhooks via HMAC-signed HTTP POST.
Modules extend this class and implement on_install/on_uninstall/etc.

Guarantees provided by the platform:
- Idempotency: same call N times = same result
- Retry: exponential backoff (3 attempts: 1s, 4s, 16s)
- Dead-letter: after retries exhausted, logged for manual intervention
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LifecycleResult:
    status: str  # "active" | "failed" | "pending"
    message: str = ""
    resources: dict[str, Any] = field(default_factory=dict)


class ModuleLifecycle:
    """Base class for module lifecycle hooks. Override any hook needed."""

    def __init__(self):
        self._handlers: dict[str, Any] = {}

    def on_install(self, func):
        self._handlers["install"] = func
        return func

    def on_uninstall(self, func):
        self._handlers["uninstall"] = func
        return func

    def on_enable(self, func):
        self._handlers["enable"] = func
        return func

    def on_disable(self, func):
        self._handlers["disable"] = func
        return func

    async def handle(
        self, event: str, tenant_id: str, config: dict | None = None
    ) -> LifecycleResult:
        handler = self._handlers.get(event)
        if handler is None:
            return LifecycleResult(
                status="active",
                message=f"No handler registered for '{event}' — noop",
            )
        try:
            return await handler(tenant_id, config or {})
        except Exception as e:
            return LifecycleResult(status="failed", message=str(e))
