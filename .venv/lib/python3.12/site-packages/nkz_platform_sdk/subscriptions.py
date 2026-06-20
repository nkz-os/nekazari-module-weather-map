"""SubscriptionRegistrar — declarative Orion-LD subscription management.

One subscription per entity type per tenant (not per parcel — filtering
happens in the module's notification handler). Idempotent by `description`.
All Orion-LD I/O goes through OrionClient (NGSI-LD compliance at SDK level).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from nkz_platform_sdk.orion import OrionClient

logger = logging.getLogger(__name__)

DESCRIPTION_PREFIX = "nkz-module"


@dataclass
class SubscriptionDef:
    type: str
    throttling: int = 30


class SubscriptionRegistrar:
    """Idempotent Orion-LD subscription manager.

    Usage at service startup::

        registrar = SubscriptionRegistrar(
            orion_url=settings.orion_ld_url,
            notification_url="http://crop-health-api-service:8000/api/crop-health/webhooks/orion",
            subscriptions=[{"type": "EOProduct", "throttling": 30}],
            module_name="crop-health",
        )
        await registrar.ensure_all(["tenant-a", "tenant-b"])
        asyncio.create_task(registrar.periodic_heal(get_tenants, interval_minutes=60))
    """

    def __init__(
        self,
        orion_url: str,
        notification_url: str,
        subscriptions: list[dict],
        module_name: str,
        context_url: str | None = None,
    ):
        self.orion_url = orion_url.rstrip("/")
        self.notification_url = notification_url
        self.context_url = context_url
        self.module_name = module_name
        self._subs: list[SubscriptionDef] = [
            SubscriptionDef(**s) if isinstance(s, dict) else s for s in subscriptions
        ]

    def _description(self, sub: SubscriptionDef) -> str:
        return f"{DESCRIPTION_PREFIX}: {sub.type} -> {self.module_name}"

    def _body(self, sub: SubscriptionDef) -> dict:
        return {
            "type": "Subscription",
            "description": self._description(sub),
            "entities": [{"type": sub.type}],
            "notification": {
                "endpoint": {"uri": self.notification_url, "accept": "application/json"},
                "format": "normalized",
            },
            "throttling": sub.throttling,
            "isActive": True,
        }

    async def ensure_all(self, tenant_ids: list[str]) -> dict:
        """Ensure subscriptions exist for all tenants. Idempotent, never raises."""
        created, skipped, errors = 0, 0, []
        for tenant_id in tenant_ids:
            client = OrionClient(
                tenant_id, base_url=self.orion_url, context_url=self.context_url
            )
            try:
                existing = await client.query_subscriptions(limit=500)
                descriptions = {s.get("description", "") for s in existing}
                for sub in self._subs:
                    if self._description(sub) in descriptions:
                        skipped += 1
                        continue
                    try:
                        await client.create_subscription(self._body(sub))
                        created += 1
                        logger.info(
                            "Subscription created: %s (tenant=%s)",
                            self._description(sub), tenant_id,
                        )
                    except Exception as e:
                        errors.append(f"{tenant_id}/{sub.type}: {e}")
                        logger.error("Subscription create failed: %s", errors[-1])
            except Exception as e:
                errors.append(f"tenant {tenant_id}: {e}")
                logger.warning("Subscription check failed for %s: %s", tenant_id, e)
            finally:
                await client.close()
        return {"created": created, "skipped": skipped, "errors": errors}

    async def periodic_heal(
        self,
        tenant_provider: Callable[[], Awaitable[list[str]]] | list[str],
        interval_minutes: int = 60,
    ) -> None:
        """Reconcile subscriptions periodically. Never exits, never raises.

        `tenant_provider` may be a static list or an async callable returning
        the current tenant list (so tenants activated after startup are healed).
        """
        while True:
            await asyncio.sleep(interval_minutes * 60)
            try:
                tenants = (
                    await tenant_provider()
                    if callable(tenant_provider)
                    else tenant_provider
                )
                result = await self.ensure_all(tenants)
                logger.info(
                    "Subscription heal: created=%d skipped=%d errors=%d",
                    result["created"], result["skipped"], len(result["errors"]),
                )
            except Exception as e:
                logger.warning("Subscription heal cycle failed: %s", e)
