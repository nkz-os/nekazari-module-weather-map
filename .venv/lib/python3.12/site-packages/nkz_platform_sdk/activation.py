"""ModuleActivation — parcel-level entity lifecycle for module backends.

ensure_entities(): idempotent placeholder creation (POST -> 409 -> PATCH).
get_status(): source-availability map for a parcel.
All Orion-LD I/O delegates to OrionClient (NGSI-LD compliance at SDK level).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from nkz_platform_sdk.orion import OrionClient

logger = logging.getLogger(__name__)


class ModuleActivation:
    """Manage Orion-LD entities for a module's per-parcel activation."""

    def __init__(
        self,
        tenant_id: str,
        orion_url: str | None = None,
        context_url: str | None = None,
    ):
        self.tenant_id = tenant_id
        self._client = OrionClient(
            tenant_id, base_url=orion_url, context_url=context_url
        )

    def _build_entity(self, parcel_urn: str, entity_type: str, id_suffix: str) -> dict:
        parcel_short = parcel_urn.split(":")[-1]
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        # Placeholders carry status/provenance only — NEVER fabricated
        # agronomic values (planting dates etc. are user/pipeline data).
        entity: dict = {
            "id": f"urn:ngsi-ld:{entity_type}:{self.tenant_id}:{parcel_short}-{id_suffix}",
            "type": entity_type,
            "hasAgriParcel": {"type": "Relationship", "object": parcel_urn},
            "status": {"type": "Property", "value": "pending"},
            "provenance": {"type": "Property", "value": "placeholder"},
            "dateCreated": {
                "type": "Property",
                "value": {"@type": "DateTime", "@value": now},
            },
        }
        if entity_type == "CropHealthAssessment":
            entity["dataFidelity"] = {"type": "Property", "value": "pending"}
        elif entity_type == "AgriSoil":
            entity["dataSource"] = {"type": "Property", "value": "pending_analysis"}
        return entity

    async def ensure_entities(self, parcel_id: str, entities: list[dict]) -> dict:
        """Idempotently create placeholder entities for a parcel.

        Args:
            parcel_id: Full AgriParcel URN.
            entities: list of {"type": str, "id_suffix": str}.

        Returns: {"created", "skipped", "errors", "entity_ids"}
        """
        created, skipped, errors, entity_ids = 0, 0, [], []
        for ent_def in entities:
            body = self._build_entity(parcel_id, ent_def["type"], ent_def["id_suffix"])
            eid = body["id"]
            entity_ids.append(eid)
            try:
                await self._client.create_entity(body)
                created += 1
                logger.info("Created %s for parcel %s", eid, parcel_id)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    attrs = {
                        k: v for k, v in body.items()
                        # dateCreated must not be refreshed on idempotent re-runs
                        if k not in ("id", "type", "@context", "dateCreated")
                    }
                    try:
                        await self._client.update_entity_attrs(eid, attrs)
                        skipped += 1
                        logger.debug("Exists, patched: %s", eid)
                    except Exception as pe:
                        errors.append(f"{ent_def['type']}: PATCH failed: {pe}")
                        logger.error("PATCH failed for %s: %s", eid, pe)
                else:
                    errors.append(
                        f"{ent_def['type']}: HTTP {e.response.status_code}"
                    )
                    logger.error("Create failed for %s: %s", eid, errors[-1])
            except Exception as e:
                errors.append(f"{ent_def['type']}: {e}")
                logger.error("Create failed for %s: %s", eid, e)
        return {
            "created": created, "skipped": skipped,
            "errors": errors, "entity_ids": entity_ids,
        }

    async def get_status(
        self, parcel_id: str, entity_types: list[str]
    ) -> dict[str, str]:
        """Map each entity type to "ok" | "unavailable" | "error" for a parcel.

        Queries both the SDM-standard `hasAgriParcel` and the legacy
        `refAgriParcel` relationship in a single OR query.
        """
        q = f'(hasAgriParcel=="{parcel_id}"|refAgriParcel=="{parcel_id}")'
        status: dict[str, str] = {}
        for entity_type in entity_types:
            try:
                found = await self._client.query_entities(
                    type=entity_type, q=q, limit=1
                )
                status[entity_type] = "ok" if found else "unavailable"
            except Exception as e:
                logger.warning("get_status(%s) failed: %s", entity_type, e)
                status[entity_type] = "error"
        return status

    async def close(self) -> None:
        await self._client.close()
