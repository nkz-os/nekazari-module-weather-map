"""Build AgriParcelRecord (FIWARE Agrifood SDM) entities for weather zonal stats.

All historized metrics are FLAT scalar Properties: telemetry-worker's
notification handler drops dict/list attribute values, so a nested blob would
never reach TimescaleDB.
"""

from __future__ import annotations

import re
from typing import Any

# weather-map internal metric name -> AgriParcelRecord attribute name.
# Standard SDM names where they exist; custom scalar names otherwise.
_METRIC_TO_ATTR = {
    "solar_radiation": "solarRadiation",
    "soil_moisture": "soilMoistureVwc",
    "soil_temperature": "soilTemperature",
    "relative_humidity": "relativeHumidity",
    "eto": "eto",
    "water_balance": "waterBalance",
    "frost_risk": "frostRisk",
    "temperature_avg": "airTemperatureAvg",
    "temperature_min": "airTemperatureMin",
}


def _parcel_short(parcel_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "-", parcel_id.split(":")[-1]).strip("-")


def build_agri_parcel_record(
    *,
    tenant_id: str,
    parcel_id: str,
    geometry: dict[str, Any],
    metrics: dict[str, float],
    observed_at: str,
) -> dict[str, Any]:
    """Return an NGSI-LD AgriParcelRecord dict (no @context; added by the SDK)."""
    ts_compact = re.sub(r"[^0-9]", "", observed_at)
    entity: dict[str, Any] = {
        "id": f"urn:ngsi-ld:AgriParcelRecord:weather-{tenant_id}-{_parcel_short(parcel_id)}-{ts_compact}",
        "type": "AgriParcelRecord",
        "hasAgriParcel": {"type": "Relationship", "object": parcel_id},
        "location": {"type": "GeoProperty", "value": geometry},
        "dateObserved": {"type": "Property", "value": {"@type": "DateTime", "@value": observed_at}},
    }
    for metric_name, value in metrics.items():
        attr = _METRIC_TO_ATTR.get(metric_name)
        if attr is None or value is None or isinstance(value, (dict, list)):
            continue
        entity[attr] = {"type": "Property", "value": float(value), "observedAt": observed_at}
    return entity
