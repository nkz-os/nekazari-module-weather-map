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
# Metric names for zone-level agronomic metrics (subset of _METRIC_TO_ATTR).
_ZONE_METRICS = {
    "tMin": "tMin",
    "tMax": "tMax",
    "eto": "eto",
    "waterBalance": "waterBalance",
}


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


def _geometry_centroid(geometry: dict[str, Any]) -> list[float]:
    """Return [lon, lat] from a GeoJSON Polygon geometry (vertex mean)."""
    try:
        if not isinstance(geometry, dict):
            return [0.0, 0.0]
        ring = geometry.get("coordinates", [])
        if not ring or not ring[0]:
            return [0.0, 0.0]
        coords = ring[0]
        # Exclude closing duplicate vertex
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if not coords:
            return [0.0, 0.0]
        n = len(coords)
        lon = sum(c[0] for c in coords) / n
        lat = sum(c[1] for c in coords) / n
        return [round(lon, 6), round(lat, 6)]
    except (KeyError, IndexError, TypeError, ZeroDivisionError):
        return [0.0, 0.0]


def build_agri_parcel_zone(
    *,
    tenant_id: str,
    parcel_id: str,
    zone: dict[str, Any],
    geometry: dict[str, Any],
    metrics: dict[str, float],
    observed_at: str,
    area_ha: float | None = None,
    sensor_nearby: bool | None = None,
    sensor_distance_m: float | None = None,
) -> dict[str, Any]:
    """Return an NGSI-LD AgriParcelZone dict (no @context; added by the SDK).

    Entity ID is STATIC (no timestamp) — attributes are updated in place via
    entityOperations/upsert each time zonal stats are recomputed.
    """
    parcel_short = _parcel_short(parcel_id)
    # zone_accumulator builds the zone dict with key "id"; accept both so each
    # zone gets a DISTINCT entity id (else all collapse to ...:unknown).
    zone_id = zone.get("zone_id") or zone.get("id") or "unknown"
    centroid = _geometry_centroid(geometry)

    entity: dict[str, Any] = {
        "id": f"urn:ngsi-ld:AgriParcelZone:{tenant_id}:{parcel_short}:{zone_id}",
        "type": "AgriParcelZone",
        "hasAgriParcel": {"type": "Relationship", "object": parcel_id},
        "location": {"type": "GeoProperty", "value": geometry},
        "dateObserved": {
            "type": "Property",
            "value": {"@type": "DateTime", "@value": observed_at},
        },
        "nkz:zoneId": {"type": "Property", "value": zone_id},
        "nkz:centroid": {"type": "Property", "value": centroid},
        "nkz:elevationMean": {"type": "Property", "value": zone.get("elevationMean", 0.0)},
        "nkz:elevationMin": {"type": "Property", "value": zone.get("elevationMin", 0.0)},
        "nkz:elevationMax": {"type": "Property", "value": zone.get("elevationMax", 0.0)},
        "nkz:aspectSector": {"type": "Property", "value": zone.get("aspectSector", "")},
        "nkz:pixelCount": {"type": "Property", "value": zone.get("pixelCount", 0)},
    }

    if area_ha is not None:
        entity["nkz:areaHa"] = {"type": "Property", "value": area_ha}

    if sensor_nearby is not None and sensor_distance_m is not None:
        entity["nkz:sensorNearby"] = {"type": "Property", "value": sensor_nearby}
        entity["nkz:sensorDistanceM"] = {"type": "Property", "value": sensor_distance_m}

    for metric_name, value in metrics.items():
        attr = _ZONE_METRICS.get(metric_name)
        if attr is None or value is None or isinstance(value, (dict, list)):
            continue
        entity[attr] = {"type": "Property", "value": float(value), "observedAt": observed_at}

    return entity
