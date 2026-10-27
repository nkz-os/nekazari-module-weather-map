"""Build AgriParcelRecord (FIWARE Agrifood SDM) entities for weather zonal stats.

All historized metrics are FLAT scalar Properties: telemetry-worker's
notification handler drops dict/list attribute values, so a nested blob would
never reach TimescaleDB.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from app.sources import HistoryReadFailed, MissingContextURL, fetch_metric_history

logger = logging.getLogger(__name__)

# weather-map internal metric name -> AgriParcelRecord attribute name.
# Standard SDM names where they exist; custom scalar names otherwise.
# Metric names for zone-level agronomic metrics (subset of _METRIC_TO_ATTR).
_ZONE_METRICS = {
    "tMin": "tMin",
    "tMax": "tMax",
    "eto": "eto",
    "waterBalance": "waterBalance",
    "solarRadiationWm2": "nkz:solarRadiationWm2",
    "frostRisk": "nkz:frostRisk",
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


class FrozenMetric(Exception):
    """A published metric shows zero variance over its last 3 samples.

    Raised instead of letting the value publish unremarked. This is the guard
    that would have caught `airTemperatureAvg = 14.934`, identical to five
    decimals, across 32 straight August days: no exception, no error log, no
    failing test, percentiles and histograms computed faithfully over a
    constant. Zero variance in 3 consecutive daily values is the same cheap
    signature shared by a constant-fed pipeline, a frozen upstream source,
    and a silent `.get(key, default)` — it doesn't matter which caused it.

    Same calling convention as `MissingWeatherInput` in `sources.py`: the
    caller must log at ERROR and skip that one metric — never let one frozen
    metric abort the whole cron run.
    """

    def __init__(self, last_three: list[float]):
        self.last_three = list(last_three)
        super().__init__(f"metric frozen at {last_three[-1]!r} for 3 consecutive samples")


def assert_metric_varies(history: list[float]) -> None:
    """Raise `FrozenMetric` if the 3 most recent values in `history` are identical.

    `AgriParcelRecord` is written once a day, so `history` is expected to be
    the last N daily values for one metric on one parcel, oldest first. Fewer
    than 3 samples is not enough to judge — two identical days happens in
    genuinely stable weather — so this returns `None` rather than guessing.
    """
    if len(history) < 3:
        return
    last_three = history[-3:]
    if len(set(last_three)) == 1:
        raise FrozenMetric(last_three)


async def guard_frozen_metrics(
    tenant_id: str, parcel_id: str, metrics: dict[str, float],
) -> None:
    """Check each about-to-publish metric against its own history; log loud,
    never block. Call this from the publish path, before the record is
    upserted — the record still publishes either way.

    A metric with fewer than 3 total samples (new parcel, or fewer than 2
    prior days on record) stays silent: `assert_metric_varies` returns
    `None` and nothing is logged — a new parcel is not a frozen one.

    A `HistoryReadFailed` (or its `MissingContextURL` subclass) is a
    different failure: the guard could not even ask the question, which must
    not be mistaken for "no history yet" (see `fetch_metric_history`). Logged
    at ERROR — as loudly as a frozen metric, because a guard that quietly
    disables itself is the same silent outcome one level up. Only that one
    metric's check is skipped; the record still publishes.
    """
    for metric_name, value in metrics.items():
        attr = _METRIC_TO_ATTR.get(metric_name)
        if attr is None or value is None or isinstance(value, (dict, list)):
            continue
        try:
            history = await fetch_metric_history(tenant_id, parcel_id, attr)
        except MissingContextURL:
            logger.error(
                "Cannot check %s for tenant=%s parcel=%s: CONTEXT_URL is not "
                "configured, history read would false-empty — skipping the "
                "variance check for this metric, publishing anyway",
                attr, tenant_id, parcel_id,
            )
            continue
        except HistoryReadFailed as exc:
            logger.error(
                "Cannot check %s for tenant=%s parcel=%s: history read failed "
                "(%s) — a broken read is NOT 'no history yet'; skipping the "
                "variance check for this metric, publishing anyway",
                attr, tenant_id, parcel_id, exc,
            )
            continue
        try:
            assert_metric_varies(history + [float(value)])
        except FrozenMetric:
            logger.error(
                "FROZEN METRIC: %s=%r repeated for 3 consecutive samples "
                "(tenant=%s, parcel=%s) — publishing anyway, needs review",
                attr, value, tenant_id, parcel_id,
            )


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


def _round_coords(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_round_coords(v) for v in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 6)
    return value


def zone_version(geometry: dict[str, Any]) -> str:
    """8-hex fingerprint of a zone geometry, stable to the 6th decimal.

    Other modules key their per-zone values on (zone id, version): a changed
    outline under the same id must invalidate them.
    """
    canonical = {
        "type": geometry.get("type"),
        "coordinates": _round_coords(geometry.get("coordinates")),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def _geometry_centroid(geometry: dict[str, Any]) -> list[float]:
    """Return [lon, lat] of a point guaranteed inside a Polygon/MultiPolygon."""
    try:
        from shapely.geometry import shape

        point = shape(geometry).representative_point()
        if point.is_empty:
            logger.warning("zone centroid: empty geometry, falling back to [0, 0]")
            return [0.0, 0.0]
        return [round(point.x, 6), round(point.y, 6)]
    except Exception:
        logger.warning("zone centroid failed, falling back to [0, 0]", exc_info=True)
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
    zone_rank: int | None = None,
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
        "nkz:zoneType": {"type": "Property", "value": "management"},
        "nkz:zoneVersion": {"type": "Property", "value": zone_version(geometry)},
        "nkz:centroid": {"type": "Property", "value": centroid},
        "nkz:elevationMean": {"type": "Property", "value": zone.get("elevationMean", 0.0)},
        "nkz:elevationMin": {"type": "Property", "value": zone.get("elevationMin", 0.0)},
        "nkz:elevationMax": {"type": "Property", "value": zone.get("elevationMax", 0.0)},
        "nkz:aspectSector": {"type": "Property", "value": zone.get("aspectSector", "")},
        "nkz:pixelCount": {"type": "Property", "value": zone.get("pixelCount", 0)},
    }

    if area_ha is None:
        # 10 m pixels -> 0.01 ha each (zone_accumulator.PIXEL_AREA_HA)
        area_ha = round(zone.get("pixelCount", 0) * 0.01, 2)
    entity["nkz:areaHa"] = {"type": "Property", "value": area_ha}

    if zone_rank is not None:
        entity["nkz:zoneRank"] = {"type": "Property", "value": zone_rank}

    if sensor_nearby is not None and sensor_distance_m is not None:
        entity["nkz:sensorNearby"] = {"type": "Property", "value": sensor_nearby}
        entity["nkz:sensorDistanceM"] = {"type": "Property", "value": sensor_distance_m}

    for metric_name, value in metrics.items():
        attr = _ZONE_METRICS.get(metric_name)
        if attr is None or value is None or isinstance(value, (dict, list)):
            continue
        entity[attr] = {"type": "Property", "value": float(value), "observedAt": observed_at}

    return entity
