"""find_nearby_sensors: a sensor is only attached to a zone within the radius."""
import math
from unittest.mock import AsyncMock, patch

import pytest

from app.sources import SENSOR_NEARBY_MAX_M, find_nearby_sensors

# ~100 m x 100 m square near lat 42.8
_LAT0, _LON0 = 42.8, -1.65
_DLAT = 100 / 110540.0
_DLON = 100 / (math.cos(math.radians(_LAT0)) * 111320.0)
_POLY = {
    "type": "Polygon",
    "coordinates": [[
        [_LON0, _LAT0], [_LON0 + _DLON, _LAT0], [_LON0 + _DLON, _LAT0 + _DLAT],
        [_LON0, _LAT0 + _DLAT], [_LON0, _LAT0],
    ]],
}


def _zone():
    return {"id": "urn:ngsi-ld:AgriParcelZone:acme:p1:z1",
            "location": {"type": "GeoProperty", "value": _POLY}}


def _device(lon, lat, did="urn:ngsi-ld:Device:d1"):
    return {"id": did, "location": {"type": "Point", "coordinates": [lon, lat]}}


def _patch(devices):
    orion = AsyncMock()
    orion.query_entities = AsyncMock(return_value=devices)
    orion.close = AsyncMock()
    return patch("app.sources.OrionClient", return_value=orion)


async def _run(devices):
    with _patch(devices):
        return (await find_nearby_sensors("acme", "p1", [_zone()]))[0]


def test_radius_constant():
    assert SENSOR_NEARBY_MAX_M == 50.0


@pytest.mark.asyncio
async def test_device_2km_away_is_not_attached():
    z = await _run([_device(_LON0 + 2000 / (math.cos(math.radians(_LAT0)) * 111320.0), _LAT0)])
    assert "nkz:sensorNearby" not in z
    assert "nkz:sensorDistanceM" not in z
    assert "sensor_nearby" not in z


@pytest.mark.asyncio
async def test_device_30m_from_edge():
    z = await _run([_device(_LON0 - 30 / (math.cos(math.radians(_LAT0)) * 111320.0), _LAT0 + _DLAT / 2)])
    assert z["nkz:sensorNearby"]["value"] is True
    assert z["nkz:sensorDistanceM"]["value"] == pytest.approx(30.0, abs=1.0)


@pytest.mark.asyncio
async def test_device_inside_is_zero():
    z = await _run([_device(_LON0 + _DLON / 2, _LAT0 + _DLAT / 2)])
    assert z["nkz:sensorNearby"]["value"] is True
    assert z["nkz:sensorDistanceM"]["value"] == 0.0


@pytest.mark.asyncio
async def test_nearest_within_radius_wins():
    near = _device(_LON0 - 10 / (math.cos(math.radians(_LAT0)) * 111320.0), _LAT0 + _DLAT / 2, "urn:ngsi-ld:Device:near")
    far = _device(_LON0 - 40 / (math.cos(math.radians(_LAT0)) * 111320.0), _LAT0 + _DLAT / 2, "urn:ngsi-ld:Device:far")
    z = await _run([far, near])
    assert "sensor_nearby" not in z
    assert z["nkz:sensorDistanceM"]["value"] == pytest.approx(10.0, abs=0.5)


def _east(m):
    return _LON0 - m / (math.cos(math.radians(_LAT0)) * 111320.0)


@pytest.mark.asyncio
async def test_boundary_50m_nearby_60m_not():
    mid = _LAT0 + _DLAT / 2
    z = await _run([_device(_east(49.5), mid)])
    assert z["nkz:sensorNearby"]["value"] is True
    z = await _run([_device(_east(60), mid)])
    assert "nkz:sensorNearby" not in z


@pytest.mark.asyncio
async def test_final_payload_is_ngsi_only():
    from app.records import build_agri_parcel_zone
    e = build_agri_parcel_zone(
        tenant_id="acme", parcel_id="urn:ngsi-ld:AgriParcel:p1",
        zone={"id": "z1", "pixelCount": 10},
        geometry=_POLY, metrics={"tMin": 1.0}, observed_at="2026-06-22T00:00:00Z",
        zone_rank=1,
    )
    with _patch([_device(_LON0 + _DLON / 2, _LAT0 + _DLAT / 2)]):
        out = (await find_nearby_sensors("acme", "p1", [e]))[0]
    assert "nkz:sensorNearby" in out
    from app.sources import upsert_agri_parcel_zones
    fake = _FakeOrion()
    with patch("app.sources.OrionClient", return_value=fake):
        await upsert_agri_parcel_zones("acme", [out])
    for k, v in fake.sent[0].items():
        if k in ("id", "type", "@context"):
            continue
        assert isinstance(v, dict) and v.get("type") in ("Property", "Relationship", "GeoProperty"), k


class _FakeOrion:
    def __init__(self, fail_404=False):
        self.deleted = []
        self.fail_404 = fail_404
        self.sent = []

    async def upsert_entities_batch(self, zones):
        self.sent = zones
        return {"upserted": len(zones), "errors": []}

    async def delete_entity_attr(self, eid, attr):
        self.deleted.append((eid, attr))
        if self.fail_404:
            import httpx
            req = httpx.Request("DELETE", "http://x")
            raise httpx.HTTPStatusError("nf", request=req, response=httpx.Response(404, request=req))

    async def close(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_404", [False, True])
async def test_upsert_clears_sensor_attrs_only_where_no_sensor(fail_404):
    from app.sources import upsert_agri_parcel_zones
    fake = _FakeOrion(fail_404)
    zones = [
        {"id": "with", "nkz:sensorNearby": {"type": "Property", "value": True}, "_sensor_lookup_ok": True},
        {"id": "without", "_sensor_lookup_ok": True},
        {"id": "unchecked"},
    ]
    with patch("app.sources.OrionClient", return_value=fake):
        await upsert_agri_parcel_zones("acme", zones)
    assert fake.deleted == [("without", "nkz:sensorNearby"), ("without", "nkz:sensorDistanceM")]
    assert fake.sent and all("_sensor_lookup_ok" not in z for z in fake.sent)


@pytest.mark.asyncio
async def test_device_fetch_failure_clears_nothing():
    from app.sources import upsert_agri_parcel_zones
    orion = AsyncMock()
    orion.query_entities = AsyncMock(side_effect=ValueError("orion down"))
    orion.close = AsyncMock()
    with patch("app.sources.OrionClient", return_value=orion):
        out = await find_nearby_sensors("acme", "p1", [_zone()])
    fake = _FakeOrion()
    with patch("app.sources.OrionClient", return_value=fake):
        await upsert_agri_parcel_zones("acme", out)
    assert fake.deleted == []


@pytest.mark.asyncio
async def test_zone_without_location_is_not_cleared():
    from app.sources import upsert_agri_parcel_zones
    with _patch([_device(_LON0, _LAT0)]):
        out = await find_nearby_sensors("acme", "p1", [{"id": "noloc"}])
    fake = _FakeOrion()
    with patch("app.sources.OrionClient", return_value=fake):
        await upsert_agri_parcel_zones("acme", out)
    assert fake.deleted == []
