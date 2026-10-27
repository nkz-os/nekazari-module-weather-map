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
    assert z["sensor_nearby"] == "urn:ngsi-ld:Device:near"
    assert z["nkz:sensorDistanceM"]["value"] == pytest.approx(10.0, abs=0.5)
