"""Regression tests for NGSI-LD geometry integration seams.

These use the REAL shapes the code receives at runtime (which the earlier
unit-test mocks did not), guarding two bugs:
1. OrionClient.get_entity() returns NORMALIZED NGSI-LD — location is a
   GeoProperty wrapper that must be unwrapped to raw GeoJSON.
2. fetch_tenant_parcels returns parcels with a `location` GeoJSON dict (no
   `lon`/`lat` keys) — the COG bbox must be derived from that geometry.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.cog_generator import _parcel_lonlat
from app.sources import fetch_agri_parcel

_POLYGON = {
    "type": "Polygon",
    "coordinates": [[[-1.50, 42.00], [-1.49, 42.00], [-1.49, 42.01], [-1.50, 42.01], [-1.50, 42.00]]],
}


@pytest.mark.asyncio
async def test_fetch_agri_parcel_unwraps_geoproperty():
    # Normalized NGSI-LD: location is a GeoProperty wrapper.
    normalized = {
        "id": "urn:ngsi-ld:AgriParcel:p1",
        "type": "AgriParcel",
        "location": {"type": "GeoProperty", "value": _POLYGON},
    }
    orion = AsyncMock()
    orion.get_entity = AsyncMock(return_value=normalized)
    orion.close = AsyncMock()
    with patch("app.sources.OrionClient", return_value=orion):
        result = await fetch_agri_parcel("asociacion-allotarra", "urn:ngsi-ld:AgriParcel:p1")
    # geometry must be the raw GeoJSON Polygon, NOT the GeoProperty wrapper.
    assert result is not None
    assert result["geometry"]["type"] == "Polygon"
    assert result["geometry"] == _POLYGON


def test_parcel_lonlat_from_location_geometry_not_origin():
    # fetch_tenant_parcels-shaped parcel: location GeoJSON, no lon/lat keys.
    parcel = {"id": "urn:ngsi-ld:AgriParcel:p1", "location": _POLYGON}
    lonlat = _parcel_lonlat(parcel)
    assert lonlat is not None
    lon, lat = lonlat
    # Must be near the parcel (~ -1.49, 42.0), NOT the (0,0) Gulf-of-Guinea default.
    assert -1.6 < lon < -1.4
    assert 41.9 < lat < 42.1
