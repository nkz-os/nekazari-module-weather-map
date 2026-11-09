"""Test that cog_generator does NOT call fetch_agri_soil with an empty parcel id."""
import inspect
import app.cog_generator as cog


def test_cog_generator_does_not_call_fetch_agri_soil_with_empty_parcel():
    src = inspect.getsource(cog)
    assert 'fetch_agri_soil(tenant_id, "")' not in src
    assert "fetch_agri_soil(tenant_id, '')" not in src


# ---------------------------------------------------------------------------
# Hydraulics from AgriSoilExtended (synthetic values only)
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock, MagicMock  # noqa: E402

import pytest  # noqa: E402

from app import sources  # noqa: E402


def _entity(horizon):
    return {
        "id": "urn:ngsi-ld:AgriSoilExtended:acme-1",
        "type": "AgriSoilExtended",
        "refAgriParcel": {"type": "Relationship", "object": "urn:ngsi-ld:AgriParcel:p1"},
        "horizons": {"type": "Property", "value": [horizon]},
    }


def _patch_orion(monkeypatch, rows):
    orion = MagicMock()
    orion.query_entities = AsyncMock(return_value=rows)
    orion.close = AsyncMock()
    monkeypatch.setattr(sources, "OrionClient", lambda tenant: orion)
    return orion


@pytest.mark.asyncio
async def test_hydraulics_read_from_extended_horizon(monkeypatch):
    """Orion stores fractions (m3/m3); the module works in % vol."""
    orion = _patch_orion(monkeypatch, [_entity({
        "availableWaterCapacity": 0.15, "fieldCapacity": 0.30, "wiltingPoint": 0.15,
    })])
    out = await sources.fetch_parcel_hydraulics("tenant-a", "urn:ngsi-ld:AgriParcel:p1")
    assert out == pytest.approx(
        {"field_capacity": 30.0, "wilting_point": 15.0, "awc": 15.0}
    )
    kwargs = orion.query_entities.call_args.kwargs
    assert kwargs["type"] == "AgriSoilExtended"
    assert "refAgriParcel" in kwargs["q"] and "hasAgriParcel" in kwargs["q"]


@pytest.mark.asyncio
async def test_hydraulics_fall_back_to_ptf_with_texture_only(monkeypatch):
    _patch_orion(monkeypatch, [_entity({"sand": 40, "silt": 40, "clay": 20})])
    out = await sources.fetch_parcel_hydraulics("tenant-a", "urn:ngsi-ld:AgriParcel:p1")
    assert out["field_capacity"] == pytest.approx(28, abs=1)
    assert out["wilting_point"] == pytest.approx(14, abs=1)


@pytest.mark.asyncio
async def test_hydraulics_none_without_horizon_data(monkeypatch):
    _patch_orion(monkeypatch, [])
    assert await sources.fetch_parcel_hydraulics("tenant-a", "p1") is None
    _patch_orion(monkeypatch, [_entity({"ph": 6.5})])
    assert await sources.fetch_parcel_hydraulics("tenant-a", "p1") is None


@pytest.mark.asyncio
async def test_hydraulics_none_when_orion_fails(monkeypatch):
    orion = _patch_orion(monkeypatch, [])
    orion.query_entities = AsyncMock(side_effect=RuntimeError("boom"))
    assert await sources.fetch_parcel_hydraulics("tenant-a", "p1") is None
