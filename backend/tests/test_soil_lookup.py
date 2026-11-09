"""Soil hydraulics lookup (AgriSoilExtended) and the no-defaults contract."""
import inspect

import app.cog_generator as cog


def test_cog_generator_has_no_legacy_agrisoil_or_texture_defaults():
    src = inspect.getsource(cog)
    assert "fetch_agri_soil" not in src
    assert "get_texture_defaults" not in src


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
    _patch_orion(monkeypatch, [_entity({"sand": 40, "silt": 40, "clay": 20, "organicCarbon": 1.45})])
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


@pytest.mark.asyncio
async def test_texture_without_organic_carbon_is_unavailable(monkeypatch):
    _patch_orion(monkeypatch, [_entity({"sand": 40, "silt": 40, "clay": 20})])
    assert await sources.fetch_parcel_hydraulics("tenant-a", "p1") is None


@pytest.mark.asyncio
async def test_percent_valued_water_content_is_rejected(monkeypatch, caplog):
    import logging
    _patch_orion(monkeypatch, [_entity({
        "availableWaterCapacity": 13.0, "fieldCapacity": 27.0, "wiltingPoint": 14.0,
    })])
    with caplog.at_level(logging.WARNING):
        assert await sources.fetch_parcel_hydraulics("tenant-a", "p1") is None
    assert "fieldCapacity" in caplog.text


@pytest.mark.asyncio
async def test_out_of_range_water_content_falls_to_ptf_with_texture(monkeypatch):
    _patch_orion(monkeypatch, [_entity({
        "fieldCapacity": 27.0, "wiltingPoint": 14.0,
        "sand": 40, "clay": 20, "organicCarbon": 1.45,
    })])
    out = await sources.fetch_parcel_hydraulics("tenant-a", "p1")
    assert out["field_capacity"] == pytest.approx(28, abs=1)


@pytest.mark.asyncio
async def test_several_entities_warns_and_uses_most_recent(monkeypatch, caplog):
    import logging
    old = _entity({"fieldCapacity": 0.20, "wiltingPoint": 0.10})
    old["modifiedAt"] = "2026-01-01T00:00:00Z"
    new = _entity({"fieldCapacity": 0.30, "wiltingPoint": 0.15})
    new["modifiedAt"] = "2026-06-01T00:00:00Z"
    _patch_orion(monkeypatch, [old, new])
    with caplog.at_level(logging.WARNING):
        out = await sources.fetch_parcel_hydraulics("tenant-a", "p1")
    assert out["field_capacity"] == pytest.approx(30.0)
    assert "2 AgriSoilExtended" in caplog.text
