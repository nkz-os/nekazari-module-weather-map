"""Test that run_for_tenant persists one AgriParcelRecord per parcel after COG gen."""
from unittest.mock import AsyncMock, patch
import pytest
from app import cog_generator


@pytest.mark.asyncio
async def test_run_for_tenant_persists_a_record_per_parcel():
    parcels = [{"id": "urn:ngsi-ld:AgriParcel:p1", "lon": -1.5, "lat": 42.0}]
    upsert = AsyncMock()
    with patch.object(cog_generator, "upsert_record", upsert), \
         patch.object(cog_generator, "compute_zonal_stats", return_value={"metrics": {"eto": {"mean": 4.2}}}), \
         patch.object(cog_generator, "_generate_and_upload_cogs", AsyncMock(return_value=None)), \
         patch.object(cog_generator, "_parcel_geometry", return_value={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}):
        await cog_generator.run_for_tenant("asociacion-allotarra", parcels, "2026-06-01", "2026-06-10")
    upsert.assert_awaited()


@pytest.mark.asyncio
async def test_run_for_tenant_persists_record_for_each_parcel():
    parcels = [
        {"id": "urn:ngsi-ld:AgriParcel:p1", "lon": -1.5, "lat": 42.0},
        {"id": "urn:ngsi-ld:AgriParcel:p2", "lon": -1.6, "lat": 42.1},
    ]
    upsert = AsyncMock()
    with patch.object(cog_generator, "upsert_record", upsert), \
         patch.object(cog_generator, "compute_zonal_stats", return_value={"metrics": {"eto": {"mean": 3.5}}}), \
         patch.object(cog_generator, "_generate_and_upload_cogs", AsyncMock(return_value=None)), \
         patch.object(cog_generator, "_parcel_geometry", return_value={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}):
        await cog_generator.run_for_tenant("asociacion-allotarra", parcels, "2026-06-01", "2026-06-10")
    assert upsert.await_count == 2
