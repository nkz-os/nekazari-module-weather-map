"""`run_for_tenant` publishes one AgriParcelRecord per parcel — y solo si el
ráster que resume es el de hoy.

Round 2 — el fallo que esta rama abrió. Ahora que la meteo puede faltar de
verdad, todos los tiles de una métrica pueden devolver `None`; entonces
`set_latest_date` no avanza y `compute_zonal_stats(date=None)` resuelve al
puntero ANTERIOR. El registro publicaba los píxeles de ayer bajo un
`observedAt` fresco: dato viejo presentado como actual, que es peor que un
hueco porque el hueco sí se ve aguas abajo.

Estos tests tampoco tocan la red: `fetch_metric_history` y la generación de
PMTiles van mockeados. Sin eso ambos ejercían la rama `except` contra un DNS
real y el CI seguía verde con el guard permanentemente roto.
"""

import sys
import types
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app import cog_generator

PARCEL_GEOMETRY = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """El cron sólo debe tocar lo que el test le da."""
    from app import records

    async def _no_history(tenant_id, parcel_id, attr_name, limit=5):
        return []

    monkeypatch.setattr(records, "fetch_metric_history", _no_history)

    # run_for_tenant importa pmtiles_generator dentro de la función y traga
    # cualquier excepción; sin este stub el paso final hablaría con MinIO.
    stub = types.ModuleType("app.pmtiles_generator")
    stub.generate_all_pmtiles = lambda *a, **kw: {}
    monkeypatch.setitem(sys.modules, "app.pmtiles_generator", stub)


def _run(parcels, stats, upsert):
    """Ejecuta run_for_tenant con el pipeline de COGs y MinIO mockeado."""
    return patch.multiple(
        cog_generator,
        upsert_record=upsert,
        compute_zonal_stats=lambda *a, **kw: stats,
        _generate_and_upload_cogs=AsyncMock(return_value=None),
        _parcel_geometry=lambda parcel: PARCEL_GEOMETRY,
    )


@pytest.mark.asyncio
async def test_run_for_tenant_persists_a_record_per_parcel():
    parcels = [{"id": "urn:ngsi-ld:AgriParcel:p1", "lon": -1.5, "lat": 42.0}]
    upsert = AsyncMock()
    stats = {"date": _today(), "metrics": {"eto": {"mean": 4.2}}}
    with _run(parcels, stats, upsert):
        await cog_generator.run_for_tenant(
            "asociacion-allotarra", parcels, "2026-06-01", "2026-06-10",
        )
    upsert.assert_awaited()


@pytest.mark.asyncio
async def test_run_for_tenant_persists_record_for_each_parcel():
    parcels = [
        {"id": "urn:ngsi-ld:AgriParcel:p1", "lon": -1.5, "lat": 42.0},
        {"id": "urn:ngsi-ld:AgriParcel:p2", "lon": -1.6, "lat": 42.1},
    ]
    upsert = AsyncMock()
    stats = {"date": _today(), "metrics": {"eto": {"mean": 3.5}}}
    with _run(parcels, stats, upsert):
        await cog_generator.run_for_tenant(
            "asociacion-allotarra", parcels, "2026-06-01", "2026-06-10",
        )
    assert upsert.await_count == 2


@pytest.mark.asyncio
async def test_yesterdays_raster_is_not_republished_as_todays_observation(caplog):
    """Todos los tiles fallaron → el puntero sigue en ayer → el registro
    resumiría píxeles de ayer con un observedAt de hoy. No se publica."""
    import logging

    parcels = [{"id": "urn:ngsi-ld:AgriParcel:p1", "lon": -1.5, "lat": 42.0}]
    upsert = AsyncMock()
    stats = {"date": "2026-08-31", "metrics": {"temperature_avg": {"mean": 14.934}}}
    with _run(parcels, stats, upsert), caplog.at_level(logging.ERROR):
        await cog_generator.run_for_tenant(
            "asociacion-allotarra", parcels, "2026-06-01", "2026-06-10",
        )

    upsert.assert_not_awaited()
    assert "2026-08-31" in caplog.text
    assert "p1" in caplog.text


@pytest.mark.asyncio
async def test_stats_without_a_date_are_not_published():
    """`compute_zonal_stats` devuelve `{"error": ...}` sin `date` cuando no hay
    ningún COG. Un registro vacío igualmente es una observación falsa."""
    parcels = [{"id": "urn:ngsi-ld:AgriParcel:p1", "lon": -1.5, "lat": 42.0}]
    upsert = AsyncMock()
    stats = {"error": "No COG data available", "metrics": {}}
    with _run(parcels, stats, upsert):
        await cog_generator.run_for_tenant(
            "asociacion-allotarra", parcels, "2026-06-01", "2026-06-10",
        )
    upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_stale_parcel_does_not_block_a_fresh_one():
    """El corte es por parcela: una parcela sin ráster de hoy no puede impedir
    que las demás publiquen."""
    parcels = [
        {"id": "urn:ngsi-ld:AgriParcel:stale", "lon": -1.5, "lat": 42.0},
        {"id": "urn:ngsi-ld:AgriParcel:fresh", "lon": -1.6, "lat": 42.1},
    ]
    upsert = AsyncMock()

    def _stats(tenant_id, geometry, metrics, date=None):
        return {
            "date": "2026-08-31" if _stats.calls.pop(0) else _today(),
            "metrics": {"eto": {"mean": 4.0}},
        }

    _stats.calls = [True, False]

    with patch.multiple(
        cog_generator,
        upsert_record=upsert,
        compute_zonal_stats=_stats,
        _generate_and_upload_cogs=AsyncMock(return_value=None),
        _parcel_geometry=lambda parcel: PARCEL_GEOMETRY,
    ):
        await cog_generator.run_for_tenant(
            "asociacion-allotarra", parcels, "2026-06-01", "2026-06-10",
        )

    assert upsert.await_count == 1
    published = upsert.await_args.args[1]
    assert "fresh" in published["hasAgriParcel"]["object"]
