"""`run_for_tenant` publishes one AgriParcelRecord per parcel — y solo si el
ráster que resume es el de hoy.

Round 2 — el fallo que esta rama abrió. Ahora que la meteo puede faltar de
verdad, todos los tiles de una métrica pueden devolver `None`; entonces
`set_latest_date` no avanza y `compute_zonal_stats(date=None)` resuelve al
puntero ANTERIOR. El registro publicaba los píxeles de ayer bajo un
`observedAt` fresco: dato viejo presentado como actual, que es peor que un
hueco porque el hueco sí se ve aguas abajo.

Estos tests tampoco tocan la red: `fetch_metric_history` va mockeado. Sin eso
ejercía la rama `except` contra un DNS
real y el CI seguía verde con el guard permanentemente roto.
"""

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
            "demo-coop", parcels, "2026-06-01", "2026-06-10",
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
            "demo-coop", parcels, "2026-06-01", "2026-06-10",
        )
    assert upsert.await_count == 2


@pytest.mark.asyncio
async def test_run_does_not_invoke_pmtiles(monkeypatch):
    """The run has no PMTiles step: no generator module, nothing imports it."""
    import importlib.util
    import sys

    assert importlib.util.find_spec("app.pmtiles_generator") is None
    monkeypatch.setitem(sys.modules, "app.pmtiles_generator", None)  # import would raise
    parcels = [{"id": "urn:ngsi-ld:AgriParcel:p1", "lon": -1.5, "lat": 42.0}]
    upsert = AsyncMock()
    stats = {"date": _today(), "metrics": {"eto": {"mean": 3.5}}}
    with _run(parcels, stats, upsert), patch.object(cog_generator.logger, "error") as err:
        await cog_generator.run_for_tenant(
            "demo-coop", parcels, "2026-06-01", "2026-06-10",
        )
    upsert.assert_awaited()
    assert not any("PMTiles" in str(c) for c in err.call_args_list)


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
            "demo-coop", parcels, "2026-06-01", "2026-06-10",
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
            "demo-coop", parcels, "2026-06-01", "2026-06-10",
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
            "demo-coop", parcels, "2026-06-01", "2026-06-10",
        )

    assert upsert.await_count == 1
    published = upsert.await_args.args[1]
    assert "fresh" in published["hasAgriParcel"]["object"]


@pytest.mark.asyncio
async def test_record_omits_soil_moisture_when_unavailable():
    parcels = [{"id": "urn:ngsi-ld:AgriParcel:p1", "lon": -1.5, "lat": 42.0}]
    upsert = AsyncMock()
    stats = {
        "date": _today(),
        "metrics": {
            "eto": {"mean": 4.2},
            "soil_moisture": {"error": "no data"},
        },
    }
    with _run(parcels, stats, upsert):
        await cog_generator.run_for_tenant(
            "demo-coop", parcels, "2026-06-01", "2026-06-10",
        )
    record = upsert.await_args.args[1]
    assert "eto" in record
    assert "soilMoistureVwc" not in record


@pytest.mark.asyncio
async def test_record_omits_soil_moisture(monkeypatch):
    """Real zonal stats: soil_moisture has no raster and no record attribute."""
    from app import stats as stats_mod

    monkeypatch.setattr(cog_generator.settings, "metrics", ["eto", "soil_moisture"])
    monkeypatch.setattr(stats_mod.settings, "metrics", ["eto", "soil_moisture"])
    monkeypatch.setattr(stats_mod, "get_latest_date", lambda t, m: _today() if m == "eto" else None)
    import numpy as np
    monkeypatch.setattr(stats_mod, "_collect_pixels", lambda *a, **kw: np.array([3.0, 4.0]))
    parcels = [{"id": "urn:ngsi-ld:AgriParcel:p1", "lon": -1.5, "lat": 42.0}]
    upsert = AsyncMock()
    with patch.multiple(
        cog_generator,
        upsert_record=upsert,
        _generate_and_upload_cogs=AsyncMock(return_value=None),
        _parcel_geometry=lambda parcel: PARCEL_GEOMETRY,
    ):
        await cog_generator.run_for_tenant("demo-coop", parcels, "2026-06-01", "2026-06-10")
    record = upsert.await_args.args[1]
    assert "soilMoistureVwc" not in record
    assert "eto" in record
