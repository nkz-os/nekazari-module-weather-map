"""Tests for the zero-variance guard — app.records.assert_metric_varies.

The incident this replaces: airTemperatureAvg = 14.934 published identical
to five decimals on 32 consecutive August days, with no error and no failing
test. Three unmoving daily values is the cheap, shared signature of a
constant-fed pipeline, a frozen upstream source, or a silent default.

Round 2: the guard was correct but unwired — nobody called it, which is the
same silent outcome in a different shape. These tests cover the wiring:
`guard_frozen_metrics` fetches a metric's own history from the broker via
`fetch_metric_history`, appends the value about to publish, and runs it
through `assert_metric_varies` — logging loud, never blocking the publish.
"""

from __future__ import annotations

import logging

import pytest

from app.records import FrozenMetric, assert_metric_varies, guard_frozen_metrics
from app.sources import MissingContextURL, fetch_metric_history


def test_flags_a_frozen_metric():
    """14.934 tres días seguidos no es meteorología, es una constante."""
    with pytest.raises(FrozenMetric):
        assert_metric_varies([14.934, 14.934, 14.934])


def test_accepts_a_metric_that_moves():
    assert assert_metric_varies([14.9, 17.3, 21.0]) is None


def test_needs_three_samples_before_judging():
    assert assert_metric_varies([14.934, 14.934]) is None


@pytest.mark.asyncio
async def test_guard_fires_through_the_publish_path(monkeypatch, caplog):
    """3 identical values reached the way records.py reaches them: fetch
    the 2 prior published values, append the one about to publish, check.
    The record still publishes — this only asserts the ERROR is loud and
    names tenant, parcel, metric, and the repeated value."""
    from app import records

    async def _fake_history(tenant_id, parcel_id, attr_name, limit=5):
        return [14.934, 14.934]

    monkeypatch.setattr(records, "fetch_metric_history", _fake_history)
    with caplog.at_level(logging.ERROR):
        await guard_frozen_metrics(
            "asociacion-allotarra",
            "urn:ngsi-ld:AgriParcel:p1",
            {"temperature_avg": 14.934},
        )

    assert "asociacion-allotarra" in caplog.text
    assert "p1" in caplog.text
    assert "airTemperatureAvg" in caplog.text
    assert "14.934" in caplog.text


@pytest.mark.asyncio
async def test_guard_stays_quiet_with_fewer_than_three_samples(monkeypatch, caplog):
    """A new parcel (no history yet) must not block publishing or log an error."""
    from app import records

    async def _fake_history(tenant_id, parcel_id, attr_name, limit=5):
        return []  # new parcel — nothing published for it before

    monkeypatch.setattr(records, "fetch_metric_history", _fake_history)
    with caplog.at_level(logging.ERROR):
        await guard_frozen_metrics(
            "t", "urn:ngsi-ld:AgriParcel:new", {"temperature_avg": 14.934},
        )

    assert caplog.records == []


@pytest.mark.asyncio
async def test_missing_context_url_is_caught_not_silently_empty(monkeypatch):
    """Without the platform @context Link, Orion expands against the default
    vocabulary and every query false-empties — indistinguishable from "no
    history yet". fetch_metric_history must refuse loudly instead of
    returning [] and letting the guard mistake a broken query for a new
    parcel."""
    from app.config import settings

    monkeypatch.setattr(settings, "context_url", "")
    with pytest.raises(MissingContextURL):
        await fetch_metric_history(
            "t", "urn:ngsi-ld:AgriParcel:p1", "airTemperatureAvg",
        )


# ======================================================================
# Ronda 3 — el guard podía deshabilitarse a sí mismo en silencio.
#
# Cualquier error que no fuera un 404 (un 400 por `orderBy`, un 500 de
# Orion, un timeout) devolvía `[]`, y `guard_frozen_metrics` no puede
# distinguir eso de "parcela nueva, aún sin histórico". El camino de
# `MissingContextURL` estaba cuidado, pero `CONTEXT_URL` tiene default no
# vacío en config.py: en producción esa rama es casi inalcanzable y la
# rama del `[]` es la viva. Un guard que se apaga solo no es un guard.
# ======================================================================


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, raise_exc=None):
        self.status_code = status_code
        self._payload = payload
        self._raise = raise_exc

    def raise_for_status(self):
        if self._raise is not None:
            raise self._raise

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, response=None, get_exc=None):
        self._response = response
        self._get_exc = get_exc
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, headers=None):
        self.calls.append({"url": url, "params": params or {}})
        if self._get_exc is not None:
            raise self._get_exc
        return self._response


def _patch_client(monkeypatch, client):
    from app import sources

    monkeypatch.setattr(sources.httpx, "AsyncClient", lambda **kw: client)
    return client


def _record(ts: str, attr: str, value: float) -> dict:
    """Un AgriParcelRecord tal y como lo escribe records.build_agri_parcel_record."""
    return {
        "id": f"urn:ngsi-ld:AgriParcelRecord:weather-asociacion-allotarra-p1-{ts}",
        "type": "AgriParcelRecord",
        attr: value,
    }


@pytest.mark.asyncio
async def test_transport_failure_is_not_an_empty_history(monkeypatch):
    """Un timeout o un 500 no es "parcela nueva". Devolver [] aquí deja el
    guard permanentemente mudo y verde."""
    import httpx

    from app.sources import HistoryReadFailed

    _patch_client(monkeypatch, _FakeClient(get_exc=httpx.ConnectTimeout("boom")))
    with pytest.raises(HistoryReadFailed):
        await fetch_metric_history("t", "urn:ngsi-ld:AgriParcel:p1", "airTemperatureAvg")


@pytest.mark.asyncio
async def test_http_error_status_is_a_read_failure(monkeypatch):
    """Un 400 por un `orderBy` no soportado degradaría el guard en silencio."""
    import httpx

    from app.sources import HistoryReadFailed

    exc = httpx.HTTPStatusError("400", request=None, response=None)
    _patch_client(monkeypatch, _FakeClient(response=_FakeResponse(400, raise_exc=exc)))
    with pytest.raises(HistoryReadFailed):
        await fetch_metric_history("t", "urn:ngsi-ld:AgriParcel:p1", "airTemperatureAvg")


@pytest.mark.asyncio
async def test_404_is_genuine_absence_not_a_failure(monkeypatch):
    """Un 404 sí es ausencia real: parcela sin histórico. Devuelve []."""
    _patch_client(monkeypatch, _FakeClient(response=_FakeResponse(404)))
    assert await fetch_metric_history(
        "t", "urn:ngsi-ld:AgriParcel:p1", "airTemperatureAvg",
    ) == []


@pytest.mark.asyncio
async def test_a_read_failure_is_logged_as_loudly_as_a_frozen_metric(monkeypatch, caplog):
    """El guard no bloquea la publicación, pero el fallo de lectura tiene que
    salir a ERROR con tenant, parcela y métrica — igual que un frozen."""
    from app import records
    from app.sources import HistoryReadFailed

    async def _boom(tenant_id, parcel_id, attr_name, limit=5):
        raise HistoryReadFailed("orion timeout")

    monkeypatch.setattr(records, "fetch_metric_history", _boom)
    with caplog.at_level(logging.ERROR):
        await guard_frozen_metrics(
            "asociacion-allotarra",
            "urn:ngsi-ld:AgriParcel:p1",
            {"temperature_avg": 14.934},
        )

    assert caplog.records, "un fallo de lectura del histórico no puede ser silencioso"
    assert caplog.records[0].levelno >= logging.ERROR
    assert "asociacion-allotarra" in caplog.text
    assert "p1" in caplog.text
    assert "airTemperatureAvg" in caplog.text


# ----------------------------------------------------------------------
# Orden: la serie la ordenamos nosotros, no el servidor.
# `orderBy=dateObserved:desc` sobre una Property anidada
# {"@type":"DateTime","@value":...} no está verificado contra Orion-LD.
# Si lo ignora, `values.reverse()` daba un orden arbitrario y el guard
# degradaba en silencio. El id que weather-map escribe lleva la marca de
# tiempo compacta al final: ese es el valor que controlamos.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_is_sorted_client_side_oldest_first(monkeypatch):
    payload = [
        _record("20260830120000", "airTemperatureAvg", 18.0),
        _record("20260901120000", "airTemperatureAvg", 21.0),
        _record("20260831120000", "airTemperatureAvg", 19.0),
    ]
    _patch_client(monkeypatch, _FakeClient(response=_FakeResponse(200, payload)))
    values = await fetch_metric_history(
        "t", "urn:ngsi-ld:AgriParcel:p1", "airTemperatureAvg",
    )
    assert values == [18.0, 19.0, 21.0]


@pytest.mark.asyncio
async def test_ignored_server_ordering_is_reported(monkeypatch, caplog):
    """Si Orion no honra el `orderBy`, el orden nuestro lo arregla, pero el
    hecho se registra: un `limit` sobre un orden arbitrario elige una
    ventana arbitraria y eso debilita el guard."""
    payload = [
        _record("20260830120000", "airTemperatureAvg", 18.0),
        _record("20260831120000", "airTemperatureAvg", 19.0),
        _record("20260901120000", "airTemperatureAvg", 21.0),
    ]  # ascendente = NO es el desc que pedimos
    _patch_client(monkeypatch, _FakeClient(response=_FakeResponse(200, payload)))
    with caplog.at_level(logging.WARNING):
        await fetch_metric_history("t", "urn:ngsi-ld:AgriParcel:p1", "airTemperatureAvg")
    assert "orderBy" in caplog.text


@pytest.mark.asyncio
async def test_another_producers_record_is_never_read_as_weather(monkeypatch):
    """AgriParcelRecord lo comparte un pipeline de fotos de campo."""
    payload = [
        _record("20260901120000", "airTemperatureAvg", 21.0),
        {
            "id": "urn:ngsi-ld:AgriParcelRecord:photo-t-p1-20260901130000",
            "airTemperatureAvg": 99.0,
        },
    ]
    _patch_client(monkeypatch, _FakeClient(response=_FakeResponse(200, payload)))
    assert await fetch_metric_history(
        "t", "urn:ngsi-ld:AgriParcel:p1", "airTemperatureAvg",
    ) == [21.0]
