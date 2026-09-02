"""El proxy de forecast tampoco fabrica: un ET0 ausente no es un ET0 de 0.

`forecast.py` sobrevivió a la limpieza porque el test de contrato no miraba este
fichero ni reconocía la forma `.get("eto_mm") or 0.0`. El consumidor es el worker
de balance hídrico del módulo de suelo: con ET0 = 0 el déficit proyectado se
queda plano y se lee como "sin estrés hídrico". Un hueco es visible; ese cero no.
"""

from __future__ import annotations

import logging

import pytest

from app.forecast import _to_worker_forecast


def test_keeps_a_complete_day():
    out = _to_worker_forecast(
        [{"date": "2026-09-02", "eto_mm": 4.2, "precip_mm": 0.0}], "p1",
    )
    assert out == [
        {"day": "2026-09-02", "et0": 4.2, "precip": 0.0, "deficitAfter": 0.0},
    ]


def test_zero_et0_is_a_real_value_not_an_absence():
    """0.0 es un ET0 legítimo (día cerrado, invierno). Se publica tal cual."""
    out = _to_worker_forecast(
        [{"date": "2026-01-02", "eto_mm": 0.0, "precip_mm": 12.0}], "p1",
    )
    assert out[0]["et0"] == 0.0


def test_drops_a_day_without_et0_instead_of_calling_it_zero(caplog):
    with caplog.at_level(logging.WARNING):
        out = _to_worker_forecast(
            [
                {"date": "2026-09-02", "eto_mm": 4.2, "precip_mm": 0.0},
                {"date": "2026-09-03", "precip_mm": 1.0},
            ],
            "urn:ngsi-ld:AgriParcel:p1",
        )
    assert [d["day"] for d in out] == ["2026-09-02"]
    assert "p1" in caplog.text


def test_drops_a_day_without_precipitation():
    out = _to_worker_forecast([{"date": "2026-09-03", "eto_mm": 4.2}], "p1")
    assert out == []


def test_drops_a_non_numeric_value():
    """Un valor presente pero no convertible se salta como uno ausente; nunca
    revienta el endpoint entero."""
    out = _to_worker_forecast(
        [{"date": "2026-09-03", "eto_mm": "n/a", "precip_mm": 1.0}], "p1",
    )
    assert out == []


@pytest.mark.asyncio
async def test_endpoint_502s_when_no_day_is_usable(monkeypatch):
    """Un forecast entero sin ET0 devuelto como lista vacía sería el mismo
    fallo silencioso en otra forma: el worker leería "sin datos" como "sin
    déficit". Se falla con 502."""
    from fastapi import HTTPException

    from app import forecast as forecast_mod

    async def _fake_parcel(tenant_id, parcel_id):
        return {"geometry": {"type": "Point", "coordinates": [-1.6, 42.8]}}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"forecast": [{"date": "2026-09-02"}, {"date": "2026-09-03"}]}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **kw):
            return _Resp()

    monkeypatch.setattr(forecast_mod, "fetch_agri_parcel", _fake_parcel)
    monkeypatch.setattr(forecast_mod.httpx, "AsyncClient", lambda **kw: _Client())

    with pytest.raises(HTTPException) as exc:
        await forecast_mod.forecast_et0(parcel_id="p1", days=7, tenant_id="t")
    assert exc.value.status_code == 502
