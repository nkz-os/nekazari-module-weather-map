"""weather-map lee la meteo del broker, no de un payload HTTP con forma propia.

El pipeline calculaba sobre constantes porque leía las claves del nivel raíz de la
respuesta de weather-api, donde viven anidadas en `forecast[]`: todos los `.get()`
fallaban y se usaban los defaults. La temperatura publicada fue 14.934 idéntica
durante 32 días. Estos tests fijan que un dato ausente no se sustituye nunca.
"""

import pytest

from app.sources import MissingWeatherInput, require


def test_require_returns_the_first_present_key():
    assert require({"b": 2.0}, "a", "b") == 2.0


def test_require_accepts_the_context_alias():
    """Orion devuelve `airTemperature`, no el `temperature` que el productor escribe."""
    assert require({"airTemperature": 21.0}, "temperature", "airTemperature") == 21.0


def test_require_raises_instead_of_defaulting():
    """Un default aquí es exactamente el bug que este trabajo elimina."""
    with pytest.raises(MissingWeatherInput) as exc:
        require({"otra": 1.0}, "t_min", "temp_min")
    assert "t_min" in str(exc.value)


def test_require_does_not_treat_zero_as_missing():
    """0.0 es un valor legítimo: precipitación cero, parcela a nivel del mar."""
    assert require({"precip_mm": 0.0}, "precip_mm") == 0.0


OBSERVED = {
    "id": "urn:ngsi-ld:WeatherObserved:t:parcel-p1",
    "type": "WeatherObserved",
    "airTemperature": 21.0,
    "humidity": 60.0,
    "windSpeed": 2.5,
    "solarRadiation": 229.4,
    "precipitation": 0.0,
    "location": {"type": "Point", "coordinates": [-2.07, 42.63, 572.8]},
    "dateObserved": {"@type": "DateTime", "@value": "2026-09-02T05:29:56Z"},
}

FORECAST = {
    "id": "urn:ngsi-ld:WeatherForecast:t:parcel-p1",
    "type": "WeatherForecast",
    "dayMinimum": {"temperature": 17.3, "relativeHumidity": 72},
    "dayMaximum": {"temperature": 32.2, "relativeHumidity": 72},
    "precipitationProbability": 5,
}


@pytest.mark.asyncio
async def test_reads_both_entities_and_flattens_them(monkeypatch):
    from app import sources

    async def _fake(tenant_id, entity_id):
        return OBSERVED if "WeatherObserved" in entity_id else FORECAST

    monkeypatch.setattr(sources, "_get_entity_keyvalues", _fake)
    out = await sources.fetch_parcel_weather("t", "urn:ngsi-ld:AgriParcel:t:p1")

    assert out["t_avg"] == 21.0          # del alias airTemperature
    assert out["rh_avg"] == 60.0         # del alias humidity
    assert out["t_min"] == 17.3          # de dayMinimum.temperature
    assert out["t_max"] == 32.2          # de dayMaximum.temperature
    assert out["solar_rad_w_m2"] == 229.4
    assert out["precip_mm"] == 0.0
    assert out["reference_altitude_m"] == 572.8   # tercera coordenada de location


@pytest.mark.asyncio
async def test_returns_none_when_the_observation_is_absent(monkeypatch):
    from app import sources

    async def _fake(tenant_id, entity_id):
        return None

    monkeypatch.setattr(sources, "_get_entity_keyvalues", _fake)
    assert await sources.fetch_parcel_weather("t", "urn:ngsi-ld:AgriParcel:t:p1") is None


@pytest.mark.asyncio
async def test_omits_the_altitude_when_location_is_2d(monkeypatch):
    """Sin tercera coordenada no hay altitud de referencia. No se inventa un 0.0:
    eso aplicaría un gradiente completo desde el nivel del mar."""
    from app import sources

    obs = dict(OBSERVED, location={"type": "Point", "coordinates": [-2.07, 42.63]})

    async def _fake(tenant_id, entity_id):
        return obs if "WeatherObserved" in entity_id else FORECAST

    monkeypatch.setattr(sources, "_get_entity_keyvalues", _fake)
    out = await sources.fetch_parcel_weather("t", "urn:ngsi-ld:AgriParcel:t:p1")
    assert "reference_altitude_m" not in out


# ----------------------------------------------------------------------
# Valor presente pero no convertible: se salta como una ausencia.
# `float()` sobre un dict, un None anidado o "n/a" lanzaba TypeError /
# ValueError, que escapa al handler de MissingWeatherInput y aborta el run
# del tenant a mitad de métrica: algunos COGs subidos, puntero volteado,
# ningún registro escrito. Un valor malformado se salta, no derriba el run.
# ----------------------------------------------------------------------


def test_require_skips_a_non_numeric_value_and_tries_the_next_key():
    assert require({"t_min": "n/a", "temperature_min": 8.5}, "t_min", "temperature_min") == 8.5


def test_require_raises_missing_when_every_candidate_is_malformed():
    with pytest.raises(MissingWeatherInput):
        require({"t_min": {"value": 8.5}}, "t_min")


def test_require_raises_missing_not_typeerror_on_a_list_value():
    with pytest.raises(MissingWeatherInput):
        require({"wind_speed_ms": [2.0, 3.0]}, "wind_speed_ms")


@pytest.mark.asyncio
async def test_malformed_observed_value_is_omitted_not_crashed(monkeypatch):
    """El extractor omite la clave; quien la necesite lanzará
    MissingWeatherInput y el tile se saltará con log ruidoso."""
    from app import sources

    obs = dict(OBSERVED, airTemperature={"value": 21.0}, windSpeed="calm")

    async def _fake(tenant_id, entity_id):
        return obs if "WeatherObserved" in entity_id else FORECAST

    monkeypatch.setattr(sources, "_get_entity_keyvalues", _fake)
    out = await sources.fetch_parcel_weather("t", "urn:ngsi-ld:AgriParcel:t:p1")

    assert "t_avg" not in out
    assert "wind_speed_ms" not in out
    assert out["rh_avg"] == 60.0  # el resto del payload sigue disponible


@pytest.mark.asyncio
async def test_malformed_altitude_or_forecast_temp_is_omitted(monkeypatch):
    from app import sources

    obs = dict(OBSERVED, location={"type": "Point", "coordinates": [-2.07, 42.63, None]})
    fcst = dict(FORECAST, dayMinimum={"temperature": "n/a"})

    async def _fake(tenant_id, entity_id):
        return obs if "WeatherObserved" in entity_id else fcst

    monkeypatch.setattr(sources, "_get_entity_keyvalues", _fake)
    out = await sources.fetch_parcel_weather("t", "urn:ngsi-ld:AgriParcel:t:p1")

    assert "reference_altitude_m" not in out
    assert "t_min" not in out
    assert out["t_max"] == 32.2


# ======================================================================
# fetch_parcel_daily — one signed, dated source (weather-api /daily)
# ======================================================================

import datetime as _dt  # noqa: E402


def _closed(day: _dt.date, precip=1.0, et0=3.0) -> dict:
    return {
        "date": day.isoformat(), "kind": "closed", "tmin_c": 10.0, "tmax_c": 25.0,
        "precip_mm": precip, "et0_mm": et0, "radiation_mj_m2": 18.0,
        "vapour_pressure_kpa": 1.2, "wind2m_ms": 1.5, "source": "x",
    }


def _running(day: _dt.date, **over) -> dict:
    rec = {
        "date": day.isoformat(), "kind": "running", "tmin_c": 11.0, "tmax_c": 28.0,
        "precip_mm": 0.0, "et0_mm": 4.0, "radiation_mj_m2": 21.0,
        "vapour_pressure_kpa": 1.3, "wind2m_ms": 2.0, "source": "x",
    }
    rec.update(over)
    return rec


def _body(running_day: _dt.date, n_closed=5, **running_over) -> dict:
    days = [_closed(running_day - _dt.timedelta(days=k)) for k in range(n_closed, 0, -1)]
    days.append(_running(running_day, **running_over))
    return {"days": days, "totals": {}, "units": {}}


class _FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("err", request=None, response=self)


def _install_fake_weather_api(monkeypatch, body, status=200, elevation=572.8):
    """Fake weather-api and parcel read; returns the list of captured calls."""
    from app import sources

    calls: list = []

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None, headers=None):
            calls.append({"url": url, "params": params, "headers": headers})
            return _FakeResponse(status, body)

    async def _parcel(tenant_id, entity_id):
        return {"id": entity_id, "elevation": elevation} if elevation is not None else {"id": entity_id}

    monkeypatch.setattr(sources.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(sources, "_get_entity_keyvalues", _parcel)
    monkeypatch.setattr(sources.settings, "hmac_secret", "unit-test-secret")
    monkeypatch.setattr(sources.settings, "weather_api_url", "http://weather-api.test")
    return calls


PARCEL_URN = "urn:ngsi-ld:AgriParcel:p1"
RUN_DAY = _dt.date(2026, 7, 6)


@pytest.mark.asyncio
async def test_fetch_parcel_daily_signed_and_parsed(monkeypatch):
    from app import sources
    from nkz_platform_sdk.crypto import verify_hmac_signature

    calls = _install_fake_weather_api(monkeypatch, _body(RUN_DAY))
    out = await sources.fetch_parcel_daily("tenant-a", PARCEL_URN)

    assert len(calls) == 1
    call = calls[0]
    assert call["url"].endswith("/api/weather/parcel/p1/daily")
    h = call["headers"]
    assert h["X-Tenant-ID"] == "tenant-a"
    assert h["X-User-ID"] == "weather-map-worker"
    assert verify_hmac_signature("unit-test-secret", h["X-Auth-Signature"], "", "tenant-a")
    assert call["params"]["forecast_days"] == 1
    assert call["params"]["fill"] == "open_meteo"
    assert out["date"] == RUN_DAY.isoformat()
    assert out["t_min"] == 11.0 and out["t_max"] == 28.0
    assert out["ea_kpa"] == 1.3 and out["u2_ms"] == 2.0 and out["rs_mj"] == 21.0
    assert out["elevation_ref_m"] == 572.8


@pytest.mark.asyncio
async def test_missing_vapour_pressure_omits_key(monkeypatch):
    from app import sources

    _install_fake_weather_api(monkeypatch, _body(RUN_DAY, vapour_pressure_kpa=None))
    out = await sources.fetch_parcel_daily("tenant-a", PARCEL_URN)
    assert "ea_kpa" not in out
    assert out["t_max"] == 28.0
    with pytest.raises(sources.MissingWeatherInput):
        sources.require(out, "ea_kpa")


@pytest.mark.asyncio
async def test_closed5_requires_five_closed_days(monkeypatch):
    from app import sources

    _install_fake_weather_api(monkeypatch, _body(RUN_DAY, n_closed=5))
    out = await sources.fetch_parcel_daily("tenant-a", PARCEL_URN)
    c5 = out["closed5"]
    assert c5["days"] == 5
    assert c5["precip_mm"] == pytest.approx(5.0)
    assert c5["et0_mm"] == pytest.approx(15.0)
    assert c5["start"] == "2026-07-01" and c5["end"] == "2026-07-05"

    _install_fake_weather_api(monkeypatch, _body(RUN_DAY, n_closed=4))
    out = await sources.fetch_parcel_daily("tenant-a", PARCEL_URN)
    assert "closed5" not in out

    # five closed days, but one has no ET0: still not a 5-day window
    body = _body(RUN_DAY, n_closed=5)
    body["days"][2]["et0_mm"] = None
    _install_fake_weather_api(monkeypatch, body)
    out = await sources.fetch_parcel_daily("tenant-a", PARCEL_URN)
    assert "closed5" not in out

    # a gap inside the last five closed days is not a 5-day window either
    body = _body(RUN_DAY, n_closed=6)
    del body["days"][3]
    _install_fake_weather_api(monkeypatch, body)
    out = await sources.fetch_parcel_daily("tenant-a", PARCEL_URN)
    assert "closed5" not in out


@pytest.mark.asyncio
async def test_doy_is_running_day(monkeypatch):
    from app import sources

    _install_fake_weather_api(monkeypatch, _body(_dt.date(2026, 7, 6)))
    out = await sources.fetch_parcel_daily("tenant-a", PARCEL_URN)
    assert out["doy"] == 187


@pytest.mark.asyncio
async def test_unreachable_or_unauthorised_returns_none(monkeypatch, caplog):
    from app import sources

    _install_fake_weather_api(monkeypatch, {"detail": "no"}, status=401)
    assert await sources.fetch_parcel_daily("tenant-a", PARCEL_URN) is None


@pytest.mark.asyncio
async def test_no_secret_means_no_call(monkeypatch):
    from app import sources

    calls = _install_fake_weather_api(monkeypatch, _body(RUN_DAY))
    monkeypatch.setattr(sources.settings, "hmac_secret", "")
    assert await sources.fetch_parcel_daily("tenant-a", PARCEL_URN) is None
    assert calls == []


@pytest.mark.asyncio
async def test_missing_parcel_elevation_omits_reference(monkeypatch):
    from app import sources

    _install_fake_weather_api(monkeypatch, _body(RUN_DAY), elevation=None)
    out = await sources.fetch_parcel_daily("tenant-a", PARCEL_URN)
    assert "elevation_ref_m" not in out


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, 0.0, -5.0, float("nan"), float("inf")])
async def test_non_positive_or_non_finite_elevation_is_unknown(monkeypatch, bad):
    """Elevation <= 0 means unknown upstream: /daily temperatures then refer to the
    model grid, so a 0 m reference would offset the per-pixel correction."""
    from app import sources

    _install_fake_weather_api(monkeypatch, _body(RUN_DAY), elevation=bad)
    out = await sources.fetch_parcel_daily("tenant-a", PARCEL_URN)
    assert "elevation_ref_m" not in out
