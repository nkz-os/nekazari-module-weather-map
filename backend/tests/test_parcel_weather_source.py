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
