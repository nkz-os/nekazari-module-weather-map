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
