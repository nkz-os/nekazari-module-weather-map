"""Contrato: la capa de física no fabrica entradas ausentes.

Este módulo publicó `airTemperatureAvg = 14.934` en 32 días consecutivos de agosto
—el default 15.0 menos la corrección de altitud— porque cada `.get(clave, literal)`
convertía una ausencia en un número plausible. Nada falló, nada avisó.

El test lee el fuente: un literal numérico como segundo argumento de `.get()` sobre
una variable meteorológica es el defecto, aunque el valor parezca razonable.
"""

import ast
import pathlib

APP = pathlib.Path(__file__).resolve().parents[1] / "app"

WEATHER_KEYS = {
    "t_avg", "t_min", "t_max", "temperature_avg", "temperature_min", "temperature_max",
    "rh_avg", "humidity", "relativeHumidity", "wind_speed_ms", "wind_speed",
    "solar_rad_w_m2", "solar_radiation", "precip_mm", "precipitation",
    "elevation_m", "elevation", "reference_altitude_m",
}


def _offending_defaults(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "get"):
            continue
        if len(node.args) != 2:
            continue
        key, default = node.args
        if not (isinstance(key, ast.Constant) and key.value in WEATHER_KEYS):
            continue
        if isinstance(default, ast.Constant) and isinstance(default.value, (int, float)):
            bad.append((node.lineno, key.value, default.value))
    return bad


def test_cog_generator_has_no_fabricated_weather_defaults():
    bad = _offending_defaults(APP / "cog_generator.py")
    assert not bad, (
        "defaults meteorológicos fabricados en cog_generator.py: "
        + "; ".join(f"línea {ln}: .get({k!r}, {d})" for ln, k, d in bad)
    )


def test_zone_accumulator_has_no_fabricated_weather_defaults():
    bad = _offending_defaults(APP / "zone_accumulator.py")
    assert not bad, (
        "defaults meteorológicos fabricados en zone_accumulator.py: "
        + "; ".join(f"línea {ln}: .get({k!r}, {d})" for ln, k, d in bad)
    )
