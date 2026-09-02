"""Contrato: la capa de física no fabrica entradas ausentes.

Este módulo publicó `airTemperatureAvg = 14.934` en 32 días consecutivos de agosto
—el default 15.0 menos la corrección de altitud— porque cada `.get(clave, literal)`
convertía una ausencia en un número plausible. Nada falló, nada avisó.

El test lee el fuente: un literal numérico como default de una variable meteorológica
es el defecto, aunque el valor parezca razonable.

Ronda 2 — el test tenía dos puntos ciegos y el mismo bug sobrevivió detrás de ambos:
solo miraba `cog_generator.py` y `zone_accumulator.py`, y solo reconocía la forma
`.get(clave, literal)`. `forecast.py` fabricaba con la otra forma, `.get(clave) or 0.0`,
en un tercer consumidor que nadie escaneaba. Ahora barre **todo** `app/` y ambas formas.
"""

import ast
import pathlib

APP = pathlib.Path(__file__).resolve().parents[1] / "app"

WEATHER_KEYS = {
    "t_avg", "t_min", "t_max", "temperature_avg", "temperature_min", "temperature_max",
    "rh_avg", "humidity", "relativeHumidity", "wind_speed_ms", "wind_speed",
    "solar_rad_w_m2", "solar_radiation", "precip_mm", "precipitation",
    "elevation_m", "elevation", "reference_altitude_m",
    "eto", "eto_mm", "et0", "evapotranspiration",
}


def _is_number(node: ast.AST) -> bool:
    """True for a numeric literal — `True`/`False` are ints in Python, exclude them."""
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    )


def _weather_key_read_by(node: ast.AST) -> str | None:
    """Nombre de la variable meteorológica que lee esta expresión, o None.

    Reconoce las cuatro formas con las que el código llega a un valor meteo:
    `d.get("t_min")`, `t_min`, `d["t_min"]` y `obj.t_min`.
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value in WEATHER_KEYS
    ):
        return node.args[0].value
    if isinstance(node, ast.Name) and node.id in WEATHER_KEYS:
        return node.id
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value in WEATHER_KEYS
    ):
        return node.slice.value
    if isinstance(node, ast.Attribute) and node.attr in WEATHER_KEYS:
        return node.attr
    return None


def _offenders_in_tree(tree: ast.AST) -> list[tuple[int, str, object]]:
    """Devuelve `(línea, clave, default)` por cada default meteorológico fabricado."""
    bad: list[tuple[int, str, object]] = []
    for node in ast.walk(tree):
        # Forma 1: `payload.get("t_min", 10.0)`
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in WEATHER_KEYS
            and _is_number(node.args[1])
        ):
            bad.append((node.lineno, node.args[0].value, node.args[1].value))
        # Forma 2: `entry.get("eto_mm") or 0.0`, `precip_mm or 0.0`.
        # Idéntica en efecto y peor de leer: un 0.0 aquí lee como "sin estrés hídrico".
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            for left, right in zip(node.values, node.values[1:]):
                key = _weather_key_read_by(left)
                if key is not None and _is_number(right):
                    bad.append((node.lineno, key, right.value))
    return bad


def _offending_defaults(path: pathlib.Path) -> list[tuple[int, str, object]]:
    return _offenders_in_tree(ast.parse(path.read_text()))


def _format(name: str, bad: list[tuple[int, str, object]]) -> str:
    return f"{name}: " + "; ".join(f"línea {ln}: clave {k!r} → {d}" for ln, k, d in bad)


def test_no_module_under_app_fabricates_a_weather_default():
    """Barrido completo de `app/`: el bug sobrevivió tres meses en el único
    consumidor que el test no miraba."""
    offenders = {}
    for path in sorted(APP.rglob("*.py")):
        bad = _offending_defaults(path)
        if bad:
            offenders[path.name] = bad
    assert not offenders, "defaults meteorológicos fabricados:\n" + "\n".join(
        _format(name, bad) for name, bad in offenders.items()
    )


def test_cog_generator_has_no_fabricated_weather_defaults():
    bad = _offending_defaults(APP / "cog_generator.py")
    assert not bad, _format("cog_generator.py", bad)


def test_zone_accumulator_has_no_fabricated_weather_defaults():
    bad = _offending_defaults(APP / "zone_accumulator.py")
    assert not bad, _format("zone_accumulator.py", bad)


def test_forecast_has_no_fabricated_weather_defaults():
    """El tercer consumidor: proxyaba weather-api y convertía un ET0 ausente en 0."""
    bad = _offending_defaults(APP / "forecast.py")
    assert not bad, _format("forecast.py", bad)


# ----------------------------------------------------------------------
# El detector, sobre sí mismo: un contrato con un punto ciego es cómo el
# bug sobrevivió. Estos casos fijan que ambas formas se reconocen.
# ----------------------------------------------------------------------


def test_detector_catches_the_get_default_form():
    bad = _offenders_in_tree(ast.parse('x = payload.get("t_min", 10.0)'))
    assert [(k, d) for _, k, d in bad] == [("t_min", 10.0)]


def test_detector_catches_the_or_default_form():
    bad = _offenders_in_tree(ast.parse('x = entry.get("eto_mm") or 0.0'))
    assert [(k, d) for _, k, d in bad] == [("eto_mm", 0.0)]


def test_detector_catches_a_bare_name_or_default():
    bad = _offenders_in_tree(ast.parse("x = precip_mm or 0.0"))
    assert [(k, d) for _, k, d in bad] == [("precip_mm", 0.0)]


def test_detector_catches_a_subscript_or_default():
    bad = _offenders_in_tree(ast.parse('x = weather["solar_radiation"] or 0.0'))
    assert [(k, d) for _, k, d in bad] == [("solar_radiation", 0.0)]


def test_detector_ignores_non_weather_keys():
    """`origin_lon`, `pixelCount` y compañía son geometría del ráster, no meteo."""
    assert _offenders_in_tree(ast.parse('x = dem.get("origin_lon", 0.0)')) == []
    assert _offenders_in_tree(ast.parse('x = row["days_with_data"] or 0')) == []


def test_detector_ignores_a_non_numeric_fallback():
    """`or None` y `or {}` no fabrican un número: propagan la ausencia."""
    assert _offenders_in_tree(ast.parse('x = entry.get("eto_mm") or None')) == []
    assert _offenders_in_tree(ast.parse('x = forecast.get("t_min") or {}')) == []
