"""HTTP clients for external services: eu-elevation, weather-api, Orion-LD.

Provides async functions that fetch DEM tiles, station weather, soil
hydraulics (AgriSoilExtended) from Orion-LD, and a stub for tenant parcels.  Every function
returns ``None`` on error — callers check for ``None`` rather than catching.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from nkz_platform_sdk import OrionClient
from nkz_platform_sdk.crypto import generate_hmac_signature

from app.config import settings
from app.downscaler import saxton_rawls_ptf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weather input extraction
# ---------------------------------------------------------------------------


class MissingWeatherInput(Exception):
    """Falta una entrada meteorológica obligatoria.

    Se lanza en vez de devolver un default. El pipeline calculó sobre constantes
    durante tres meses porque un `.get(clave, 15.0)` convertía la ausencia en un
    número plausible. Un hueco es honesto; un número inventado no.
    """

    def __init__(self, keys):
        self.keys = tuple(keys)
        super().__init__(f"missing weather input, tried: {', '.join(self.keys)}")


def _as_float(value: Any) -> float | None:
    """`float(value)` o `None` si el valor no es un escalar numérico.

    Un `float()` desnudo sobre un dict anidado, una lista o `"n/a"` lanza
    TypeError/ValueError, que escapa al handler de `MissingWeatherInput` y
    aborta el run del tenant a mitad de métrica — algunos COGs subidos, el
    puntero volteado, ningún registro escrito. Un valor malformado debe
    saltarse como uno ausente, no derribar el run.
    """
    if value is None or isinstance(value, (dict, list, tuple, set, bool)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def require(payload: dict, *keys) -> float:
    """Primera clave presente entre `keys`. Lanza `MissingWeatherInput` si ninguna.

    Acepta varios nombres porque el @context de la plataforma no es inyectivo: la
    compactación devuelve el alias que el escritor no usó (`airTemperature` donde
    el productor escribió `temperature`).

    Un valor presente pero no convertible a float cuenta como ausente: se
    prueba la siguiente clave y, si ninguna sirve, se lanza
    `MissingWeatherInput` — nunca TypeError/ValueError, que nadie captura.
    """
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        as_float = _as_float(value)
        if as_float is not None:
            return as_float
        logger.warning(
            "weather input %r present but not numeric (%r) — treating as missing",
            key, value,
        )
    raise MissingWeatherInput(keys)


async def _get_entity_keyvalues(tenant_id: str, entity_id: str) -> dict | None:
    """GET a single NGSI-LD entity in ``keyValues`` form, or ``None`` on 404/error.

    Same header construction as ``fetch_tenant_parcels`` (Link header carrying the
    platform ``@context``) — without it Orion silently expands attribute names
    against the default vocabulary and the request is a false-empty, not an error.
    """
    headers = {
        "NGSILD-Tenant": tenant_id,
        "Fiware-Service": tenant_id,
        "Fiware-ServicePath": "/",
        "Link": f'<{settings.context_url}>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{settings.orion_url}/ngsi-ld/v1/entities/{entity_id}",
                params={"options": "keyValues"},
                headers=headers,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
    except Exception:
        logger.exception(
            "_get_entity_keyvalues(tenant=%s, entity=%s) failed",
            tenant_id, entity_id,
        )
        return None


async def fetch_parcel_weather(tenant_id: str, parcel_id: str) -> dict | None:
    """Meteo por parcela desde el broker: WeatherObserved + WeatherForecast.

    Devuelve un dict plano con las claves que el motor de física necesita, o None si
    no hay observación. Una clave cuyo dato no exista se OMITE: quien la necesite
    lanzará MissingWeatherInput y el tile se saltará con un log ruidoso.
    """
    suffix = parcel_id.split(":")[-1]
    observed = await _get_entity_keyvalues(
        tenant_id, f"urn:ngsi-ld:WeatherObserved:{tenant_id}:parcel-{suffix}"
    )
    if not observed:
        return None
    forecast = await _get_entity_keyvalues(
        tenant_id, f"urn:ngsi-ld:WeatherForecast:{tenant_id}:parcel-{suffix}"
    ) or {}

    out: dict = {}

    def _put(key, source, *names):
        for name in names:
            raw = source.get(name)
            if raw is None:
                continue
            value = _as_float(raw)
            if value is not None:
                out[key] = value
                return
            logger.warning(
                "%s=%r on %s is not numeric — omitting %s",
                name, raw, parcel_id, key,
            )

    # Alias del @context: Orion devuelve el término que el productor NO usó.
    _put("t_avg", observed, "airTemperature", "temperature")
    # rh_avg, wind_speed_ms and precip_mm are extracted but unused by callers
    # (ET0 reads /daily); kept because tests and the alias handling cover them.
    _put("rh_avg", observed, "humidity", "relativeHumidity")
    _put("wind_speed_ms", observed, "windSpeed")
    _put("solar_rad_w_m2", observed, "solarRadiation")
    _put("precip_mm", observed, "precipitation")

    for key, attr in (("t_min", "dayMinimum"), ("t_max", "dayMaximum")):
        block = forecast.get(attr) or {}
        if isinstance(block, dict):
            _put(key, block, "temperature")

    coords = (observed.get("location") or {}).get("coordinates") or []
    if len(coords) >= 3:
        altitude = _as_float(coords[2])
        if altitude is not None:
            out["reference_altitude_m"] = altitude

    observed_at = observed.get("dateObserved")
    if isinstance(observed_at, dict):
        observed_at = observed_at.get("@value")
    if observed_at:
        out["observed_at"] = observed_at

    return out


# Days of closed weather the water balance sums, and the user id presented to
# weather-api (it only needs to be non-empty and identify this caller).
_BALANCE_DAYS = 5
_DAILY_CALLER = "weather-map-worker"


def _closed5(days: list[dict]) -> dict | None:
    """Sum of the last 5 CLOSED days, or None unless all 5 are whole.

    "Whole" means five consecutive dates, each with a numeric precipitation
    and ET0. A shorter or gappy window is not presented as 5 days.
    """
    closed = sorted(
        (d for d in days if d.get("kind") == "closed" and d.get("date")),
        key=lambda d: d["date"],
    )[-_BALANCE_DAYS:]
    if len(closed) < _BALANCE_DAYS:
        return None
    try:
        dates = [date.fromisoformat(d["date"]) for d in closed]
    except ValueError:
        return None
    for prev, nxt in zip(dates, dates[1:]):
        if nxt - prev != timedelta(days=1):
            return None
    precip = [_as_float(d.get("precip_mm")) for d in closed]
    et0 = [_as_float(d.get("et0_mm")) for d in closed]
    if any(v is None for v in precip) or any(v is None for v in et0):
        return None
    return {
        "precip_mm": float(sum(precip)),
        "et0_mm": float(sum(et0)),
        "days": _BALANCE_DAYS,
        "start": closed[0]["date"],
        "end": closed[-1]["date"],
    }


async def fetch_parcel_daily(tenant_id: str, parcel_id: str) -> dict | None:
    """One parcel's weather from weather-api ``/daily``: the running day's ET0
    inputs and the last five closed days.

    Returns ``None`` when the call cannot be made or fails (no secret, network
    error, 401, 5xx): the caller skips the parcel and nothing is invented.
    Otherwise a flat dict in which every key whose value is missing is
    OMITTED, so ``require`` raises ``MissingWeatherInput``:

    ``date`` / ``doy`` (the running day), ``t_min``, ``t_max``, ``ea_kpa``,
    ``u2_ms`` (already at 2 m), ``rs_mj`` (MJ m-2 d-1, horizontal),
    ``elevation_ref_m`` (the parcel altitude those temperatures refer to) and
    ``closed5`` (sums over 5 whole closed days, else omitted).
    """
    secret = settings.hmac_secret
    if not secret:
        logger.error(
            "fetch_parcel_daily(tenant=%s, parcel=%s): HMAC_SECRET is not set; "
            "weather-api rejects unsigned calls, so none is made",
            tenant_id, parcel_id,
        )
        return None

    suffix = parcel_id.split(":")[-1]
    today = datetime.now(timezone.utc).date()
    params = {
        "start": (today - timedelta(days=_BALANCE_DAYS)).isoformat(),
        "end": today.isoformat(),
        "forecast_days": 1,
        "fill": "open_meteo",
    }
    headers = {
        "X-Tenant-ID": tenant_id,
        "X-User-ID": _DAILY_CALLER,
        "X-Auth-Signature": generate_hmac_signature(secret, "", tenant_id),
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{settings.weather_api_url}/api/weather/parcel/{suffix}/daily",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            body = resp.json()
    except Exception:
        logger.exception(
            "fetch_parcel_daily(tenant=%s, parcel=%s) failed", tenant_id, parcel_id,
        )
        return None

    days = body.get("days") if isinstance(body, dict) else None
    if not isinstance(days, list):
        logger.error(
            "fetch_parcel_daily(tenant=%s, parcel=%s): no days in the response",
            tenant_id, parcel_id,
        )
        return None

    out: dict = {}
    running = next((d for d in days if d.get("kind") == "running"), None)
    if running is not None:
        try:
            run_date = date.fromisoformat(running["date"])
        except (KeyError, ValueError):
            run_date = None
        if run_date is not None:
            out["date"] = run_date.isoformat()
            out["doy"] = run_date.timetuple().tm_yday
        for key, field in (
            ("t_min", "tmin_c"),
            ("t_max", "tmax_c"),
            ("ea_kpa", "vapour_pressure_kpa"),
            ("u2_ms", "wind2m_ms"),
            ("rs_mj", "radiation_mj_m2"),
        ):
            value = _as_float(running.get(field))
            if value is not None:
                out[key] = value

    parcel = await _get_entity_keyvalues(
        tenant_id, parcel_id if parcel_id.startswith("urn:") else f"urn:ngsi-ld:AgriParcel:{parcel_id}",
    ) or {}
    elevation = parcel.get("elevation")
    if isinstance(elevation, dict):
        elevation = elevation.get("value")
    elevation = _as_float(elevation)
    # weather-api treats an elevation <= 0 as unknown and then returns
    # temperatures for the model grid, not for the parcel: such a value is not a
    # usable reference altitude.
    if elevation is not None and math.isfinite(elevation) and elevation > 0:
        out["elevation_ref_m"] = elevation
    elif elevation is not None:
        logger.warning(
            "fetch_parcel_daily(tenant=%s, parcel=%s): elevation %r is not a "
            "usable reference altitude (needs a finite value > 0)",
            tenant_id, parcel_id, elevation,
        )

    closed5 = _closed5(days)
    if closed5 is not None:
        out["closed5"] = closed5
    return out


async def fetch_parcel_scalars(
    tenant_id: str, parcel_id: str, *, observed: bool, forecast: bool,
) -> dict:
    """Parcel scalars for the `/stats` fallback, reading only what is asked for.

    Same inputs the raster job uses in `fetch_parcel_weather`: the running
    WeatherObserved (`parcel-<id>`) for `t_avg` and `solar_rad_w_m2`, and the
    WeatherForecast `dayMinimum` for `t_min`. The closed-day `-daily` entity is
    never read. Each read happens only when its flag is set. A key whose data is
    absent is omitted. `observed_at` / `valid_from` carry the source's own time.
    """
    suffix = parcel_id.split(":")[-1]
    out: dict = {}

    def _num(raw):
        return _as_float(raw) if raw is not None else None

    def _stamp(raw):
        if isinstance(raw, dict):
            raw = raw.get("@value")
        return raw if isinstance(raw, str) and raw else None

    if observed:
        obs = await _get_entity_keyvalues(
            tenant_id, f"urn:ngsi-ld:WeatherObserved:{tenant_id}:parcel-{suffix}"
        ) or {}
        for key, names in (
            ("t_avg", ("airTemperature", "temperature")),
            ("solar_rad_w_m2", ("solarRadiation",)),
        ):
            for name in names:
                value = _num(obs.get(name))
                if value is not None:
                    out[key] = value
                    break
        stamp = _stamp(obs.get("dateObserved"))
        if stamp:
            out["observed_at"] = stamp
    if forecast:
        fc = await _get_entity_keyvalues(
            tenant_id, f"urn:ngsi-ld:WeatherForecast:{tenant_id}:parcel-{suffix}"
        ) or {}
        block = fc.get("dayMinimum")
        if isinstance(block, dict):
            value = _num(block.get("temperature"))
            if value is not None:
                out["t_min"] = value
        stamp = _stamp(fc.get("validFrom"))
        if stamp:
            out["valid_from"] = stamp
    return out


class HistoryReadFailed(Exception):
    """The history read did not complete — this is NOT "no history yet".

    Every non-404 outcome (a 400 on `orderBy`, an Orion 500, a timeout, a
    malformed body) used to return `[]`, which `guard_frozen_metrics` cannot
    tell apart from a brand-new parcel. A guard that quietly disables itself
    on the first broken read is the same silent failure this whole branch
    exists to remove, one level up. Raise; the caller logs it as loudly as a
    frozen metric and skips only that one check.
    """


class MissingContextURL(HistoryReadFailed):
    """`CONTEXT_URL` is not configured — refusing a history read that would
    otherwise silently false-empty.

    Without the platform `@context` Link header, Orion expands attribute and
    type names against the default vocabulary and a query returns zero
    matches — a broken query that LOOKS exactly like "no history yet" (see
    `_get_entity_keyvalues`). A guard that can't tell "new parcel" from
    "couldn't even ask the question" is not a guard; refuse instead of
    returning `[]`.
    """


_HISTORY_PAGE = 1000  # Orion-LD's maximum page size


async def fetch_metric_history(
    tenant_id: str, parcel_id: str, attr_name: str, limit: int = 5,
) -> list[float]:
    """Return up to *limit* prior published values of *attr_name* for one
    parcel's own `AgriParcelRecord` history, oldest first.

    Returns ``[]`` ONLY when there is genuinely no history (a 404, or no
    record of this producer). Any read that did not complete raises
    `HistoryReadFailed` — a transport failure must never be mistaken for a
    new parcel, and neither must an unset `CONTEXT_URL` (`MissingContextURL`,
    a subclass).

    Orion-LD answers 501 to `orderBy` on this query, so none is sent: every
    record of the parcel is read page by page and ordered here, client-side,
    on the compact timestamp weather-map itself writes into the record id;
    the last *limit* are kept. The link is queried as `hasAgriParcel`; records
    are stored as `refAgriParcel`, but both names share one IRI so the query
    matches them (checked against the live broker).

    `AgriParcelRecord` is shared with a field-photo pipeline (ids prefixed
    `photo-`, also related to the same parcel via `hasAgriParcel`) — records
    are filtered to weather-map's own `weather-` id prefix so a photo upload
    is never read as a weather sample. Same broker-read shape as
    `fetch_tenant_parcels` (manual `@context` Link header) plus the
    `q=hasAgriParcel==` filter.
    """
    if not settings.context_url:
        raise MissingContextURL()

    headers = {
        "NGSILD-Tenant": tenant_id,
        "Fiware-Service": tenant_id,
        "Fiware-ServicePath": "/",
        "Link": f'<{settings.context_url}>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
    }
    data: list = []
    offset = 0
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                resp = await client.get(
                    f"{settings.orion_url}/ngsi-ld/v1/entities",
                    params={
                        "type": "AgriParcelRecord",
                        "q": f'hasAgriParcel=="{parcel_id}"',
                        "attrs": attr_name,
                        "options": "keyValues",
                        "limit": _HISTORY_PAGE,
                        "offset": offset,
                    },
                    headers=headers,
                )
                if resp.status_code == 404:
                    break  # genuine absence: no (more) records for this parcel
                resp.raise_for_status()
                page = resp.json()
                if not isinstance(page, list):
                    raise HistoryReadFailed(
                        f"history read for {attr_name} on {parcel_id} returned "
                        f"{type(page).__name__}, expected a list"
                    )
                data.extend(page)
                if len(page) < _HISTORY_PAGE:
                    break
                offset += _HISTORY_PAGE
    except HistoryReadFailed:
        raise
    except Exception as exc:
        logger.exception(
            "fetch_metric_history(tenant=%s, parcel=%s, attr=%s) failed",
            tenant_id, parcel_id, attr_name,
        )
        raise HistoryReadFailed(
            f"history read failed for {attr_name} on {parcel_id}: {exc!r}"
        ) from exc

    samples: list[tuple[str, float]] = []
    for entity in data:
        entity_id = entity.get("id", "")
        if not entity_id.rsplit(":", 1)[-1].startswith("weather-"):
            continue  # another producer's record (e.g. field-photo `photo-...`)
        stamp = _record_timestamp(entity_id)
        if stamp is None:
            logger.warning(
                "AgriParcelRecord %s carries no sortable timestamp in its id "
                "— excluded from the variance history", entity_id,
            )
            continue
        value = entity.get(attr_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            samples.append((stamp, value))

    samples.sort(key=lambda sample: sample[0])  # oldest first — the guard's contract
    samples = samples[-limit:]
    return [float(value) for _, value in samples]


def _record_timestamp(entity_id: str) -> str | None:
    """Compact timestamp weather-map writes at the end of its own record ids.

    `build_agri_parcel_record` builds
    ``urn:ngsi-ld:AgriParcelRecord:weather-{tenant}-{parcel}-{YYYYMMDDHHMMSS}``.
    Lexicographic order over that suffix IS chronological order, and it is a
    value this module writes — unlike the server-side `orderBy`.
    """
    tail = entity_id.rsplit(":", 1)[-1].rsplit("-", 1)[-1]
    return tail if tail.isdigit() else None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tms_tile_to_bbox(z: int, x: int, y: int) -> dict[str, float]:
    """Convert TMS tile coordinates to a geographic bounding box (EPSG:4326).

    Parameters
    ----------
    z : int
        Zoom level.
    x : int
        Tile column (0 … 2**z − 1).
    y : int
        Tile row (0 … 2**z − 1), origin at 85°N.

    Returns
    -------
    dict[str, float]
        ``{min_lon, min_lat, max_lon, max_lat}`` in decimal degrees.
    """
    n = 2.0 ** z
    min_lon = x / n * 360.0 - 180.0
    max_lon = (x + 1) / n * 360.0 - 180.0

    min_lat = math.degrees(
        math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    )
    max_lat = math.degrees(
        math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    )

    return {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_dem_tile(z: int, x: int, y: int) -> dict[str, Any] | None:
    """Fetch a DEM elevation grid for a TMS tile from the eu-elevation service.

    Returns the JSON response (e.g. a 2-D height array / GeoTIFF metadata)
    or ``None`` on any error (network, HTTP error, parse issue).
    """
    bbox = tms_tile_to_bbox(z, x, y)
    params: dict[str, Any] = {
        "min_lon": bbox["min_lon"],
        "min_lat": bbox["min_lat"],
        "max_lon": bbox["max_lon"],
        "max_lat": bbox["max_lat"],
        "resolution_m": 10,
        "purpose": "weather",
    }
    try:
        headers: dict[str, str] = {}
        if settings.internal_service_secret:
            headers["X-Internal-Service-Secret"] = settings.internal_service_secret
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{settings.elevation_service_url}/api/elevation/raster",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception:
        logger.exception(
            "fetch_dem_tile(z=%d, x=%d, y=%d) failed", z, x, y
        )
        return None


def _attr_value(entity: dict, name: str) -> Any:
    """Value of an NGSI-LD attribute, normalized or keyValues form."""
    attr = entity.get(name)
    if isinstance(attr, dict):
        return attr.get("value", attr.get("object"))
    return attr


# Organic matter (% weight) from organic carbon (% weight): the Van Bemmelen
# factor 1.724 (OM = OC x 1.724). GENERAL KNOWLEDGE, not verified against a
# primary source here: verify before relying on it.
_OC_TO_OM = 1.724
_OM_MAX_PCT = 20.0  # plausible upper bound for mineral topsoil organic matter


def _fraction(h: dict, attr: str) -> float | None:
    """A water-content attribute as a fraction in (0, 1), or None.

    Orion stores m3/m3. A value outside (0, 1) (e.g. a percentage written
    by mistake) is rejected with a WARNING instead of being scaled.
    """
    v = _as_float(h.get(attr))
    if v is None:
        return None
    if not 0.0 < v < 1.0:
        logger.warning("Soil attribute %s=%r is not a fraction in (0, 1); ignored", attr, v)
        return None
    return v


async def fetch_parcel_hydraulics(
    tenant_id: str, parcel_id: str
) -> dict[str, float] | None:
    """Soil hydraulics of a parcel from its ``AgriSoilExtended`` entity.

    Reads the top horizon (``horizons[0]``, as the risk module does). Orion
    stores water contents as fractions (m3/m3); the result is in % vol:
    ``{"field_capacity", "wilting_point", "awc"}``. If the horizon has no
    valid water contents but has sand, clay and organic carbon, they come
    from the Saxton & Rawls pedotransfer function. Returns ``None`` when
    there is no entity or no usable data: never a default.
    """
    orion = OrionClient(tenant_id)
    try:
        rows = await orion.query_entities(
            type="AgriSoilExtended",
            q=f'refAgriParcel=="{parcel_id}"|hasAgriParcel=="{parcel_id}"',
            limit=5,
            options="sysAttrs",
        )
    except Exception:
        logger.exception(
            "fetch_parcel_hydraulics(tenant=%s, parcel=%s) failed",
            tenant_id, parcel_id,
        )
        return None
    finally:
        await orion.close()

    if not rows:
        return None
    entity = rows[0]
    if len(rows) > 1:
        dated = [r for r in rows if _attr_value(r, "modifiedAt") or r.get("modifiedAt")]
        if dated:
            entity = max(
                dated, key=lambda r: str(_attr_value(r, "modifiedAt") or r.get("modifiedAt")),
            )
        logger.warning(
            "%d AgriSoilExtended match parcel %s; using %s",
            len(rows), parcel_id,
            "the most recently modified" if dated else "the first",
        )
    horizons = _attr_value(entity, "horizons")
    if not (isinstance(horizons, list) and horizons and isinstance(horizons[0], dict)):
        return None
    h = horizons[0]

    fc = _fraction(h, "fieldCapacity")
    wp = _fraction(h, "wiltingPoint")
    awc = _fraction(h, "availableWaterCapacity")
    out = None
    if fc is not None and wp is not None:
        if awc is None:
            awc = fc - wp
        out = {
            "field_capacity": fc * 100.0,
            "wilting_point": wp * 100.0,
            "awc": awc * 100.0,
        }
    else:
        sand = _as_float(h.get("sand"))
        clay = _as_float(h.get("clay"))
        oc = _as_float(h.get("organicCarbon"))
        if sand is None or clay is None or oc is None:
            return None
        # organicCarbon is % weight (live AgriSoilExtended horizons and the soil
        # module's writer agree). Anything giving OM outside (0, 20] % is a unit
        # slip (e.g. g/kg) or noise: unavailable, never fed to the PTF.
        om = oc * _OC_TO_OM
        if not 0.0 < om <= _OM_MAX_PCT:
            logger.warning(
                "Soil organicCarbon=%r gives organic matter %.1f %%, outside "
                "(0, %.0f] %%; hydraulics unavailable for parcel %s",
                oc, om, _OM_MAX_PCT, parcel_id,
            )
            return None
        out = saxton_rawls_ptf(sand, clay, om)

    if not (out["awc"] > 0 and out["field_capacity"] > out["wilting_point"] >= 0):
        logger.warning("Unusable hydraulics for parcel %s: %s", parcel_id, out)
        return None
    return out


async def upsert_record(tenant_id: str, entity: dict) -> None:
    """Create an NGSI-LD entity (AgriParcelRecord) via the SDK.

    Same-second id collisions are tolerated: a 409/duplicate is logged and
    ignored (append-only timeseries).
    """
    orion = OrionClient(tenant_id)
    try:
        await orion.create_entity(entity)
    except Exception as exc:  # noqa: BLE001
        if "already exists" in str(exc).lower() or "409" in str(exc):
            logger.info("AgriParcelRecord %s already exists, skipping", entity.get("id"))
            return
        logger.warning("upsert_record(tenant=%s) failed: %s", tenant_id, exc)
    finally:
        await orion.close()


async def fetch_entity_attr(
    tenant_id: str,
    entity_id: str,
    attr_name: str,
) -> Any | None:
    """Fetch a single NGSI-LD attribute from an entity in Orion-LD via the SDK.

    Returns the attribute's ``value`` (with keyValues simplification) or
    ``None`` if the entity/attribute does not exist or on error.
    """
    orion = OrionClient(tenant_id)
    try:
        data = await orion.get_entity(entity_id)
        return data.get(attr_name)
    except Exception:
        logger.exception(
            "fetch_entity_attr(tenant=%s, entity=%s, attr=%s) failed",
            tenant_id, entity_id, attr_name,
        )
        return None
    finally:
        await orion.close()


_PARCEL_PAGE_SIZE = 100
_PARCEL_MAX_PAGES = 100


async def fetch_tenant_parcels(tenant_id: str) -> list[dict[str, Any]]:
    """Fetch all AgriParcel entities for a tenant from Orion-LD.

    Uses ``POST /ngsi-ld/v1/entityOperations/query`` to retrieve
    entities of type ``AgriParcel`` with their ``location`` geometry.

    Returns a list of dicts with ``id``, ``location`` (GeoJSON geometry),
    and optionally ``name`` / ``description``.

    Pages through every parcel (limit + offset until a short page).
    Returns an empty list on error or if no parcels exist.
    """
    if not settings.context_url:
        # An empty CONTEXT_URL turns the Link header into `<>`, which Orion
        # answers with 400: say what is wrong instead of failing downstream.
        logger.error(
            "fetch_tenant_parcels(tenant=%s): CONTEXT_URL is not set; "
            "cannot list parcels",
            tenant_id,
        )
        return []
    headers = {
        "NGSILD-Tenant": tenant_id,
        "Fiware-Service": tenant_id,
        "Fiware-ServicePath": "/",
        "Content-Type": "application/json",
        "Link": f'<{settings.context_url}>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
    }
    # NGSI-LD POST /entityOperations/query body MUST be a Query object with the
    # type selector under `entities` — a bare {"type": "AgriParcel"} is a 400.
    payload: dict[str, Any] = {
        "type": "Query",
        "entities": [{"type": "AgriParcel"}],
        "attrs": ["location", "name", "description"],
    }
    parcels: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Orion pages at 20 by default: ask for explicit pages and keep
            # going until one comes back short.
            for page_no in range(_PARCEL_MAX_PAGES):
                resp = await client.post(
                    f"{settings.orion_url}/ngsi-ld/v1/entityOperations/query",
                    params={
                        "options": "keyValues",
                        "limit": _PARCEL_PAGE_SIZE,
                        "offset": page_no * _PARCEL_PAGE_SIZE,
                    },
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list):
                    logger.warning(
                        "fetch_tenant_parcels(tenant=%s): unexpected response type %s",
                        tenant_id, type(data).__name__,
                    )
                    return []

                for entity in data:
                    location = entity.get("location")
                    if location is None:
                        continue
                    parcel: dict[str, Any] = {
                        "id": entity.get("id"),
                        "location": location,
                    }
                    if entity.get("name") is not None:
                        parcel["name"] = entity["name"]
                    if entity.get("description") is not None:
                        parcel["description"] = entity["description"]
                    parcels.append(parcel)

                if len(data) < _PARCEL_PAGE_SIZE:
                    return parcels
            logger.error(
                "fetch_tenant_parcels(tenant=%s): more than %d parcels, "
                "refusing a truncated listing",
                tenant_id, _PARCEL_MAX_PAGES * _PARCEL_PAGE_SIZE,
            )
            return []
    except Exception:
        # Never hand back a partial listing as if it were the whole tenant.
        logger.exception(
            "fetch_tenant_parcels(tenant=%s) failed", tenant_id,
        )
        return []


async def fetch_agri_parcel(
    tenant_id: str, parcel_id: str
) -> dict[str, Any] | None:
    """Fetch an ``AgriParcel`` NGSI-LD entity from Orion-LD for geometry via SDK.

    Returns a dict with the parcel's GeoJSON ``location`` geometry and
    ``agriParcelOf`` (tenant/enterprise), or ``None`` if not found.
    """
    # Normalize: strip prefix if already has it
    eid = parcel_id if parcel_id.startswith("urn:ngsi-ld:AgriParcel:") else \
        f"urn:ngsi-ld:AgriParcel:{parcel_id}"
    orion = OrionClient(tenant_id)
    try:
        data = await orion.get_entity(eid)
        # get_entity returns NORMALIZED NGSI-LD: location is
        # {"type": "GeoProperty", "value": <GeoJSON>}. Unwrap to the raw GeoJSON.
        raw_loc = data.get("location") or data.get("geometry")
        if isinstance(raw_loc, dict) and raw_loc.get("type") == "GeoProperty":
            geometry = raw_loc.get("value")
        else:
            geometry = raw_loc
        if geometry is None:
            logger.warning(
                "fetch_agri_parcel(tenant=%s, parcel=%s): no location geometry",
                tenant_id, parcel_id,
            )
            return None
        return {
            "id": data.get("id", eid),
            "geometry": geometry,
            "name": data.get("name"),
            "description": data.get("description"),
        }
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.warning(
                "fetch_agri_parcel(tenant=%s, parcel=%s): entity not found (404)",
                tenant_id, parcel_id,
            )
        else:
            logger.exception(
                "fetch_agri_parcel(tenant=%s, parcel=%s) HTTP error",
                tenant_id, parcel_id,
            )
        return None
    except Exception:
        logger.exception(
            "fetch_agri_parcel(tenant=%s, parcel=%s) failed",
            tenant_id, parcel_id,
        )
        return None
    finally:
        await orion.close()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute great-circle distance in metres between two WGS-84 points.

    Uses the standard haversine formula with Earth radius R = 6,371,000 m.
    """
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return float(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def _unwrap_location(location: Any) -> dict | None:
    """Unwrap an NGSI-LD GeoProperty to a raw GeoJSON geometry dict.

    Handles both the normalized format
    ``{"type": "GeoProperty", "value": <GeoJSON>}`` and the simplified
    ``keyValues`` format where the value *is* the GeoJSON geometry.

    Returns ``None`` if *location* cannot be interpreted as a GeoJSON
    geometry (e.g. it is ``None`` or not a dict).
    """
    if isinstance(location, dict) and location.get("type") == "GeoProperty":
        return location.get("value")
    if isinstance(location, dict) and "type" in location and "coordinates" in location:
        return location  # already raw GeoJSON
    return None


def _centroid(geometry: dict) -> tuple[float, float] | None:
    """Return ``(lon, lat)`` centroid from a GeoJSON geometry dict.

    Supports ``Point``, ``Polygon`` and ``MultiPolygon``.
    For ``Polygon`` / ``MultiPolygon`` the centroid is the arithmetic
    mean of the exterior-ring vertex coordinates.
    """
    if not isinstance(geometry, dict):
        return None
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return None
    if geom_type == "Point":
        return (float(coords[0]), float(coords[1]))
    if geom_type == "Polygon":
        ring = coords[0]
        lon = sum(float(c[0]) for c in ring) / len(ring)
        lat = sum(float(c[1]) for c in ring) / len(ring)
        return (lon, lat)
    if geom_type == "MultiPolygon":
        lons, lats = [], []
        for polygon in coords:
            ring = polygon[0]
            lons.extend(float(c[0]) for c in ring)
            lats.extend(float(c[1]) for c in ring)
        return (sum(lons) / len(lons), sum(lats) / len(lats))
    return None


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------


_SENSOR_ATTRS = ("nkz:sensorNearby", "nkz:sensorDistanceM")

_LOOKUP_OK_KEY = "_sensor_lookup_ok"

# Attributes (local names) whose COG lookup succeeded for this zone this run.
COG_CHECKED_KEY = "_cog_checked"
_COG_ATTRS = ("nkz:solarRadiationWm2", "nkz:frostRisk")
_PRIVATE_KEYS = (_LOOKUP_OK_KEY, COG_CHECKED_KEY)
_ZONE_LIST_LIMIT = 1000


def _present(entity: dict, attr: str) -> bool:
    """True when a keyValues entity carries *attr* (compact or full-IRI key)."""
    local = attr.split(":", 1)[-1]
    return any(
        k == attr or k.endswith("/" + local) or k.endswith("#" + local)
        for k in entity
    )


async def _list_carriers(orion: Any, tenant_id: str, parcel_urn: str) -> dict[str, dict] | None:
    """Zone entities of a parcel as they currently are, by id (None on failure)."""
    try:
        rows = await orion.query_entities(
            type="AgriParcelZone",
            q=f'hasAgriParcel=="{parcel_urn}"',
            options="keyValues",
            limit=_ZONE_LIST_LIMIT,
        )
        if not isinstance(rows, list):
            raise ValueError(f"unexpected zone listing: {type(rows).__name__}")
        if len(rows) >= _ZONE_LIST_LIMIT:
            logger.warning("zone listing for %s hit the %d cap", parcel_urn, _ZONE_LIST_LIMIT)
        return {r["id"]: r for r in rows if isinstance(r, dict) and r.get("id")}
    except Exception:
        logger.warning(
            "zone listing failed for %s (tenant=%s); stale attributes not cleared",
            parcel_urn, tenant_id, exc_info=True,
        )
        return None


async def _delete_attr(orion: Any, tenant_id: str, eid: str, attr: str) -> None:
    try:
        await orion.delete_entity_attr(eid, attr)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404:
            logger.debug("no %s to clear on %s", attr, eid)
        else:
            logger.warning(
                "clear %s on %s failed (tenant=%s)", attr, eid, tenant_id, exc_info=True,
            )


async def _clear_stale_attrs(orion: Any, tenant_id: str, zones: list[dict]) -> None:
    """Delete attributes this run no longer produces, only where they exist.

    The batch upsert keeps attributes absent from the payload, so a device that
    moved away, or a COG that no longer covers a zone, would leave stale values
    forever. One listing per parcel finds the zones that currently carry the
    attributes; only those get a DELETE. Sensor attributes are cleared only for
    zones marked by a SUCCESSFUL device lookup, radiation/frost only for the
    metrics whose COG lookup succeeded. A failed listing clears nothing.
    """
    parcels: dict[str, dict | None] = {}
    for zone in zones:
        if not (zone.get(_LOOKUP_OK_KEY) or zone.get(COG_CHECKED_KEY)):
            continue
        parcel = (zone.get("hasAgriParcel") or {}).get("object")
        if parcel and parcel not in parcels:
            parcels[parcel] = await _list_carriers(orion, tenant_id, parcel)

    for zone in zones:
        parcel = (zone.get("hasAgriParcel") or {}).get("object")
        current = (parcels.get(parcel) or {}).get(zone["id"])
        if current is None:
            continue
        stale: list[str] = []
        if zone.get(_LOOKUP_OK_KEY) and "nkz:sensorNearby" not in zone:
            stale += [a for a in _SENSOR_ATTRS if _present(current, a)]
        checked = zone.get(COG_CHECKED_KEY) or set()
        stale += [
            a for a in _COG_ATTRS
            if a.split(":", 1)[1] in checked and a not in zone and _present(current, a)
        ]
        for attr in stale:
            await _delete_attr(orion, tenant_id, zone["id"], attr)


async def upsert_agri_parcel_zones(tenant_id: str, zones: list[dict]) -> None:
    """Batch upsert ``AgriParcelZone`` entities via Orion-LD SDK.

    Uses ``OrionClient.upsert_entities_batch()`` to create or replace all
    zone entities in a single request.  Attributes on existing entities are
    updated; new entities are created.

    Parameters
    ----------
    tenant_id : str
        The tenant / FIWARE service.
    zones : list[dict]
        NGSI-LD entity dicts (from ``build_agri_parcel_zone``).
    """
    if not zones:
        logger.info(
            "upsert_agri_parcel_zones(tenant=%s): no zones to upsert", tenant_id,
        )
        return

    orion = OrionClient(tenant_id)
    try:
        payload = [
            {k: v for k, v in z.items() if k not in _PRIVATE_KEYS} for z in zones
        ]
        result = await orion.upsert_entities_batch(payload)
        logger.info(
            "upsert_agri_parcel_zones(tenant=%s): upserted %d zone(s)",
            tenant_id, result.get("upserted", 0),
        )
        if result.get("errors"):
            logger.warning(
                "upsert_agri_parcel_zones(tenant=%s): %d error(s)",
                tenant_id, len(result["errors"]),
            )
        await _clear_stale_attrs(orion, tenant_id, zones)
    except Exception:
        logger.exception(
            "upsert_agri_parcel_zones(tenant=%s) failed for %d zone(s)",
            tenant_id, len(zones),
        )
    finally:
        await orion.close()


SENSOR_NEARBY_MAX_M = 50.0


_BBOX_PAD_DEG = 0.001  # ~100 m: cheap prefilter well above SENSOR_NEARBY_MAX_M
_DEVICE_LIST_LIMIT = 1000


def _zone_projector(zone_geom: dict):
    """Project a zone to local metres ONCE; return ``(zone_m, project, bbox)``.

    Equirectangular approximation around the zone centroid
    (x = dlon*cos(lat0)*111320, y = dlat*110540): error < 1 % at zone scale.
    ``bbox`` is the zone's lon/lat bounds expanded by ``_BBOX_PAD_DEG``.
    """
    from shapely.geometry import shape
    from shapely.ops import transform

    zone = shape(zone_geom)
    lon0, lat0 = zone.centroid.x, zone.centroid.y
    kx = math.cos(math.radians(lat0)) * 111320.0

    def _proj(x, y, z=None):
        return ((x - lon0) * kx, (y - lat0) * 110540.0)

    minx, miny, maxx, maxy = zone.bounds
    bbox = (minx - _BBOX_PAD_DEG, miny - _BBOX_PAD_DEG,
            maxx + _BBOX_PAD_DEG, maxy + _BBOX_PAD_DEG)
    return transform(_proj, zone), _proj, bbox


def _project_device(device_geom: dict, proj):
    from shapely.geometry import shape
    from shapely.ops import transform

    return transform(proj, shape(device_geom))


def _device_distance_m(zone_m, proj, device_geom: dict, device_id: str | None = None) -> float | None:
    """Distance in metres from a device to the (already projected) zone."""
    try:
        return float(zone_m.distance(_project_device(device_geom, proj)))
    except Exception:
        logger.warning(
            "zone/device distance failed for device %s", device_id, exc_info=True,
        )
        return None


def _in_bbox(device_geom: dict, bbox) -> bool:
    c = _centroid(device_geom)
    if c is None:
        return False
    return bbox[0] <= c[0] <= bbox[2] and bbox[1] <= c[1] <= bbox[3]


async def find_nearby_sensors(
    tenant_id: str,
    parcel_id: str,
    zones: list[dict],
) -> list[dict]:
    """Enrich zone descriptors with the nearest IoT sensor.

    Queries Orion-LD for ``Device`` entities associated with the tenant,
    then for each zone finds the closest device within SENSOR_NEARBY_MAX_M
    of the zone (metres, 0 when inside); farther devices are ignored.
    Adds ``sensor_nearby`` (device URN) and ``sensor_distance_m``
    (distance in metres, rounded to 1 decimal) to each zone dict.

    Parameters
    ----------
    tenant_id : str
        The tenant / FIWARE service.
    parcel_id : str
        The parcel identifier (for logging).
    zones : list[dict]
        Zone descriptors (NGSI-LD entity dicts with a ``location``
        attribute).

    Returns
    -------
    list[dict]
        Enriched zone dicts; original zones if no devices found.
    """
    if not zones:
        return []

    # Fetch all Device entities for this tenant via SDK
    orion = OrionClient(tenant_id)
    devices: list[dict[str, Any]] = []
    try:
        data = await orion.query_entities(
            type="Device",
            attrs="location,name",
            options="keyValues",
            limit=_DEVICE_LIST_LIMIT,
        )
        if not isinstance(data, list):
            raise ValueError(f"unexpected Device query result: {type(data).__name__}")
        if len(data) >= _DEVICE_LIST_LIMIT:
            logger.warning(
                "find_nearby_sensors(tenant=%s, parcel=%s): Device query hit the %d cap; "
                "some devices may be ignored", tenant_id, parcel_id, _DEVICE_LIST_LIMIT,
            )
        for entity in data:
            raw_loc = entity.get("location")
            loc = _unwrap_location(raw_loc)
            if loc is None:
                continue
            devices.append({
                "id": entity.get("id"),
                "location": loc,
            })
    except Exception:
        logger.exception(
            "find_nearby_sensors(tenant=%s, parcel=%s): failed to fetch devices",
            tenant_id, parcel_id,
        )
        return list(zones)  # return unenriched
    finally:
        await orion.close()

    if not devices:
        logger.info(
            "find_nearby_sensors(tenant=%s, parcel=%s): no devices found",
            tenant_id, parcel_id,
        )

    # Enrich each zone with the nearest device
    enriched: list[dict[str, Any]] = []
    for zone in zones:
        zone_copy = dict(zone)
        raw_loc = zone_copy.get("location") or zone_copy.get("geometry")
        loc = _unwrap_location(raw_loc)
        if loc is None:
            enriched.append(zone_copy)
            continue

        # Lookup succeeded and this zone was evaluated: the upsert may clear
        # stale sensor attributes on it. Private marker, stripped before send.
        zone_copy[_LOOKUP_OK_KEY] = True
        best_dist = float("inf")
        best_sensor: str | None = None

        try:
            zone_m, proj, bbox = _zone_projector(loc)
        except Exception:
            logger.warning("zone projection failed for %s", zone.get("id"), exc_info=True)
            enriched.append(zone_copy)
            continue

        for device in devices:
            if not _in_bbox(device["location"], bbox):
                continue
            d = _device_distance_m(zone_m, proj, device["location"], device["id"])
            if d is None or d > SENSOR_NEARBY_MAX_M:
                continue
            if d < best_dist:
                best_dist = d
                best_sensor = device["id"]

        if best_sensor is not None:
            dist = round(best_dist, 1)
            zone_copy["nkz:sensorNearby"] = {"type": "Property", "value": True}
            zone_copy["nkz:sensorDistanceM"] = {"type": "Property", "value": dist}

        enriched.append(zone_copy)

    logger.info(
        "find_nearby_sensors(tenant=%s, parcel=%s): enriched %d zone(s) with %d device(s)",
        tenant_id, parcel_id, len(enriched), len(devices),
    )
    return enriched
