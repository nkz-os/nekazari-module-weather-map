"""Tests for zone accumulator helpers."""
import numpy as np
from app.zone_accumulator import _compute_area_ha, _geom_geojson_bbox


def test_compute_area_ha():
    import pytest
    assert _compute_area_ha(100) == 1.0
    assert _compute_area_ha(0) == 0.0
    assert _compute_area_ha(1520) == pytest.approx(15.2)


def test_geom_geojson_bbox_polygon():
    geom = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
    }
    bbox = _geom_geojson_bbox(geom)
    assert bbox == (0, 0, 10, 10)


def test_geom_geojson_bbox_multipolygon():
    geom = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
            [[[5, 5], [7, 5], [7, 7], [5, 7], [5, 5]]],
        ],
    }
    bbox = _geom_geojson_bbox(geom)
    assert bbox == (0, 0, 7, 7)


def test_geom_geojson_bbox_empty():
    assert _geom_geojson_bbox({}) == (0.0, 0.0, 0.0, 0.0)


def test_gradient_flat_surface():
    from app.zone_accumulator import _gradient
    elev = np.ones((10, 10), dtype=float)
    dzdx, dzdy = _gradient(elev, 0.0001)
    assert dzdx.shape == (10, 10)
    assert dzdy.shape == (10, 10)
    assert np.all(dzdx[1:-1, 1:-1] == 0.0)
    assert np.all(dzdy[1:-1, 1:-1] == 0.0)


# ======================================================================
# process_parcel — comportamiento, no "es invocable".
#
# Mismos dos invariantes que `generate_cog_for_tile`: sin meteo no se
# fabrica nada, y la corrección de altitud va contra `reference_altitude_m`
# (location[2] de WeatherObserved), nunca contra el grid del DEM.
# ======================================================================

import logging
from unittest.mock import AsyncMock

import pytest

PARCEL = {"id": "urn:ngsi-ld:AgriParcel:p1", "lon": -1.65, "lat": 42.8}

WEATHER = {
    "t_min": 17.3,
    "t_max": 32.2,
    "rh_avg": 60.0,
    "wind_speed_ms": 2.5,
    "solar_rad_w_m2": 229.4,
    "reference_altitude_m": 572.8,
}

DEM_BASE_ELEVATION = 1200.0


def _dem(rows: int = 12, cols: int = 12) -> dict:
    return {
        "elevations": [
            [DEM_BASE_ELEVATION + r * 2.0 + c * 0.5 for c in range(cols)]
            for r in range(rows)
        ],
        "origin_lon": -1.65,
        "origin_lat": 42.8,
        "pixel_size_deg": 0.0001,
        "rows": rows,
        "cols": cols,
    }


@pytest.fixture
def zone_env(monkeypatch):
    """DEM fijo, sin sensores, zonas pequeñas admitidas."""
    from app import zone_accumulator as za

    monkeypatch.setattr(za, "fetch_dem_tile", AsyncMock(return_value=_dem()))
    monkeypatch.setattr(za.settings, "zones_min_pixels", 1)

    async def _no_sensors(tenant_id, parcel_id, zones):
        return list(zones)

    monkeypatch.setattr(za, "find_nearby_sensors", _no_sensors)
    return za


@pytest.mark.asyncio
async def test_absent_observation_yields_no_zone_entities(zone_env, monkeypatch):
    monkeypatch.setattr(zone_env, "fetch_parcel_weather", AsyncMock(return_value=None))
    assert await zone_env.process_parcel("t", PARCEL, "2026-09-02") == []


@pytest.mark.asyncio
async def test_incomplete_weather_publishes_nothing_and_computes_nothing(
    zone_env, monkeypatch, caplog,
):
    physics_calls: list = []
    incomplete = {k: v for k, v in WEATHER.items() if k != "t_max"}

    monkeypatch.setattr(
        zone_env, "fetch_parcel_weather", AsyncMock(return_value=incomplete),
    )
    monkeypatch.setattr(
        zone_env, "correct_temperature", lambda *a: physics_calls.append(a),
    )

    with caplog.at_level(logging.ERROR):
        entities = await zone_env.process_parcel("t", PARCEL, "2026-09-02")

    assert entities == []
    assert physics_calls == []
    assert "t_max" in caplog.text


@pytest.mark.asyncio
async def test_zone_correction_uses_the_observation_altitude(zone_env, monkeypatch):
    seen: list[float] = []

    def _spy(t_base, station_elevation_m, pixel_elevations):
        seen.append(station_elevation_m)
        return np.asarray(pixel_elevations, dtype=float) * 0.0 + t_base

    monkeypatch.setattr(
        zone_env, "fetch_parcel_weather", AsyncMock(return_value=dict(WEATHER)),
    )
    monkeypatch.setattr(zone_env, "correct_temperature", _spy)

    entities = await zone_env.process_parcel("t", PARCEL, "2026-09-02")

    assert entities, "con meteo completa se esperan entidades AgriParcelZone"
    assert set(seen) == {572.8}
    assert DEM_BASE_ELEVATION not in seen
