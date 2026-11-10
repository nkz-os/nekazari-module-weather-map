"""Tests for the COG generator module.

Focuses on pure helper functions (``_tile_to_bbox`` and ``_bbox_to_tiles``)
which have no external dependencies.  Async integration tests that exercise
``generate_cog_for_tile`` require mocks and are kept minimal.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.cog_generator import _bbox_to_tiles, _tile_to_bbox


# ======================================================================
# _tile_to_bbox
# ======================================================================


class TestTileToBbox:
    """Tile-to-bbox conversion — pure math, no external deps."""

    def test_zoom_0(self):
        """At zoom 0 the single tile covers the whole world."""
        min_lon, min_lat, max_lon, max_lat = _tile_to_bbox(0, 0, 0)
        assert min_lon == -180.0
        assert max_lon == 180.0
        assert min_lat < -80.0  # Approximate south pole
        assert max_lat > 80.0  # Approximate north pole

    def test_navarra_tile(self):
        """Tile 14/8117/6038 should cover roughly the Navarra (Spain) area."""
        min_lon, min_lat, max_lon, max_lat = _tile_to_bbox(14, 8117, 6038)
        assert -2.0 < min_lon < -1.0  # Western Navarra
        assert 42.0 < min_lat < 43.0   # Southern Navarra
        assert max_lon > min_lon
        assert max_lat > min_lat

    def test_min_less_than_max(self):
        """Output respects min < max for both longitude and latitude."""
        for z in (0, 5, 10, 14, 18):
            min_l, min_la, max_l, max_la = _tile_to_bbox(z, 0, 0)
            assert min_l < max_l, f"z={z}: min_lon >= max_lon"
            assert min_la < max_la, f"z={z}: min_lat >= max_lat"

    def test_zoom_increases_precision(self):
        """Higher zoom → narrower tiles (smaller longitude extent)."""
        bbox_z10 = _tile_to_bbox(10, 500, 500)
        bbox_z14 = _tile_to_bbox(14, 500, 500)
        lon_span_z10 = bbox_z10[2] - bbox_z10[0]
        lon_span_z14 = bbox_z14[2] - bbox_z14[0]
        assert lon_span_z14 < lon_span_z10

    def test_antimeridian_behaviour(self):
        """Tile near the antimeridian yields consistent bounds."""
        min_lon, _, max_lon, _ = _tile_to_bbox(2, 3, 0)
        assert -180.0 <= min_lon <= 180.0
        assert -180.0 <= max_lon <= 180.0
        assert min_lon < max_lon

    def test_southern_hemisphere(self):
        """Tiles in the southern hemisphere have negative latitudes."""
        _, min_lat, _, max_lat = _tile_to_bbox(5, 16, 20)
        # Upper row indices = southern hemisphere in TMS
        assert min_lat < 0.0
        assert max_lat < 0.0

    def test_tile_centre_is_within_bbox(self):
        """The centre of a tile falls within its own bbox."""
        z, x, y = 12, 2048, 1024
        min_lon, min_lat, max_lon, max_lat = _tile_to_bbox(z, x, y)
        centre_lon = (min_lon + max_lon) / 2.0
        centre_lat = (min_lat + max_lat) / 2.0
        assert min_lon <= centre_lon <= max_lon
        assert min_lat <= centre_lat <= max_lat

    def test_consistent_zoom_1_partition(self):
        """Zoom-1 tiles correctly partition the world."""
        # NW quadrant
        nw_min_lon, nw_min_lat, nw_max_lon, nw_max_lat = _tile_to_bbox(1, 0, 0)
        # SE quadrant
        se_min_lon, se_min_lat, se_max_lon, se_max_lat = _tile_to_bbox(1, 1, 1)
        # NW left edge touches -180
        assert nw_min_lon == -180.0
        # SE right edge touches 180
        assert se_max_lon == 180.0
        # NW top edge is near the north pole (85°)
        assert nw_max_lat > 80.0
        # SE bottom edge is near the south pole (-85°)
        assert se_min_lat < -80.0
        # NW bottom and SE top meet at the equator
        assert abs(nw_min_lat) < 1e-10
        assert abs(se_max_lat) < 1e-10


# ======================================================================
# _bbox_to_tiles
# ======================================================================


class TestBboxToTiles:
    """Bbox-to-tiles conversion — pure math, no external deps."""

    def test_few_tiles_at_low_zoom(self):
        """A small bbox at low zoom produces a small number of tiles."""
        tiles = _bbox_to_tiles(-2.0, 42.0, -1.0, 43.0, zoom=5)
        assert len(tiles) > 0
        assert len(tiles) < 10

    def test_more_tiles_at_higher_zoom(self):
        """The same bbox at a higher zoom produces more tiles."""
        low = _bbox_to_tiles(-2.0, 42.0, -1.0, 43.0, zoom=10)
        high = _bbox_to_tiles(-2.0, 42.0, -1.0, 43.0, zoom=14)
        assert len(low) < len(high)

    def test_tiles_are_valid_tms(self):
        """Every returned tile coordinate is a valid TMS triplet."""
        tiles = _bbox_to_tiles(-10.0, 35.0, 5.0, 50.0, zoom=10)
        for z, x, y in tiles:
            assert z == 10
            assert 0 <= x < 2**z
            assert 0 <= y < 2**z

    def test_empty_bbox_yields_tiles(self):
        """Even a point-like bbox should yield at least one tile."""
        tiles = _bbox_to_tiles(-1.65, 42.8, -1.65, 42.8, zoom=14)
        assert len(tiles) >= 1

    def test_large_bbox(self):
        """A large bbox covering most of Europe produces many tiles."""
        tiles = _bbox_to_tiles(-10.0, 35.0, 30.0, 60.0, zoom=8)
        assert len(tiles) > 50

    def test_all_returned_tiles_cover_bbox(self):
        """The union of returned tiles fully covers the input bbox."""
        bbox = (-3.0, 42.0, 0.0, 44.0)
        tiles = _bbox_to_tiles(*bbox, zoom=12)
        # Each tile's bbox should overlap the input bbox
        for z, x, y in tiles:
            t_min_lon, t_min_lat, t_max_lon, t_max_lat = _tile_to_bbox(z, x, y)
            # Check overlap: partial overlap is sufficient
            assert t_max_lon > bbox[0]  # tile right > bbox left
            assert t_min_lon <= bbox[2]  # tile left <= bbox right (allow exact boundary)
            assert t_max_lat > bbox[1]  # tile top > bbox bottom
            assert t_min_lat < bbox[3]  # tile bottom < bbox top

    def test_tile_count_scales_with_area(self):
        """Doubling the area approximately doubles the tile count."""
        small = _bbox_to_tiles(-1.0, 42.0, 0.0, 43.0, zoom=12)
        large = _bbox_to_tiles(-2.0, 41.0, 1.0, 44.0, zoom=12)
        # The large bbox has roughly 6× the area
        assert len(large) > len(small)


# ======================================================================
# generate_cog_for_tile — comportamiento, no "es invocable"
#
# Esta función solo tenía tests de "callable / is async": por eso un cambio
# de firma pasó desapercibido y en verde. Los dos invariantes de toda esta
# rama son (1) sin meteo no se fabrica nada, y (2) la corrección de altitud
# se hace RESPECTO A la altitud a la que ya está bajado el dato —
# `location[2]` de WeatherObserved— nunca respecto al grid del DEM.
# ======================================================================


PARCEL = {"id": "urn:ngsi-ld:AgriParcel:p1", "lon": -1.65, "lat": 42.8}

WEATHER = {
    "t_avg": 21.0,
    "t_min": 17.3,
    "t_max": 32.2,
    "rh_avg": 60.0,
    "wind_speed_ms": 2.5,
    "solar_rad_w_m2": 229.4,
    "precip_mm": 0.0,
    "reference_altitude_m": 572.8,   # tercera coordenada de location
}

# Elevaciones del tile MUY lejos de los 572.8 m de referencia: si la
# corrección se hiciera contra el grid, el test lo vería.
DEM_BASE_ELEVATION = 1200.0


def _dem(rows: int = 8, cols: int = 8) -> dict:
    return {
        "elevations": [
            [DEM_BASE_ELEVATION + r * 10.0 + c for c in range(cols)]
            for r in range(rows)
        ],
        "origin_lon": -1.65,
        "origin_lat": 42.8,
        "pixel_size_deg": 0.0001,
        "cols": cols,
        "rows": rows,
    }


async def _tile(cg, metric="temperature_avg"):
    return await cg.generate_cog_for_tile(
        "demo-coop", metric, 14, 8117, 6038,
        "2026-09-01", "2026-09-02", 42.8, -1.65, parcels=[PARCEL],
    )


class TestGenerateCogForTile:
    """Comportamiento con y sin meteo — sin HTTP real."""

    @pytest.mark.asyncio
    async def test_absent_observation_yields_no_tile(self, monkeypatch):
        from app import cog_generator as cg

        monkeypatch.setattr(cg, "fetch_dem_tile", AsyncMock(return_value=_dem()))
        monkeypatch.setattr(cg, "fetch_parcel_weather", AsyncMock(return_value=None))
        assert await _tile(cg) is None

    @pytest.mark.asyncio
    async def test_incomplete_weather_returns_none_and_fabricates_nothing(
        self, monkeypatch, caplog,
    ):
        """Falta t_min: se salta el tile con log ERROR y la física NO se llama.
        El bug original era exactamente esto resuelto con un 10.0."""
        from app import cog_generator as cg

        physics_calls: list = []
        incomplete = {k: v for k, v in WEATHER.items() if k != "t_min"}

        monkeypatch.setattr(cg, "fetch_dem_tile", AsyncMock(return_value=_dem()))
        monkeypatch.setattr(cg, "fetch_parcel_weather", AsyncMock(return_value=incomplete))
        monkeypatch.setattr(
            cg, "correct_temperature", lambda *a: physics_calls.append(a),
        )

        with caplog.at_level(logging.ERROR):
            result = await _tile(cg)

        assert result is None
        assert physics_calls == [], "no se calcula nada sobre una entrada ausente"
        assert "t_min" in caplog.text

    @pytest.mark.asyncio
    async def test_altitude_correction_uses_the_observation_altitude(self, monkeypatch):
        """`station_elevation` es `reference_altitude_m` (location[2]), no la
        elevación del grid del DEM: la corrección es RELATIVA a la altitud a
        la que el dato ya viene bajado."""
        from app import cog_generator as cg

        seen: dict = {}

        def _spy(t_base, station_elevation_m, pixel_elevations):
            seen["t_base"] = t_base
            seen["station"] = station_elevation_m
            return np.asarray(pixel_elevations, dtype=float) * 0.0 + t_base

        monkeypatch.setattr(cg, "fetch_dem_tile", AsyncMock(return_value=_dem()))
        monkeypatch.setattr(cg, "fetch_parcel_weather", AsyncMock(return_value=dict(WEATHER)))
        monkeypatch.setattr(cg, "correct_temperature", _spy)

        result = await _tile(cg)

        assert result is not None and len(result) > 0
        assert seen["t_base"] == 21.0
        assert seen["station"] == 572.8
        assert seen["station"] != DEM_BASE_ELEVATION

    @pytest.mark.asyncio
    async def test_weather_is_read_once_per_parcel_per_run(self, monkeypatch):
        """2 GETs a Orion por (métrica x tile), sin caché, con un AsyncClient
        nuevo cada vez: siete métricas por un número de tiles sin tope son
        miles de conexiones por tenant y por cron contra un broker que ya
        tuvo un deadlock por agotamiento de conexiones. La meteo es la misma
        para las siete métricas y para todo tile con la misma parcela."""
        from app import cog_generator as cg

        reads: list[str] = []

        async def _weather(tenant_id, parcel_id):
            reads.append(parcel_id)
            return dict(WEATHER)

        monkeypatch.setattr(cg, "fetch_dem_tile", AsyncMock(return_value=_dem()))
        monkeypatch.setattr(cg, "fetch_parcel_weather", _weather)
        monkeypatch.setattr(cg, "upload_cog", lambda *a, **kw: True)
        monkeypatch.setattr(cg, "set_latest_date", lambda *a, **kw: True)
        monkeypatch.setattr(
            cg.settings, "metrics",
            ["temperature_avg", "temperature_min", "frost_risk"],
        )

        await cg._generate_and_upload_cogs(
            "demo-coop", [PARCEL], "2026-09-01", "2026-09-02", 14,
            "2026-09-02",
        )

        assert reads == ["urn:ngsi-ld:AgriParcel:p1"], (
            f"la meteo de una parcela se leyó {len(reads)} veces en un run"
        )


class TestSoilMoistureWithoutSoil:
    @pytest.mark.asyncio
    async def test_no_soil_makes_metric_unavailable_no_defaults(self, monkeypatch):
        from app import cog_generator as cg

        monkeypatch.setattr(cg, "fetch_dem_tile", AsyncMock(return_value=_dem()))
        monkeypatch.setattr(cg, "fetch_parcel_weather", AsyncMock(return_value=dict(WEATHER)))
        monkeypatch.setattr(cg, "fetch_parcel_hydraulics", AsyncMock(return_value=None))
        assert await _tile(cg, "soil_moisture") is None
        assert not hasattr(cg, "get_texture_defaults")

    @pytest.mark.asyncio
    async def test_soil_moisture_uses_real_hydraulics(self, monkeypatch):
        from app import cog_generator as cg

        monkeypatch.setattr(cg, "fetch_dem_tile", AsyncMock(return_value=_dem()))
        monkeypatch.setattr(cg, "fetch_parcel_weather", AsyncMock(return_value=dict(WEATHER)))
        monkeypatch.setattr(
            cg, "fetch_parcel_hydraulics",
            AsyncMock(return_value={"field_capacity": 30.0, "wilting_point": 15.0, "awc": 15.0}),
        )
        five = {"precip_mm": 4.0, "et0_mm": 20.0, "days": 5, "start": "2026-08-27", "end": "2026-08-31"}
        monkeypatch.setattr(
            cg, "fetch_parcel_daily", AsyncMock(return_value={"closed5": five}),
        )
        assert await _tile(cg, "soil_moisture") is not None

        monkeypatch.setattr(cg, "fetch_parcel_daily", AsyncMock(return_value={}))
        assert await _tile(cg, "soil_moisture") is None

    @pytest.mark.asyncio
    async def test_no_usable_geometry_clears_all_pointers(self, monkeypatch, caplog):
        import logging
        from app import cog_generator as cg

        cleared: list = []
        monkeypatch.setattr(cg, "clear_latest_pointer", lambda t, m: cleared.append((t, m)))
        monkeypatch.setattr(cg.settings, "metrics", ["temperature_avg", "eto"])
        with caplog.at_level(logging.WARNING):
            await cg._generate_and_upload_cogs(
                "tenant-a", [{"id": "p"}], "2026-09-01", "2026-09-02", 14, "2026-09-02",
            )
        assert cleared == [("tenant-a", "temperature_avg"), ("tenant-a", "eto")]
        assert "tenant-a" in caplog.text and "clearing raster pointers" in caplog.text

    @pytest.mark.asyncio
    async def test_no_parcels_clears_all_pointers(self, monkeypatch, caplog):
        import logging
        from app import cog_generator as cg

        cleared: list = []
        monkeypatch.setattr(cg, "clear_latest_pointer", lambda t, m: cleared.append((t, m)))
        monkeypatch.setattr(cg.settings, "metrics", ["temperature_avg", "eto"])
        with caplog.at_level(logging.WARNING):
            await cg.run_for_tenant("tenant-a", [], "2026-09-01", "2026-09-02")
        assert cleared == [("tenant-a", "temperature_avg"), ("tenant-a", "eto")]
        assert "tenant-a" in caplog.text and "No parcels" in caplog.text

    @pytest.mark.asyncio
    async def test_job_makes_no_soil_tiles_and_clears_pointer(self, monkeypatch):
        from app import cog_generator as cg

        seen: list = []

        async def _fake_tile(tenant_id, metric, *a, **kw):
            seen.append(metric)
            return b"cog"

        hydraulics = AsyncMock(return_value=None)
        monkeypatch.setattr(cg, "generate_cog_for_tile", _fake_tile)
        monkeypatch.setattr(cg, "fetch_parcel_hydraulics", hydraulics)
        monkeypatch.setattr(cg, "upload_cog", lambda *a, **kw: True)
        monkeypatch.setattr(cg, "set_latest_date", lambda *a, **kw: True)
        cleared: list = []
        monkeypatch.setattr(cg, "clear_latest_pointer", lambda t, m: cleared.append((t, m)))
        monkeypatch.setattr(cg.settings, "metrics", ["temperature_avg", "soil_moisture"])

        await cg._generate_and_upload_cogs(
            "tenant-a", [PARCEL], "2026-09-01", "2026-09-02", 14, "2026-09-02",
        )
        assert seen and set(seen) == {"temperature_avg"}
        assert cleared == [("tenant-a", "soil_moisture")]
        hydraulics.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_metric_with_no_tiles_clears_latest_pointer(self, monkeypatch):
        from app import cog_generator as cg

        monkeypatch.setattr(cg, "fetch_dem_tile", AsyncMock(return_value=_dem()))
        monkeypatch.setattr(cg, "fetch_parcel_weather", AsyncMock(return_value=dict(WEATHER)))
        monkeypatch.setattr(cg, "fetch_parcel_hydraulics", AsyncMock(return_value=None))
        monkeypatch.setattr(cg, "upload_cog", lambda *a, **kw: True)
        monkeypatch.setattr(cg, "set_latest_date", lambda *a, **kw: True)
        cleared: list = []
        monkeypatch.setattr(cg, "clear_latest_pointer", lambda t, m: cleared.append((t, m)))
        monkeypatch.setattr(cg.settings, "metrics", ["temperature_avg", "soil_moisture"])

        await cg._generate_and_upload_cogs(
            "tenant-a", [PARCEL], "2026-09-01", "2026-09-02", 14, "2026-09-02",
        )
        assert cleared == [("tenant-a", "soil_moisture")]


# ======================================================================
# eto / water_balance: inputs from /daily, each pixel from its own parcel
# ======================================================================

import rasterio  # noqa: E402
from rasterio import MemoryFile  # noqa: E402

PX = 0.0001
ORIGIN_LON, ORIGIN_LAT = -1.65, 42.8


def _wide_dem(rows: int = 10, cols: int = 24) -> dict:
    return {
        "elevations": [[500.0 + r + c * 0.1 for c in range(cols)] for r in range(rows)],
        "origin_lon": ORIGIN_LON,
        "origin_lat": ORIGIN_LAT,
        "pixel_size_deg": PX,
        "rows": rows,
        "cols": cols,
    }


def _box(col0: int, col1: int, rows: int = 10) -> dict:
    """Polygon covering pixel columns [col0, col1) of the wide DEM, full height."""
    w = ORIGIN_LON + col0 * PX
    e = ORIGIN_LON + col1 * PX
    s = ORIGIN_LAT
    n = ORIGIN_LAT + rows * PX
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


PARCEL_A = {"id": "urn:ngsi-ld:AgriParcel:a", "location": _box(0, 5)}
PARCEL_B = {"id": "urn:ngsi-ld:AgriParcel:b", "location": _box(19, 24)}


def _daily(t_max=30.0, closed5=None, **over) -> dict:
    d = {
        "date": "2026-07-06", "doy": 187, "t_min": 15.0, "t_max": t_max,
        "ea_kpa": 1.3, "u2_ms": 2.0, "rs_mj": 22.0, "elevation_ref_m": 500.0,
    }
    if closed5 is not None:
        d["closed5"] = closed5
    d.update(over)
    return d


def _c5(precip, et0):
    return {"precip_mm": precip, "et0_mm": et0, "days": 5, "start": "2026-07-01", "end": "2026-07-05"}


def _read(cog: bytes) -> np.ndarray:
    with MemoryFile(cog) as mf, mf.open() as ds:
        return ds.read(1)


async def _daily_tile(cg, monkeypatch, metric, by_parcel, parcels=(PARCEL_A, PARCEL_B)):
    async def _fetch(tenant_id, parcel_id):
        return by_parcel.get(parcel_id)

    monkeypatch.setattr(cg, "fetch_dem_tile", AsyncMock(return_value=_wide_dem()))
    monkeypatch.setattr(cg, "fetch_parcel_daily", _fetch)
    return await cg.generate_cog_for_tile(
        "demo-coop", metric, 14, 8117, 6038,
        "2026-07-06", "2026-07-06", 42.8, -1.65, parcels=list(parcels),
    )


@pytest.mark.asyncio
async def test_pixel_takes_containing_parcel_weather(monkeypatch):
    from app import cog_generator as cg

    cog = await _daily_tile(cg, monkeypatch, "eto", {
        PARCEL_A["id"]: _daily(t_max=20.0),
        PARCEL_B["id"]: _daily(t_max=36.0),
    })
    arr = _read(cog)
    cool = arr[:, 0:5]
    hot = arr[:, 19:24]
    assert np.all(np.isfinite(cool)) and np.all(np.isfinite(hot))
    assert hot.mean() > cool.mean() + 0.5


@pytest.mark.asyncio
async def test_pixels_outside_parcels_are_nodata(monkeypatch):
    from app import cog_generator as cg

    cog = await _daily_tile(cg, monkeypatch, "eto", {
        PARCEL_A["id"]: _daily(), PARCEL_B["id"]: _daily(),
    })
    arr = _read(cog)
    # columns 8..15 are more than the clip margin away from both parcels
    assert np.all(np.isnan(arr[:, 8:16]))
    assert np.all(np.isfinite(arr[:, 0:5]))


@pytest.mark.asyncio
async def test_eto_parcel_missing_an_input_is_skipped_not_filled(monkeypatch, caplog):
    from app import cog_generator as cg

    no_ea = {k: v for k, v in _daily().items() if k != "ea_kpa"}
    with caplog.at_level(logging.ERROR):
        cog = await _daily_tile(cg, monkeypatch, "eto", {
            PARCEL_A["id"]: no_ea, PARCEL_B["id"]: _daily(),
        })
    arr = _read(cog)
    assert np.all(np.isnan(arr[:, 0:5]))
    assert np.all(np.isfinite(arr[:, 19:24]))
    assert "ea_kpa" in caplog.text


@pytest.mark.asyncio
async def test_eto_with_every_parcel_skipped_yields_no_tile(monkeypatch):
    from app import cog_generator as cg

    no_ea = {k: v for k, v in _daily().items() if k != "ea_kpa"}
    assert await _daily_tile(cg, monkeypatch, "eto", {
        PARCEL_A["id"]: no_ea, PARCEL_B["id"]: no_ea,
    }) is None
    # weather-api unreachable: same
    assert await _daily_tile(cg, monkeypatch, "eto", {}) is None


@pytest.mark.asyncio
async def test_doy_comes_from_the_running_day(monkeypatch):
    from app import cog_generator as cg

    seen: list = []
    real = cg.compute_eto

    def _spy(*args):
        seen.append(args[-1])
        return real(*args)

    monkeypatch.setattr(cg, "compute_eto", _spy)
    await _daily_tile(cg, monkeypatch, "eto", {PARCEL_A["id"]: _daily(doy=187)}, parcels=[PARCEL_A])
    assert seen == [187]


@pytest.mark.asyncio
async def test_water_balance_is_closed_5day_sum(monkeypatch):
    from app import cog_generator as cg

    cog = await _daily_tile(cg, monkeypatch, "water_balance", {
        PARCEL_A["id"]: _daily(closed5=_c5(10.0, 25.0)),
        PARCEL_B["id"]: _daily(closed5=_c5(2.0, 12.0)),
    })
    arr = _read(cog)
    assert np.allclose(arr[:, 0:5], -15.0)
    assert np.allclose(arr[:, 19:24], -10.0)


@pytest.mark.asyncio
async def test_water_balance_absent_when_window_incomplete(monkeypatch):
    from app import cog_generator as cg

    cog = await _daily_tile(cg, monkeypatch, "water_balance", {
        PARCEL_A["id"]: _daily(closed5=_c5(10.0, 25.0)),
        PARCEL_B["id"]: _daily(),            # fewer than 5 closed days: no closed5
    })
    arr = _read(cog)
    assert np.allclose(arr[:, 0:5], -15.0)
    assert np.all(np.isnan(arr[:, 19:24]))


@pytest.mark.asyncio
async def test_raster_ea_is_clamped_to_pixel_tmin_saturation(monkeypatch):
    """Parcel reference far below the pixels (they are colder): a humid-day ea equal
    to e0(Tmin_ref) would exceed e0(Tmin_px); the clamp must bring it under."""
    from app import cog_generator as cg
    from app.downscaler import _saturation_vapour_pressure

    seen: dict = {}
    real = cg.compute_eto

    def _spy(t_min, t_max, ea, *rest):
        seen["t_min"], seen["ea"] = np.asarray(t_min), np.asarray(ea)
        return real(t_min, t_max, ea, *rest)

    monkeypatch.setattr(cg, "compute_eto", _spy)
    humid = _daily(elevation_ref_m=100.0, ea_kpa=float(_saturation_vapour_pressure(15.0)))
    cog = await _daily_tile(cg, monkeypatch, "eto", {PARCEL_A["id"]: humid}, parcels=[PARCEL_A])
    assert cog is not None
    bound = _saturation_vapour_pressure(seen["t_min"])
    assert np.any(humid["ea_kpa"] > bound), "the clamp must be exercised"
    assert np.all(seen["ea"] <= bound + 1e-12)


def test_margin_pixels_take_the_parcel_and_exact_polygon_beats_neighbour_margin():
    from app.cog_generator import _parcel_pixel_labels
    from rasterio.transform import from_bounds

    rows, cols = 4, 24
    tr = from_bounds(ORIGIN_LON, ORIGIN_LAT, ORIGIN_LON + cols * PX, ORIGIN_LAT + rows * PX, cols, rows)
    a = {"id": "a", "location": _box(0, 5, rows)}
    b = {"id": "b", "location": _box(6, 12, rows)}  # 1 pixel gap, inside both margins
    labels = _parcel_pixel_labels([a, b], tr, rows, cols)
    assert np.all(labels[:, 0:5] == 0) and np.all(labels[:, 6:12] == 1)
    # column 5 is in neither polygon: it takes a parcel from the margin pass
    assert np.all(labels[:, 5] >= 0)
    # b's exact pixels next to a are not stolen by a's margin
    assert np.all(labels[:, 6] == 1)
    # far from every parcel: nodata
    assert np.all(labels[:, 18:24] == -1)
    # the margin does not stretch beyond ~20 m (2 pixels): column 14+ is free
    assert np.all(labels[:, 14] == -1)


def test_record_balance_window_from_daily_cache(monkeypatch):
    import app.cog_generator as cg
    written = []
    monkeypatch.setattr(cg, "set_window_meta", lambda t, m, meta: written.append((t, m, meta)))
    c5 = {"start": "2026-06-08", "end": "2026-06-12", "days": 5}
    cg._record_balance_window("t1", "2026-06-13", {"daily:a": {"closed5": c5}, "daily:b": None, "x": {}})
    assert written == [("t1", "water_balance", {
        "date": "2026-06-13", "window_start": "2026-06-08",
        "window_end": "2026-06-12", "days": 5})]
    written.clear()
    cg._record_balance_window("t1", "2026-06-13", {"daily:a": {}})
    assert written == []
