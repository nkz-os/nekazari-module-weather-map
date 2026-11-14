"""Tests for zone accumulator helpers."""
import numpy as np
from app.zone_accumulator import _compute_area_ha, _geom_geojson_bbox


def test_compute_area_ha_from_latitude():
    import pytest
    # 10 m x 10 m pixels at the equator: 0.01 ha each.
    side = 10.0 / 111_320.0
    assert _compute_area_ha(np.zeros(100), side, side) == pytest.approx(1.0)
    assert _compute_area_ha(np.zeros(0), side, side) == 0.0
    # A pixel 0.0001 deg wide and high at 42.8 N is ~8.17 m x 11.13 m.
    expected = (0.0001 * 111_320.0 * np.cos(np.radians(42.8))) * (0.0001 * 111_320.0) / 10_000.0
    assert _compute_area_ha(np.full(1, 42.8), 0.0001, 0.0001) == pytest.approx(expected)
    assert _compute_area_ha(np.full(1, 42.8), 0.0001, 0.0001) < 0.0111  # not the old flat 0.01 x 1.24


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


def _synthetic_dem(origin_lat=42.0, size=0.001, n=10):
    return {"origin_lon": -1.0, "origin_lat": origin_lat, "pixel_size_deg": size,
            "rows": n, "cols": n}


def test_zone_geometry_north_rows_land_north():
    from shapely.geometry import shape
    from app.dem_grid import dem_transform
    from app.zone_accumulator import _zone_geometry
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:5, :] = True
    geom = _zone_geometry(mask, dem_transform(_synthetic_dem()), None, 0.001)
    assert shape(geom).bounds[1] >= 42.005 - 1e-6


def test_zone_geometry_two_tiles_do_not_swap():
    from shapely.geometry import shape
    from app.dem_grid import dem_transform
    from app.zone_accumulator import _zone_geometry
    mask = np.ones((10, 10), dtype=bool)
    south = shape(_zone_geometry(mask, dem_transform(_synthetic_dem(42.0)), None, 0.001))
    north = shape(_zone_geometry(mask, dem_transform(_synthetic_dem(42.01)), None, 0.001))
    assert 42.0 - 1e-6 <= south.bounds[1] and south.bounds[3] <= 42.01 + 1e-6
    assert 42.01 - 1e-6 <= north.bounds[1] and north.bounds[3] <= 42.02 + 1e-6


def test_zone_geometry_is_clipped_and_valid():
    from shapely.geometry import Polygon, mapping, shape
    from app.dem_grid import dem_transform
    from app.zone_accumulator import _zone_geometry
    parcel = Polygon([(-0.998, 42.001), (-0.991, 42.002), (-0.995, 42.009)])
    mask = np.ones((10, 10), dtype=bool)
    geom = shape(_zone_geometry(mask, dem_transform(_synthetic_dem()), mapping(parcel), 0.001))
    assert geom.is_valid
    assert geom.within(parcel.buffer(1e-9))
    assert geom.area >= 0.99 * parcel.area


def _sectors_for_north_south_drop(drop_m):
    from nkz_platform_sdk.gis.terrain import aspect_degrees, slope_degrees
    from app.downscaler import discretize_aspect
    size = 0.0001
    elev = np.repeat(np.linspace(drop_m, 0.0, 10)[:, None], 10, axis=1)  # row 0 = north, highest
    return discretize_aspect(
        aspect_degrees(elev, size, 42.8, size), slope_degrees(elev, size, 42.8, size),
    )[1:-1, 1:-1]


def test_gentle_slope_is_flat_sector():
    assert np.all(_sectors_for_north_south_drop(0.5) == 0)


def test_south_facing_slope_is_S_sector():
    assert np.all(_sectors_for_north_south_drop(30.0) == 5)


def test_terrain_slope_is_metric():
    """30 m drop over 30 rows of ~11 m: ~5 deg, not the ~90 deg of metres over degrees."""
    from app import zone_accumulator as za
    assert not hasattr(za, "_gradient")
    elev = np.repeat(np.linspace(30.0, 0.0, 30)[:, None], 30, axis=1)
    slope, _ = za._terrain(elev, 0.0001, 42.8, 0.0001)
    assert 3.0 < float(np.median(slope[5:-5, 5:-5])) < 8.0


def test_terrain_east_west_slope_uses_latitude():
    """East-west plane at 42.8 N: a pixel is 8.17 m wide, not 11.13 m."""
    from app import zone_accumulator as za
    size = 0.0001
    dx_m = size * 111_320.0 * np.cos(np.radians(42.8))
    step = np.tan(np.radians(10.0)) * dx_m
    elev = np.repeat((np.arange(30) * step)[None, :], 30, axis=0)
    slope, _ = za._terrain(elev, size, 42.8, size)
    assert float(np.median(slope[8:-8, 8:-8])) == pytest.approx(10.0, abs=0.5)


# ======================================================================
# process_parcel — comportamiento, no "es invocable".
#
# Mismos dos invariantes que `generate_cog_for_tile`: sin meteo no se
# fabrica nada, y la corrección de altitud va contra `reference_altitude_m`
# (altitud de referencia de /daily), nunca contra el grid del DEM.
# ======================================================================

import logging  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402

import pytest  # noqa: E402

PARCEL = {"id": "urn:ngsi-ld:AgriParcel:p1", "lon": -1.65, "lat": 42.8}

WEATHER = {
    "date": "2026-09-02",
    "doy": 245,
    "t_min": 17.3,
    "t_max": 32.2,
    "ea_kpa": 1.4,
    "u2_ms": 2.5,
    "rs_mj": 19.8,
    "elevation_ref_m": 572.8,
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
    monkeypatch.setattr(zone_env, "fetch_parcel_daily", AsyncMock(return_value=None))
    assert await zone_env.process_parcel("t", PARCEL, "2026-09-02") is None


@pytest.mark.asyncio
async def test_incomplete_weather_publishes_nothing_and_computes_nothing(
    zone_env, monkeypatch, caplog,
):
    physics_calls: list = []
    incomplete = {k: v for k, v in WEATHER.items() if k != "t_max"}

    monkeypatch.setattr(
        zone_env, "fetch_parcel_daily", AsyncMock(return_value=incomplete),
    )
    monkeypatch.setattr(
        zone_env, "correct_temperature", lambda *a: physics_calls.append(a),
    )

    with caplog.at_level(logging.ERROR):
        entities = await zone_env.process_parcel("t", PARCEL, "2026-09-02")

    assert entities is None
    assert physics_calls == []
    assert "t_max" in caplog.text


@pytest.mark.asyncio
async def test_zone_correction_uses_the_observation_altitude(zone_env, monkeypatch):
    seen: list[float] = []

    def _spy(t_base, station_elevation_m, pixel_elevations):
        seen.append(station_elevation_m)
        return np.asarray(pixel_elevations, dtype=float) * 0.0 + t_base

    monkeypatch.setattr(
        zone_env, "fetch_parcel_daily", AsyncMock(return_value=dict(WEATHER)),
    )
    monkeypatch.setattr(zone_env, "correct_temperature", _spy)

    entities = await zone_env.process_parcel("t", PARCEL, "2026-09-02")

    assert entities, "con meteo completa se esperan entidades AgriParcelZone"
    assert set(seen) == {572.8}
    assert DEM_BASE_ELEVATION not in seen


def test_assign_zone_ranks_north_to_south_then_west_to_east():
    from app.zone_accumulator import _assign_zone_ranks
    zones = [
        {"id": "s", "centroid": [0.0, 1.0]},
        {"id": "ne", "centroid": [2.0, 5.0]},
        {"id": "nw", "centroid": [1.0, 5.0]},
    ]
    assert _assign_zone_ranks(zones) == {"nw": 1, "ne": 2, "s": 3}


def _colliding_zones():
    return [
        {"id": "zabc-e1-N", "centroid": [1.0, 1.0]},
        {"id": "zabc-e1-N", "centroid": [2.0, 5.0]},
        {"id": "zabc-e1-N", "centroid": [1.0, 5.0]},
        {"id": "zxyz-e2-S", "centroid": [0.0, 3.0]},
    ]


def test_dedupe_zone_ids_suffixes_in_north_then_west_order():
    from app.zone_accumulator import _dedupe_zone_ids
    out = _dedupe_zone_ids(_colliding_zones())
    by_centroid = {tuple(z["centroid"]): z["id"] for z in out}
    assert by_centroid[(1.0, 5.0)] == "zabc-e1-N"
    assert by_centroid[(2.0, 5.0)] == "zabc-e1-N-2"
    assert by_centroid[(1.0, 1.0)] == "zabc-e1-N-3"
    assert by_centroid[(0.0, 3.0)] == "zxyz-e2-S"
    assert len({z["id"] for z in out}) == len(out)


def test_dedupe_zone_ids_is_noop_for_unique_ids():
    from app.zone_accumulator import _dedupe_zone_ids
    zones = [{"id": "a", "centroid": [0.0, 0.0]}, {"id": "b", "centroid": [1.0, 1.0]}]
    assert _dedupe_zone_ids(zones) is zones


def test_assign_zone_ranks_stays_contiguous_with_duplicate_ids(caplog):
    from app.zone_accumulator import _assign_zone_ranks
    with caplog.at_level(logging.ERROR):
        ranks = _assign_zone_ranks(_colliding_zones())
    assert sorted(ranks.values()) == [1, 2, 3, 4]
    assert len(ranks) == 4
    assert "Duplicate zone ids" in caplog.text


@pytest.mark.asyncio
async def test_colliding_zone_ids_yield_distinct_entities(zone_env, monkeypatch):
    monkeypatch.setattr(
        zone_env, "fetch_parcel_daily", AsyncMock(return_value=dict(WEATHER)),
    )
    monkeypatch.setattr(zone_env.geohash2, "encode", lambda *a, **k: "samecel")
    monkeypatch.setattr(zone_env.settings, "zones_elevation_band_m", 100000.0)

    def _two_labels(elev, aspect, slope, **kwargs):
        labels = np.zeros(elev.shape, dtype=int)
        labels[: elev.shape[0] // 2, :] = 1
        labels[elev.shape[0] // 2 :, :] = 2
        return [], labels

    monkeypatch.setattr(zone_env, "compute_zones", _two_labels)
    monkeypatch.setattr(
        zone_env, "_zone_geometry",
        lambda *a, **k: {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
    )

    # Both labelled halves must lie inside the parcel: zone stats are parcel-masked.
    entities = await zone_env.process_parcel("t", _covering(), "2026-09-02")

    ids = [e["id"] for e in entities]
    assert len(ids) == len(set(ids))
    assert len(entities) >= 2
    assert any(i.endswith("-2") for i in ids), "test must exercise a collision"
    ranks = sorted(e["nkz:zoneRank"]["value"] for e in entities)
    assert ranks == list(range(1, len(entities) + 1))


@pytest.mark.asyncio
async def test_zone_gets_modelled_slope_radiation_and_no_frost(zone_env, monkeypatch):
    """Zone radiation is the Allen et al. slope model, not the horizontal raster value."""
    from app import slope_radiation as sr

    monkeypatch.setattr(zone_env, "fetch_parcel_daily", AsyncMock(return_value=dict(WEATHER)))
    calls = []
    real = sr.slope_radiation_wm2

    def _spy(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(zone_env, "slope_radiation_wm2", _spy)
    entities = await zone_env.process_parcel("t", PARCEL, "2026-09-02")
    assert entities and calls
    rs_wm2 = WEATHER["rs_mj"] * 1e6 / 86400.0
    for e in entities:
        v = e["nkz:solarRadiationWm2"]["value"]
        assert 0.0 < v < 2.0 * rs_wm2
        assert "nkz:frostRisk" not in e
        assert "frostRisk" in e["_cog_checked"] and "solarRadiationWm2" in e["_cog_checked"]
    # horizontal value goes in as the measurement, doy and ea from /daily
    assert all(c[0] == pytest.approx(rs_wm2) and c[2] == WEATHER["doy"] and c[6] == WEATHER["ea_kpa"] for c in calls)


def test_zone_radiation_north_slope_below_south_slope_in_winter():
    from app import zone_accumulator as za

    def _zone(aspect):
        return {
            "id": "z", "centroid": [-1.65, 42.8], "elevationMean": 500.0,
            za._TERRAIN_KEY: (np.full(50, 20.0), np.full(50, aspect)),
        }

    kw = {"doy": 355, "rs_mj": 5.0, "ea_kpa": 0.8}
    south, _ = za._zone_radiation(_zone(180.0), **kw)
    north, _ = za._zone_radiation(_zone(0.0), **kw)
    flat, _ = za._zone_radiation({**_zone(0.0), za._TERRAIN_KEY: (np.zeros(50), np.zeros(50))}, **kw)
    assert south["solarRadiationWm2"] > flat["solarRadiationWm2"] > north["solarRadiationWm2"]
    assert flat["solarRadiationWm2"] == pytest.approx(5.0 * 1e6 / 86400.0, abs=0.1)


def test_zone_radiation_failure_is_isolated(monkeypatch, caplog):
    from app import zone_accumulator as za

    def _boom(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(za, "slope_radiation_wm2", _boom)
    zone = {"id": "z", "centroid": [0.0, 42.0], za._TERRAIN_KEY: (np.ones(3), np.ones(3))}
    with caplog.at_level(logging.WARNING):
        values, checked = za._zone_radiation(zone, doy=100, rs_mj=15.0, ea_kpa=1.0)
    assert values == {}
    assert "solarRadiationWm2" not in checked  # a stale value is kept, not erased
    assert "Slope radiation failed" in caplog.text and "model down" in caplog.text


def test_zone_without_terrain_pixels_gives_no_radiation():
    from app import zone_accumulator as za
    assert za._zone_radiation({"id": "z1", "centroid": [0, 42]}, doy=100, rs_mj=15.0, ea_kpa=1.0) == (
        {}, {"frostRisk"},
    )


@pytest.mark.asyncio
async def test_zone_eto_uses_daily_inputs(zone_env, monkeypatch):
    """Each zone's ET0 is compute_eto on the /daily running-day inputs, with the
    running day's doy (not today's) and the zone's Tmin/Tmax."""
    seen: list = []
    real = zone_env.compute_eto

    def _spy(t_min, t_max, ea, u2, rs_h, rs_s, z, lat, doy):
        seen.append({"ea": ea, "u2": u2, "rs_h": rs_h, "doy": doy,
                     "t_min": float(t_min[0]), "t_max": float(t_max[0])})
        return real(t_min, t_max, ea, u2, rs_h, rs_s, z, lat, doy)

    monkeypatch.setattr(zone_env, "fetch_parcel_daily", AsyncMock(return_value=dict(WEATHER)))
    monkeypatch.setattr(zone_env, "compute_eto", _spy)
    entities = await zone_env.process_parcel("t", PARCEL, "2026-09-02")

    assert entities and len(seen) == len(entities)
    for call in seen:
        assert call["ea"] == 1.4
        assert call["u2"] == 2.5
        assert call["rs_h"] == 19.8
        assert call["doy"] == 245


@pytest.mark.asyncio
async def test_zone_missing_vapour_pressure_publishes_nothing(zone_env, monkeypatch, caplog):
    no_ea = {k: v for k, v in WEATHER.items() if k != "ea_kpa"}
    monkeypatch.setattr(zone_env, "fetch_parcel_daily", AsyncMock(return_value=no_ea))
    with caplog.at_level(logging.ERROR):
        assert await zone_env.process_parcel("t", PARCEL, "2026-09-02") is None
    assert "ea_kpa" in caplog.text


@pytest.mark.asyncio
async def test_sloped_north_half_zone_lies_north(zone_env, monkeypatch):
    """Rows 0-5 (north) fall to the south, rows 6-11 are flat: the S zone is the northern one."""
    dem = _dem()
    dem["elevations"] = [
        [DEM_BASE_ELEVATION + (5 - r) * 3.0 if r <= 5 else DEM_BASE_ELEVATION for _ in range(12)]
        for r in range(12)
    ]
    monkeypatch.setattr(zone_env, "fetch_dem_tile", AsyncMock(return_value=dem))
    monkeypatch.setattr(zone_env, "fetch_parcel_daily", AsyncMock(return_value=WEATHER))

    covering = {**PARCEL, "location": {"type": "Polygon", "coordinates": [[
        [-1.65, 42.8], [-1.6488, 42.8], [-1.6488, 42.8012], [-1.65, 42.8012], [-1.65, 42.8],
    ]]}}
    entities = await zone_env.process_parcel("t", covering, "2026-09-02")

    def _val(e, key):
        v = e.get(f"nkz:{key}")
        return v.get("value") if isinstance(v, dict) else v

    by_sector = {_val(e, "aspectSector"): _val(e, "centroid") for e in entities}
    assert "S" in by_sector and "flat" in by_sector
    assert by_sector["S"][1] > 42.8006  # centre of the 12-row grid: the S zone is north of it


def test_integer_dem_gentle_ramps_stay_flat():
    """IGN serves whole metres: a gentle ramp must not become 1 m stripes."""
    from app.downscaler import discretize_aspect
    from app.zone_accumulator import _terrain
    size = 8.97e-5  # ~10 m
    for degrees in (0.5, 1.0, 1.5):
        drop_per_row = np.tan(np.radians(degrees)) * size * 111_320.0
        elev = np.round(np.repeat((np.arange(60)[::-1] * drop_per_row)[:, None], 60, axis=1))
        slope, aspect = _terrain(elev, size, 42.8, size)
        sectors = discretize_aspect(aspect, slope)[3:-3, 3:-3]
        assert np.all(sectors == 0), degrees


def test_integer_dem_real_slope_keeps_its_aspect():
    from app.downscaler import discretize_aspect
    from app.zone_accumulator import _terrain
    size = 8.97e-5
    drop_per_row = np.tan(np.radians(5.0)) * size * 111_320.0
    elev = np.round(np.repeat((np.arange(60)[::-1] * drop_per_row)[:, None], 60, axis=1))
    slope, aspect = _terrain(elev, size, 42.8, size)
    assert np.all(discretize_aspect(aspect, slope)[3:-3, 3:-3] == 5)


def _covering(lat_top=42.8012):
    return {**PARCEL, "location": {"type": "Polygon", "coordinates": [[
        [-1.65, 42.8], [-1.6488, 42.8], [-1.6488, lat_top], [-1.65, lat_top], [-1.65, 42.8],
    ]]}}


@pytest.mark.asyncio
async def test_zone_stats_describe_only_the_parcel_part(zone_env, monkeypatch):
    """A tile-wide zone clipped to the south half: centroid and pixels of that half only."""
    dem = _dem()
    dem["elevations"] = [[DEM_BASE_ELEVATION] * 12 for _ in range(12)]
    monkeypatch.setattr(zone_env, "fetch_dem_tile", AsyncMock(return_value=dem))
    monkeypatch.setattr(zone_env, "fetch_parcel_daily", AsyncMock(return_value=WEATHER))

    entities = await zone_env.process_parcel("t", _covering(lat_top=42.8006), "2026-09-02")

    assert len(entities) == 1
    centroid = entities[0]["nkz:centroid"]["value"]
    assert centroid[1] < 42.8006
    assert entities[0]["nkz:pixelCount"]["value"] <= 72


@pytest.mark.asyncio
async def test_invalid_parcel_geometry_is_repaired(zone_env, monkeypatch):
    dem = _dem()
    dem["elevations"] = [[DEM_BASE_ELEVATION] * 12 for _ in range(12)]
    monkeypatch.setattr(zone_env, "fetch_dem_tile", AsyncMock(return_value=dem))
    monkeypatch.setattr(zone_env, "fetch_parcel_daily", AsyncMock(return_value=WEATHER))
    bowtie = {**PARCEL, "location": {"type": "Polygon", "coordinates": [[
        [-1.65, 42.8], [-1.6488, 42.8012], [-1.6488, 42.8], [-1.65, 42.8012], [-1.65, 42.8],
    ]]}}
    assert await zone_env.process_parcel("t", bowtie, "2026-09-02")


@pytest.mark.asyncio
async def test_missing_weather_is_undecided_not_empty(zone_env, monkeypatch):
    monkeypatch.setattr(zone_env, "fetch_parcel_daily", AsyncMock(return_value=None))
    assert await zone_env.process_parcel("t", _covering(), "2026-09-02") is None


@pytest.mark.asyncio
async def test_parcel_without_zones_drops_its_stale_zones(zone_env, monkeypatch):
    from app import zone_accumulator as za
    deleted: dict = {}

    async def _delete(tenant_id, parcel_id, keep_ids):
        deleted[parcel_id] = keep_ids

    monkeypatch.setattr(za, "fetch_tenant_parcels", AsyncMock(return_value=[PARCEL]))
    monkeypatch.setattr(za, "process_parcel", AsyncMock(return_value=[]))
    monkeypatch.setattr(za, "upsert_agri_parcel_zones", AsyncMock())
    monkeypatch.setattr(za, "_delete_stale_zones", _delete)
    await za.run_for_tenant("t")
    assert deleted == {PARCEL["id"]: set()}


@pytest.mark.asyncio
async def test_undecided_parcel_keeps_its_zones(zone_env, monkeypatch):
    from app import zone_accumulator as za
    deleted: dict = {}

    async def _delete(tenant_id, parcel_id, keep_ids):
        deleted[parcel_id] = keep_ids

    monkeypatch.setattr(za, "fetch_tenant_parcels", AsyncMock(return_value=[PARCEL]))
    monkeypatch.setattr(za, "process_parcel", AsyncMock(return_value=None))
    monkeypatch.setattr(za, "upsert_agri_parcel_zones", AsyncMock())
    monkeypatch.setattr(za, "_delete_stale_zones", _delete)
    await za.run_for_tenant("t")
    assert deleted == {}


@pytest.mark.asyncio
async def test_zone_with_a_sliver_inside_the_parcel_is_dropped(zone_env, monkeypatch):
    """A zone that only grazes the parcel edge is not published as a zone of it."""
    dem = _dem()
    dem["elevations"] = [[DEM_BASE_ELEVATION] * 12 for _ in range(12)]
    monkeypatch.setattr(zone_env, "fetch_dem_tile", AsyncMock(return_value=dem))
    monkeypatch.setattr(zone_env, "fetch_parcel_daily", AsyncMock(return_value=WEATHER))
    monkeypatch.setattr(zone_env.settings, "zones_min_parcel_ha", 0.1)

    def _two_labels(elev, aspect, slope, **kwargs):
        labels = np.full(elev.shape, 1, dtype=int)
        labels[0:2, -1] = 2  # two pixels in the north-east corner: ~0.02 ha inside the parcel
        return [], labels

    monkeypatch.setattr(zone_env, "compute_zones", _two_labels)
    entities = await zone_env.process_parcel("t", _covering(), "2026-09-02")
    assert len(entities) == 1


def test_zone_area_ha_of_the_clipped_geometry():
    from app.zone_accumulator import _geometry_area_ha
    square = {"type": "Polygon", "coordinates": [[
        [-1.65, 42.8], [-1.6488, 42.8], [-1.6488, 42.8009], [-1.65, 42.8009], [-1.65, 42.8],
    ]]}
    # ~98 m (E-W at 42.8 N) x ~100 m (N-S) ~= 0.98 ha
    assert _geometry_area_ha(square) == pytest.approx(0.98, rel=0.03)


def _val(e, key):
    v = e.get(f"nkz:{key}", e.get(key))
    return v.get("value") if isinstance(v, dict) else v


@pytest.mark.asyncio
async def test_zone_tmin_is_the_reference_and_tmax_is_lapse_corrected(zone_env, monkeypatch):
    """Valley/hill zones at 1200 m against a 572.8 m reference: Tmin never lapses."""
    monkeypatch.setattr(zone_env, "fetch_parcel_daily", AsyncMock(return_value=dict(WEATHER)))
    entities = await zone_env.process_parcel("t", PARCEL, "2026-09-02")
    assert entities
    for e in entities:
        assert _val(e, "tMin") == round(WEATHER["t_min"], 1)
        expected_tmax = WEATHER["t_max"] - 0.0065 * (_val(e, "elevationMean") - WEATHER["elevation_ref_m"])
        assert _val(e, "tMax") == pytest.approx(expected_tmax, abs=0.06)
        assert _val(e, "tMax") < WEATHER["t_max"] - 3.0


@pytest.mark.asyncio
async def test_zone_eto_gets_uncorrected_tmin(zone_env, monkeypatch):
    monkeypatch.setattr(zone_env, "fetch_parcel_daily", AsyncMock(return_value=dict(WEATHER)))
    seen = []
    real = zone_env.compute_eto

    def _spy(t_min, t_max, *a, **k):
        seen.append(float(np.asarray(t_min).ravel()[0]))
        return real(t_min, t_max, *a, **k)

    monkeypatch.setattr(zone_env, "compute_eto", _spy)
    assert await zone_env.process_parcel("t", PARCEL, "2026-09-02")
    assert seen and set(seen) == {WEATHER["t_min"]}


@pytest.mark.asyncio
async def test_zone_areas_sum_to_the_parcel_area(zone_env, monkeypatch):
    """At 42.8 N a 0.0001 deg pixel is ~0.0091 ha, not 0.01: totals match the polygon."""
    from app.zone_accumulator import _geometry_area_ha
    parcel = _covering()
    monkeypatch.setattr(zone_env, "fetch_parcel_daily", AsyncMock(return_value=dict(WEATHER)))
    entities = await zone_env.process_parcel("t", parcel, "2026-09-02")
    assert entities
    total = sum(_val(e, "areaHa") for e in entities)
    parcel_ha = _geometry_area_ha(parcel["location"])
    assert total == pytest.approx(parcel_ha, rel=0.03)


@pytest.mark.asyncio
async def test_terrain_gets_grid_latitude_and_row_height(zone_env, monkeypatch):
    seen = {}
    real = zone_env._terrain

    def _spy(elev, size, latitude_deg, pixel_height_deg):
        seen.update(size=size, lat=latitude_deg, h=pixel_height_deg)
        return real(elev, size, latitude_deg, pixel_height_deg)

    monkeypatch.setattr(zone_env, "_terrain", _spy)
    monkeypatch.setattr(zone_env, "fetch_parcel_daily", AsyncMock(return_value=dict(WEATHER)))
    await zone_env.process_parcel("t", PARCEL, "2026-09-02")
    # grid: south edge 42.8, 12 rows of 0.0001 -> centre 42.8006
    assert seen["lat"] == pytest.approx(42.8006, abs=1e-9)
    assert seen["h"] == pytest.approx(0.0001)
    assert seen["size"] == pytest.approx(0.0001)
