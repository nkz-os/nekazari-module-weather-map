"""Placing eu-elevation /raster grids: row 0 is the north edge."""
import pytest

from app.dem_grid import dem_transform, north_edge, pixel_center_grids


def _dem(**extra):
    return {"origin_lon": -1.0, "origin_lat": 42.0, "pixel_size_deg": 0.001,
            "rows": 10, "cols": 10, **extra}


def test_north_edge_prefers_declared_north_lat():
    assert north_edge(_dem(north_lat=42.0101)) == 42.0101
    assert north_edge(_dem()) == pytest.approx(42.01)


def test_pixel_centers_row0_is_north():
    lon_g, lat_g = pixel_center_grids(_dem())
    assert lat_g.shape == (10, 10)
    assert lat_g[0, 0] == pytest.approx(42.0095, abs=1e-9)
    assert lat_g[-1, 0] == pytest.approx(42.0005, abs=1e-9)
    assert lon_g[0, 0] == pytest.approx(-0.9995, abs=1e-9)


def test_transform_maps_row0_to_north():
    lon, lat = dem_transform(_dem()) * (0.5, 0.5)
    assert lon == pytest.approx(-0.9995, abs=1e-9)
    assert lat == pytest.approx(42.0095, abs=1e-9)
