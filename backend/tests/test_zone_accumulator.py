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
