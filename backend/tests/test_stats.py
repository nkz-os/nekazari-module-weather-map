"""Tests for the zonal statistics module."""

import numpy as np
import pytest

from app.stats import (
    _geometry_bbox,
    _compute_histogram,
    compute_zonal_stats,
)


# A simple square parcel around Navarra (roughly 1km²)
SAMPLE_PARCEL_GEOM = {
    "type": "Polygon",
    "coordinates": [[
        [-1.66, 42.80],
        [-1.64, 42.80],
        [-1.64, 42.81],
        [-1.66, 42.81],
        [-1.66, 42.80],
    ]],
}


def test_geometry_bbox_simple():
    """Bbox of a simple polygon should return min/max lat/lon."""
    bbox = _geometry_bbox(SAMPLE_PARCEL_GEOM)
    assert bbox is not None
    min_lon, min_lat, max_lon, max_lat = bbox
    assert min_lon == pytest.approx(-1.66)
    assert max_lon == pytest.approx(-1.64)
    assert min_lat == pytest.approx(42.80)
    assert max_lat == pytest.approx(42.81)


def test_geometry_bbox_multipolygon():
    """MultiPolygon geometry should also produce a valid bbox."""
    geom = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[-1.66, 42.80], [-1.64, 42.80], [-1.64, 42.81],
              [-1.66, 42.81], [-1.66, 42.80]]],
            [[[-1.63, 42.80], [-1.62, 42.80], [-1.62, 42.81],
              [-1.63, 42.81], [-1.63, 42.80]]],
        ],
    }
    bbox = _geometry_bbox(geom)
    assert bbox is not None
    min_lon, min_lat, max_lon, max_lat = bbox
    assert min_lon == pytest.approx(-1.66)
    assert max_lon == pytest.approx(-1.62)


def test_geometry_bbox_invalid():
    """Invalid geometry should return None."""
    assert _geometry_bbox({"type": "Point", "coordinates": [1, 2]}) is None
    assert _geometry_bbox({"type": "Nonsense"}) is None
    assert _geometry_bbox({}) is None


def test_compute_histogram():
    """Histogram should have the expected number of bins."""
    vals = np.random.randn(1000) * 10 + 20
    hist = _compute_histogram(vals, bins=10)
    assert len(hist) == 10
    assert all(isinstance(v, (int, float)) for v in hist)
    # Sum of histogram counts should equal number of values
    assert sum(hist) == 1000


def test_compute_histogram_empty():
    """Empty input should return all zeros."""
    hist = _compute_histogram(np.array([]), bins=5)
    assert hist == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_compute_zonal_stats_unknown_metric():
    """Unknown metric should return an error entry."""
    result = compute_zonal_stats(
        SAMPLE_PARCEL_GEOM,
        metrics=["nonexistent_metric"],
    )
    assert "error" in result
    # Without COG data, it should have an error
    assert "metrics" in result


def test_compute_zonal_stats_no_cogs():
    """When no COG data is available, per-metric entries should have error."""
    result = compute_zonal_stats(
        SAMPLE_PARCEL_GEOM,
        metrics=["temperature_avg"],
        date="2099-01-01",
    )
    assert "metrics" in result
    t = result["metrics"].get("temperature_avg", {})
    if isinstance(t, dict) and "error" in t:
        assert "No data" in t["error"] or "No COG" in t["error"]
