"""Tests for sources.py — HTTP clients for external services.

Tests pure path/format logic only — no network calls.
"""

from __future__ import annotations

import math

import pytest

from app.sources import tms_tile_to_bbox
from app.minio_io import cog_path


class TestDemTileBbox:
    """TMS tile coordinate → geographic bbox conversion."""

    def test_fetch_dem_tile_bbox(self):
        """TMS 14/8557/5302 converts to expected geographic bbox (rough Navarra area)."""
        bbox = tms_tile_to_bbox(14, 8557, 5302)

        # Re-compute expected values from the same formula
        n = 2.0 ** 14
        exp_min_lon = 8557 / n * 360.0 - 180.0
        exp_max_lon = (8557 + 1) / n * 360.0 - 180.0
        exp_min_lat = math.degrees(
            math.atan(math.sinh(math.pi * (1 - 2 * (5302 + 1) / n)))
        )
        exp_max_lat = math.degrees(
            math.atan(math.sinh(math.pi * (1 - 2 * 5302 / n)))
        )

        assert bbox["min_lon"] == pytest.approx(exp_min_lon)
        assert bbox["max_lon"] == pytest.approx(exp_max_lon)
        assert bbox["min_lat"] == pytest.approx(exp_min_lat)
        assert bbox["max_lat"] == pytest.approx(exp_max_lat)

        # Basic sanity: bbox expands eastward and northward
        assert bbox["min_lon"] < bbox["max_lon"]
        assert bbox["min_lat"] < bbox["max_lat"]

    def test_zoom_0_single_tile(self):
        """Zoom 0 covers the whole world in one tile."""
        bbox = tms_tile_to_bbox(0, 0, 0)
        assert bbox["min_lon"] == pytest.approx(-180.0)
        assert bbox["max_lon"] == pytest.approx(180.0)
        assert bbox["min_lat"] == pytest.approx(-85.051129, abs=1e-5)
        assert bbox["max_lat"] == pytest.approx(85.051129, abs=1e-5)

    def test_zoom_15_equator_tile(self):
        """Tile at equator should have roughly square aspect ratio."""
        z, x, y = 15, 16384, 16384  # equator (y = n/2 for z=15)
        bbox = tms_tile_to_bbox(z, x, y)
        # Width in degrees
        lon_span = bbox["max_lon"] - bbox["min_lon"]
        lat_span = bbox["max_lat"] - bbox["min_lat"]
        # At equator, lat and lon spans are roughly equal (symmetry of Mercator)
        assert lon_span == pytest.approx(0.010986, abs=1e-5)
        assert lat_span == pytest.approx(0.010986, abs=1e-5)

    def test_bbox_order(self):
        """Bbox keys are in order min_lon → max_lat."""
        for z, x, y in [(5, 10, 15), (10, 500, 500), (16, 30000, 20000)]:
            bbox = tms_tile_to_bbox(z, x, y)
            assert bbox["min_lon"] < bbox["max_lon"]
            assert bbox["min_lat"] < bbox["max_lat"]


class TestCogPathFormat:
    """COG path format used in sources/minio context."""

    def test_cog_path_format(self):
        """Verify cog path matches expected pattern."""
        path = cog_path("tenant1", "temperature_avg", "2026-06-10", 14, 8557, 5302)
        assert path == "cogs/tenant1/temperature_avg/2026-06-10/14/8557/5302.tif"
