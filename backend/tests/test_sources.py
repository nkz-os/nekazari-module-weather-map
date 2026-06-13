"""Tests for sources.py — HTTP clients for external services.

Tests pure path/format logic only — no network calls.
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.sources import fetch_tenant_parcels, tms_tile_to_bbox
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


# ======================================================================
# fetch_tenant_parcels
# ======================================================================


class TestFetchTenantParcels:
    """Tests for fetch_tenant_parcels — mocked HTTP calls."""

    @staticmethod
    def _mock_client(response_data, raise_for_status=None):
        """Create a patched httpx.AsyncClient that returns *response_data*."""
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        if raise_for_status:
            mock_response.raise_for_status.side_effect = raise_for_status
        else:
            mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response

        patcher = patch("httpx.AsyncClient", return_value=mock_client)
        return patcher

    @pytest.mark.asyncio
    async def test_returns_parcels_with_location_name(self):
        """Parses id, location, and name from Orion response."""
        response_data = [
            {
                "id": "urn:ngsi-ld:AgriParcel:parcel-a",
                "type": "AgriParcel",
                "location": {
                    "type": "Point",
                    "coordinates": [-1.65, 42.8],
                },
                "name": "Parcela A",
            },
        ]
        with self._mock_client(response_data) as _:
            result = await fetch_tenant_parcels("test-tenant")

        assert len(result) == 1
        assert result[0]["id"] == "urn:ngsi-ld:AgriParcel:parcel-a"
        assert result[0]["location"] == {
            "type": "Point",
            "coordinates": [-1.65, 42.8],
        }
        assert result[0]["name"] == "Parcela A"
        assert "description" not in result[0]

    @pytest.mark.asyncio
    async def test_returns_parcels_with_description(self):
        """Includes description when present, omits name when absent."""
        response_data = [
            {
                "id": "urn:ngsi-ld:AgriParcel:parcel-b",
                "type": "AgriParcel",
                "location": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
                "description": "Southern plot",
            },
        ]
        with self._mock_client(response_data) as _:
            result = await fetch_tenant_parcels("test-tenant")

        assert len(result) == 1
        assert result[0]["id"] == "urn:ngsi-ld:AgriParcel:parcel-b"
        assert "name" not in result[0]
        assert result[0]["description"] == "Southern plot"

    @pytest.mark.asyncio
    async def test_skips_entities_without_location(self):
        """Entities missing a location geometry are skipped."""
        response_data = [
            {
                "id": "urn:ngsi-ld:AgriParcel:valid",
                "type": "AgriParcel",
                "location": {"type": "Point", "coordinates": [0.0, 0.0]},
            },
            {
                "id": "urn:ngsi-ld:AgriParcel:no-location",
                "type": "AgriParcel",
                # no "location" key
            },
            {
                "id": "urn:ngsi-ld:AgriParcel:null-location",
                "type": "AgriParcel",
                "location": None,
            },
        ]
        with self._mock_client(response_data) as _:
            result = await fetch_tenant_parcels("test-tenant")

        assert len(result) == 1
        assert result[0]["id"] == "urn:ngsi-ld:AgriParcel:valid"

    @pytest.mark.asyncio
    async def test_empty_list_on_empty_response(self):
        """Returns empty list when Orion returns no entities."""
        with self._mock_client([]) as _:
            result = await fetch_tenant_parcels("test-tenant")

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_list_on_http_error(self):
        """Returns empty list on HTTP error (e.g. 404)."""
        http_error = httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )
        with self._mock_client([], raise_for_status=http_error) as _:
            result = await fetch_tenant_parcels("test-tenant")

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_list_on_network_error(self):
        """Returns empty list when the HTTP request itself fails."""
        with patch(
            "httpx.AsyncClient",
            side_effect=ConnectionError("DNS resolution failed"),
        ):
            result = await fetch_tenant_parcels("test-tenant")

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_list_on_non_list_response(self):
        """Returns empty list if Orion returns a non-list (e.g. error object)."""
        with self._mock_client({"error": "Internal"}) as _:
            result = await fetch_tenant_parcels("test-tenant")

        assert result == []

    @pytest.mark.asyncio
    async def test_limits_to_100_parcels(self):
        """Only the first 100 parcels are returned when more exist."""
        many_parcels = [
            {
                "id": f"urn:ngsi-ld:AgriParcel:p{i}",
                "type": "AgriParcel",
                "location": {"type": "Point", "coordinates": [i, 0.0]},
            }
            for i in range(150)
        ]
        with self._mock_client(many_parcels) as _:
            result = await fetch_tenant_parcels("test-tenant")

        assert len(result) == 100


class TestCogPathFormat:
    """COG path format used in sources/minio context."""

    def test_cog_path_format(self):
        """Verify cog path matches expected pattern."""
        path = cog_path("tenant1", "temperature_avg", "2026-06-10", 14, 8557, 5302)
        assert path == "cogs/tenant1/temperature_avg/2026-06-10/14/8557/5302.tif"
