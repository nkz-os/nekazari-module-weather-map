"""Verify GET /stats read-only behaviour and POST /stats persist behaviour.

- GET /stats MUST NOT call upsert_record.
- POST /stats MUST call upsert_record exactly once.

All dependencies are patched at the ``app.tiles`` module level so no real
Orion-LD or MinIO connections are required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

TENANT = "asociacion-allotarra"
AUTH_HEADERS = {"X-Tenant-ID": TENANT, "X-User-ID": "test-user"}

SAMPLE_GEOM = {
    "type": "Polygon",
    "coordinates": [[
        [-1.66, 42.80],
        [-1.64, 42.80],
        [-1.64, 42.81],
        [-1.66, 42.81],
        [-1.66, 42.80],
    ]],
}

SAMPLE_PARCEL = {
    "id": "urn:ngsi-ld:AgriParcel:allotarra-field-1",
    "geometry": SAMPLE_GEOM,
    "name": "Field 1",
    "description": None,
}

SAMPLE_STATS = {
    "parcel_geojson": SAMPLE_GEOM,
    "date": "2026-06-13",
    "metrics": {
        "eto": {
            "mean": 4.2,
            "min": 3.0,
            "max": 5.5,
            "std": 0.6,
            "p25": 3.8,
            "p50": 4.2,
            "p75": 4.8,
            "pixel_count": 80,
            "histogram": [2, 4, 8, 12, 16, 14, 10, 8, 4, 2],
        },
    },
}


@pytest.fixture
def client():
    return TestClient(app)


class TestGetStatsReadOnly:
    """GET /stats/{parcel_id} must never call upsert_record."""

    @patch("app.tiles.upsert_record", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", return_value=SAMPLE_STATS)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_get_does_not_call_upsert_record(
        self,
        mock_fetch: AsyncMock,
        mock_stats: MagicMock,
        mock_upsert: AsyncMock,
        client,
    ):
        """GET must not persist anything — upsert_record must not be called."""
        mock_fetch.return_value = SAMPLE_PARCEL

        resp = client.get(
            "/api/weather-map/stats/allotarra-field-1",
            params={"metrics": "eto"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        mock_upsert.assert_not_awaited()

    @patch("app.tiles.upsert_record", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", return_value=SAMPLE_STATS)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_get_returns_stats_without_persisting(
        self,
        mock_fetch: AsyncMock,
        mock_stats: MagicMock,
        mock_upsert: AsyncMock,
        client,
    ):
        """GET returns stats payload; upsert not called even on repeated calls."""
        mock_fetch.return_value = SAMPLE_PARCEL

        for _ in range(2):
            resp = client.get(
                "/api/weather-map/stats/allotarra-field-1",
                params={"metrics": "eto"},
                headers=AUTH_HEADERS,
            )
            assert resp.status_code == 200

        mock_upsert.assert_not_awaited()


class TestPostStatsPersists:
    """POST /stats/{parcel_id} must call upsert_record exactly once."""

    @patch("app.tiles.upsert_record", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", return_value=SAMPLE_STATS)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_post_calls_upsert_record_once(
        self,
        mock_fetch: AsyncMock,
        mock_stats: MagicMock,
        mock_upsert: AsyncMock,
        client,
    ):
        """POST must call upsert_record exactly once per request."""
        mock_fetch.return_value = SAMPLE_PARCEL

        resp = client.post(
            "/api/weather-map/stats/allotarra-field-1",
            params={"metrics": "eto"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        mock_upsert.assert_awaited_once()

    @patch("app.tiles.upsert_record", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", return_value=SAMPLE_STATS)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_post_response_shape(
        self,
        mock_fetch: AsyncMock,
        mock_stats: MagicMock,
        mock_upsert: AsyncMock,
        client,
    ):
        """POST response includes status, id, and stats."""
        mock_fetch.return_value = SAMPLE_PARCEL

        resp = client.post(
            "/api/weather-map/stats/allotarra-field-1",
            params={"metrics": "eto"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "persisted"
        assert data["id"].startswith("urn:ngsi-ld:AgriParcelRecord:")
        assert "stats" in data
        assert "metrics" in data["stats"]

    @patch("app.tiles.upsert_record", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", return_value=SAMPLE_STATS)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_post_passes_correct_tenant_to_upsert(
        self,
        mock_fetch: AsyncMock,
        mock_stats: MagicMock,
        mock_upsert: AsyncMock,
        client,
    ):
        """upsert_record is called with the tenant from the X-Tenant-ID header."""
        mock_fetch.return_value = SAMPLE_PARCEL

        resp = client.post(
            "/api/weather-map/stats/allotarra-field-1",
            params={"metrics": "eto"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        call_args = mock_upsert.call_args[0]
        assert call_args[0] == TENANT

    @patch("app.tiles.upsert_record", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats")
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_post_parcel_not_found_returns_404_no_upsert(
        self,
        mock_fetch: AsyncMock,
        mock_stats: MagicMock,
        mock_upsert: AsyncMock,
        client,
    ):
        """POST returns 404 when parcel missing; upsert is never called."""
        mock_fetch.return_value = None

        resp = client.post(
            "/api/weather-map/stats/missing",
            params={"metrics": "eto"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 404
        mock_upsert.assert_not_awaited()
