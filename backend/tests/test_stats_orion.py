"""Mock Orion-LD tests for sources and stats endpoint.

Tests the HTTP client functions in ``app.sources`` (``fetch_agri_parcel``,
``fetch_entity_attr``, ``upsert_record``) by mocking
``httpx.AsyncClient`` / ``OrionClient``, and the stats endpoints by mocking
the source functions at the ``app.tiles`` level.

All tests are independent of network / real Orion-LD.

Changes from original (hardening PR-A):
- Removed ``write_entity_attrs`` (deleted from sources — GET is now read-only).
- Added ``X-Tenant-ID`` header to every endpoint request (auth is now mandatory).
- ``TestWriteEntityAttrs`` replaced by ``TestUpsertRecord`` exercising the new
  ``upsert_record()`` helper.
- GET-write tests rewired to POST ``/stats`` (the persist path).
"""

from __future__ import annotations

import copy
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.sources import fetch_agri_parcel, fetch_entity_attr, upsert_record


# ═══════════════════════════════════════════════════════════════════
# Shared test data
# ═══════════════════════════════════════════════════════════════════

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

SAMPLE_PARCEL_RESPONSE = {
    "id": "urn:ngsi-ld:AgriParcel:test123",
    "location": SAMPLE_PARCEL_GEOM,
    "name": "Parcela de prueba",
    "description": "A test parcel in Navarra",
}

AUTH_HEADERS = {"X-Tenant-ID": "test-tenant", "X-User-ID": "test-user"}


def _fresh_stats_two_metrics() -> dict:
    """Factory: returns fresh copy of two-metric stats (temperature_avg + water_balance).

    The endpoint mutates the stats dict in-place (adding parcel_id, phenology, etc.),
    so every test needs its own independent copy.
    """
    return {
        "parcel_geojson": copy.deepcopy(SAMPLE_PARCEL_GEOM),
        "date": "2026-06-01",
        "metrics": {
            "temperature_avg": {
                "mean": 22.5,
                "min": 18.0,
                "max": 28.0,
                "std": 2.5,
                "p25": 20.0,
                "p50": 22.5,
                "p75": 25.0,
                "pixel_count": 100,
                "histogram": [5, 10, 15, 20, 15, 10, 10, 5, 5, 5],
                "heat_stress_pct": 0.0,
                "frost_pct": 0.0,
            },
            "water_balance": {
                "mean": 5.0,
                "min": -10.0,
                "max": 20.0,
                "std": 5.0,
                "p25": 0.0,
                "p50": 5.0,
                "p75": 10.0,
                "pixel_count": 100,
                "histogram": [5, 10, 15, 20, 15, 10, 10, 5, 5, 5],
                "deficit_area_pct": 10.0,
                "severe_deficit_pct": 2.0,
            },
        },
    }


def _fresh_stats_single() -> dict:
    """Factory: returns fresh copy of single-metric stats (temperature_avg only)."""
    return {
        "parcel_geojson": SAMPLE_PARCEL_GEOM,
        "date": "2026-06-01",
        "metrics": {
            "temperature_avg": {
                "mean": 22.5,
                "min": 18.0,
                "max": 28.0,
                "std": 2.5,
                "p25": 20.0,
                "p50": 22.5,
                "p75": 25.0,
                "pixel_count": 100,
                "histogram": [5, 10, 15, 20, 15, 10, 10, 5, 5, 5],
                "heat_stress_pct": 0.0,
                "frost_pct": 0.0,
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Fixture: httpx.AsyncClient mock for source-function tests
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_async_client():
    """Patch ``app.sources.httpx.AsyncClient`` and return the mock instance.

    The mock behaves as a valid async context manager so that::

        async with httpx.AsyncClient(...) as client:
            resp = await client.get(...)

    works correctly.  The default ``get`` / ``post`` return a generic
    ``MagicMock`` response.  Individual tests replace the return value
    or JSON payload as needed.
    """
    with patch("app.sources.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.__aexit__.return_value = None

        default_resp = MagicMock(spec=httpx.Response)
        default_resp.raise_for_status = MagicMock()
        default_resp.json = MagicMock(return_value={})
        default_resp.status_code = 200
        mock_instance.get = AsyncMock(return_value=default_resp)
        mock_instance.post = AsyncMock(return_value=default_resp)

        yield mock_instance


# ═══════════════════════════════════════════════════════════════════
# Source-function tests — fetch_agri_parcel
# ═══════════════════════════════════════════════════════════════════


class TestFetchAgriParcel:
    """``fetch_agri_parcel()`` — resolve AgriParcel geometry from Orion-LD."""

    @pytest.mark.asyncio
    async def test_success(self, mock_async_client):
        """Returns parcel data when Orion responds with a valid entity."""
        mock_async_client.get.return_value.json.return_value = SAMPLE_PARCEL_RESPONSE

        result = await fetch_agri_parcel("tenant_navarra", "test123")

        assert result is not None
        assert result["id"] == "urn:ngsi-ld:AgriParcel:test123"
        assert result["geometry"] == SAMPLE_PARCEL_GEOM
        assert result["name"] == "Parcela de prueba"

    @pytest.mark.asyncio
    async def test_sends_ngsi_ld_headers(self, mock_async_client):
        """fetch_agri_parcel uses OrionClient (SDK) for the Orion-LD call.

        The SDK handles header injection internally, so we verify the right
        entity ID is passed to OrionClient.get_entity rather than inspecting
        raw httpx call kwargs.
        """
        mock_async_client.get.return_value.json.return_value = SAMPLE_PARCEL_RESPONSE

        with patch("app.sources.OrionClient") as mock_cls:
            mock_orion = AsyncMock()
            mock_cls.return_value = mock_orion
            mock_orion.get_entity = AsyncMock(return_value=SAMPLE_PARCEL_RESPONSE)
            mock_orion.close = AsyncMock()

            result = await fetch_agri_parcel("tenant_navarra", "test123")

        mock_cls.assert_called_once_with("tenant_navarra")
        mock_orion.get_entity.assert_awaited_once_with(
            "urn:ngsi-ld:AgriParcel:test123"
        )
        assert result is not None
        assert result["geometry"] == SAMPLE_PARCEL_GEOM

    @pytest.mark.asyncio
    async def test_normalizes_entity_id(self, mock_async_client):
        """Adds ``urn:ngsi-ld:AgriParcel:`` prefix when missing."""
        mock_async_client.get.return_value.json.return_value = SAMPLE_PARCEL_RESPONSE

        await fetch_agri_parcel("default", "urn:ngsi-ld:AgriParcel:already_prefixed")
        called_url = mock_async_client.get.call_args[0][0]
        assert "already_prefixed" in called_url
        # Count occurrences of the prefix — should be exactly one
        assert called_url.count("urn:ngsi-ld:AgriParcel:") == 1

    @pytest.mark.asyncio
    async def test_404_returns_none(self, mock_async_client):
        """Orion-LD 404 is swallowed, function returns None."""
        from httpx import HTTPStatusError

        error_resp = MagicMock(spec=httpx.Response)
        error_resp.status_code = 404
        error_resp.raise_for_status.side_effect = HTTPStatusError(
            "Not Found", request=MagicMock(), response=error_resp,
        )
        error_resp.json = MagicMock(return_value={"error": "Not Found"})
        mock_async_client.get = AsyncMock(return_value=error_resp)

        result = await fetch_agri_parcel("tenant_navarra", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_500_returns_none(self, mock_async_client):
        """Orion-LD 500 is swallowed, function returns None."""
        from httpx import HTTPStatusError

        error_resp = MagicMock(spec=httpx.Response)
        error_resp.status_code = 500
        error_resp.raise_for_status.side_effect = HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=error_resp,
        )
        error_resp.json = MagicMock(return_value={"error": "Internal"})
        mock_async_client.get = AsyncMock(return_value=error_resp)

        result = await fetch_agri_parcel("tenant_navarra", "broken")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_location_returns_none(self, mock_async_client):
        """Entity without ``location`` / ``geometry`` returns None."""
        mock_async_client.get.return_value.json.return_value = {
            "id": "urn:ngsi-ld:AgriParcel:geomless",
            "name": "No Geometry",
        }

        result = await fetch_agri_parcel("default", "geomless")
        assert result is None

    @pytest.mark.asyncio
    async def test_network_error_returns_none(self, mock_async_client):
        """Connection error (e.g. timeout) returns None."""
        mock_async_client.get.side_effect = httpx.TimeoutException(
            "Connection timed out"
        )

        result = await fetch_agri_parcel("default", "timeout")
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# Source-function tests — fetch_entity_attr
# ═══════════════════════════════════════════════════════════════════


class TestFetchEntityAttr:
    """``fetch_entity_attr()`` — read a single NGSI-LD attribute."""

    @pytest.mark.asyncio
    async def test_success(self, mock_async_client):
        """Returns attribute value when entity contains the requested attr."""
        mock_async_client.get.return_value.json.return_value = {
            "id": "urn:ngsi-ld:AgriCrop:wheat_mid-season",
            "kc": 1.15,
            "ky": 1.0,
            "name": "Wheat mid-season",
        }

        result = await fetch_entity_attr("tenant1", "urn:ngsi-ld:AgriCrop:wheat_mid-season", "kc")

        assert result == 1.15

    @pytest.mark.asyncio
    async def test_missing_attr_returns_none(self, mock_async_client):
        """Entity exists but lacks requested attribute → None."""
        mock_async_client.get.return_value.json.return_value = {
            "id": "urn:ngsi-ld:AgriCrop:wheat_mid-season",
            "name": "Wheat mid-season",
        }

        result = await fetch_entity_attr("tenant1", "crop123", "kc")
        assert result is None

    @pytest.mark.asyncio
    async def test_404_returns_none(self, mock_async_client):
        """Entity not found (404) returns None."""
        from httpx import HTTPStatusError

        error_resp = MagicMock(spec=httpx.Response)
        error_resp.status_code = 404
        error_resp.raise_for_status.side_effect = HTTPStatusError(
            "Not Found", request=MagicMock(), response=error_resp,
        )
        error_resp.json = MagicMock(return_value={"error": "Not Found"})
        mock_async_client.get = AsyncMock(return_value=error_resp)

        result = await fetch_entity_attr("tenant1", "nonexistent", "kc")
        assert result is None

    @pytest.mark.asyncio
    async def test_uses_correct_url_and_headers(self, mock_async_client):
        """fetch_entity_attr uses OrionClient (SDK) which handles headers.

        Verify the right tenant and entity ID are passed to the SDK.
        """
        with patch("app.sources.OrionClient") as mock_cls:
            mock_orion = AsyncMock()
            mock_cls.return_value = mock_orion
            mock_orion.get_entity = AsyncMock(return_value={"kc": 1.15})
            mock_orion.close = AsyncMock()

            result = await fetch_entity_attr("t1", "urn:ngsi-ld:AgriCrop:wheat_mid", "kc")

        mock_cls.assert_called_once_with("t1")
        mock_orion.get_entity.assert_awaited_once_with("urn:ngsi-ld:AgriCrop:wheat_mid")
        assert result == 1.15


# ═══════════════════════════════════════════════════════════════════
# Source-function tests — upsert_record
# (replaces the removed write_entity_attrs tests)
# ═══════════════════════════════════════════════════════════════════


class TestUpsertRecord:
    """``upsert_record()`` — create an AgriParcelRecord entity via SDK."""

    @pytest.mark.asyncio
    async def test_calls_orion_create_entity(self):
        """Delegates to OrionClient.create_entity with the entity dict."""
        entity = {
            "id": "urn:ngsi-ld:AgriParcelRecord:weather-test-parcel-20260601T000000Z",
            "type": "AgriParcelRecord",
        }
        with patch("app.sources.OrionClient") as mock_cls:
            mock_orion = AsyncMock()
            mock_cls.return_value = mock_orion
            mock_orion.create_entity = AsyncMock()
            mock_orion.close = AsyncMock()

            await upsert_record("test-tenant", entity)

            mock_cls.assert_called_once_with("test-tenant")
            mock_orion.create_entity.assert_awaited_once_with(entity)

    @pytest.mark.asyncio
    async def test_tolerates_409_duplicate(self):
        """A 409-style exception (duplicate entity) is swallowed — no raise."""
        entity = {"id": "urn:ngsi-ld:AgriParcelRecord:dup", "type": "AgriParcelRecord"}
        with patch("app.sources.OrionClient") as mock_cls:
            mock_orion = AsyncMock()
            mock_cls.return_value = mock_orion
            mock_orion.create_entity = AsyncMock(
                side_effect=Exception("Entity already exists (409)")
            )
            mock_orion.close = AsyncMock()

            # Must not raise
            await upsert_record("test-tenant", entity)

    @pytest.mark.asyncio
    async def test_logs_non_409_failure(self):
        """Non-duplicate SDK failures are logged but not raised."""
        entity = {"id": "urn:ngsi-ld:AgriParcelRecord:fail", "type": "AgriParcelRecord"}
        with patch("app.sources.OrionClient") as mock_cls:
            mock_orion = AsyncMock()
            mock_cls.return_value = mock_orion
            mock_orion.create_entity = AsyncMock(
                side_effect=Exception("Connection refused")
            )
            mock_orion.close = AsyncMock()

            # Must not raise
            await upsert_record("test-tenant", entity)

    @pytest.mark.asyncio
    async def test_always_closes_orion_client(self):
        """OrionClient.close() is always called (via finally)."""
        entity = {"id": "urn:ngsi-ld:AgriParcelRecord:close-test", "type": "AgriParcelRecord"}
        with patch("app.sources.OrionClient") as mock_cls:
            mock_orion = AsyncMock()
            mock_cls.return_value = mock_orion
            mock_orion.create_entity = AsyncMock(side_effect=Exception("boom"))
            mock_orion.close = AsyncMock()

            await upsert_record("test-tenant", entity)

            mock_orion.close.assert_awaited_once()


# ═══════════════════════════════════════════════════════════════════
# Endpoint tests — GET /api/weather-map/stats/{parcel_id}
# GET is READ-ONLY — it never calls upsert_record.
# ═══════════════════════════════════════════════════════════════════


class TestParcelZonalStatsEndpoint:
    """``parcel_zonal_stats()`` GET endpoint — mocked at ``app.tiles`` level."""

    # ── auth required ───────────────────────────────────────────────

    def test_missing_tenant_header_returns_401(self, client):
        """GET without X-Tenant-ID returns 401."""
        resp = client.get(
            "/api/weather-map/stats/test123",
            params={"metrics": "temperature_avg"},
        )
        assert resp.status_code == 401

    # ── geometry override (no Orion-LD) ──────────────────────────────

    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a, **kw: _fresh_stats_single())
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_explicit_geometry_bypasses_orion_lookup(
        self, mock_fetch_parcel: AsyncMock, mock_compute: MagicMock, client,
    ):
        """When ``geometry`` is provided, no Orion-LD fetch is needed."""
        geom_json = json.dumps(SAMPLE_PARCEL_GEOM)
        resp = client.get(
            "/api/weather-map/stats/test123",
            params={
                "metrics": "temperature_avg",
                "geometry": geom_json,
            },
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()

        assert data["parcel_id"] == "test123"
        assert "metrics" in data
        assert "temperature_avg" in data["metrics"]

        # GET is read-only — fetch_agri_parcel NOT called for geometry override
        mock_fetch_parcel.assert_not_awaited()

    # ── parcel lookup from Orion-LD ─────────────────────────────────

    @patch("app.tiles.fetch_entity_attr", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a, **kw: _fresh_stats_two_metrics())
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_parcel_lookup_success(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_compute: MagicMock,
        mock_fetch_attr: AsyncMock,
        client,
    ):
        """Parcel looked up from Orion-LD, stats computed and returned."""
        mock_fetch_parcel.return_value = {
            "id": "urn:ngsi-ld:AgriParcel:parcel123",
            "geometry": SAMPLE_PARCEL_GEOM,
            "name": "Campo Norte",
        }

        resp = client.get(
            "/api/weather-map/stats/parcel123",
            params={"metrics": "temperature_avg,water_balance"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["parcel_id"] == "parcel123"
        assert data["parcel_name"] == "Campo Norte"
        assert "temperature_avg" in data["metrics"]
        assert "water_balance" in data["metrics"]

        mock_fetch_parcel.assert_awaited_once()
        _parcel_arg = mock_fetch_parcel.call_args[0]
        assert _parcel_arg[1] == "parcel123"

    # ── parcel not found ────────────────────────────────────────────

    @patch("app.tiles.fetch_entity_attr", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a, **kw: _fresh_stats_two_metrics())
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_parcel_not_found_returns_404(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_compute: MagicMock,
        mock_fetch_attr: AsyncMock,
        client,
    ):
        """When Orion-LD has no parcel, endpoint returns 404."""
        mock_fetch_parcel.return_value = None

        resp = client.get(
            "/api/weather-map/stats/nonexistent",
            params={"metrics": "temperature_avg"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    # ── invalid metrics ─────────────────────────────────────────────

    @patch("app.tiles.fetch_entity_attr", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats")
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_invalid_metrics_returns_400(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_compute: MagicMock,
        mock_fetch_attr: AsyncMock,
        client,
    ):
        """Unrecognised metric names produce a 400 error before any IO."""
        fake_geom = json.dumps(SAMPLE_PARCEL_GEOM)

        resp = client.get(
            "/api/weather-map/stats/test123",
            params={"metrics": "temperature_avg,invalid_metric", "geometry": fake_geom},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 400
        assert "invalid_metric" in resp.json()["detail"].lower() or \
               "unknown" in resp.json()["detail"].lower()

        mock_fetch_parcel.assert_not_awaited()
        mock_compute.assert_not_called()

    # ── phenology params ────────────────────────────────────────────

    @patch("app.tiles.fetch_entity_attr", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a, **kw: _fresh_stats_single())
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_phenology_fetched_when_crop_and_stage_provided(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_compute: MagicMock,
        mock_fetch_attr: AsyncMock,
        client,
    ):
        """crop+stage triggers fetch_entity_attr for Kc and Ky."""
        mock_fetch_parcel.return_value = {
            "id": "urn:ngsi-ld:AgriParcel:parcel1",
            "geometry": SAMPLE_PARCEL_GEOM,
            "name": "Parcel",
        }
        mock_fetch_attr.side_effect = [1.15, 1.0]

        resp = client.get(
            "/api/weather-map/stats/parcel1",
            params={
                "metrics": "temperature_avg",
                "crop": "wheat",
                "stage": "mid-season",
            },
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["phenology"] == {
            "crop": "wheat",
            "stage": "mid-season",
            "kc": 1.15,
            "ky": 1.0,
        }

        assert mock_fetch_attr.await_count == 2
        crop_eid = mock_fetch_attr.call_args_list[0][0][1]
        assert "wheat" in crop_eid
        assert "mid-season" in crop_eid

    @patch("app.tiles.fetch_entity_attr", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a, **kw: _fresh_stats_single())
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_phenology_with_ky_missing(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_compute: MagicMock,
        mock_fetch_attr: AsyncMock,
        client,
    ):
        """Kc found but Ky missing — phenology has Kc and null Ky."""
        mock_fetch_parcel.return_value = {
            "id": "urn:ngsi-ld:AgriParcel:parcel1",
            "geometry": SAMPLE_PARCEL_GEOM,
            "name": "Parcel",
        }
        mock_fetch_attr.side_effect = [1.15, None]

        resp = client.get(
            "/api/weather-map/stats/parcel1",
            params={
                "metrics": "temperature_avg",
                "crop": "wheat",
                "stage": "mid-season",
            },
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["phenology"]["kc"] == 1.15
        assert data["phenology"]["ky"] is None

    @patch("app.tiles.fetch_entity_attr", return_value=None)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a, **kw: _fresh_stats_single())
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_phenology_kc_not_found_no_phenology_in_response(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_compute: MagicMock,
        mock_fetch_attr: AsyncMock,
        client,
    ):
        """When Kc is not found, no ``phenology`` key in response."""
        mock_fetch_parcel.return_value = {
            "id": "urn:ngsi-ld:AgriParcel:parcel1",
            "geometry": SAMPLE_PARCEL_GEOM,
            "name": "Parcel",
        }

        resp = client.get(
            "/api/weather-map/stats/parcel1",
            params={
                "metrics": "temperature_avg",
                "crop": "wheat",
                "stage": "mid-season",
            },
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "phenology" not in data or data["phenology"] is None

        resp = client.get(
            "/api/weather-map/stats/parcel1",
            params={
                "metrics": "temperature_avg",
                "crop": "wheat",
                "stage": "mid-season",
            },
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "phenology" not in data or data["phenology"] is None

    # ── GET is read-only: upsert_record must NOT be called ──────────

    @patch("app.tiles.upsert_record", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a, **kw: _fresh_stats_single())
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_get_does_not_call_upsert_record(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_compute: MagicMock,
        mock_upsert: AsyncMock,
        client,
    ):
        """GET /stats never calls upsert_record — it is read-only."""
        mock_fetch_parcel.return_value = {
            "id": "urn:ngsi-ld:AgriParcel:readonly",
            "geometry": SAMPLE_PARCEL_GEOM,
            "name": "ReadOnly Parcel",
        }

        resp = client.get(
            "/api/weather-map/stats/readonly",
            params={"metrics": "temperature_avg"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        mock_upsert.assert_not_awaited()

    # ── response shape includes parcel_id ───────────────────────────

    @patch("app.tiles.fetch_entity_attr", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a, **kw: _fresh_stats_two_metrics())
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_response_has_correct_shape(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_compute: MagicMock,
        mock_fetch_attr: AsyncMock,
        client,
    ):
        """Response includes parcel_id, metrics with expected fields."""
        mock_fetch_parcel.return_value = {
            "id": "urn:ngsi-ld:AgriParcel:shape_test",
            "geometry": SAMPLE_PARCEL_GEOM,
            "name": "Shape Test",
        }

        resp = client.get(
            "/api/weather-map/stats/shape_test",
            params={"metrics": "temperature_avg,water_balance"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "parcel_id" in data
        assert "parcel_geojson" in data
        assert "date" in data
        assert "metrics" in data

        for metric_name in ("temperature_avg", "water_balance"):
            assert metric_name in data["metrics"]
            m = data["metrics"][metric_name]
            assert "mean" in m
            assert "min" in m
            assert "max" in m
            assert "std" in m
            assert "pixel_count" in m


# ═══════════════════════════════════════════════════════════════════
# Endpoint tests — POST /api/weather-map/stats/{parcel_id}
# POST is the persist path — calls upsert_record.
# ═══════════════════════════════════════════════════════════════════


class TestPersistZonalStatsEndpoint:
    """``persist_zonal_stats()`` POST endpoint — mocked at ``app.tiles`` level."""

    # ── auth required ───────────────────────────────────────────────

    def test_missing_tenant_header_returns_401(self, client):
        """POST without X-Tenant-ID returns 401."""
        resp = client.post(
            "/api/weather-map/stats/test123",
            params={"metrics": "temperature_avg"},
        )
        assert resp.status_code == 401

    # ── parcel not found ────────────────────────────────────────────

    @patch("app.tiles.upsert_record", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats")
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_parcel_not_found_returns_404(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_compute: MagicMock,
        mock_upsert: AsyncMock,
        client,
    ):
        """When parcel is not in Orion-LD, POST returns 404."""
        mock_fetch_parcel.return_value = None

        resp = client.post(
            "/api/weather-map/stats/missing_parcel",
            params={"metrics": "temperature_avg"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 404
        mock_upsert.assert_not_awaited()

    # ── success: persists and returns status ────────────────────────

    @patch("app.tiles.upsert_record", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a, **kw: _fresh_stats_single())
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_persist_calls_upsert_record_once(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_compute: MagicMock,
        mock_upsert: AsyncMock,
        client,
    ):
        """POST calls upsert_record exactly once and returns persisted status."""
        mock_fetch_parcel.return_value = {
            "id": "urn:ngsi-ld:AgriParcel:parcel_persist",
            "geometry": SAMPLE_PARCEL_GEOM,
            "name": "Persist Test",
        }

        resp = client.post(
            "/api/weather-map/stats/parcel_persist",
            params={"metrics": "temperature_avg"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "persisted"
        assert "id" in data
        assert "stats" in data
        assert "metrics" in data["stats"]

        mock_upsert.assert_awaited_once()

    # ── upsert receives AgriParcelRecord entity ──────────────────────

    @patch("app.tiles.upsert_record", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a, **kw: _fresh_stats_single())
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_upsert_receives_agri_parcel_record(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_compute: MagicMock,
        mock_upsert: AsyncMock,
        client,
    ):
        """upsert_record is called with a valid AgriParcelRecord entity."""
        mock_fetch_parcel.return_value = {
            "id": "urn:ngsi-ld:AgriParcel:record_check",
            "geometry": SAMPLE_PARCEL_GEOM,
            "name": "Record Check",
        }

        resp = client.post(
            "/api/weather-map/stats/record_check",
            params={"metrics": "temperature_avg"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        mock_upsert.assert_awaited_once()

        call_args = mock_upsert.call_args[0]
        tenant_arg = call_args[0]
        entity_arg = call_args[1]

        assert tenant_arg == "test-tenant"
        assert entity_arg["type"] == "AgriParcelRecord"
        assert entity_arg["id"].startswith("urn:ngsi-ld:AgriParcelRecord:weather-")
        assert "airTemperatureAvg" in entity_arg  # temperature_avg → airTemperatureAvg
        assert entity_arg["airTemperatureAvg"]["type"] == "Property"
        assert entity_arg["airTemperatureAvg"]["value"] == 22.5  # mean from _fresh_stats_single

    # ── default metric is eto ────────────────────────────────────────

    @patch("app.tiles.upsert_record", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats")
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_default_metric_is_eto(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_compute: MagicMock,
        mock_upsert: AsyncMock,
        client,
    ):
        """When no metrics param is supplied, defaults to 'eto'."""
        mock_fetch_parcel.return_value = {
            "id": "urn:ngsi-ld:AgriParcel:eto_test",
            "geometry": SAMPLE_PARCEL_GEOM,
            "name": "ETo Test",
        }
        mock_compute.return_value = {
            "parcel_geojson": SAMPLE_PARCEL_GEOM,
            "date": "2026-06-01",
            "metrics": {"eto": {"mean": 4.2, "min": 3.0, "max": 5.5, "std": 0.5,
                                 "p25": 3.8, "p50": 4.2, "p75": 4.8, "pixel_count": 50,
                                 "histogram": [1] * 10}},
        }

        resp = client.post(
            "/api/weather-map/stats/eto_test",
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        mock_compute.assert_called_once()
        metric_list_arg = mock_compute.call_args[0][2]
        assert metric_list_arg == ["eto"]
