"""Mock Orion-LD tests for sources and stats endpoint.

Tests the HTTP client functions in ``app.sources`` (``fetch_agri_parcel``,
``fetch_entity_attr``, ``write_entity_attrs``) by mocking
``httpx.AsyncClient``, and the ``parcel_zonal_stats`` endpoint by mocking
the source functions at the ``app.tiles`` level.

All tests are independent of network / real Orion-LD.
"""

from __future__ import annotations

import copy
import json
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import httpx
import pytest

from app.sources import fetch_agri_parcel, fetch_entity_attr, write_entity_attrs


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
        """Request includes NGSILD-Tenant, Fiware-Service, and keyValues."""
        mock_async_client.get.return_value.json.return_value = SAMPLE_PARCEL_RESPONSE

        await fetch_agri_parcel("tenant_navarra", "test123")

        mock_async_client.get.assert_called_once()
        _call_url = mock_async_client.get.call_args[0][0]
        _call_kw = mock_async_client.get.call_args[1]
        assert "/ngsi-ld/v1/entities/urn:ngsi-ld:AgriParcel:test123" in _call_url
        assert _call_kw["headers"]["NGSILD-Tenant"] == "tenant_navarra"
        assert _call_kw["headers"]["Fiware-Service"] == "tenant_navarra"
        assert _call_kw["headers"]["Fiware-ServicePath"] == "/"
        assert _call_kw["params"] == {"options": "keyValues"}

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
        """GET the entity with keyValues and correct NGSI-LD headers."""
        mock_async_client.get.return_value.json.return_value = {"kc": 1.15}

        await fetch_entity_attr("t1", "urn:ngsi-ld:AgriCrop:wheat_mid", "kc")

        mock_async_client.get.assert_called_once()
        _call = mock_async_client.get.call_args
        url = _call[0][0]
        kwargs = _call[1]
        assert "urn:ngsi-ld:AgriCrop:wheat_mid" in url
        assert kwargs["headers"]["NGSILD-Tenant"] == "t1"
        assert kwargs["headers"]["Fiware-Service"] == "t1"
        assert kwargs["params"]["options"] == "keyValues"


# ═══════════════════════════════════════════════════════════════════
# Source-function tests — write_entity_attrs
# ═══════════════════════════════════════════════════════════════════


class TestWriteEntityAttrs:
    """``write_entity_attrs()`` — write NGSI-LD attributes via POST /attrs."""

    @pytest.mark.asyncio
    async def test_success_returns_true(self, mock_async_client):
        """Successful POST returns True."""
        result = await write_entity_attrs("tenant1", "parcel123", {"weatherStats": {"type": "Property", "value": {"mean": 22.5}}})
        assert result is True

    @pytest.mark.asyncio
    async def test_posts_to_attrs_endpoint_with_options_append(self, mock_async_client):
        """POST to ``/ngsi-ld/v1/entities/{id}/attrs?options=append``."""
        await write_entity_attrs("t1", "parcel123", {"weatherStats": {"type": "Property", "value": {}}})

        mock_async_client.post.assert_called_once()
        _call = mock_async_client.post.call_args
        url = _call[0][0]
        params = _call[1].get("params", {})
        assert "/ngsi-ld/v1/entities/parcel123/attrs" in url
        assert params == {"options": "append"}

    @pytest.mark.asyncio
    async def test_sends_correct_headers(self, mock_async_client):
        """NGSILD-Tenant, Fiware-Service, Content-Type are sent."""
        await write_entity_attrs("navarra", "parcel1", {"weatherStats": {"type": "Property", "value": {}}})

        mock_async_client.post.assert_called_once()
        headers = mock_async_client.post.call_args[1].get("headers", {})
        assert headers["NGSILD-Tenant"] == "navarra"
        assert headers["Fiware-Service"] == "navarra"
        assert headers["Fiware-ServicePath"] == "/"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_passes_attrs_as_json_body(self, mock_async_client):
        """The ``attrs`` dict is sent as the JSON body."""
        attrs = {
            "weatherStats": {
                "type": "Property",
                "value": {"mean": 22.5, "pixel_count": 100},
                "observedAt": "2026-06-01T00:00:00Z",
            },
        }

        await write_entity_attrs("t1", "parcel1", attrs)

        mock_async_client.post.assert_called_once()
        sent_json = mock_async_client.post.call_args[1].get("json", {})
        assert sent_json == attrs
        assert sent_json["weatherStats"]["value"]["mean"] == 22.5

    @pytest.mark.asyncio
    async def test_500_returns_false(self, mock_async_client):
        """Orion-LD 500 does not propagate — returns False."""
        from httpx import HTTPStatusError

        error_resp = MagicMock(spec=httpx.Response)
        error_resp.status_code = 500
        error_resp.raise_for_status.side_effect = HTTPStatusError(
            "Internal", request=MagicMock(), response=error_resp,
        )
        mock_async_client.post = AsyncMock(return_value=error_resp)

        result = await write_entity_attrs("t1", "parcel1", {"k": "v"})
        assert result is False


# ═══════════════════════════════════════════════════════════════════
# Endpoint tests — GET /api/weather-map/stats/{parcel_id}
# ═══════════════════════════════════════════════════════════════════


class TestParcelZonalStatsEndpoint:
    """``parcel_zonal_stats()`` endpoint — mocked at ``app.tiles`` level."""

    # ── geometry override (no Orion-LD) ──────────────────────────────

    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a,**kw: _fresh_stats_single())
    @patch("app.tiles.write_entity_attrs", new_callable=AsyncMock)
    def test_explicit_geometry_bypasses_orion_lookup(
        self, mock_write: AsyncMock, mock_compute: MagicMock, client,
    ):
        """When ``geometry`` is provided, no Orion-LD fetch is needed."""
        geom_json = json.dumps(SAMPLE_PARCEL_GEOM)
        resp = client.get(
            "/api/weather-map/stats/test123",
            params={
                "metrics": "temperature_avg",
                "geometry": geom_json,
            },
        )

        assert resp.status_code == 200
        data = resp.json()

        # Stats are returned with parcel_id
        assert data["parcel_id"] == "test123"
        assert "metrics" in data
        assert "temperature_avg" in data["metrics"]

        # Orion functions were NOT called (no fetch_agri_parcel, no fetch_entity_attr)
        # write_entity_attrs SHOULD be called regardless
        mock_write.assert_awaited_once()

    # ── parcel lookup from Orion-LD ─────────────────────────────────

    @patch("app.tiles.fetch_entity_attr", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a,**kw: _fresh_stats_two_metrics())
    @patch("app.tiles.write_entity_attrs", new_callable=AsyncMock)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_parcel_lookup_success(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_write: AsyncMock,
        mock_compute: MagicMock,
        mock_fetch_attr: AsyncMock,
        client,
    ):
        """Parcel looked up from Orion-LD, stats computed and written."""
        mock_fetch_parcel.return_value = {
            "id": "urn:ngsi-ld:AgriParcel:parcel123",
            "geometry": SAMPLE_PARCEL_GEOM,
            "name": "Campo Norte",
        }

        resp = client.get(
            "/api/weather-map/stats/parcel123",
            params={"metrics": "temperature_avg,water_balance"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["parcel_id"] == "parcel123"
        assert data["parcel_name"] == "Campo Norte"
        assert "temperature_avg" in data["metrics"]
        assert "water_balance" in data["metrics"]

        # fetch_agri_parcel was called with the right tenant + parcel
        mock_fetch_parcel.assert_awaited_once()
        _parcel_arg = mock_fetch_parcel.call_args[0]
        assert _parcel_arg[1] == "parcel123"

        # write_entity_attrs was called
        mock_write.assert_awaited_once()

    # ── parcel not found ────────────────────────────────────────────

    @patch("app.tiles.fetch_entity_attr", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a,**kw: _fresh_stats_two_metrics())
    @patch("app.tiles.write_entity_attrs", new_callable=AsyncMock)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_parcel_not_found_returns_404(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_write: AsyncMock,
        mock_compute: MagicMock,
        mock_fetch_attr: AsyncMock,
        client,
    ):
        """When Orion-LD has no parcel, endpoint returns 404."""
        mock_fetch_parcel.return_value = None

        resp = client.get(
            "/api/weather-map/stats/nonexistent",
            params={"metrics": "temperature_avg"},
        )

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    # ── invalid metrics ─────────────────────────────────────────────

    @patch("app.tiles.fetch_entity_attr", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats")
    @patch("app.tiles.write_entity_attrs", new_callable=AsyncMock)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_invalid_metrics_returns_400(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_write: AsyncMock,
        mock_compute: MagicMock,
        mock_fetch_attr: AsyncMock,
        client,
    ):
        """Unrecognised metric names produce a 400 error before any IO."""
        fake_geom = json.dumps(SAMPLE_PARCEL_GEOM)

        resp = client.get(
            "/api/weather-map/stats/test123",
            params={"metrics": "temperature_avg,invalid_metric", "geometry": fake_geom},
        )

        assert resp.status_code == 400
        assert "invalid_metric" in resp.json()["detail"].lower() or \
               "unknown" in resp.json()["detail"].lower()

        # No Orion / compute calls were made
        mock_fetch_parcel.assert_not_awaited()
        mock_compute.assert_not_called()
        mock_write.assert_not_awaited()

    # ── phenology params ────────────────────────────────────────────

    @patch("app.tiles.fetch_entity_attr", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a,**kw: _fresh_stats_single())
    @patch("app.tiles.write_entity_attrs", new_callable=AsyncMock)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_phenology_fetched_when_crop_and_stage_provided(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_write: AsyncMock,
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
        # fetch_entity_attr returns Kc first, then Ky
        mock_fetch_attr.side_effect = [1.15, 1.0]

        resp = client.get(
            "/api/weather-map/stats/parcel1",
            params={
                "metrics": "temperature_avg",
                "crop": "wheat",
                "stage": "mid-season",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["phenology"] == {
            "crop": "wheat",
            "stage": "mid-season",
            "kc": 1.15,
            "ky": 1.0,
        }

        # Verify the crop entity IDs used
        assert mock_fetch_attr.await_count == 2
        crop_eid = mock_fetch_attr.call_args_list[0][0][1]
        assert "wheat" in crop_eid
        assert "mid-season" in crop_eid

    @patch("app.tiles.fetch_entity_attr", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a,**kw: _fresh_stats_single())
    @patch("app.tiles.write_entity_attrs", new_callable=AsyncMock)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_phenology_with_ky_missing(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_write: AsyncMock,
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
        # Kc found (numeric), Ky not found (None)
        mock_fetch_attr.side_effect = [1.15, None]

        resp = client.get(
            "/api/weather-map/stats/parcel1",
            params={
                "metrics": "temperature_avg",
                "crop": "wheat",
                "stage": "mid-season",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["phenology"]["kc"] == 1.15
        assert data["phenology"]["ky"] is None

    @patch("app.tiles.fetch_entity_attr", return_value=None)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a,**kw: _fresh_stats_single())
    @patch("app.tiles.write_entity_attrs", new_callable=AsyncMock)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_phenology_kc_not_found_no_phenology_in_response(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_write: AsyncMock,
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
        # Kc not found (returns None or non-numeric)
        # fetch_entity_attr is patched with return_value=None via @patch

        resp = client.get(
            "/api/weather-map/stats/parcel1",
            params={
                "metrics": "temperature_avg",
                "crop": "wheat",
                "stage": "mid-season",
            },
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
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "phenology" not in data or data["phenology"] is None

    # ── write failure does not block ────────────────────────────────

    @patch("app.tiles.fetch_entity_attr", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a,**kw: _fresh_stats_single())
    @patch("app.tiles.write_entity_attrs", new_callable=AsyncMock)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_write_failure_still_returns_stats(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_write: AsyncMock,
        mock_compute: MagicMock,
        mock_fetch_attr: AsyncMock,
        client,
    ):
        """When writing to Orion-LD fails, the endpoint still returns stats."""
        mock_fetch_parcel.return_value = {
            "id": "urn:ngsi-ld:AgriParcel:parcel1",
            "geometry": SAMPLE_PARCEL_GEOM,
            "name": "Parcel",
        }
        # write_entity_attrs returns False (simulating Orion error)
        mock_write.return_value = False

        resp = client.get(
            "/api/weather-map/stats/parcel1",
            params={"metrics": "temperature_avg"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "metrics" in data
        assert "temperature_avg" in data["metrics"]

    # ── write_entity_attrs receives correct data ────────────────────

    @patch("app.tiles.fetch_entity_attr", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a,**kw: _fresh_stats_single())
    @patch("app.tiles.write_entity_attrs", new_callable=AsyncMock)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_write_entity_attrs_receives_correct_arguments(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_write: AsyncMock,
        mock_compute: MagicMock,
        mock_fetch_attr: AsyncMock,
        client,
    ):
        """Verifies write_entity_attrs is called with correct tenant, parcel and NGSI-LD attrs."""
        mock_fetch_parcel.return_value = {
            "id": "urn:ngsi-ld:AgriParcel:parcel_write_test",
            "geometry": SAMPLE_PARCEL_GEOM,
            "name": "Write Test",
        }

        resp = client.get(
            "/api/weather-map/stats/parcel_write_test",
            params={"metrics": "temperature_avg"},
        )

        assert resp.status_code == 200

        # Verify write_entity_attrs was called
        mock_write.assert_awaited_once()
        call_args = mock_write.call_args[0]
        tenant_arg = call_args[0]
        entity_arg = call_args[1]
        attrs_arg = call_args[2]

        assert tenant_arg == "default"  # default tenant_id from query param
        assert entity_arg == "parcel_write_test"
        assert "weatherStats" in attrs_arg
        assert attrs_arg["weatherStats"]["type"] == "Property"
        assert "value" in attrs_arg["weatherStats"]
        assert "observedAt" in attrs_arg["weatherStats"]

    # ── response shape includes parcel_id ───────────────────────────

    @patch("app.tiles.fetch_entity_attr", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", side_effect=lambda *a,**kw: _fresh_stats_two_metrics())
    @patch("app.tiles.write_entity_attrs", new_callable=AsyncMock)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_response_has_correct_shape(
        self,
        mock_fetch_parcel: AsyncMock,
        mock_write: AsyncMock,
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
