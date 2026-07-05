"""Tests for zones and GDD endpoints — tenant auth wiring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from app.config import settings

AUTH_HEADERS = {"X-Tenant-ID": "asociacion-allotarra", "X-User-ID": "u1"}


def test_zones_router_exists():
    from app.zones import router

    assert router is not None


def test_zones_missing_auth_returns_401(client):
    resp = client.get("/api/weather-map/zones/p1")
    assert resp.status_code == 401


def test_gdd_missing_auth_returns_401(client):
    resp = client.get(
        "/api/weather-map/gdd?parcel_id=p1&season_start=2026-01-01",
    )
    assert resp.status_code == 401


def test_zones_tenant_from_header_not_query(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"id": "urn:ngsi-ld:AgriParcelZone:t:p1:z0"}]
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_orion = AsyncMock()
    mock_orion._get_headers = AsyncMock(return_value={"NGSILD-Tenant": "asociacion-allotarra"})
    mock_orion.close = AsyncMock()

    with patch("app.zones.OrionClient", return_value=mock_orion), patch(
        "app.zones.httpx.AsyncClient", return_value=mock_client
    ):
        resp = client.get(
            "/api/weather-map/zones/p1?tenant_id=evil",
            headers=AUTH_HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json()["parcel_id"] == "urn:ngsi-ld:AgriParcel:p1"
    assert len(resp.json()["zones"]) == 1

    call_kwargs = mock_client.get.call_args.kwargs
    assert call_kwargs["headers"]["NGSILD-Tenant"] == "asociacion-allotarra"


def test_gdd_rejects_foreign_parcel(client):
    with patch("app.zones.fetch_agri_parcel", new=AsyncMock(return_value=None)), patch.object(
        settings, "postgres_url", "postgres://test"
    ):
        resp = client.get(
            "/api/weather-map/gdd"
            "?parcel_id=urn:ngsi-ld:AgriParcel:other-tenant:secret"
            "&season_start=2026-01-01",
            headers=AUTH_HEADERS,
        )
    assert resp.status_code == 404


def test_gdd_uses_tenant_scoped_parcel(client):
    fake_parcel = {
        "id": "urn:ngsi-ld:AgriParcel:asociacion-allotarra:p1",
        "geometry": {"type": "Point", "coordinates": [-1.0, 42.0]},
    }
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.close = AsyncMock()

    with patch("app.zones.fetch_agri_parcel", new=AsyncMock(return_value=fake_parcel)), patch.object(
        settings, "postgres_url", "postgres://test"
    ), patch("app.zones.asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
        resp = client.get(
            "/api/weather-map/gdd?parcel_id=p1&season_start=2026-01-01",
            headers=AUTH_HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json()["source"] == "regional_knn"
    mock_conn.fetch.assert_awaited_once()
    assert mock_conn.fetch.await_args.args[1] == "urn:ngsi-ld:AgriParcel:p1"
