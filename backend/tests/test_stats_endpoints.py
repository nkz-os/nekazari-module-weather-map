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

TENANT = "demo-coop"
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
    "id": "urn:ngsi-ld:AgriParcel:demo-field-1",
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
            "/api/weather-map/stats/demo-field-1",
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
                "/api/weather-map/stats/demo-field-1",
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
            "/api/weather-map/stats/demo-field-1",
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
            "/api/weather-map/stats/demo-field-1",
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
            "/api/weather-map/stats/demo-field-1",
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


NO_DATA = {"error": "No data for parcel at this metric/date"}


def _uncovered_stats(metrics):
    return {
        "parcel_geojson": SAMPLE_GEOM,
        "date": "2026-06-13",
        "metrics": {m: dict(NO_DATA) for m in metrics},
    }


WEATHER = {
    "t_avg": 11.5, "t_min": 2.0, "solar_rad_w_m2": 180.0,
    "observed_at": "2026-07-01T09:00:00Z", "valid_from": "2026-07-02T00:00:00Z",
}


def _get(client, metrics, **extra):
    return client.get(
        "/api/weather-map/stats/demo-field-1",
        params={"metrics": metrics, **extra},
        headers=AUTH_HEADERS,
    )


class TestStatsLabelsAndScalarFallback:
    @patch("app.tiles.fetch_parcel_scalars", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", return_value=SAMPLE_STATS)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_stats_adds_kind_and_valid_date_keeping_date(
        self, mock_fetch, mock_stats, mock_weather, client
    ):
        mock_fetch.return_value = SAMPLE_PARCEL
        data = _get(client, "eto").json()
        assert data["date"] == "2026-06-13"
        assert data["valid_date"] == "2026-06-13"
        assert data["kind"] == "forecast_snapshot"
        assert data["metrics"]["eto"]["mean"] == 4.2

    @patch("app.stats.get_window_meta")
    @patch("app.tiles.fetch_parcel_scalars", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats")
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_water_balance_payload_labels_window(
        self, mock_fetch, mock_stats, mock_weather, mock_meta, client
    ):
        mock_fetch.return_value = SAMPLE_PARCEL
        mock_stats.return_value = {
            "parcel_geojson": SAMPLE_GEOM, "date": "2026-06-13",
            "metrics": {"water_balance": {"mean": -7.0, "deficit_area_pct": 100.0}},
        }
        mock_meta.return_value = {
            "date": "2026-06-13", "window_start": "2026-06-08", "window_end": "2026-06-12",
        }
        wb = _get(client, "water_balance").json()["metrics"]["water_balance"]
        assert wb["kind"] == "closed_days"
        assert wb["window_start"] == "2026-06-08"
        assert wb["window_end"] == "2026-06-12"
        assert wb["days"] == 5
        assert wb["et0_source"] == "weather-api daily (provider)"
        assert wb["mean"] == -7.0 and wb["deficit_area_pct"] == 100.0

    @patch("app.stats.get_window_meta")
    @patch("app.tiles.fetch_parcel_scalars", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats")
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_water_balance_window_not_invented_when_unrecorded(
        self, mock_fetch, mock_stats, mock_weather, mock_meta, client
    ):
        mock_fetch.return_value = SAMPLE_PARCEL
        mock_stats.return_value = {
            "parcel_geojson": SAMPLE_GEOM, "date": "2026-06-13",
            "metrics": {"water_balance": {"mean": -7.0}},
        }
        # Metadata of another raster date must not be attributed to this one.
        mock_meta.return_value = {
            "date": "2026-06-01", "window_start": "2026-05-27", "window_end": "2026-05-31",
        }
        wb = _get(client, "water_balance").json()["metrics"]["water_balance"]
        assert wb["kind"] == "closed_days" and wb["days"] == 5
        assert wb["window_start"] is None and wb["window_end"] is None

    @patch("app.tiles.fetch_parcel_scalars", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", return_value=SAMPLE_STATS)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_eto_payload_labels_method(
        self, mock_fetch, mock_stats, mock_weather, client
    ):
        mock_fetch.return_value = SAMPLE_PARCEL
        data = _get(client, "eto").json()
        eto = data["metrics"]["eto"]
        assert eto["kind"] == "forecast_running_day"
        assert eto["method"] == "FAO-56 PM daily"
        assert eto["mean"] == 4.2
        assert data["kind"] == "forecast_snapshot"

    @patch("app.tiles.fetch_parcel_scalars", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats")
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_uncovered_parcel_gets_scalar_fallback_marked(
        self, mock_fetch, mock_stats, mock_weather, client
    ):
        mock_fetch.return_value = SAMPLE_PARCEL
        names = ["temperature_avg", "temperature_min", "solar_radiation"]
        mock_stats.return_value = _uncovered_stats(names)
        mock_weather.return_value = dict(WEATHER)
        m = _get(client, ",".join(names)).json()["metrics"]
        expect = {"temperature_avg": 11.5, "temperature_min": 2.0, "solar_radiation": 180.0}
        for name, value in expect.items():
            stamp = "2026-07-02" if name == "temperature_min" else "2026-07-01"
            assert m[name]["fallback"] == "parcel_scalar"
            assert m[name]["std"] is None
            assert (m[name]["mean"], m[name]["min"], m[name]["max"]) == (value,) * 3
            assert m[name]["valid_date"] == stamp
        assert m["temperature_min"]["valid_date"] == "2026-07-02"
        assert m["temperature_min"]["observed_at"] == "2026-07-02T00:00:00Z"

    @patch("app.tiles.fetch_parcel_scalars", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats")
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_fallback_frost_risk_uses_same_function_as_raster(
        self, mock_fetch, mock_stats, mock_weather, client
    ):
        import numpy as np
        from app.downscaler import compute_frost_risk

        mock_fetch.return_value = SAMPLE_PARCEL
        mock_stats.return_value = _uncovered_stats(["frost_risk"])
        mock_weather.return_value = dict(WEATHER)
        m = _get(client, "frost_risk").json()["metrics"]["frost_risk"]
        expected = float(compute_frost_risk(np.array([2.0]), np.array([0.0]))[0])
        assert m["mean"] == pytest.approx(expected)
        assert m["fallback"] == "parcel_scalar"
        assert m["std"] is None

    @patch("app.tiles.fetch_parcel_scalars", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats")
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_fallback_without_broker_data_keeps_error(
        self, mock_fetch, mock_stats, mock_weather, client
    ):
        mock_fetch.return_value = SAMPLE_PARCEL
        names = ["temperature_min", "frost_risk", "solar_radiation"]
        mock_stats.return_value = _uncovered_stats(names)
        mock_weather.return_value = None
        m = _get(client, ",".join(names)).json()["metrics"]
        for name in names:
            assert m[name] == NO_DATA
        # Observation present but the needed key absent: still no invented value.
        mock_weather.return_value = {"t_avg": 10.0}
        m = _get(client, ",".join(names)).json()["metrics"]
        for name in names:
            assert m[name] == NO_DATA

    @patch("app.tiles.fetch_parcel_scalars", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", return_value=SAMPLE_STATS)
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_covered_parcel_never_uses_fallback(
        self, mock_fetch, mock_stats, mock_weather, client
    ):
        mock_fetch.return_value = SAMPLE_PARCEL
        data = _get(client, "eto").json()
        assert "fallback" not in data["metrics"]["eto"]
        mock_weather.assert_not_awaited()

    @patch("app.tiles.fetch_parcel_scalars", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats")
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_explicit_date_never_uses_fallback(
        self, mock_fetch, mock_stats, mock_weather, client
    ):
        mock_fetch.return_value = SAMPLE_PARCEL
        mock_stats.return_value = _uncovered_stats(["temperature_min"])
        mock_weather.return_value = dict(WEATHER)
        m = _get(client, "temperature_min", date="2026-06-13").json()["metrics"]
        assert m["temperature_min"] == NO_DATA
        mock_weather.assert_not_awaited()

    @patch("app.tiles.fetch_parcel_scalars", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats")
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_other_metrics_keep_their_error(
        self, mock_fetch, mock_stats, mock_weather, client
    ):
        mock_fetch.return_value = SAMPLE_PARCEL
        mock_stats.return_value = _uncovered_stats(["eto"])
        mock_weather.return_value = dict(WEATHER)
        assert _get(client, "eto").json()["metrics"]["eto"] == NO_DATA


NO_RASTER = {"error": "No COG data available", "metrics": {}}


class TestNoRasterAndSources:
    @patch("app.tiles.fetch_parcel_scalars", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", return_value=dict(NO_RASTER))
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_tenant_without_raster_gets_scalar_fallback(
        self, mock_fetch, mock_stats, mock_weather, client
    ):
        mock_fetch.return_value = SAMPLE_PARCEL
        mock_weather.return_value = dict(WEATHER)
        data = _get(client, "temperature_min,eto").json()
        assert "error" not in data
        assert data["valid_date"] is None
        assert data["kind"] == "parcel_scalar"
        assert data["metrics"]["temperature_min"]["fallback"] == "parcel_scalar"
        assert data["metrics"]["eto"] == NO_DATA

    @patch("app.tiles.fetch_parcel_scalars", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", return_value=dict(NO_RASTER))
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_no_raster_and_no_broker_data_keeps_original_payload(
        self, mock_fetch, mock_stats, mock_weather, client
    ):
        mock_fetch.return_value = SAMPLE_PARCEL
        mock_weather.return_value = {}
        data = _get(client, "temperature_min").json()
        assert data["error"] == "No COG data available"
        assert data["metrics"] == {}
        assert "kind" not in data and "valid_date" not in data

    @patch("app.tiles.fetch_parcel_scalars", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats", return_value=dict(NO_RASTER))
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_no_raster_explicit_date_never_uses_fallback(
        self, mock_fetch, mock_stats, mock_weather, client
    ):
        mock_fetch.return_value = SAMPLE_PARCEL
        data = _get(client, "temperature_min", date="2026-06-13").json()
        assert data["error"] == "No COG data available"
        mock_weather.assert_not_awaited()

    @patch("app.tiles.fetch_parcel_scalars", new_callable=AsyncMock)
    @patch("app.tiles.compute_zonal_stats")
    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_stale_raster_and_fallback_metric_have_different_dates(
        self, mock_fetch, mock_stats, mock_weather, client
    ):
        mock_fetch.return_value = SAMPLE_PARCEL
        stats = _uncovered_stats(["temperature_min"])
        stats["metrics"]["eto"] = SAMPLE_STATS["metrics"]["eto"]
        mock_stats.return_value = stats
        mock_weather.return_value = dict(WEATHER)
        data = _get(client, "temperature_min,eto").json()
        assert data["valid_date"] == "2026-06-13"
        assert data["kind"] == "forecast_snapshot"
        assert data["metrics"]["temperature_min"]["valid_date"] == "2026-07-02"
        assert "valid_date" not in data["metrics"]["eto"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metrics,expected",
        [
            (["temperature_avg"], ["WeatherObserved"]),
            (["solar_radiation"], ["WeatherObserved"]),
            (["temperature_min"], ["WeatherForecast"]),
            (["frost_risk"], ["WeatherForecast"]),
            (["temperature_avg", "frost_risk"], ["WeatherObserved", "WeatherForecast"]),
        ],
    )
    async def test_reads_only_what_the_metrics_need(self, metrics, expected):
        from app import sources
        from app.stats import fallback_sources

        obs, fc = fallback_sources(metrics)
        with patch.object(
            sources, "_get_entity_keyvalues", new_callable=AsyncMock
        ) as get:
            get.return_value = {"airTemperature": 10.0}
            await sources.fetch_parcel_scalars(
                "tenant-a", "urn:ngsi-ld:AgriParcel:p1", observed=obs, forecast=fc
            )
        ids = [c.args[1] for c in get.await_args_list]
        assert [i.split(":")[2] for i in ids] == expected
        assert not any(i.endswith("-daily") or "daily" in i for i in ids)


class TestSoilMoistureUnavailable:
    """soil_moisture stays accepted but reports no data with a reason."""

    @patch("app.tiles.fetch_agri_parcel", new_callable=AsyncMock)
    def test_get_stats_soil_moisture_is_unavailable_not_400(self, mock_fetch, client):
        mock_fetch.return_value = SAMPLE_PARCEL

        resp = client.get(
            "/api/weather-map/stats/demo-field-1",
            params={"metrics": "soil_moisture"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        entry = resp.json()["metrics"]["soil_moisture"]
        assert entry["error"] == "No data for parcel at this metric/date"
        assert "Not available" in entry["reason"]
        assert "mean" not in entry
