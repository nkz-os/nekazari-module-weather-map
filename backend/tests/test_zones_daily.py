"""GET /zones/{parcel}/daily — per-zone daily series from the hypertable."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

from app.config import settings

AUTH = {"X-Tenant-ID": "demo-coop", "X-User-ID": "u1"}
PID = "urn:ngsi-ld:AgriParcel:demo-coop:p1"
URL = "/api/weather-map/zones/p1/daily"


def _row(zone, day, t_min, t_max, eto):
    return {"nkz_zone_id": zone, "day": date.fromisoformat(day),
            "t_min": t_min, "t_max": t_max, "eto": eto}


def _patches(rows, parcel=True):
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=rows)
    conn.close = AsyncMock()
    return conn, (
        patch("app.zones.fetch_agri_parcel",
              new=AsyncMock(return_value={"id": PID} if parcel else None)),
        patch("app.zones.asyncpg.connect", new=AsyncMock(return_value=conn)),
        patch.object(settings, "postgres_url", "postgres://test"),
    )


def _get(client, qs="?start=2026-06-01&end=2026-06-03", rows=(), parcel=True):
    conn, ps = _patches(list(rows), parcel)
    with ps[0], ps[1], ps[2]:
        return conn, client.get(URL + qs, headers=AUTH)


def test_daily_reaches_handler_not_catch_all(client):
    conn, resp = _get(client)
    assert resp.status_code == 200
    assert resp.json()["parcel_id"] == PID or resp.json()["parcel_id"].endswith("p1")
    assert "zones" in resp.json() and "days" in resp.json()
    conn.fetch.assert_awaited_once()


def test_daily_groups_zones_by_date(client):
    rows = [
        _row("zA", "2026-06-01", 10.0, 20.0, 3.0),
        _row("zA", "2026-06-02", 11.0, 21.0, 3.1),
        _row("zA", "2026-06-03", 12.0, 22.0, None),
        _row("zB", "2026-06-01", 9.0, 19.0, 2.9),
        _row("zB", "2026-06-02", 9.5, 19.5, 3.0),
        _row("zB", "2026-06-03", 10.5, 20.5, 3.2),
    ]
    conn, resp = _get(client, rows=rows)
    body = resp.json()
    assert body["days"] == 3
    assert [d["date"] for d in body["zones"]["zA"]] == ["2026-06-01", "2026-06-02", "2026-06-03"]
    assert body["zones"]["zA"][0] == {"date": "2026-06-01", "tMin": 10.0, "tMax": 20.0, "eto": 3.0}
    assert body["zones"]["zA"][2]["eto"] is None
    assert len(body["zones"]["zB"]) == 3


def test_daily_sql_keeps_latest_run_per_day(client):
    conn, resp = _get(client)
    sql = conn.fetch.await_args.args[0]
    assert "DISTINCT ON (nkz_zone_id, DATE(observed_at))" in sql
    assert "observed_at DESC" in sql
    args = conn.fetch.await_args.args[1:]
    assert args[0] == "urn:ngsi-ld:AgriParcel:p1"
    assert args[1].date() == date(2026, 6, 1)
    assert args[2].date() == date(2026, 6, 4)  # end + 1 day, exclusive


def test_daily_rejects_range_over_400_days(client):
    conn, resp = _get(client, "?start=2025-01-01&end=2026-02-06")  # 401 days
    assert resp.status_code == 400
    conn.fetch.assert_not_awaited()


def test_daily_accepts_400_days(client):
    _, resp = _get(client, "?start=2025-01-01&end=2026-02-05")
    assert resp.status_code == 200


def test_daily_rejects_end_before_start(client):
    _, resp = _get(client, "?start=2026-06-03&end=2026-06-01")
    assert resp.status_code == 400


def test_daily_rejects_malformed_dates(client):
    _, resp = _get(client, "?start=2026-6-1&end=2026-06-03")
    assert resp.status_code == 400
    _, resp = _get(client, "?start=2026-06-01&end=2026-13-45")
    assert resp.status_code == 400


def test_daily_503_without_postgres(client):
    with patch.object(settings, "postgres_url", ""):
        resp = client.get(URL + "?start=2026-06-01&end=2026-06-03", headers=AUTH)
    assert resp.status_code == 503


def test_daily_404_unknown_parcel(client):
    _, resp = _get(client, parcel=False)
    assert resp.status_code == 404


def test_daily_requires_auth(client):
    assert client.get(URL + "?start=2026-06-01&end=2026-06-03").status_code == 401
