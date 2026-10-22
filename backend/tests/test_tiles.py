"""Tests for tile serving logic."""

import numpy as np
import pytest

from app.tiles import router


def test_tile_router_exists():
    """Router should have routes defined."""
    assert len(router.routes) > 0


def test_unknown_metric_returns_404(client):
    """Request with unknown metric should return 404."""
    resp = client.get(
        "/api/weather-map/tiles/invalid_metric/14/8557/5302.png",
        headers={"X-Tenant-ID": "test-tenant", "X-User-ID": "test-user"},
    )
    assert resp.status_code == 404


def test_latest_unknown_metric_returns_404(client):
    resp = client.get(
        "/api/weather-map/latest/invalid_metric",
        headers={"X-Tenant-ID": "test-tenant", "X-User-ID": "test-user"},
    )
    assert resp.status_code == 404


def test_latest_metric_no_data_returns_404(client, monkeypatch):
    monkeypatch.setattr("app.tiles.get_latest_date", lambda _t, _m: None)
    resp = client.get(
        "/api/weather-map/latest/temperature_avg",
        headers={"X-Tenant-ID": "test-tenant", "X-User-ID": "test-user"},
    )
    assert resp.status_code == 404


def test_latest_metric_returns_date(client, monkeypatch):
    monkeypatch.setattr("app.tiles.get_latest_date", lambda _t, _m: "2026-06-10")
    resp = client.get(
        "/api/weather-map/latest/temperature_avg",
        headers={"X-Tenant-ID": "test-tenant", "X-User-ID": "test-user"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"metric": "temperature_avg", "date": "2026-06-10"}


@pytest.mark.asyncio
async def test_apply_color_scale_tile():
    """apply_color_scale should work on 256x256 array."""
    from app.color_scales import apply_color_scale
    data = np.random.randn(256, 256) * 10 + 20
    rgba = apply_color_scale(data, "temperature_avg")
    assert rgba.shape == (256, 256, 4)
    assert rgba.dtype == np.uint8
