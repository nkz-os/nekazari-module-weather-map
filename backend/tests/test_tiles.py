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
        headers={"X-Tenant-ID": "test-tenant"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_apply_color_scale_tile():
    """apply_color_scale should work on 256x256 array."""
    from app.color_scales import apply_color_scale
    data = np.random.randn(256, 256) * 10 + 20
    rgba = apply_color_scale(data, "temperature_avg")
    assert rgba.shape == (256, 256, 4)
    assert rgba.dtype == np.uint8
