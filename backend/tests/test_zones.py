"""Tests for zones endpoint."""
def test_zones_router_exists():
    from app.zones import router
    assert router is not None
