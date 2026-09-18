from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_tile_requires_tenant_header():
    r = client.get("/api/weather-map/tiles/eto/8/100/100.png")
    assert r.status_code == 401


def test_valid_signed_token_passes_auth(monkeypatch):
    from app import config
    from app.tile_auth import make_tile_token
    monkeypatch.setattr(config.settings, "tile_token_secret", "test-secret")
    token, _ = make_tile_token("montiko", "eto")
    # No COG data -> 404 proves the token got past auth (not a 401).
    monkeypatch.setattr("app.tiles.get_latest_date", lambda t, m: None)
    r = client.get(f"/api/weather-map/tiles/eto/8/100/100.png?tenant=montiko&token={token}")
    assert r.status_code == 404


def test_invalid_signed_token_is_401(monkeypatch):
    from app import config
    monkeypatch.setattr(config.settings, "tile_token_secret", "test-secret")
    r = client.get(
        "/api/weather-map/tiles/eto/8/100/100.png"
        "?tenant=montiko&token=deadbeef:9999999999"
    )
    assert r.status_code == 401


def test_signed_token_is_bound_to_tenant_and_metric(monkeypatch):
    from app import config
    from app.tile_auth import make_tile_token
    monkeypatch.setattr(config.settings, "tile_token_secret", "test-secret")
    token, _ = make_tile_token("montiko", "eto")
    monkeypatch.setattr("app.tiles.get_latest_date", lambda t, m: None)
    r = client.get(f"/api/weather-map/tiles/eto/8/100/100.png?tenant=other&token={token}")
    assert r.status_code == 401


def test_stats_requires_tenant_header():
    r = client.get("/api/weather-map/stats/urn:ngsi-ld:AgriParcel:p1?metrics=eto")
    assert r.status_code == 401
