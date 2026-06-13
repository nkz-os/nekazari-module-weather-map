from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_tile_requires_tenant_header():
    r = client.get("/api/weather-map/tiles/eto/8/100/100.png")
    assert r.status_code == 401


def test_stats_requires_tenant_header():
    r = client.get("/api/weather-map/stats/urn:ngsi-ld:AgriParcel:p1?metrics=eto")
    assert r.status_code == 401
