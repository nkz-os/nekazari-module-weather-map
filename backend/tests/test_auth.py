from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.auth import require_tenant


def _app():
    app = FastAPI()

    @app.get("/probe")
    def probe(tenant: str = Depends(require_tenant)):
        return {"tenant": tenant}

    return app


def test_missing_tenant_header_is_rejected():
    client = TestClient(_app())
    assert client.get("/probe").status_code in (401, 403)


def test_tenant_taken_from_header_not_query():
    client = TestClient(_app())
    resp = client.get(
        "/probe?tenant_id=evil",
        headers={"X-Tenant-ID": "asociacion-allotarra", "X-User-ID": "u1"},
    )
    assert resp.status_code == 200
    assert resp.json()["tenant"] == "asociacion-allotarra"
