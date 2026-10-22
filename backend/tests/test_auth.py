import logging
import time

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from nkz_platform_sdk.crypto import generate_hmac_signature

from app import auth
from app.auth import require_tenant

SECRET = "s3cret"


def _app():
    app = FastAPI()

    @app.get("/probe")
    def probe(tenant: str = Depends(require_tenant)):
        return {"tenant": tenant}

    return app


@pytest.fixture
def client():
    return TestClient(_app())


@pytest.fixture
def enforce(monkeypatch):
    monkeypatch.setenv("REQUIRE_HMAC_SIGNATURE", "true")
    monkeypatch.setenv("HMAC_SECRET", SECRET)


def _headers(tenant="acme", user="u1", sig=None, token=None):
    h = {"X-Tenant-ID": tenant, "X-User-ID": user}
    if sig is not None:
        h["X-Auth-Signature"] = sig
    if token is not None:
        h["Authorization"] = f"Bearer {token}"
    return h


def test_missing_tenant_header_is_rejected(client):
    assert client.get("/probe").status_code in (401, 403)


def test_tenant_taken_from_header_not_query(client):
    resp = client.get("/probe?tenant_id=evil", headers=_headers())
    assert resp.status_code == 200
    assert resp.json()["tenant"] == "acme"


def test_module_signature_accepted(enforce, client):
    sig = generate_hmac_signature(SECRET, "", "acme")
    resp = client.get("/probe", headers=_headers(sig=sig))
    assert resp.status_code == 200
    assert resp.json()["tenant"] == "acme"


def test_gateway_signature_over_user_token_accepted(enforce, client):
    sig = generate_hmac_signature(SECRET, "tok", "acme")
    resp = client.get("/probe", headers=_headers(sig=sig, token="tok"))
    assert resp.status_code == 200


def test_signature_for_other_tenant_rejected(enforce, client):
    sig = generate_hmac_signature(SECRET, "", "acme")
    resp = client.get("/probe", headers=_headers(tenant="victim", sig=sig))
    assert resp.status_code == 401


def test_signature_over_other_token_rejected(enforce, client):
    sig = generate_hmac_signature(SECRET, "tok", "acme")
    resp = client.get("/probe", headers=_headers(sig=sig, token="other"))
    assert resp.status_code == 401


def test_wrong_secret_rejected(enforce, client):
    sig = generate_hmac_signature("other", "", "acme")
    assert client.get("/probe", headers=_headers(sig=sig)).status_code == 401


def test_expired_signature_rejected(enforce, client):
    sig = generate_hmac_signature(SECRET, "", "acme", timestamp=int(time.time()) - 301)
    assert client.get("/probe", headers=_headers(sig=sig)).status_code == 401


def test_unsigned_rejected_when_enforced(enforce, client):
    assert client.get("/probe", headers=_headers()).status_code == 401


def test_missing_user_rejected_even_if_signed(enforce, client):
    sig = generate_hmac_signature(SECRET, "", "acme")
    resp = client.get("/probe", headers={"X-Tenant-ID": "acme", "X-Auth-Signature": sig})
    assert resp.status_code == 401


def test_enforced_without_secret_is_503(monkeypatch, client):
    monkeypatch.setenv("REQUIRE_HMAC_SIGNATURE", "true")
    monkeypatch.delenv("HMAC_SECRET", raising=False)
    sig = generate_hmac_signature(SECRET, "", "acme")
    assert client.get("/probe", headers=_headers(sig=sig)).status_code == 503


def test_enforcement_is_the_default(monkeypatch, client):
    monkeypatch.delenv("REQUIRE_HMAC_SIGNATURE", raising=False)
    monkeypatch.setenv("HMAC_SECRET", SECRET)
    assert client.get("/probe", headers=_headers()).status_code == 401


def test_log_only_serves_unsigned_and_logs_caller(monkeypatch, client, caplog):
    monkeypatch.setenv("REQUIRE_HMAC_SIGNATURE", "false")
    monkeypatch.setenv("HMAC_SECRET", SECRET)
    auth._UNSIGNED_SEEN.clear()
    with caplog.at_level(logging.WARNING, logger="app.auth"):
        resp = client.get("/probe", headers=_headers(user="legacy-worker"))
    assert resp.status_code == 200
    assert "Unsigned request" in caplog.text
    assert "legacy-worker" in caplog.text


def test_log_only_still_rejects_invalid_signature(monkeypatch, client):
    monkeypatch.setenv("REQUIRE_HMAC_SIGNATURE", "false")
    monkeypatch.setenv("HMAC_SECRET", SECRET)
    sig = generate_hmac_signature("other", "", "acme")
    assert client.get("/probe", headers=_headers(sig=sig)).status_code == 401
