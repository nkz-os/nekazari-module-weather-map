"""Test fixtures."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _unsigned_requests_allowed(monkeypatch):
    """Route tests send bare tenant headers; signatures are covered in test_auth."""
    monkeypatch.setenv("REQUIRE_HMAC_SIGNATURE", "false")
    monkeypatch.delenv("HMAC_SECRET", raising=False)


@pytest.fixture
def app():
    from app.main import app
    return app


@pytest.fixture
def client(app):
    return TestClient(app)
