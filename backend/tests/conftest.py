"""Test fixtures."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    from app.main import app
    return app


@pytest.fixture
def client(app):
    return TestClient(app)
