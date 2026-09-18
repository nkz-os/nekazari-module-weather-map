"""Tile tokens follow the platform's canonical HMAC format and are bound to tenant+metric."""

from app import config
from app.tile_auth import make_tile_token, verify_tile_token


def _set_secret(monkeypatch, secret="tile-test-secret"):
    monkeypatch.setattr(config.settings, "tile_token_secret", secret)


def test_token_roundtrip_and_binding(monkeypatch):
    _set_secret(monkeypatch)
    token, _ = make_tile_token("montiko", "temperature_avg")
    assert verify_tile_token(token, "montiko", "temperature_avg") is True
    # the token is bound to both tenant and metric
    assert verify_tile_token(token, "montiko", "eto") is False
    assert verify_tile_token(token, "allotarra", "temperature_avg") is False


def test_token_uses_full_hex_and_sig_first(monkeypatch):
    _set_secret(monkeypatch)
    token, expiry = make_tile_token("montiko", "temperature_avg")
    digest, _, exp = token.partition(":")
    assert len(digest) == 64  # full SHA-256 hex, never truncated
    assert exp == str(expiry)  # signature first, expiry second


def test_expired_token_is_rejected(monkeypatch):
    _set_secret(monkeypatch)
    token, _ = make_tile_token("montiko", "temperature_avg", ttl=-10)
    assert verify_tile_token(token, "montiko", "temperature_avg") is False


def test_malformed_token_is_rejected(monkeypatch):
    _set_secret(monkeypatch)
    assert verify_tile_token(None, "montiko", "temperature_avg") is False
    assert verify_tile_token("", "montiko", "temperature_avg") is False
    assert verify_tile_token("no-colon", "montiko", "temperature_avg") is False
    assert verify_tile_token("abc:not-an-int", "montiko", "temperature_avg") is False


def test_tampered_token_is_rejected(monkeypatch):
    _set_secret(monkeypatch)
    token, _ = make_tile_token("montiko", "temperature_avg")
    digest, sep, exp = token.partition(":")
    tampered = f"{'0' * 64}{sep}{exp}"
    assert verify_tile_token(tampered, "montiko", "temperature_avg") is False
