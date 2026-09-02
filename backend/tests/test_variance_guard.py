"""Tests for the zero-variance guard — app.records.assert_metric_varies.

The incident this replaces: airTemperatureAvg = 14.934 published identical
to five decimals on 32 consecutive August days, with no error and no failing
test. Three unmoving daily values is the cheap, shared signature of a
constant-fed pipeline, a frozen upstream source, or a silent default.

Round 2: the guard was correct but unwired — nobody called it, which is the
same silent outcome in a different shape. These tests cover the wiring:
`guard_frozen_metrics` fetches a metric's own history from the broker via
`fetch_metric_history`, appends the value about to publish, and runs it
through `assert_metric_varies` — logging loud, never blocking the publish.
"""

from __future__ import annotations

import logging

import pytest

from app.records import FrozenMetric, assert_metric_varies, guard_frozen_metrics
from app.sources import MissingContextURL, fetch_metric_history


def test_flags_a_frozen_metric():
    """14.934 tres días seguidos no es meteorología, es una constante."""
    with pytest.raises(FrozenMetric):
        assert_metric_varies([14.934, 14.934, 14.934])


def test_accepts_a_metric_that_moves():
    assert assert_metric_varies([14.9, 17.3, 21.0]) is None


def test_needs_three_samples_before_judging():
    assert assert_metric_varies([14.934, 14.934]) is None


@pytest.mark.asyncio
async def test_guard_fires_through_the_publish_path(monkeypatch, caplog):
    """3 identical values reached the way records.py reaches them: fetch
    the 2 prior published values, append the one about to publish, check.
    The record still publishes — this only asserts the ERROR is loud and
    names tenant, parcel, metric, and the repeated value."""
    from app import records

    async def _fake_history(tenant_id, parcel_id, attr_name, limit=5):
        return [14.934, 14.934]

    monkeypatch.setattr(records, "fetch_metric_history", _fake_history)
    with caplog.at_level(logging.ERROR):
        await guard_frozen_metrics(
            "asociacion-allotarra",
            "urn:ngsi-ld:AgriParcel:p1",
            {"temperature_avg": 14.934},
        )

    assert "asociacion-allotarra" in caplog.text
    assert "p1" in caplog.text
    assert "airTemperatureAvg" in caplog.text
    assert "14.934" in caplog.text


@pytest.mark.asyncio
async def test_guard_stays_quiet_with_fewer_than_three_samples(monkeypatch, caplog):
    """A new parcel (no history yet) must not block publishing or log an error."""
    from app import records

    async def _fake_history(tenant_id, parcel_id, attr_name, limit=5):
        return []  # new parcel — nothing published for it before

    monkeypatch.setattr(records, "fetch_metric_history", _fake_history)
    with caplog.at_level(logging.ERROR):
        await guard_frozen_metrics(
            "t", "urn:ngsi-ld:AgriParcel:new", {"temperature_avg": 14.934},
        )

    assert caplog.records == []


@pytest.mark.asyncio
async def test_missing_context_url_is_caught_not_silently_empty(monkeypatch):
    """Without the platform @context Link, Orion expands against the default
    vocabulary and every query false-empties — indistinguishable from "no
    history yet". fetch_metric_history must refuse loudly instead of
    returning [] and letting the guard mistake a broken query for a new
    parcel."""
    from app.config import settings

    monkeypatch.setattr(settings, "context_url", "")
    with pytest.raises(MissingContextURL):
        await fetch_metric_history(
            "t", "urn:ngsi-ld:AgriParcel:p1", "airTemperatureAvg",
        )
