"""Tests for the zero-variance guard — app.records.assert_metric_varies.

The incident this replaces: airTemperatureAvg = 14.934 published identical
to five decimals on 32 consecutive August days, with no error and no failing
test. Three unmoving daily values is the cheap, shared signature of a
constant-fed pipeline, a frozen upstream source, or a silent default.
"""

from __future__ import annotations

import pytest

from app.records import FrozenMetric, assert_metric_varies


def test_flags_a_frozen_metric():
    """14.934 tres días seguidos no es meteorología, es una constante."""
    with pytest.raises(FrozenMetric):
        assert_metric_varies([14.934, 14.934, 14.934])


def test_accepts_a_metric_that_moves():
    assert assert_metric_varies([14.9, 17.3, 21.0]) is None


def test_needs_three_samples_before_judging():
    assert assert_metric_varies([14.934, 14.934]) is None
