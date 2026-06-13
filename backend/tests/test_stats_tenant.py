"""Test that compute_zonal_stats threads the caller's tenant_id through to COG reads."""

from unittest.mock import patch

import numpy as np

from app.stats import compute_zonal_stats

GEOM = {"type": "Polygon", "coordinates": [[[-1.5, 42.0], [-1.49, 42.0], [-1.49, 42.01], [-1.5, 42.01], [-1.5, 42.0]]]}


def test_cog_read_uses_caller_tenant_not_default():
    seen = {}

    def fake_collect(tenant_id, geometry, metric, date, zoom=14):
        seen["tenant"] = tenant_id
        return np.array([1.0, 2.0], dtype="float32")

    with patch("app.stats._collect_pixels", side_effect=fake_collect), \
         patch("app.stats.get_latest_date", return_value="2026-06-10"):
        compute_zonal_stats("asociacion-allotarra", GEOM, ["eto"], date="2026-06-10")

    assert seen["tenant"] == "asociacion-allotarra"
