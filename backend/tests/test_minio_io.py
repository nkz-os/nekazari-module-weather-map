"""Tests for minio_io.py — MinIO S3 read/write helpers.

Tests path construction only — no network calls.
"""

from __future__ import annotations


from app.minio_io import cog_path, latest_pointer_path


class TestCogPath:
    """COG tile path construction."""

    def test_cog_path(self):
        """Verify path construction for a typical COG tile."""
        path = cog_path("tenant_demo", "eto", "2026-06-10", 14, 8557, 5302)
        assert path == "cogs/tenant_demo/eto/2026-06-10/14/8557/5302.tif"

    def test_cog_path_zoom_prefix(self):
        """Path includes zoom as a subdirectory."""
        path = cog_path("t1", "temperature_min", "2026-01-01", 10, 500, 300)
        assert path.startswith("cogs/t1/temperature_min/2026-01-01/10/")
        assert path.endswith(".tif")

    def test_cog_path_different_tile(self):
        """Different tile coordinates produce different paths."""
        path_a = cog_path("t", "m", "d", 12, 1000, 2000)
        path_b = cog_path("t", "m", "d", 12, 1001, 2000)
        assert path_a != path_b

    def test_cog_path_special_characters(self):
        """Tenant IDs with hyphens/underscores are handled."""
        path = cog_path("my-tenant_1", "soil_moisture", "2026-06-12", 8, 128, 64)
        assert path == "cogs/my-tenant_1/soil_moisture/2026-06-12/8/128/64.tif"

    def test_cog_path_includes_metric(self):
        """Metric name appears in the path."""
        t_path = cog_path("t1", "temperature_avg", "d", 10, 100, 100)
        wb_path = cog_path("t1", "water_balance", "d", 10, 100, 100)
        assert "temperature_avg" in t_path
        assert "water_balance" in wb_path

    def test_cog_path_includes_date(self):
        """Date string appears in the path."""
        path = cog_path("t1", "m", "2026-06-15", 10, 100, 100)
        assert "2026-06-15" in path


class TestLatestPointerPath:
    """Latest-date pointer file path."""

    def test_latest_pointer_path(self):
        """Verify pointer path matches expected pattern."""
        path = latest_pointer_path("tenant_demo", "eto")
        assert path == "cogs/tenant_demo/eto/latest.txt"

    def test_latest_pointer_path_tenant_variation(self):
        """Different tenants and metrics produce different paths."""
        path1 = latest_pointer_path("tenant_a", "temperature_avg")
        path2 = latest_pointer_path("tenant_b", "temperature_avg")
        path3 = latest_pointer_path("tenant_a", "water_balance")
        assert path1 != path2
        assert path1 != path3
        assert path2 != path3

    def test_latest_pointer_ends_with_txt(self):
        """Pointer file always ends with .txt."""
        path = latest_pointer_path("any", "any")
        assert path.endswith(".txt")

    def test_latest_pointer_in_cogs_prefix(self):
        """Pointer file lives under cogs/."""
        path = latest_pointer_path("t1", "m")
        assert path.startswith("cogs/")
