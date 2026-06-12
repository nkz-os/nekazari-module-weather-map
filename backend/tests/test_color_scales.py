"""Tests for color scale palettes and lookup tables."""

import numpy as np
import pytest

from app.color_scales import (
    PALETTES,
    build_lut,
    apply_color_scale,
    _get_stops,
)
from app.config import settings


class TestPalettes:
    """Validation of PALETTES dict structure."""

    def test_all_metrics_have_palettes(self):
        """Every metric in settings.metrics must have a palette."""
        for metric in settings.metrics:
            assert metric in PALETTES, f"Missing palette for metric: {metric}"

    def test_palette_stops_ascending(self):
        """Value stops must be in ascending order."""
        for metric, stops in PALETTES.items():
            values = [s[0] for s in stops]
            for i in range(1, len(values)):
                assert values[i] > values[i - 1], (
                    f"{metric}: stop {i} value {values[i]} <= {values[i - 1]}"
                )

    def test_palette_values_within_range(self):
        """Values must be finite numbers."""
        for metric, stops in PALETTES.items():
            for stop in stops:
                assert np.isfinite(stop[0]), f"{metric}: non-finite value {stop[0]}"

    def test_palette_rgba_valid(self):
        """RGBA values must be 0-255 integers."""
        for metric, stops in PALETTES.items():
            for stop in stops:
                for i, channel in enumerate(["R", "G", "B", "A"]):
                    val = stop[i + 1]
                    assert isinstance(val, int), f"{metric}: {channel}={val} not int"
                    assert 0 <= val <= 255, f"{metric}: {channel}={val} out of range"


class TestBuildLut:
    """Lookup table generation."""

    def test_build_lut_256(self):
        """LUT must have 256 entries × 4 channels."""
        lut = build_lut("temperature_avg")
        assert lut.shape == (256, 4)
        assert lut.dtype == np.uint8

    def test_build_lut_first_color(self):
        """First LUT entry should match first stop color."""
        lut = build_lut("temperature_avg")
        first_stop = PALETTES["temperature_avg"][0]
        expected = np.array(first_stop[1:5], dtype=np.uint8)
        np.testing.assert_array_equal(lut[0], expected)

    def test_build_lut_last_color(self):
        """Last LUT entry should match last stop color."""
        lut = build_lut("temperature_avg")
        last_stop = PALETTES["temperature_avg"][-1]
        expected = np.array(last_stop[1:5], dtype=np.uint8)
        np.testing.assert_array_equal(lut[-1], expected)

    def test_build_lut_custom_n_colors(self):
        """Custom n_colors produces correct shape."""
        lut = build_lut("temperature_avg", n_colors=128)
        assert lut.shape == (128, 4)

    def test_build_lut_all_metrics(self):
        """All metrics produce valid LUTs."""
        for metric in PALETTES:
            lut = build_lut(metric)
            assert lut.shape == (256, 4)
            assert lut.dtype == np.uint8
            # First and last entries match stops
            first_stop = PALETTES[metric][0]
            last_stop = PALETTES[metric][-1]
            np.testing.assert_array_equal(lut[0], np.array(first_stop[1:5], dtype=np.uint8))
            np.testing.assert_array_equal(lut[-1], np.array(last_stop[1:5], dtype=np.uint8))

    def test_build_lut_interpolation(self):
        """Interior LUT entries should be interpolated between stops."""
        lut = build_lut("temperature_avg")
        # Check that an interior entry differs from the first stop
        # (which would mean no interpolation happened)
        first_color = np.array(PALETTES["temperature_avg"][0][1:5], dtype=np.uint8)
        # Entry ~64 is roughly 1/4 of the range — should be interpolated
        mid_idx = lut.shape[0] // 4
        assert not np.array_equal(lut[mid_idx], first_color), (
            "Interior LUT entry matches first stop — no interpolation"
        )


class TestApplyColorScale:
    """Applying color scale to 2D arrays."""

    def test_apply_color_scale_shape(self):
        """Returns H×W×4 uint8 array."""
        data = np.random.uniform(0, 30, (50, 100)).astype(np.float32)
        result = apply_color_scale(data, "temperature_avg")
        assert result.shape == (50, 100, 4)

    def test_apply_color_scale_dtype(self):
        """Returns uint8."""
        data = np.random.uniform(0, 30, (10, 10)).astype(np.float32)
        result = apply_color_scale(data, "temperature_avg")
        assert result.dtype == np.uint8

    def test_nodata_transparent(self):
        """NaN pixels should have alpha=0."""
        data = np.random.uniform(0, 30, (10, 10)).astype(np.float32)
        data[3, 5] = np.nan
        result = apply_color_scale(data, "temperature_avg")
        assert result[3, 5, 3] == 0, "NaN pixel should have alpha=0"

    def test_inf_nodata(self):
        """Inf pixels should have alpha=0."""
        data = np.random.uniform(0, 30, (10, 10)).astype(np.float32)
        data[3, 5] = np.inf
        data[7, 2] = -np.inf
        result = apply_color_scale(data, "temperature_avg")
        assert result[3, 5, 3] == 0, "Inf pixel should have alpha=0"
        assert result[7, 2, 3] == 0, "-Inf pixel should have alpha=0"

    def test_vmin_vmax_clipping(self):
        """Values below min → first color, above max → last color."""
        data = np.array([[-100.0, 50.0]], dtype=np.float32)
        result = apply_color_scale(data, "temperature_avg")
        # -100 is below vmin (-5), so it should get first stop color
        first_stop = PALETTES["temperature_avg"][0]
        np.testing.assert_array_equal(
            result[0, 0],
            np.array(first_stop[1:5], dtype=np.uint8),
        )
        # 50 is above vmax (45), so it should get last stop color
        last_stop = PALETTES["temperature_avg"][-1]
        np.testing.assert_array_equal(
            result[0, 1],
            np.array(last_stop[1:5], dtype=np.uint8),
        )

    def test_unknown_metric(self):
        """Unknown metric should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            apply_color_scale(np.zeros((5, 5), dtype=np.float32), "nonexistent_metric")

    def test_apply_color_scale_normal_values(self):
        """Normal values produce non-transparent output."""
        data = np.full((5, 5), 20.0, dtype=np.float32)
        result = apply_color_scale(data, "temperature_avg")
        assert np.all(result[:, :, 3] == 255), "Normal values should be fully opaque"
        # At 20°C, should map close to the yellow-green stop
        # (may differ by ±1 due to LUT quantization)
        expected = np.array(PALETTES["temperature_avg"][3][1:5], dtype=np.uint8)
        np.testing.assert_allclose(result[0, 0], expected, atol=1)

    def test_apply_color_scale_partial_nodata(self):
        """Mix of normal and NaN pixels."""
        data = np.full((4, 4), 15.0, dtype=np.float32)
        data[0, 0] = np.nan
        data[1, 1] = np.inf
        data[2, 2] = -np.inf
        result = apply_color_scale(data, "temperature_avg")
        assert result[0, 0, 3] == 0  # NaN transparent
        assert result[1, 1, 3] == 0  # Inf transparent
        assert result[2, 2, 3] == 0  # -Inf transparent
        assert result[3, 3, 3] == 255  # Normal opaque


class TestInternalHelpers:
    """Internal helper functions."""

    def test_get_stops_returns_correct(self):
        """_get_stops returns correct stops for valid metric."""
        stops = _get_stops("temperature_avg")
        assert stops is PALETTES["temperature_avg"]

    def test_get_stops_unknown_metric(self):
        """_get_stops raises ValueError for unknown metric."""
        with pytest.raises(ValueError, match="Unknown metric: nonexistent"):
            _get_stops("nonexistent")
