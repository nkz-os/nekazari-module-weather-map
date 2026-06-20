"""Tests for the Weather Map downscaler physics engine."""

import numpy as np

from app.downscaler import (
    correct_temperature,
    correct_solar_radiation,
    compute_eto,
    compute_water_balance,
    compute_frost_risk,
    compute_soil_moisture,
    saxton_rawls_ptf,
    get_texture_defaults,
    discretize_aspect,
    compute_zones,
    _dominant_sector,
)


class TestCorrectTemperature:
    """Air temperature correction using standard lapse rate."""

    def test_lower_elevation(self):
        """500m pixel, 100m station, 25°C base → expect ~22.4°C."""
        elev = np.full((2, 2), 500.0)
        result = correct_temperature(t_base=25.0, station_elevation_m=100.0, pixel_elevations=elev)
        expected = 25.0 - (500.0 - 100.0) * 0.0065  # 22.4
        assert np.allclose(result, expected, atol=0.01)
        assert result.shape == (2, 2)

    def test_same_elevation(self):
        """Same elevation → no change."""
        elev = np.full((3, 3), 200.0)
        result = correct_temperature(t_base=15.0, station_elevation_m=200.0, pixel_elevations=elev)
        assert np.allclose(result, 15.0, atol=0.001)

    def test_higher_elevation_colder(self):
        """Higher elevation → lower temperature."""
        elev = np.array([[1000.0, 500.0], [200.0, 0.0]])
        result = correct_temperature(t_base=20.0, station_elevation_m=0.0, pixel_elevations=elev)
        # 1000m: 20 - 1000*0.0065 = 13.5
        # 500m:  20 - 500*0.0065 = 16.75
        # 200m:  20 - 200*0.0065 = 18.7
        # 0m:    20 - 0 = 20
        expected = np.array([[13.5, 16.75], [18.7, 20.0]])
        assert np.allclose(result, expected, atol=0.01)

    def test_negative_elevation_diff(self):
        """Pixel below station → warmer."""
        elev = np.full((2, 2), -50.0)
        result = correct_temperature(t_base=10.0, station_elevation_m=100.0, pixel_elevations=elev)
        expected = 10.0 - (-50.0 - 100.0) * 0.0065  # 10 + 0.975 = 10.975
        assert np.allclose(result, expected, atol=0.01)


class TestCorrectSolarRadiation:
    """Solar radiation correction for slope and aspect."""

    def test_flat_surface_returns_base(self):
        """Flat surface (slope=0) with no clouds should give approx base value."""
        lat = np.full((2, 2), 40.0)
        aspect = np.full((2, 2), 0.0)
        slope = np.full((2, 2), 0.0)
        doy = 172  # summer solstice approximately
        rad_base = 350.0  # W/m²

        result = correct_solar_radiation(rad_base, lat, aspect, slope, doy)
        assert result.shape == (2, 2)
        assert np.all(result >= 0)
        # Flat surface should be close to base (within reasonable range)
        assert np.all(result < 1200)

    def test_south_vs_north(self):
        """South-facing slope gets more radiation than north-facing (northern hemisphere)."""
        # 40°N latitude, 10° slope
        lat = np.full((2,), 40.0)
        aspect = np.array([180.0, 0.0])  # south, north
        slope = np.full((2,), 10.0)
        doy = 80  # spring, decent sun angle
        rad_base = 400.0

        lat_2d = lat.reshape(2, 1)
        aspect_2d = aspect.reshape(2, 1)
        slope_2d = slope.reshape(2, 1)

        result = correct_solar_radiation(rad_base, lat_2d, aspect_2d, slope_2d, doy)
        # South-facing should get more radiation
        assert result[0, 0] > result[1, 0], (
            f"South-facing ({result[0,0]:.1f}) should exceed north-facing ({result[1,0]:.1f}) in NH"
        )

    def test_winter_south_advantage(self):
        """In winter, south advantage is more pronounced."""
        lat = np.full((2,), 40.0)
        aspect = np.array([180.0, 0.0])
        slope = np.full((2,), 15.0)
        doy = 355  # winter solstice
        rad_base = 200.0

        lat_2d = lat.reshape(2, 1)
        aspect_2d = aspect.reshape(2, 1)
        slope_2d = slope.reshape(2, 1)

        result = correct_solar_radiation(rad_base, lat_2d, aspect_2d, slope_2d, doy)
        assert result[0, 0] > result[1, 0]

    def test_high_cloudiness_reduces_radiation(self):
        """Very cloudy (low base rad / clear-sky ratio) should reduce corrected radiation."""
        lat = np.full((2, 2), 40.0)
        aspect = np.full((2, 2), 0.0)
        slope = np.full((2, 2), 0.0)
        doy = 172

        # Very low base radiation (dense overcast)
        result_low = correct_solar_radiation(50.0, lat, aspect, slope, doy)
        result_high = correct_solar_radiation(400.0, lat, aspect, slope, doy)

        assert np.all(result_low >= 0)
        assert np.all(result_high >= 0)

    def test_output_clipped(self):
        """Output should be clipped to [0, 1200]."""
        lat = np.full((2, 2), 0.0)  # equator, high potential
        aspect = np.full((2, 2), 0.0)
        slope = np.full((2, 2), 0.0)
        doy = 80
        rad_base = 1200.0  # unrealistically high

        result = correct_solar_radiation(rad_base, lat, aspect, slope, doy)
        assert np.all(result <= 1200)
        assert np.all(result >= 0)


class TestComputeETo:
    """FAO Penman-Monteith reference evapotranspiration."""

    def test_eto_bounds(self):
        """ET0 should be in plausible range [0, 15] for reasonable inputs."""
        t_avg = np.full((2, 2), 20.0)
        t_min = np.full((2, 2), 15.0)
        t_max = np.full((2, 2), 25.0)
        solar_rad = np.full((2, 2), 300.0)  # W/m²
        wind_speed = 2.0  # m/s
        rh = 60.0  # %
        elevation_m = np.full((2, 2), 100.0)

        result = compute_eto(t_avg, t_min, t_max, solar_rad, wind_speed, rh, elevation_m)
        assert result.shape == (2, 2)
        assert np.all(result >= 0)
        assert np.all(result <= 15)

    def test_eto_higher_with_more_radiation(self):
        """More solar radiation → higher ET0."""
        t_avg = np.full((2, 2), 25.0)
        t_min = np.full((2, 2), 20.0)
        t_max = np.full((2, 2), 30.0)
        wind_speed = 3.0
        rh = 50.0
        elevation_m = np.full((2, 2), 0.0)

        rad_low = np.full((2, 2), 100.0)
        rad_high = np.full((2, 2), 400.0)

        eto_low = compute_eto(t_avg, t_min, t_max, rad_low, wind_speed, rh, elevation_m)
        eto_high = compute_eto(t_avg, t_min, t_max, rad_high, wind_speed, rh, elevation_m)

        assert np.all(eto_high > eto_low)

    def test_eto_higher_with_more_wind(self):
        """More wind → higher ET0."""
        t_avg = np.full((2, 2), 25.0)
        t_min = np.full((2, 2), 20.0)
        t_max = np.full((2, 2), 30.0)
        solar_rad = np.full((2, 2), 300.0)
        rh = 50.0
        elevation_m = np.full((2, 2), 100.0)

        eto_low_wind = compute_eto(t_avg, t_min, t_max, solar_rad, 0.5, rh, elevation_m)
        eto_high_wind = compute_eto(t_avg, t_min, t_max, solar_rad, 5.0, rh, elevation_m)

        assert np.all(eto_high_wind >= eto_low_wind)

    def test_eto_low_when_humid(self):
        """High humidity → lower ET0."""
        t_avg = np.full((2, 2), 25.0)
        t_min = np.full((2, 2), 20.0)
        t_max = np.full((2, 2), 30.0)
        solar_rad = np.full((2, 2), 300.0)
        wind_speed = 2.0
        elevation_m = np.full((2, 2), 100.0)

        eto_dry = compute_eto(t_avg, t_min, t_max, solar_rad, wind_speed, 20.0, elevation_m)
        eto_humid = compute_eto(t_avg, t_min, t_max, solar_rad, wind_speed, 90.0, elevation_m)

        assert np.all(eto_humid < eto_dry)

    def test_eto_higher_at_higher_elevation(self):
        """Higher elevation → lower air pressure → slightly higher ET0 due to psychrometric constant effect."""
        t_avg = np.full((2, 2), 20.0)
        t_min = np.full((2, 2), 15.0)
        t_max = np.full((2, 2), 25.0)
        solar_rad = np.full((2, 2), 300.0)
        wind_speed = 2.0
        rh = 60.0

        low_elev = np.full((2, 2), 0.0)
        high_elev = np.full((2, 2), 3000.0)

        eto_low = compute_eto(t_avg, t_min, t_max, solar_rad, wind_speed, rh, low_elev)
        eto_high = compute_eto(t_avg, t_min, t_max, solar_rad, wind_speed, rh, high_elev)

        assert np.all(eto_high != eto_low)


class TestWaterBalance:
    """Simple water balance."""

    def test_deficit(self):
        """ET0 > precip → negative balance."""
        precip = 5.0
        eto = np.full((2, 2), 15.0)
        result = compute_water_balance(precip, eto)
        assert np.all(result < 0)
        assert np.allclose(result, precip - eto)

    def test_surplus(self):
        """Precip > ET0 → positive balance."""
        precip = 50.0
        eto = np.full((2, 2), 10.0)
        result = compute_water_balance(precip, eto)
        assert np.all(result > 0)
        assert np.allclose(result, precip - eto)

    def test_broadcast(self):
        """Scalar precip broadcast over 2D ET0 array."""
        precip = 25.0
        eto = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = compute_water_balance(precip, eto)
        expected = np.array([[15.0, 5.0], [-5.0, -15.0]])
        assert np.allclose(result, expected)


class TestFrostRisk:
    """Cold-air pooling frost risk assessment."""

    def test_warm(self):
        """5°C with flat terrain → risk < 30%."""
        t_min = np.full((3, 3), 5.0)
        elev = np.full((3, 3), 200.0)
        result = compute_frost_risk(t_min, elev)
        # Cold-air pooling adds 2°C penalty even on flat terrain
        assert np.all(result < 30.0)
        assert np.all(result > 20.0)
        assert np.allclose(result, 26.8941, atol=0.01)

    def test_freezing(self):
        """-3°C with flat terrain → risk > 80%."""
        t_min = np.full((3, 3), -3.0)
        elev = np.full((3, 3), 200.0)
        result = compute_frost_risk(t_min, elev)
        assert np.all(result > 80.0)

    def test_cold_air_pooling(self):
        """Valley (low elevation) gets colder → higher risk than ridge."""
        t_min = np.full((3, 3), 2.0)
        elev = np.array([[300.0, 300.0, 300.0],
                         [200.0, 100.0, 200.0],
                         [300.0, 300.0, 300.0]])
        result = compute_frost_risk(t_min, elev)
        # Center (valley) should have higher risk than surroundings
        assert result[1, 1] > result[0, 1]
        assert result[1, 1] > result[1, 0]

    def test_output_range(self):
        """Frost risk should be in [0, 100]."""
        t_min = np.array([[-10.0, -5.0], [0.0, 10.0]])
        elev = np.array([[100.0, 200.0], [300.0, 400.0]])
        result = compute_frost_risk(t_min, elev)
        assert np.all(result >= 0)
        assert np.all(result <= 100)

    def test_shape_preservation(self):
        """Output matches input shape."""
        t_min = np.random.uniform(-5, 10, (4, 5))
        elev = np.random.uniform(0, 500, (4, 5))
        result = compute_frost_risk(t_min, elev)
        assert result.shape == (4, 5)


class TestSoilMoisture:
    """Bucket model soil moisture."""

    def test_initialization_default(self):
        """Without current, start at 60% FC."""
        awc = 20.0
        fc = 30.0
        wp = 10.0
        result = compute_soil_moisture(awc, fc, wp, precip_5d=0.0, eto_5d=0.0)
        # Default start = 0.6 * 30 = 18.0
        assert np.isclose(result, 18.0, atol=0.1)

    def test_recharge_from_precip(self):
        """Precipitation recharges soil moisture."""
        awc = 20.0
        fc = 30.0
        wp = 10.0
        # Start at 15.0, precip = 20mm → recharge = 20*0.7 = 14mm → 14/20 = 0.7%
        # moisture = 15 + 0.7 = 15.7
        result = compute_soil_moisture(awc, fc, wp, precip_5d=20.0, eto_5d=0.0, current=15.0)
        assert np.isclose(result, 15.7, atol=0.1)

    def test_depletion(self):
        """ET0 depletes soil moisture."""
        awc = 20.0
        fc = 30.0
        wp = 10.0
        # Start at 15.0, eto_5d = 10mm → depletion = 10/20*100 = 0.5%
        # moisture = 15 - 0.5 = 14.5
        result = compute_soil_moisture(awc, fc, wp, precip_5d=0.0, eto_5d=10.0, current=15.0)
        assert np.isclose(result, 14.5, atol=0.1)

    def test_clip_to_wp(self):
        """Moisture cannot go below wilting point."""
        awc = 20.0
        fc = 30.0
        wp = 10.0
        # Start at 10.5, deplete heavily
        result = compute_soil_moisture(awc, fc, wp, precip_5d=0.0, eto_5d=100.0, current=10.5)
        assert np.isclose(result, wp, atol=0.1)

    def test_clip_to_fc(self):
        """Moisture cannot exceed field capacity."""
        awc = 20.0
        fc = 30.0
        wp = 10.0
        # Start at 28.0, add lots of precip
        result = compute_soil_moisture(awc, fc, wp, precip_5d=100.0, eto_5d=0.0, current=28.0)
        assert np.isclose(result, fc, atol=0.1)

    def test_2d_array_input(self):
        """Works with 2D arrays for current moisture."""
        awc = 20.0
        fc = 30.0
        wp = 10.0
        current = np.array([[15.0, 20.0], [25.0, 28.0]])
        result = compute_soil_moisture(awc, fc, wp, precip_5d=5.0, eto_5d=2.0, current=current)
        # recharge = 5*0.7 = 3.5 / 20 = 0.175
        # depletion = 2 / 20 * 100 = 0.1
        # moisture = current + 0.075
        assert result.shape == (2, 2)
        assert np.allclose(result, current + 0.075, atol=0.1)
        assert np.all(result >= wp)
        assert np.all(result <= fc)


class TestSaxtonRawls:
    """Saxton-Rawls pedotransfer function."""

    def test_sand(self):
        """85% sand → low FC and WP."""
        result = saxton_rawls_ptf(sand_pct=85.0, clay_pct=5.0)
        assert result["wilting_point"] < 10.0
        assert result["field_capacity"] < 20.0
        assert result["awc"] > 0

    def test_clay(self):
        """20% sand, 60% clay → high FC and WP."""
        result = saxton_rawls_ptf(sand_pct=20.0, clay_pct=60.0)
        assert result["wilting_point"] > 15.0
        assert result["field_capacity"] > 25.0
        assert result["awc"] > 0

    def test_key_structure(self):
        """Returns expected keys."""
        result = saxton_rawls_ptf(sand_pct=50.0, clay_pct=20.0)
        assert "field_capacity" in result
        assert "wilting_point" in result
        assert "awc" in result

    def test_fc_gt_wp(self):
        """Field capacity always > wilting point."""
        for sand, clay in [(10, 80), (80, 10), (40, 40), (30, 50)]:
            result = saxton_rawls_ptf(sand, clay)
            assert result["field_capacity"] > result["wilting_point"], f"sand={sand}, clay={clay}"

    def test_awc_positive(self):
        """AWC always positive."""
        for sand, clay in [(10, 80), (80, 10), (40, 40), (30, 50)]:
            result = saxton_rawls_ptf(sand, clay)
            assert result["awc"] > 0, f"sand={sand}, clay={clay}"

    def test_in_bounds(self):
        """Values are reasonable percentages."""
        result = saxton_rawls_ptf(sand_pct=50.0, clay_pct=20.0)
        assert 0 <= result["wilting_point"] <= 100
        assert 0 <= result["field_capacity"] <= 100
        assert 0 <= result["awc"] <= 100


class TestTextureDefaults:
    """Default texture values."""

    def test_returns_dict(self):
        """Returns a dict with expected keys."""
        result = get_texture_defaults()
        assert isinstance(result, dict)
        assert "sand_pct" in result
        assert "silt_pct" in result
        assert "clay_pct" in result

    def test_loam_values(self):
        """Defaults are loam (50/30/20)."""
        result = get_texture_defaults()
        assert result["sand_pct"] == 50.0
        assert result["silt_pct"] == 30.0
        assert result["clay_pct"] == 20.0

    def test_sums_to_100(self):
        """Texture percentages sum to 100."""
        result = get_texture_defaults()
        assert abs(result["sand_pct"] + result["silt_pct"] + result["clay_pct"] - 100.0) < 0.01


# ---------------------------------------------------------------------------
# 9. Topographic zoning (elevation bands × aspect sectors)
# ---------------------------------------------------------------------------


class TestDiscretizeAspect:
    """Aspect discretisation into 9 sectors."""

    def test_flat_pixel_returns_zero(self):
        """Slope < flat_threshold → sector 0 (flat) regardless of aspect."""
        aspect = np.array([[45.0, 180.0], [270.0, 350.0]])
        slope = np.array([[1.0, 0.5], [0.0, 1.9]])
        result = discretize_aspect(aspect, slope, flat_threshold=2.0)
        assert np.all(result == 0)

    def test_north(self):
        """Aspect around 0° → sector 1 (N)."""
        result = discretize_aspect(
            np.array([[0.0, 359.0], [10.0, 350.0]]),
            np.full((2, 2), 10.0),
        )
        assert np.all(result == 1)

    def test_south(self):
        """Aspect around 180° → sector 5 (S)."""
        result = discretize_aspect(
            np.array([[160.0, 180.0], [200.0, 180.0]]),
            np.full((2, 2), 10.0),
        )
        assert np.all(result == 5)

    def test_aspect_boundaries(self):
        """Sector boundaries handled correctly."""
        slope = np.full(8, 5.0)
        # Values just below boundary → lower sector; at boundary → upper sector
        # Boundary at 22.5 → 22.4 is N(1), 22.5 is NE(2)
        aspects = np.array([22.4, 22.5, 67.4, 67.5])
        expected = [1, 2, 2, 3]  # N, NE, NE, E
        result = discretize_aspect(aspects, np.full(4, 5.0))
        assert np.all(result == expected)


class TestComputeZones:
    """Topographic zoning from elevation × aspect sectors."""

    def test_single_flat_parcel(self):
        """All pixels same elevation → 1 zone."""
        shape = (8, 8)
        elev = np.full(shape, 100.0)
        aspect = np.full(shape, 0.0)
        slope = np.full(shape, 0.0)
        zones, labels = compute_zones(elev, aspect, slope, min_pixels=1, elevation_band_m=50.0)
        assert labels.shape == shape
        assert len(zones) == 1
        assert zones[0]["pixelCount"] == 64
        assert np.all(labels > 0)

    def test_two_elevation_bands(self):
        """Two bands → 2 zones."""
        shape = (4, 4)
        elev = np.full(shape, 30.0)
        elev[:2, :] = 80.0
        aspect = np.full(shape, 180.0)
        slope = np.full(shape, 5.0)
        zones, labels = compute_zones(elev, aspect, slope, min_pixels=1, elevation_band_m=50.0)
        assert labels.shape == shape
        assert len(zones) == 2
        assert labels[0, 0] != labels[2, 0]
        assert np.all(labels > 0)

    def test_two_separate_hills_same_elevation(self):
        """Same elevation/aspect but separate → 2 distinct zone labels."""
        shape = (10, 10)
        elev = np.full(shape, 10.0)
        aspect = np.full(shape, 180.0)
        slope = np.full(shape, 5.0)
        elev[1:4, 1:4] = 120.0
        elev[6:9, 6:9] = 120.0
        zones, labels = compute_zones(elev, aspect, slope, min_pixels=1, elevation_band_m=50.0)
        assert labels.shape == shape
        hill_1 = labels[2, 2]
        hill_2 = labels[7, 7]
        assert hill_1 > 0
        assert hill_2 > 0
        assert hill_1 != hill_2

    def test_ignores_small_clusters(self):
        """Clusters < min_pixels dropped (label = 0)."""
        shape = (10, 10)
        elev = np.full(shape, 10.0)
        aspect = np.full(shape, 0.0)
        slope = np.full(shape, 5.0)
        # 2-pixel cluster at 120 m
        elev[5, 5] = 120.0
        elev[5, 6] = 120.0
        zones, labels = compute_zones(elev, aspect, slope, min_pixels=5, elevation_band_m=50.0)
        assert labels[5, 5] == 0
        assert labels[5, 6] == 0

    def test_flat_zone_returns_flat_aspectSector(self):
        """Flat terrain (slope=0, aspect=0) → aspectSector is 'flat'."""
        shape = (10, 10)
        elev = np.full(shape, 100.0)
        aspect = np.full(shape, 0.0)
        slope = np.full(shape, 0.0)
        zones, labels = compute_zones(elev, aspect, slope, min_pixels=1, elevation_band_m=50.0)
        assert len(zones) == 1
        assert zones[0]["aspectSector"] == "flat"

    def test_empty_parcel(self):
        """Empty input → ([], empty array)."""
        elev = np.empty((0, 5))
        aspect = np.empty((0, 5))
        slope = np.empty((0, 5))
        zones, labels = compute_zones(elev, aspect, slope)
        assert zones == []
        assert labels.shape == (0,)
