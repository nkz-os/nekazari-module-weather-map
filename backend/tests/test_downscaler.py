"""Tests for the Weather Map downscaler physics engine."""

import pytest
import numpy as np

from app.downscaler import (
    correct_temperature,
    correct_solar_radiation,
    compute_eto,
    extraterrestrial_radiation,
    clear_sky_radiation,
    ea_from_rh,
    compute_water_balance,
    compute_frost_risk,
    compute_soil_moisture,
    saxton_rawls_ptf,
    discretize_aspect,
    compute_zones,
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
    """FAO-56 Penman-Monteith daily reference evapotranspiration."""

    @staticmethod
    def _eto(t_min=15.0, t_max=25.0, ea=1.2, u2=2.0, rs=20.0, rs_surf=None,
             z=100.0, lat=40.0, doy=180):
        shape = (2, 2)
        return compute_eto(
            np.full(shape, t_min), np.full(shape, t_max), ea, u2, rs,
            np.full(shape, rs if rs_surf is None else rs_surf),
            np.full(shape, z), np.full(shape, lat), doy,
        )

    def test_fao56_example18_et0(self):
        """FAO-56 Example 18 (Brussels, 6 July): ET0 = 3.88 mm/day."""
        result = compute_eto(
            np.array([12.3]), np.array([21.5]), 1.409, 2.078, 22.07,
            np.array([22.07]), np.array([100.0]), np.array([50.80]), 187,
        )
        assert float(result[0]) == pytest.approx(3.88, abs=0.01)

    def test_ra_example8(self):
        """FAO-56 Example 8: 20 deg S, 3 September -> Ra = 32.2 MJ m-2 d-1."""
        ra = extraterrestrial_radiation(np.array([-20.0]), 246)
        assert float(ra[0]) == pytest.approx(32.2, abs=0.1)

    def test_ra_example18(self):
        """FAO-56 Example 18: 50 deg 48' N (50.80), J = 187 -> Ra = 41.09 MJ m-2 d-1."""
        ra = extraterrestrial_radiation(np.array([50.80]), 187)
        assert float(ra[0]) == pytest.approx(41.09, abs=0.05)

    def test_rso_example18(self):
        """FAO-56 Example 18: Ra = 41.09, z = 100 m -> Rso = 30.90 MJ m-2 d-1."""
        rso = clear_sky_radiation(np.array([41.09]), np.array([100.0]))
        assert float(rso[0]) == pytest.approx(30.90, abs=0.01)

    def test_rso_example11(self):
        """FAO-56 Example 11: Ra = 25.1, sea level -> Rso = 0.75 Ra = 18.8."""
        rso = clear_sky_radiation(np.array([25.1]), np.array([0.0]))
        assert float(rso[0]) == pytest.approx(18.8, abs=0.05)

    def test_ea_array_high_pixel_keeps_vpd_non_negative(self):
        """ea derived per pixel from RH and the pixel's own Tmin/Tmax: es - ea >= 0 at +1000 m."""
        t_min = np.array([15.0, 15.0 - 6.5])
        t_max = np.array([25.0, 25.0 - 6.5])
        ea = ea_from_rh(t_min, t_max, 60.0)
        es = (0.6108 * np.exp(17.27 * t_max / (t_max + 237.3))
              + 0.6108 * np.exp(17.27 * t_min / (t_min + 237.3))) / 2.0
        assert np.all(ea <= es)
        assert np.all(es - ea >= 0)
        result = compute_eto(
            t_min, t_max, ea, 2.0, 20.0, np.full(2, 20.0),
            np.array([100.0, 1100.0]), np.full(2, 40.0), 180,
        )
        assert result.shape == (2,) and np.all(result >= 0)

    def test_rs_ratio_clipped(self):
        """Rs above Rso gives the same ET0 as Rs = Rso."""
        ra = extraterrestrial_radiation(np.array([40.0]), 180)
        rso = float(clear_sky_radiation(ra, np.array([100.0]))[0])
        at_rso = self._eto(rs=rso, rs_surf=rso)
        above = self._eto(rs=rso * 1.3, rs_surf=rso)
        np.testing.assert_allclose(above, at_rso)

    def test_overcast_floor(self):
        """Rs/Rso is floored at 0.25: Rs = 0 and Rs = 0.25 Rso agree in Rnl."""
        ra = extraterrestrial_radiation(np.array([40.0]), 180)
        rso = float(clear_sky_radiation(ra, np.array([100.0]))[0])
        zero = self._eto(rs=0.0, rs_surf=0.0)
        floor = self._eto(rs=0.25 * rso, rs_surf=0.0)
        np.testing.assert_allclose(zero, floor)

    def test_higher_wind_raises_et0_in_dry_air(self):
        low = self._eto(ea=0.5, u2=0.5)
        high = self._eto(ea=0.5, u2=5.0)
        assert np.all(high > low)

    def test_no_negative_et0_in_winter_high_latitude(self):
        result = self._eto(t_min=-15.0, t_max=-8.0, ea=0.2, rs=0.0, lat=70.0, doy=355)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

    def test_eto_bounds(self):
        result = self._eto()
        assert result.shape == (2, 2)
        assert np.all(result >= 0) and np.all(result <= 15)

    def test_eto_higher_with_more_radiation(self):
        low = self._eto(rs=8.0)
        high = self._eto(rs=22.0)
        assert np.all(high > low)

    def test_eto_low_when_humid(self):
        dry = self._eto(ea=0.6)
        humid = self._eto(ea=2.2)
        assert np.all(humid < dry)

    def test_eto_differs_with_elevation(self):
        assert np.all(self._eto(z=3000.0) != self._eto(z=0.0))

    def test_cap_logs_warning(self, caplog):
        with caplog.at_level("WARNING"):
            result = self._eto(t_min=35.0, t_max=50.0, ea=0.1, u2=15.0, rs=35.0, lat=0.0, doy=80)
        assert np.all(result <= 15.0)
        warns = [r for r in caplog.records if r.levelname == "WARNING" and "capped" in r.message]
        assert len(warns) == 1


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

    def test_ptf_loam_matches_saxton_rawls_table(self):
        p = saxton_rawls_ptf(40.0, 20.0, om_pct=2.5)
        assert p["field_capacity"] == pytest.approx(28, abs=1)
        assert p["wilting_point"] == pytest.approx(14, abs=1)
        assert p["awc"] == pytest.approx(14, abs=2)

    def test_ptf_orders_sand_loam_clay(self):
        sand = saxton_rawls_ptf(85.0, 5.0, om_pct=2.5)
        loam = saxton_rawls_ptf(40.0, 20.0, om_pct=2.5)
        clay = saxton_rawls_ptf(20.0, 60.0, om_pct=2.5)
        assert sand["field_capacity"] < loam["field_capacity"] < clay["field_capacity"]
        assert sand["wilting_point"] < loam["wilting_point"] < clay["wilting_point"]

    def test_sand(self):
        """85% sand → low FC and WP."""
        result = saxton_rawls_ptf(sand_pct=85.0, clay_pct=5.0, om_pct=2.5)
        assert result["wilting_point"] < 10.0
        assert result["field_capacity"] < 20.0
        assert result["awc"] > 0

    def test_clay(self):
        """20% sand, 60% clay → high FC and WP."""
        result = saxton_rawls_ptf(sand_pct=20.0, clay_pct=60.0, om_pct=2.5)
        assert result["wilting_point"] > 15.0
        assert result["field_capacity"] > 25.0
        assert result["awc"] > 0

    def test_key_structure(self):
        """Returns expected keys."""
        result = saxton_rawls_ptf(sand_pct=50.0, clay_pct=20.0, om_pct=2.5)
        assert "field_capacity" in result
        assert "wilting_point" in result
        assert "awc" in result

    def test_fc_gt_wp(self):
        """Field capacity always > wilting point."""
        for sand, clay in [(10, 80), (80, 10), (40, 40), (30, 50)]:
            result = saxton_rawls_ptf(sand, clay, om_pct=2.5)
            assert result["field_capacity"] > result["wilting_point"], f"sand={sand}, clay={clay}"

    def test_awc_positive(self):
        """AWC always positive."""
        for sand, clay in [(10, 80), (80, 10), (40, 40), (30, 50)]:
            result = saxton_rawls_ptf(sand, clay, om_pct=2.5)
            assert result["awc"] > 0, f"sand={sand}, clay={clay}"

    def test_in_bounds(self):
        """Values are reasonable percentages."""
        result = saxton_rawls_ptf(sand_pct=50.0, clay_pct=20.0, om_pct=2.5)
        assert 0 <= result["wilting_point"] <= 100
        assert 0 <= result["field_capacity"] <= 100
        assert 0 <= result["awc"] <= 100


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
