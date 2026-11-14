"""Slope radiation model (Allen, Trezza & Tasumi 2006, Agric. For. Meteorol. 139:55-73).

The paper prints no numeric worked example: its Table 4 reports error statistics
against measurements (not model outputs for given inputs) and its results are
figures (Figs. 1-6) without tabulated values. So there are NO "published" test
vectors here. The model is checked by:
  * identities the paper states (Eq. 4/35 horizontal reduction, Eq. 32 limits,
    Eq. 39-42 closure, Eq. 3 vs Eq. 14);
  * brute-force numerical integration of Eq. 3 (the quantity Eqs. 5/6/50/51 and
    Appendix A compute analytically), including slopes with two beam periods;
  * the qualitative behaviour the paper describes in Section 6 (steep
    north-facing slope: no beam in winter, two beam periods in summer).
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from app import slope_radiation as sr

APP = Path(sr.__file__).parent
LAT = 42.8
ELEV = 500.0
EA = 1.0


def _numeric_ra_mean(doy, lat_deg, slope_deg, aspect_compass):
    """24-h mean extraterrestrial W m-2 on the slope by midpoint integration of Eq. 3 (Eq. 1)."""
    phi, s = math.radians(lat_deg), math.radians(slope_deg)
    g = math.radians(aspect_compass - 180.0)
    d = sr._declination(doy)
    n = 200_000
    w = (np.arange(n) + 0.5) / n * 2 * np.pi - np.pi
    cos_t = (
        np.sin(d) * np.sin(phi) * np.cos(s)
        - np.sin(d) * np.cos(phi) * np.sin(s) * np.cos(g)
        + np.cos(d) * np.cos(phi) * np.cos(s) * np.cos(w)
        + np.cos(d) * np.sin(phi) * np.sin(s) * np.cos(g) * np.cos(w)
        + np.cos(d) * np.sin(g) * np.sin(s) * np.sin(w)
    )
    cos_h = np.sin(d) * np.sin(phi) + np.cos(d) * np.cos(phi) * np.cos(w)  # Eq. 4
    beam = np.where((cos_t > 0) & (cos_h > 0), cos_t, 0.0)
    return sr.GSC * sr._inv_d2(doy) * beam.sum() * (2 * np.pi / n) / (2 * np.pi)


def _analytic_ra_mean(doy, lat_deg, slope_deg, aspect_compass):
    phi, s = math.radians(lat_deg), math.radians(slope_deg)
    g = math.radians(aspect_compass - 180.0)
    d = sr._declination(doy)
    periods = sr.integration_periods(d, phi, s, g)
    return sr._extraterrestrial_slope_mean(d, phi, s, g, sr._inv_d2(doy), periods) if periods else 0.0


def _rs(doy, slope, aspect, rs=150.0, lat=LAT):
    return sr.slope_radiation_wm2(rs, lat, doy, slope, aspect, ELEV, EA)


# --- identity -------------------------------------------------------------------------

@pytest.mark.parametrize("doy", [1, 80, 172, 266, 355])
@pytest.mark.parametrize("rs", [20.0, 120.0, 250.0])
def test_flat_surface_equals_horizontal(doy, rs):
    """Slope 0: Eq. 3 -> Eq. 4, fB = 1, fi = 1, no reflected term, Eq. 41+42 close to tsw."""
    assert _rs(doy, 0.0, 0.0, rs) == pytest.approx(rs, rel=0.005)


def test_horizontal_extraterrestrial_matches_eq35_and_integral_of_eq4():
    doy = 100
    phi = math.radians(LAT)
    d = sr._declination(doy)
    ra_hor = sr._extraterrestrial_horizontal_mean(d, phi, sr._sunset_angle(d, phi), sr._inv_d2(doy))
    assert ra_hor == pytest.approx(_numeric_ra_mean(doy, LAT, 0.0, 0.0), rel=1e-4)


def test_sky_view_factor_limits_eq32():
    assert sr.sky_view_factor(0.0) == pytest.approx(1.0)
    assert sr.sky_view_factor(math.pi / 2) == pytest.approx(0.75 - 0.25)  # 0.5 s / pi = 0.25


def test_eq14_equals_eq3():
    d, phi, s, g, w = 0.2, math.radians(42.8), 0.5, 1.1, -0.7
    a, b, c = sr._abc(d, phi, s, g)
    eq3 = (
        math.sin(d) * math.sin(phi) * math.cos(s) - math.sin(d) * math.cos(phi) * math.sin(s) * math.cos(g)
        + math.cos(d) * math.cos(phi) * math.cos(s) * math.cos(w)
        + math.cos(d) * math.sin(phi) * math.sin(s) * math.cos(g) * math.cos(w)
        + math.cos(d) * math.sin(g) * math.sin(s) * math.sin(w)
    )
    assert sr._cos_theta(a, b, c, w) == pytest.approx(eq3, abs=1e-12)


def test_eq22_reduces_to_eq26_on_the_horizontal():
    doy = 150
    phi = math.radians(LAT)
    d = sr._declination(doy)
    ws = sr._sunset_angle(d, phi)
    periods = sr.integration_periods(d, phi, 0.0, 0.0)
    assert periods == [(pytest.approx(-ws), pytest.approx(ws))]
    assert sr._sin_beta24(d, phi, 0.0, 0.0, periods) == pytest.approx(sr._sin_beta24_horizontal(d, phi, ws), rel=1e-9)


# --- integration limits vs brute force (Eqs. 5, 6, 13, 44, 50, 51, Appendix A) -----------

@pytest.mark.parametrize("doy", [1, 60, 100, 172, 220, 300, 355])
@pytest.mark.parametrize("lat", [35.0, 42.8, 50.0, 60.0])
@pytest.mark.parametrize("slope", [10.0, 30.0, 60.0, 90.0])
@pytest.mark.parametrize("aspect", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
def test_analytic_extraterrestrial_matches_numerical_integration(doy, lat, slope, aspect):
    got = _analytic_ra_mean(doy, lat, slope, aspect)
    want = _numeric_ra_mean(doy, lat, slope, aspect)
    assert got == pytest.approx(want, abs=0.05, rel=2e-3)


def test_steep_north_slope_has_two_beam_periods_in_summer_and_none_in_winter():
    """Paper Section 6: a vertical north-facing slope sees no beam in fall/winter and
    two separate beam periods per day in spring/summer (paper: doy 81-254 at 39.7/44 N)."""
    phi = math.radians(44.0)
    g = math.radians(180.0)  # compass north -> paper gamma = 180 deg
    summer = sr.integration_periods(sr._declination(172), phi, math.pi / 2, g)
    winter = sr.integration_periods(sr._declination(355), phi, math.pi / 2, g)
    assert len(summer) == 2
    assert winter == []


# --- behaviour ------------------------------------------------------------------------

def test_winter_south_beats_flat_beats_north():
    s, f, n = _rs(355, 20.0, 180.0, 60.0), _rs(355, 0.0, 0.0, 60.0), _rs(355, 20.0, 0.0, 60.0)
    assert s > f > n


def test_summer_ordering_at_42n_20deg():
    """On the summer solstice the paper's model gives a 20 deg slope less than the flat
    surface in every direction at 42.8 N (high sun: tilt only costs beam); south is still
    the best of the slopes and north the worst."""
    flat = _rs(172, 0.0, 0.0, 300.0)
    south, east, north = (_rs(172, 20.0, a, 300.0) for a in (180.0, 90.0, 0.0))
    assert south > east > north
    assert flat > north


def test_shaded_north_slope_in_winter_gets_diffuse_and_reflected_only():
    doy, rs, slope = 355, 60.0, 60.0
    phi = math.radians(LAT)
    assert sr.integration_periods(sr._declination(doy), phi, math.radians(slope), math.pi) == []
    got = _rs(doy, slope, 0.0, rs)
    d = sr._declination(doy)
    tsw = rs / sr._extraterrestrial_horizontal_mean(d, phi, sr._sunset_angle(d, phi), sr._inv_d2(doy))
    f_i = sr.sky_view_factor(math.radians(slope))
    kd = tsw - sr._k_b_actual(tsw)
    assert got == pytest.approx(rs * (f_i * kd / tsw + sr.DEFAULT_ALBEDO * (1 - f_i)), rel=1e-9)
    assert got > 0.0


def test_no_nan_or_negative_over_the_grid():
    for doy in range(1, 367):
        for lat in (35.0, 42.8, 50.0, 60.0):
            for slope in (0.0, 5.0, 20.0, 40.0, 60.0):
                for aspect in range(0, 360, 45):
                    v = _rs(doy, slope, float(aspect), 100.0, lat)
                    assert math.isfinite(v) and v >= 0.0, (doy, lat, slope, aspect, v)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        sr.slope_radiation_wm2(100.0, -10.0, 100, 10.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        sr.slope_radiation_wm2(float("nan"), 40.0, 100, 10.0, 0.0, 0.0, 1.0)


# --- scope: internal only --------------------------------------------------------------

def test_slope_model_is_used_only_by_the_zone_accumulator():
    users = sorted(
        p.name for p in APP.glob("*.py")
        if p.name != "slope_radiation.py"
        and re.search(r"slope_radiation|slope_radiation_wm2", p.read_text())
    )
    assert users == ["zone_accumulator.py"]


def test_old_correction_is_gone():
    assert "correct_solar_radiation" not in (APP / "downscaler.py").read_text()
