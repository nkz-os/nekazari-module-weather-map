"""Daily global radiation on a uniform slope, translated from a horizontal measurement.

Allen, R.G., Trezza, R., Tasumi, M. (2006). Analytical integrated functions for
daily solar radiation on slopes. Agric. For. Meteorol. 139:55-73.

INTERNAL USE ONLY: feeds the zone attribute ``nkz:solarRadiationWm2``
("modelled from terrain"). It must not reach any public raster or ``/stats``
metric, whose solar radiation is the horizontal measurement.

Conventions (the paper's): latitude positive north, slope ``s`` in radians
(0 = horizontal), aspect ``gamma`` 0 = due south, +90 deg = west, -90 deg = east,
180 deg = north; hour angle ``w`` 0 at solar noon, negative in the morning.
Northern hemisphere only. The public wrapper takes a compass aspect
(0 = north, clockwise) and converts: ``gamma = aspect - 180``.

Equation numbers below are the paper's. Daily radiations are 24-h means in
W m-2 of surface.
"""

from __future__ import annotations

import math

GSC = 1367.0  # W m-2, solar constant (paper, Eq. 1)
DEFAULT_ALBEDO = 0.20  # paper, Table 3 (Golden site: sparse grass, light soil)
_TOL = 1e-6


def _declination(doy: int) -> float:
    """Solar declination (rad). The paper does not give it: FAO-56 Eq. 24."""
    return 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)


def _inv_d2(doy: int) -> float:
    """1/d^2, inverse squared earth-sun distance: Eq. 2 (d^2 = 1 / (1 + 0.033 cos))."""
    return 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)


def _pressure_kpa(elevation_m: float) -> float:
    """Atmospheric pressure (kPa) from elevation: FAO-56 Eq. 7 (as ASCE-EWRI)."""
    return 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26


def _sunset_angle(delta: float, phi: float) -> float:
    """Horizontal sunset hour angle (Eq. 8), with Appendix A step A limits.

    Polar day (sun never sets) gives pi, polar night (never rises) gives 0.
    """
    return math.acos(max(-1.0, min(1.0, -math.tan(delta) * math.tan(phi))))


def _abc(delta: float, phi: float, s: float, gamma: float) -> tuple[float, float, float]:
    """Coefficients of cos(theta) = -a + b cos(w) + c sin(w): Eqs. 11a-c, Eq. 14.

    The paper's Eq. 11a defines ``a`` as the negative of Eq. 3's constant term;
    Eq. 3 and Table 2 fix the sign (the minus of ``-a`` is lost in the printed
    Eq. 14).
    """
    a = math.sin(delta) * math.cos(phi) * math.sin(s) * math.cos(gamma) \
        - math.sin(delta) * math.sin(phi) * math.cos(s)
    b = math.cos(delta) * math.cos(phi) * math.cos(s) \
        + math.cos(delta) * math.sin(phi) * math.sin(s) * math.cos(gamma)
    c = math.cos(delta) * math.sin(s) * math.sin(gamma)
    return a, b, c


def _cos_theta(a: float, b: float, c: float, w: float) -> float:
    """Eq. 14 (equivalent to Eq. 3): cosine of the incidence angle."""
    return -a + b * math.cos(w) + c * math.sin(w)


def _wrap(w: float) -> float:
    """Angle in (-pi, pi]."""
    w = math.fmod(w, 2.0 * math.pi)
    if w > math.pi:
        w -= 2.0 * math.pi
    elif w <= -math.pi:
        w += 2.0 * math.pi
    return w


def integration_periods(
    delta: float, phi: float, s: float, gamma: float,
) -> list[tuple[float, float]]:
    """Hour-angle periods in which the sun's beam strikes the slope (Section 2.2, App. A).

    Candidate limits come from the quadratic solution Eq. 13a/13b (the same
    pair as Eq. 44a/44b). Each sine has two angles (``w`` and ``pi - w``); the
    candidates that satisfy cos(theta) = 0 (Eq. 14) are the true sunrise/sunset
    on the slope. Together with the horizontal sunrise/sunset of Eq. 8 they cut
    the day into intervals, and an interval counts when cos(theta) > 0 at its
    midpoint. This is what Appendix A steps B-D select (limits, sun never
    rises on the slope, and two beam periods per day when Eq. 7 holds) but as
    one root-and-sign test instead of the printed chain of inequalities, whose
    signs are lost in the PDF text layer. Verified against numerical
    integration in the tests. Returns ``[]`` when the slope never sees the
    beam, one period in the common case, two when the slope shades itself at
    midday.
    """
    ws = _sunset_angle(delta, phi)
    if ws <= 0.0:
        return []  # polar night: no beam on any slope (Appendix A step A)
    a, b, c = _abc(delta, phi, s, gamma)
    roots: list[float] = []
    den = b * b + c * c
    if den > 1e-12:
        # Argument of the radical limited to >= 0 (Appendix A step D.i).
        rad = math.sqrt(max(b * b + c * c - a * a, 0.0))
        for sin_w in ((a * c - b * rad) / den, (a * c + b * rad) / den):  # Eq. 13a, 13b
            sin_w = max(-1.0, min(1.0, sin_w))
            w0 = math.asin(sin_w)
            for w in (w0, _wrap(math.pi - w0)):
                if abs(_cos_theta(a, b, c, w)) < _TOL and -ws < w < ws:
                    roots.append(w)
    cuts = sorted({-ws, ws, *roots})
    periods: list[tuple[float, float]] = []
    for w1, w2 in zip(cuts, cuts[1:]):
        if w2 - w1 <= 1e-9:
            continue
        if _cos_theta(a, b, c, 0.5 * (w1 + w2)) > 0.0:
            if periods and abs(periods[-1][1] - w1) < 1e-9:
                periods[-1] = (periods[-1][0], w2)
            else:
                periods.append((w1, w2))
    return periods


def _f_params(periods: list[tuple[float, float]]) -> tuple[float, float, float, float, float]:
    """f1..f5 of Table 1 (sums over the periods when there are two)."""
    f1 = f2 = f3 = f4 = f5 = 0.0
    for w1, w2 in periods:
        f1 += math.sin(w2) - math.sin(w1)
        f2 += math.cos(w2) - math.cos(w1)
        f3 += w2 - w1
        f4 += math.sin(2 * w2) - math.sin(2 * w1)
        f5 += math.sin(w2) ** 2 - math.sin(w1) ** 2
    return f1, f2, f3, f4, f5


def _extraterrestrial_slope_mean(
    delta: float, phi: float, s: float, gamma: float, inv_d2: float,
    periods: list[tuple[float, float]],
) -> float:
    """24-h mean extraterrestrial radiation on the slope, W m-2: Eqs. 5, 6, 50, 51.

    Eq. 6 gives the integral in W rad m-2; dividing by 2*pi gives the 24-h mean
    that the paper reports (it is the same conversion as in Eq. 35).
    """
    f1, f2, f3, _, _ = _f_params(periods)
    integral = (
        math.sin(delta) * math.sin(phi) * math.cos(s) * f3
        - math.sin(delta) * math.cos(phi) * math.sin(s) * math.cos(gamma) * f3
        + math.cos(delta) * math.cos(phi) * math.cos(s) * f1
        + math.cos(delta) * math.sin(phi) * math.sin(s) * math.cos(gamma) * f1
        - math.cos(delta) * math.sin(s) * math.sin(gamma) * f2
    )
    return max(GSC * inv_d2 * integral / (2.0 * math.pi), 0.0)


def _extraterrestrial_horizontal_mean(delta: float, phi: float, ws: float, inv_d2: float) -> float:
    """24-h mean extraterrestrial radiation on a horizontal surface, W m-2: Eq. 35."""
    return GSC * inv_d2 / math.pi * (
        math.sin(delta) * math.sin(phi) * ws + math.cos(delta) * math.cos(phi) * math.sin(ws)
    )


def _sin_beta24(
    delta: float, phi: float, s: float, gamma: float, periods: list[tuple[float, float]],
) -> float:
    """Ra-weighted mean sine of the solar elevation over the day: Eq. 22 with Tables 1-2."""
    a, b, c = _abc(delta, phi, s, gamma)
    g = math.sin(delta) * math.sin(phi)
    h = math.cos(delta) * math.cos(phi)
    f1, f2, f3, f4, f5 = _f_params(periods)
    num = (b * g - a * h) * f1 - c * g * f2 + (0.5 * b * h - a * g) * f3 + 0.25 * b * h * f4 \
        + 0.5 * c * h * f5
    den = b * f1 - c * f2 - a * f3
    if den <= 1e-12:
        return 0.0
    return max(num / den, 0.0)  # "limited to >= 0" (Section 3.1)


def _sin_beta24_horizontal(delta: float, phi: float, ws: float) -> float:
    """Eq. 26: Eq. 22 for a horizontal surface."""
    g = math.sin(delta) * math.sin(phi)
    h = math.cos(delta) * math.cos(phi)
    den = 2.0 * (g * ws + h * math.sin(ws))
    if den <= 1e-12:
        return 0.0
    num = 2 * g * g * ws + 4 * g * h * math.sin(ws) + h * h * (ws + 0.5 * math.sin(2 * ws))
    return max(num / den, 0.0)


def _k_bo(sin_b: float, pressure_kpa: float, w_mm: float, kt: float = 1.0) -> float:
    """Clear-sky direct-beam clearness index: Eq. 17 (Kt = 1, clean air)."""
    sb = max(sin_b, 1e-6)
    return 0.98 * math.exp(-0.00146 * pressure_kpa / (kt * sb) - 0.075 * (w_mm / sb) ** 0.4)


def _k_b_actual(tsw: float) -> float:
    """Actual direct-beam index on the horizontal from the transmissivity: Eq. 41a-c."""
    if tsw >= 0.42:
        k = 1.56 * tsw - 0.55
    elif tsw > 0.175:
        k = 0.022 - 0.280 * tsw + 0.828 * tsw ** 2 + 0.765 * tsw ** 3
    else:
        k = 0.016 * tsw
    return max(0.0, min(k, tsw))


def sky_view_factor(slope_rad: float) -> float:
    """Isotropic diffuse reduction with slope: Eq. 32, f_i = 0.75 + 0.25 cos(s) - 0.5 s / pi."""
    return 0.75 + 0.25 * math.cos(slope_rad) - 0.5 * slope_rad / math.pi


def slope_radiation_wm2(
    rs_horizontal_wm2: float,
    latitude_deg: float,
    doy: int,
    slope_deg: float,
    aspect_deg: float,
    elevation_m: float,
    ea_kpa: float,
    albedo: float = DEFAULT_ALBEDO,
) -> float:
    """24-h mean global radiation on a slope from the horizontal measurement (W m-2 of slope).

    Eq. 38 with isotropic diffuse (f_ia = f_i, Eq. 32)::

        Rs = Rs_hor * ( fB * KBhor/tsw + fi * KDhor/tsw + albedo * (1 - fi) )

    - ``tsw = Rs_hor / Rahor`` (Eq. 39), ``Rahor`` from Eq. 35;
    - ``KBhor`` from Eq. 41, ``KDhor = tsw - KBhor`` (Eq. 42);
    - ``fB = (KBo Ra) / (KBohor Rahor)`` (Eq. 34): ``Ra`` on the slope from
      Eqs. 5/6 with the integration limits of Appendix A (``integration_periods``),
      ``KBo`` and ``KBohor`` from Eq. 17 with the weighted sin(beta) of Eqs. 22 / 26,
      the pressure from the elevation and precipitable water from Eq. 18.

    The value is the irradiance per m2 of the SLOPE surface; the per-horizontal-area
    reprojection of Eq. 43 is NOT applied. A fully shaded slope (no beam) gets only
    the diffuse and reflected terms. ``aspect_deg`` is compass (0 = north, clockwise).
    Northern hemisphere only.
    """
    if not 0.0 <= latitude_deg < 89.0:
        raise ValueError(f"latitude {latitude_deg} outside the supported northern range [0, 89)")
    if not 0.0 <= slope_deg <= 90.0:
        raise ValueError(f"slope {slope_deg} outside [0, 90] degrees")
    rs = float(rs_horizontal_wm2)
    if not math.isfinite(rs) or rs < 0.0:
        raise ValueError(f"horizontal radiation {rs_horizontal_wm2} is not a finite non-negative number")

    phi = math.radians(latitude_deg)
    delta = _declination(doy)
    inv_d2 = _inv_d2(doy)
    ws = _sunset_angle(delta, phi)
    ra_hor = _extraterrestrial_horizontal_mean(delta, phi, ws, inv_d2)
    if ra_hor <= 1e-9:
        return rs  # polar night: no sun, nothing to translate

    s = math.radians(slope_deg)
    gamma = math.radians(aspect_deg - 180.0)  # compass -> paper (0 = south, + = west)
    f_i = sky_view_factor(s)

    tsw = rs / ra_hor  # Eq. 39
    k_b_hor = _k_b_actual(tsw)  # Eq. 41
    k_d_hor = tsw - k_b_hor  # Eq. 42

    periods = integration_periods(delta, phi, s, gamma)
    ra_slope = _extraterrestrial_slope_mean(delta, phi, s, gamma, inv_d2, periods) if periods else 0.0
    if ra_slope > 0.0:
        pressure = _pressure_kpa(elevation_m)
        w_mm = 0.14 * ea_kpa * pressure + 2.1  # Eq. 18
        k_bo = _k_bo(_sin_beta24(delta, phi, s, gamma, periods), pressure, w_mm)
        k_bo_hor = _k_bo(_sin_beta24_horizontal(delta, phi, ws), pressure, w_mm)
        f_b = (k_bo * ra_slope) / (k_bo_hor * ra_hor)  # Eq. 34
    else:
        f_b = 0.0

    ratio = f_b * k_b_hor / tsw + f_i * k_d_hor / tsw + albedo * (1.0 - f_i) if tsw > 0 else 0.0
    return max(rs * ratio, 0.0)
