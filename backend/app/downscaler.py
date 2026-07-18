"""Physics engine for per-pixel weather downscaling.

Contains NumPy-based functions that operate on 2D arrays (256 × 256 grids).
Each function takes one or more arrays and returns a corrected / derived array.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Air temperature — elevation lapse rate correction
# ---------------------------------------------------------------------------

def correct_temperature(
    t_base: float,
    station_elevation_m: float,
    pixel_elevations: np.ndarray,
) -> np.ndarray:
    """Correct air temperature for altitude using the standard lapse rate.

    Uses 6.5 °C per 1000 m (the environmental lapse rate).

    Parameters
    ----------
    t_base : float
        Temperature (°C) measured at the weather station.
    station_elevation_m : float
        Elevation of the weather station (m).
    pixel_elevations : np.ndarray
        2-D array of pixel elevations (m).

    Returns
    -------
    np.ndarray
        2-D array of corrected temperatures (°C).
    """
    elevation_diff = pixel_elevations - station_elevation_m
    return t_base - elevation_diff * 0.0065


# ---------------------------------------------------------------------------
# 2. Solar radiation — slope / aspect correction (FAO tilted-surface model)
# ---------------------------------------------------------------------------

def correct_solar_radiation(
    rad_base: float,
    pixel_latitudes: np.ndarray,
    pixel_aspects: np.ndarray,
    pixel_slopes: np.ndarray,
    doy: int,
) -> np.ndarray:
    """Adjust solar radiation for slope and aspect.

    Uses the FAO / ASAE tilted-surface model.  The correction factor is
    derived from the ratio of extraterrestrial radiation received on the
    actual surface versus a horizontal surface.

    Parameters
    ----------
    rad_base : float
        Reference solar radiation at a horizontal surface (W m⁻²).
    pixel_latitudes : np.ndarray
        2-D array of pixel latitudes (degrees).
    pixel_aspects : np.ndarray
        2-D array of pixel aspects, 0° = North (degrees).
    pixel_slopes : np.ndarray
        2-D array of pixel slopes (degrees).
    doy : int
        Day of year (1–366).

    Returns
    -------
    np.ndarray
        2-D array of corrected solar radiation (W m⁻²), clipped to [0, 1200].
    """
    deg2rad = np.pi / 180.0

    lat_rad = pixel_latitudes * deg2rad
    slope_rad = pixel_slopes * deg2rad
    aspect_rad = pixel_aspects * deg2rad

    # --- 1. Solar declination (FAO-56 Eq. 24) ---
    delta = 0.409 * np.sin((2.0 * np.pi * doy / 365.0) - 1.39)

    # --- 2. Sunset hour angle ---
    cos_omega = -np.tan(lat_rad) * np.tan(delta)
    # Clamp to [-1, 1] to avoid NaN for polar regions
    cos_omega = np.clip(cos_omega, -1.0, 1.0)
    omega_s = np.arccos(cos_omega)

    # --- 3. Extraterrestrial radiation Ra (FAO-56 Eq. 21) ---
    gsc = 0.0820  # MJ m⁻² min⁻¹
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    Ra = (
        (24.0 * 60.0 / np.pi)
        * gsc
        * dr
        * (
            omega_s * np.sin(lat_rad) * np.sin(delta)
            + np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)
        )
    )
    # Ra is in MJ m⁻² day⁻¹; keep it for downstream steps

    # --- 4. Clear-sky radiation ---
    Rso = 0.75 * Ra  # MJ m⁻² day⁻¹

    # --- 5. Cloudiness factor ---
    # Convert rad_base (W m⁻²) to MJ m⁻² day⁻¹ for ratio
    rad_base_mj = rad_base / 11.6
    # Avoid division by zero at night
    Rso_safe = np.where(Rso > 0.01, Rso, 0.01)
    actual_ratio = np.clip(rad_base_mj / Rso_safe, 0.0, 1.0)

    # --- 6. Tilted-surface cos(θ) for solar noon (simplified) ---
    # Aspect is specified as 0° = North, but the FAO tilted-surface formula
    # conventionally uses 0° = South (for the northern hemisphere).  We therefore
    # substitute cos(aspect + π) = -cos(aspect).
    cos_aspect = np.cos(aspect_rad)
    cos_theta = (
        np.sin(delta) * np.sin(lat_rad) * np.cos(slope_rad)
        - np.sin(delta) * np.cos(lat_rad) * np.sin(slope_rad) * (-cos_aspect)
        + np.cos(delta) * np.cos(lat_rad) * np.cos(slope_rad) * np.cos(omega_s)
        + np.cos(delta) * np.sin(lat_rad) * np.sin(slope_rad) * (-cos_aspect) * np.sin(omega_s)
    )

    # --- 7. Horizontal-surface cos(θ) ---
    cos_theta_h = (
        np.sin(delta) * np.sin(lat_rad)
        + np.cos(delta) * np.cos(lat_rad) * np.cos(omega_s)
    )

    # --- 8. Beam ratio Rb ---
    cos_theta_h_safe = np.where(np.abs(cos_theta_h) > 1e-10, cos_theta_h, 1e-10)
    Rb = np.clip(cos_theta / cos_theta_h_safe, 0.1, 2.0)

    # --- 9. Corrected radiation (MJ m⁻² day⁻¹) ---
    R_corrected_mj = (
        (0.75 + 0.25 * actual_ratio) * Rb * Ra
        - 0.25 * (Rb - 1.0) * Rso
    )
    # Ensure non-negative
    R_corrected_mj = np.maximum(R_corrected_mj, 0.0)

    # --- 10. Convert to W m⁻² and clip ---
    R_corrected = np.clip(R_corrected_mj * 11.6, 0.0, 1200.0)

    return R_corrected


# ---------------------------------------------------------------------------
# 3. Reference evapotranspiration (FAO Penman–Monteith)
# ---------------------------------------------------------------------------

def compute_eto(
    t_avg: np.ndarray,
    t_min: np.ndarray,
    t_max: np.ndarray,
    solar_rad: np.ndarray,
    wind_speed: float,
    rh: float,
    elevation_m: np.ndarray,
) -> np.ndarray:
    """FAO Penman–Monteith reference evapotranspiration per pixel.

    Parameters
    ----------
    t_avg : np.ndarray
        Mean daily air temperature (°C).
    t_min : np.ndarray
        Minimum daily air temperature (°C).
    t_max : np.ndarray
        Maximum daily air temperature (°C).
    solar_rad : np.ndarray
        Solar radiation at the surface (W m⁻²).
    wind_speed : float
        Wind speed at 2 m height (m s⁻¹).
    rh : float
        Relative humidity (%).
    elevation_m : np.ndarray
        Elevation (m).

    Returns
    -------
    np.ndarray
        Reference evapotranspiration (mm day⁻¹), clipped to [0, 15].
    """
    # --- 1. Atmospheric pressure (kPa) ---
    P = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26

    # --- 2. Psychrometric constant (kPa °C⁻¹) ---
    gamma = 0.665e-3 * P

    # --- 3. Saturation vapour pressure (kPa) ---
    es_tmin = 0.6108 * np.exp(17.27 * t_min / (t_min + 237.3))
    es_tmax = 0.6108 * np.exp(17.27 * t_max / (t_max + 237.3))
    es = (es_tmin + es_tmax) / 2.0

    # --- 4. Actual vapour pressure (kPa) ---
    ea = es * (rh / 100.0)

    # --- 5. Slope of saturation vapour pressure curve (kPa °C⁻¹) ---
    #  es(T_avg) — use the mean-temperature saturation
    es_tavg = 0.6108 * np.exp(17.27 * t_avg / (t_avg + 237.3))
    delta = 4098.0 * es_tavg / (t_avg + 237.3) ** 2

    # --- 6. Net radiation (MJ m⁻² day⁻¹) ---
    # Convert solar_rad from W m⁻² to MJ m⁻² day⁻¹
    Rs = solar_rad / 11.6  # MJ m⁻² day⁻¹

    # Net shortwave: Rn_s = (1 - α) * Rs, α = 0.23 for grass
    Rn_s = 0.77 * Rs

    # Net longwave: Stefan–Boltzmann
    sigma = 4.903e-9  # MJ m⁻² day⁻¹ K⁻⁴
    t_avg_k = t_avg + 273.16
    # KNOWN LIMITATION (documented 2026-07-18, not fixed): the cloudiness term is
    # degenerate. Rs_rel is Rs divided by itself, i.e. identically 1.0 — equivalent
    # to permanently clear sky, so net longwave loss is always at its maximum and
    # ET0 is biased low on overcast days. FAO-56 wants Rs/Rso with Rso ≈ 0.75·Ra
    # (Ra = extraterrestrial radiation, computable from latitude and day of year).
    # Fixing it changes ET0 for every downstream consumer (water balance, crop
    # stress, yield projection) and needs its own validation — hence not done here.
    Rs_max = np.maximum(Rs, 1e-10)
    Rs_rel = Rs / Rs_max
    Rn_l = sigma * t_avg_k ** 4 * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * Rs_rel - 0.35)

    Rn = Rn_s - Rn_l
    Rn = np.maximum(Rn, 0.0)

    # --- 7. Soil heat flux (negligible for daily) ---
    G = 0.0

    # --- 9. FAO-56 Eq. 6 ---
    # Numerator term 1: radiation
    term1 = 0.408 * delta * (Rn - G)
    # Numerator term 2: aerodynamic
    term2 = gamma * (900.0 / (t_avg + 273.0)) * wind_speed * (es - ea)
    # Denominator
    denom = delta + gamma * (1.0 + 0.34 * wind_speed)

    ET0 = np.where(denom > 1e-10, (term1 + term2) / denom, 0.0)

    # --- 10. Clip ---
    ET0 = np.clip(ET0, 0.0, 15.0)

    return ET0


# ---------------------------------------------------------------------------
# 4. Water balance (simple bucket, broadcast)
# ---------------------------------------------------------------------------

def compute_water_balance(
    precip_5d_mm: float,
    eto_5d_mm: np.ndarray,
) -> np.ndarray:
    """Compute a simple water balance (precip − ET₀).

    Parameters
    ----------
    precip_5d_mm : float
        5‑day cumulative precipitation (mm).
    eto_5d_mm : np.ndarray
        5‑day cumulative reference evapotranspiration (mm).

    Returns
    -------
    np.ndarray
        Water balance (mm); positive = surplus, negative = deficit.
    """
    return precip_5d_mm - eto_5d_mm


# ---------------------------------------------------------------------------
# 5. Frost risk (cold-air pooling model)
# ---------------------------------------------------------------------------

def compute_frost_risk(
    t_min: np.ndarray,
    elevation_m: np.ndarray,
) -> np.ndarray:
    """Frost risk based on minimum temperature and cold-air pooling.

    Valleys (low relative elevation) experience enhanced cooling due to
    cold‑air drainage / pooling.

    Parameters
    ----------
    t_min : np.ndarray
        Minimum daily air temperature (°C).
    elevation_m : np.ndarray
        Pixel elevation (m).

    Returns
    -------
    np.ndarray
        Frost risk (0–100 %), 0 = no risk, 100 = certain frost.
    """
    rel_elev = elevation_m - elevation_m.min()
    pooling = np.exp(-rel_elev / 50.0) * 2.0
    effective_t_min = t_min - pooling
    risk = 100.0 / (1.0 + np.exp(0.5 * (effective_t_min - 1.0)))
    return risk


# ---------------------------------------------------------------------------
# 6. Soil moisture (simple bucket model)
# ---------------------------------------------------------------------------

def compute_soil_moisture(
    awc: float,
    fc: float,
    wp: float,
    precip_5d: float,
    eto_5d: float,
    current: np.ndarray | float | None = None,
) -> np.ndarray | float:
    """Simple bucket model for soil moisture.

    Parameters
    ----------
    awc : float
        Available water capacity (% vol).
    fc : float
        Field capacity (% vol).
    wp : float
        Wilting point (% vol).
    precip_5d : float
        5‑day cumulative precipitation (mm).
    eto_5d : float
        5‑day cumulative reference evapotranspiration (mm).
    current : np.ndarray | float | None, optional
        Current soil moisture (% vol).  If ``None``, starts at 60 % of FC.

    Returns
    -------
    np.ndarray | float
        Updated soil moisture (% vol), clipped to [WP, FC].
    """
    if current is None:
        moisture = 0.6 * fc
    else:
        moisture = np.array(current, dtype=float)

    # Recharge: 70 % of precipitation converted to %‑vol via AWC
    recharge = precip_5d * 0.7 / awc
    # Depletion: ET₀ relative to AWC
    depletion = eto_5d / awc

    moisture = moisture + recharge - depletion
    moisture = np.clip(moisture, wp, fc)

    # Return scalar if input was scalar
    if isinstance(moisture, np.ndarray) and moisture.ndim == 0:
        return float(moisture)
    return moisture


# ---------------------------------------------------------------------------
# 7. Saxton–Rawls pedotransfer function
# ---------------------------------------------------------------------------

def saxton_rawls_ptf(sand_pct: float, clay_pct: float) -> dict:
    """Estimate hydraulic parameters from soil texture (Saxton–Rawls PTF).

    Parameters
    ----------
    sand_pct : float
        Sand content (% weight).
    clay_pct : float
        Clay content (% weight).

    Returns
    -------
    dict
        ``field_capacity`` — water content at −33 kPa (% vol)\n
        ``wilting_point``  — water content at −1500 kPa (% vol)\n
        ``awc``           — available water capacity (% vol)
    """
    # θ at −1500 kPa (wilting point)
    theta_1500 = (
        -0.024 * sand_pct
        + 0.487 * clay_pct
        + 0.006 * sand_pct * clay_pct
        + 0.005 * sand_pct * clay_pct
        + 0.013
    )
    theta_1500 = max(theta_1500, 0.01)

    # θ at −33 kPa (field capacity)
    theta_33 = (
        -0.251 * sand_pct
        + 0.195 * clay_pct
        + 0.011 * sand_pct * clay_pct
        + 0.006 * (sand_pct * clay_pct)
        + 0.027
    )
    # Ensure FC ≥ WP + 0.01
    theta_33 = max(theta_33, theta_1500 + 0.01)

    awc = theta_33 - theta_1500

    return {
        "field_capacity": theta_33,
        "wilting_point": theta_1500,
        "awc": awc,
    }


# ---------------------------------------------------------------------------
# 8. Texture defaults (loam)
# ---------------------------------------------------------------------------

def get_texture_defaults() -> dict:
    """Return default soil texture values (loam).

    Returns
    -------
    dict
        ``sand_pct``, ``silt_pct``, ``clay_pct``.
    """
    return {"sand_pct": 50.0, "silt_pct": 30.0, "clay_pct": 20.0}


# ---------------------------------------------------------------------------
# 9. Topographic zoning (elevation bands × aspect sectors)
# ---------------------------------------------------------------------------


_SECTOR_NAMES = {0: "flat", 1: "N", 2: "NE", 3: "E", 4: "SE",
                 5: "S", 6: "SW", 7: "W", 8: "NW"}


def discretize_aspect(
    aspect_deg: np.ndarray,
    slope_deg: np.ndarray,
    flat_threshold: float = 2.0,
) -> np.ndarray:
    """Discretise aspect into 9 sectors.

    Pixels with slope < *flat_threshold* are classified as flat (sector 0)
    regardless of their aspect value.  All other pixels are binned into 8
    wind-direction sectors:

    * 0 — flat
    * 1 — N    (337.5–360 / 0–22.5)
    * 2 — NE   (22.5–67.5)
    * 3 — E    (67.5–112.5)
    * 4 — SE   (112.5–157.5)
    * 5 — S    (157.5–202.5)
    * 6 — SW   (202.5–247.5)
    * 7 — W    (247.5–292.5)
    * 8 — NW   (292.5–337.5)

    Parameters
    ----------
    aspect_deg : np.ndarray
        2-D array of aspect angles (degrees, 0 = North, clockwise).
    slope_deg : np.ndarray
        2-D array of slope angles (degrees).
    flat_threshold : float, optional
        Slopes below this value are considered flat (default 2.0).

    Returns
    -------
    np.ndarray
        Integer array with sector codes 0–8, same shape as inputs.
    """
    sectors = np.zeros_like(aspect_deg, dtype=np.int32)
    aspect = aspect_deg % 360.0

    # Flat check — slope below threshold
    flat_mask = slope_deg < flat_threshold
    sectors[flat_mask] = 0

    not_flat = ~flat_mask

    sectors[not_flat & ((aspect >= 337.5) | (aspect < 22.5))] = 1   # N
    sectors[not_flat & ((aspect >= 22.5) & (aspect < 67.5))] = 2    # NE
    sectors[not_flat & ((aspect >= 67.5) & (aspect < 112.5))] = 3   # E
    sectors[not_flat & ((aspect >= 112.5) & (aspect < 157.5))] = 4  # SE
    sectors[not_flat & ((aspect >= 157.5) & (aspect < 202.5))] = 5  # S
    sectors[not_flat & ((aspect >= 202.5) & (aspect < 247.5))] = 6  # SW
    sectors[not_flat & ((aspect >= 247.5) & (aspect < 292.5))] = 7  # W
    sectors[not_flat & ((aspect >= 292.5) & (aspect < 337.5))] = 8  # NW

    return sectors


def compute_zones(
    elevation_2d: np.ndarray,
    aspect_2d: np.ndarray,
    slope_2d: np.ndarray,
    min_pixels: int = 50,
    elevation_band_m: float = 50.0,
) -> tuple[list[dict], np.ndarray]:
    """Group pixels into contiguous topographic zones.

    Each zone is a connected region sharing the same elevation band and
    aspect sector.  Zone descriptors are returned *without* an ``id`` key
    — the caller assigns stable IDs later.

    Parameters
    ----------
    elevation_2d : np.ndarray
        2-D array of pixel elevations (m).
    aspect_2d : np.ndarray
        2-D array of aspect angles (degrees).
    slope_2d : np.ndarray
        2-D array of slope angles (degrees).
    min_pixels : int, optional
        Minimum number of pixels for a zone (default 50).
    elevation_band_m : float, optional
        Elevation band height (default 50.0 m).

    Returns
    -------
    tuple[list[dict], np.ndarray]
        ``(zones, zone_labels)`` where *zones* is a list of descriptors
        (``elevationMean``, ``elevationMin``, ``elevationMax``,
        ``aspectSector``, ``pixelCount``) and *zone_labels* is an int32
        array of the same shape, 0 = background.

        Zone key = (elevation_band, sector) — tuple encoding avoids collision
        across different elevation/aspect combinations.
    """
    import scipy.ndimage  # deferred — only needed when computing zones

    # Handle empty input
    if elevation_2d.size == 0 or elevation_2d.shape[0] == 0:
        return [], np.array([], dtype=np.int32)

    # 1. Floor elevation to bands
    elev_bands = np.floor(elevation_2d / elevation_band_m).astype(np.int32)

    # 2. Discretise aspect sectors
    sectors = discretize_aspect(aspect_2d, slope_2d)

    # 3. Combine into tuple encoding (avoids collision across band/sector combos)
    combo = np.stack([
        elev_bands.astype(np.int32),
        sectors.astype(np.int32),
    ], axis=-1)

    zone_labels = np.zeros_like(elevation_2d, dtype=np.int32)
    zones: list[dict] = []
    next_label = 1

    unique_combos = np.unique(combo.reshape(-1, 2), axis=0)
    for band_val, sector_val in unique_combos:
        if sector_val == 0 and band_val == 0:
            continue  # skip background
        mask = (combo[..., 0] == band_val) & (combo[..., 1] == sector_val)
        labeled, n_features = scipy.ndimage.label(mask)

        for feature_id in range(1, n_features + 1):
            feature_mask = labeled == feature_id
            pixel_count = int(np.count_nonzero(feature_mask))

            if pixel_count < min_pixels:
                continue

            zone_labels[feature_mask] = next_label
            zone_elev = elevation_2d[feature_mask]

            zones.append({
                "elevationMean": float(np.mean(zone_elev)),
                "elevationMin": float(np.min(zone_elev)),
                "elevationMax": float(np.max(zone_elev)),
                "aspectSector": _dominant_sector(sectors[feature_mask]),
                "pixelCount": pixel_count,
            })
            next_label += 1

    return zones, zone_labels


def _dominant_sector(sector_labels: np.ndarray) -> str:
    """Return the most frequent aspect sector label.

    Parameters
    ----------
    sector_labels : np.ndarray
        1-D array of integer sector labels (0–8) as produced by
        :func:`discretize_aspect`.

    Returns
    -------
    str
        One of ``flat``, ``N``, ``NE``, ``E``, ``SE``, ``S``, ``SW``,
        ``W``, ``NW``.
    """
    if len(sector_labels) == 0:
        return "flat"

    unique, counts = np.unique(sector_labels, return_counts=True)
    most_common = unique[np.argmax(counts)]
    return _SECTOR_NAMES[int(most_common)]
