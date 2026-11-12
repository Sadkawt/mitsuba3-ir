"""
Solar geometry and clear-sky irradiance.

* Solar position: the algorithm of Michalsky (1988), Solar Energy 40(3),
  227-235 (accuracy ~0.01 deg between 1950 and 2050), as used by NREL's
  SOLPOS.
* Clear-sky irradiance: Bird & Hulstrom (1981), SERI/TR-642-761, "A
  simplified clear sky model for direct and diffuse insolation on horizontal
  surfaces"; inputs are pressure, aerosol optical depths at 380/500 nm,
  precipitable water, ozone and ground albedo.
* Clouds: Kasten & Czeplak (1980), Solar Energy 24(2), 177-189: global
  irradiance reduced by (1 - 0.75 N^3.4), diffuse fraction 0.3 + 0.7 N^2.
"""

import datetime as _dt
import numpy as np

#: Nominal total solar irradiance at 1 AU [W m^-2] (IAU 2015)
SOLAR_CONSTANT = 1361.0


def day_of_year(date):
    """Day of year (1-366) of a ``datetime.date`` / ``datetime`` / 'YYYY-MM-DD'."""
    if isinstance(date, str):
        date = _dt.date.fromisoformat(date)
    return date.timetuple().tm_yday


def extraterrestrial_irradiance(doy):
    """Normal extraterrestrial irradiance [W m^-2] with the Sun-Earth distance correction."""
    return SOLAR_CONSTANT * (1.0 + 0.033 * np.cos(2.0 * np.pi * (np.asarray(doy) - 3) / 365.0))


def julian_date(dt_utc):
    """Julian date of a naive UTC ``datetime``."""
    y, m, d = dt_utc.year, dt_utc.month, dt_utc.day
    h = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5 + h / 24.0


def solar_position(latitude_deg, longitude_deg, dt_utc):
    """
    Solar elevation and azimuth (degrees) for a naive UTC ``datetime`` (or an
    array-like of them), following Michalsky (1988). Azimuth is measured
    clockwise from north; longitude is positive east. Also returns the
    declination [deg] and the equation of time [minutes].

    Returns
    -------
    dict with 'elevation', 'azimuth', 'declination', 'equation_of_time',
    'hour_angle' (all floats or arrays in degrees / minutes).
    """
    single = not isinstance(dt_utc, (list, tuple, np.ndarray))
    dts = [dt_utc] if single else list(dt_utc)
    jd = np.array([julian_date(t) for t in dts])
    hour = np.array([t.hour + t.minute / 60.0 + t.second / 3600.0 for t in dts])

    delta = jd - 2451545.0
    mnlong = np.mod(280.460 + 0.9856474 * delta, 360.0)
    mnanom = np.deg2rad(np.mod(357.528 + 0.9856003 * delta, 360.0))
    eclong = np.deg2rad(np.mod(mnlong + 1.915 * np.sin(mnanom) + 0.020 * np.sin(2 * mnanom), 360.0))
    oblqec = np.deg2rad(23.439 - 0.0000004 * delta)

    ra = np.arctan2(np.cos(oblqec) * np.sin(eclong), np.cos(eclong))
    ra = np.mod(ra, 2 * np.pi)
    dec = np.arcsin(np.sin(oblqec) * np.sin(eclong))

    gmst = np.mod(6.697375 + 0.0657098242 * delta + hour, 24.0)
    lmst = np.mod(gmst + longitude_deg / 15.0, 24.0)
    ha = np.deg2rad(lmst * 15.0) - ra
    ha = np.where(ha < -np.pi, ha + 2 * np.pi, ha)
    ha = np.where(ha > np.pi, ha - 2 * np.pi, ha)

    lat = np.deg2rad(latitude_deg)
    el = np.arcsin(np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha))
    az = np.arcsin(np.clip(-np.cos(dec) * np.sin(ha) / np.maximum(np.cos(el), 1e-9), -1.0, 1.0))
    elc = np.arcsin(np.sin(dec) / np.maximum(np.sin(lat), 1e-9)) if abs(latitude_deg) > 1e-9 else np.full_like(el, np.pi / 2)
    cond = el >= elc
    az = np.where(cond, np.pi - az, az)
    az = np.where((~cond) & (ha > 0), az + 2 * np.pi, az)
    az = np.mod(az, 2 * np.pi)

    # Equation of time [min]: apparent solar time - mean solar time
    # Equation of time [min] = 4 min/deg * (mean longitude - right ascension)
    eot = 4.0 * (mnlong - np.rad2deg(ra))
    eot = (eot + 720.0) % 1440.0 - 720.0

    out = dict(elevation=np.rad2deg(el), azimuth=np.rad2deg(az), declination=np.rad2deg(dec),
               equation_of_time=eot, hour_angle=np.rad2deg(ha))
    if single:
        out = {k: float(v[0]) for k, v in out.items()}
    return out


def sun_direction(elevation_deg, azimuth_deg, up=(0, 0, 1), north=(0, 1, 0)):
    """
    Unit vector *towards the sun* in world coordinates for a scene whose
    zenith is ``up`` and whose geographic north is ``north`` (east = north x up).
    Azimuth is clockwise from north.
    """
    up = np.asarray(up, dtype=np.float64)
    up /= np.linalg.norm(up)
    north = np.asarray(north, dtype=np.float64)
    north = north - up * (north @ up)
    north /= np.linalg.norm(north)
    east = np.cross(north, up)
    el = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    d = (np.cos(el) * np.sin(az))[..., None] * east + (np.cos(el) * np.cos(az))[..., None] * north \
        + np.sin(el)[..., None] * up
    return d


def air_mass(zenith_deg):
    """Relative optical air mass (Kasten & Young 1989)."""
    z = np.asarray(zenith_deg, dtype=np.float64)
    zc = np.minimum(z, 90.0)
    return 1.0 / (np.cos(np.deg2rad(zc)) + 0.50572 * (96.07995 - zc) ** -1.6364)


def bird_clear_sky(zenith_deg, doy, pressure_hPa=1013.25, aod500=0.1, aod380=None,
                   precipitable_water_cm=1.5, ozone_cm=0.3, albedo=0.2):
    """
    Bird & Hulstrom (1981) clear-sky model.

    Returns (DNI, DHI, GHI) in W m^-2: direct normal, diffuse horizontal and
    global horizontal irradiance (zero for the sun below the horizon).
    """
    z = np.asarray(zenith_deg, dtype=np.float64)
    up_sun = z < 90.0
    cosz = np.cos(np.deg2rad(np.minimum(z, 89.999)))
    AM = 1.0 / (cosz + 0.15 * (93.885 - np.minimum(z, 89.999)) ** -1.25)
    AMp = AM * pressure_hPa / 1013.25
    E0 = extraterrestrial_irradiance(doy)
    if aod380 is None:
        aod380 = 1.3 * aod500         # typical Angstrom exponent ~1

    T_r = np.exp(-0.0903 * AMp ** 0.84 * (1.0 + AMp - AMp ** 1.01))
    X_o = ozone_cm * AM
    T_o = 1.0 - 0.1611 * X_o * (1.0 + 139.48 * X_o) ** -0.3035 \
        - 0.002715 * X_o / (1.0 + 0.044 * X_o + 0.0003 * X_o ** 2)
    T_um = np.exp(-0.0127 * AMp ** 0.26)
    X_w = precipitable_water_cm * AM
    T_w = 1.0 - 2.4959 * X_w / ((1.0 + 79.034 * X_w) ** 0.6828 + 6.385 * X_w)
    tau_a = 0.2758 * aod380 + 0.35 * aod500
    T_a = np.exp(-tau_a ** 0.873 * (1.0 + tau_a - tau_a ** 0.7088) * AM ** 0.9108)
    T_aa = 1.0 - 0.1 * (1.0 - AM + AM ** 1.06) * (1.0 - T_a)
    T_as = T_a / T_aa
    Ba = 0.84

    dni = 0.9662 * E0 * T_r * T_o * T_um * T_w * T_a
    r_s = 0.0685 + (1.0 - Ba) * (1.0 - T_as)
    dhi = 0.79 * E0 * cosz * T_o * T_um * T_w * T_aa * (0.5 * (1.0 - T_r) + Ba * (1.0 - T_as)) \
        / (1.0 - AM + AM ** 1.02)
    ghi = (dni * cosz + dhi) / (1.0 - albedo * r_s)
    dhi = ghi - dni * cosz
    dni = np.where(up_sun, dni, 0.0)
    dhi = np.where(up_sun, np.maximum(dhi, 0.0), 0.0)
    ghi = np.where(up_sun, ghi, 0.0)
    return dni, dhi, ghi


def cloud_correction(dni, dhi, ghi, zenith_deg, cloud_cover):
    """
    Kasten & Czeplak (1980) all-sky correction for cloud cover N (0-1, okta/8):
    GHI_cloudy = GHI_clear (1 - 0.75 N^3.4), diffuse fraction 0.3 + 0.7 N^2.
    The direct component follows from the remainder.
    """
    N = float(np.clip(cloud_cover, 0.0, 1.0))
    if N <= 0.0:
        return dni, dhi, ghi
    cosz = np.cos(np.deg2rad(np.minimum(np.asarray(zenith_deg, dtype=np.float64), 89.999)))
    ghi_c = ghi * (1.0 - 0.75 * N ** 3.4)
    frac = min(0.3 + 0.7 * N ** 2, 1.0)
    dhi_c = np.maximum(ghi_c * frac, dhi * (1.0 - 0.75 * N ** 3.4))
    dhi_c = np.minimum(dhi_c, ghi_c)
    dni_c = np.where(cosz > 1e-3, (ghi_c - dhi_c) / np.maximum(cosz, 1e-3), 0.0)
    return np.maximum(dni_c, 0.0), dhi_c, ghi_c
