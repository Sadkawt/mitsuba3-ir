"""
Radiometric utilities for infrared rendering.

Units follow the convention of the IR extension (report, assumption A1):
wavelengths in nanometres, temperatures in kelvin, spectral radiance in
W m^-2 sr^-1 nm^-1, band radiance in W m^-2 sr^-1.
"""

import numpy as np

# Exact SI (2019) constants
H_PLANCK = 6.62607015e-34      # J s
C_LIGHT = 299792458.0          # m s^-1
K_BOLTZMANN = 1.380649e-23     # J K^-1
SIGMA_SB = 5.670374419e-8      # W m^-2 K^-4 (derived, exact to the digits given)
C1L = 2.0 * H_PLANCK * C_LIGHT**2            # first radiation constant (radiance form), W m^2 sr^-1
C2 = H_PLANCK * C_LIGHT / K_BOLTZMANN        # second radiation constant, m K

# Nominal solar parameters (IAU 2015 Resolution B3, Prsa et al. 2016)
T_SUN = 5772.0                 # K
R_SUN = 6.957e8                # m
AU = 1.495978707e11            # m
SUN_GEOMETRY = (R_SUN / AU)**2 # (R_sun / 1 AU)^2 = 2.163e-5; solar solid angle = pi * this

#: Standard infrared bands (nm). Ranges follow common EO/IR engineering usage
#: (Vollmer & Moellmann 2018 sec. 1.2; Rogalski 2011); 'lwir_bolometer' is the
#: typical response window of uncooled microbolometer cameras.
BANDS = {
    'swir':           (900.0, 1700.0),    # InGaAs
    'swir_ext':       (900.0, 2500.0),    # extended InGaAs / HgCdTe
    'mwir':           (3000.0, 5000.0),   # InSb, HgCdTe
    'lwir':           (8000.0, 14000.0),  # HgCdTe
    'lwir_bolometer': (7500.0, 13500.0),  # microbolometer
    'visible':        (380.0, 780.0),
}


def band_range(band):
    """Return (lambda_min, lambda_max) in nm for a band name or a 2-tuple."""
    if isinstance(band, str):
        try:
            return BANDS[band.lower()]
        except KeyError:
            raise ValueError(f"Unknown band '{band}'. Known bands: {sorted(BANDS)}")
    lo, hi = float(band[0]), float(band[1])
    if not (0 < lo < hi):
        raise ValueError(f"Invalid band range {band}")
    return lo, hi


def planck(lambda_nm, T):
    """
    Planck's law: blackbody spectral radiance B_lambda(lambda, T).

    Parameters
    ----------
    lambda_nm : array_like
        Wavelength(s) in nanometres.
    T : array_like
        Absolute temperature(s) in kelvin (broadcast against ``lambda_nm``).

    Returns
    -------
    ndarray
        Spectral radiance in W m^-2 sr^-1 nm^-1.
    """
    lam = np.asarray(lambda_nm, dtype=np.float64) * 1e-9
    T = np.asarray(T, dtype=np.float64)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        x = C2 / (lam * T)
        B = C1L / lam**5 / np.expm1(x) * 1e-9
    return np.where(np.isfinite(B), B, 0.0)


def planck_dT(lambda_nm, T):
    """Temperature derivative dB/dT in W m^-2 sr^-1 nm^-1 K^-1."""
    lam = np.asarray(lambda_nm, dtype=np.float64) * 1e-9
    T = np.asarray(T, dtype=np.float64)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        x = C2 / (lam * T)
        ex = np.exp(x)
        d = C1L / lam**5 * ex * x / (T * (ex - 1.0)**2) * 1e-9
    return np.where(np.isfinite(d), d, 0.0)


def _srf_on_grid(lam, srf):
    if srf is None:
        return np.ones_like(lam)
    srf_lam, srf_val = np.asarray(srf[0], dtype=np.float64), np.asarray(srf[1], dtype=np.float64)
    return np.interp(lam, srf_lam, srf_val, left=0.0, right=0.0)


def band_radiance(lambda_min, lambda_max, T, srf=None, n=4001):
    """
    Band-integrated blackbody radiance  int_{lmin}^{lmax} B(lambda, T) R(lambda) dlambda.

    ``srf`` is an optional (wavelengths_nm, response) pair; without it the
    response is 1 over the band. Returns W m^-2 sr^-1. ``T`` may be an array.
    """
    lam = np.linspace(float(lambda_min), float(lambda_max), int(n))
    R = _srf_on_grid(lam, srf)
    T = np.atleast_1d(np.asarray(T, dtype=np.float64))
    B = planck(lam[None, :], T[:, None]) * R[None, :]
    out = np.trapezoid(B, lam, axis=1)
    return out[0] if out.size == 1 else out


def apparent_temperature(L_band, lambda_min, lambda_max, srf=None,
                         T_range=(100.0, 3000.0), n_table=4096):
    """
    Invert the band-integrated Planck function: the temperature of a blackbody
    that produces band radiance ``L_band`` in the given band / response.

    This is the standard definition of apparent (brightness) temperature used
    by thermal cameras. Values outside the invertible range are clipped to
    ``T_range``. Works element-wise on arrays (e.g. a rendered image).
    """
    Ts = np.geomspace(T_range[0], T_range[1], int(n_table))
    Ls = band_radiance(lambda_min, lambda_max, Ts, srf=srf)
    L = np.asarray(L_band, dtype=np.float64)
    return np.interp(L, Ls, Ts, left=Ts[0], right=Ts[-1])


def boxcar_srf(lambda_min, lambda_max):
    """Ideal unit response over a band as a Mitsuba 'regular' spectrum dict."""
    lo, hi = band_range((lambda_min, lambda_max))
    return {'type': 'regular', 'wavelength_min': lo, 'wavelength_max': hi,
            'values': '1, 1'}


def srf_dict(wavelengths_nm, values):
    """Tabulated response as a Mitsuba 'irregular' spectrum dict."""
    w = np.asarray(wavelengths_nm, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if w.ndim != 1 or w.shape != v.shape or np.any(np.diff(w) <= 0):
        raise ValueError("wavelengths must be a strictly increasing 1-D array matching values")
    return {'type': 'irregular',
            'wavelengths': ', '.join(f'{x:.6g}' for x in w),
            'values': ', '.join(f'{x:.6g}' for x in v)}


def band_film(band, width, height, srf=None, name=None, rfilter='gaussian',
              component_format='float32', **extra):
    """
    Build a ``specfilm`` dictionary for one infrared band.

    Parameters
    ----------
    band : str or (lmin, lmax)
        Band name from :data:`BANDS` or explicit range in nm.
    width, height : int
    srf : optional (wavelengths_nm, response)
        Detector spectral response. Default: ideal unit response over the band.
    name : str, optional
        Channel name (default: band name or 'band').
    extra : passed through to the film (e.g. crop parameters).

    The resulting channel contains  int L_lambda R(lambda) dlambda  in
    W m^-2 sr^-1 (report, section 2.3 and 5).
    """
    lo, hi = band_range(band)
    if name is None:
        name = band if isinstance(band, str) else 'band'
    film = {'type': 'specfilm', 'width': int(width), 'height': int(height),
            'component_format': component_format,
            'rfilter': {'type': rfilter}}
    film[name] = boxcar_srf(lo, hi) if srf is None else srf_dict(*srf)
    film.update(extra)
    return film


def solar_irradiance_toa(lambda_nm, T_sun=T_SUN):
    """
    Top-of-atmosphere solar spectral irradiance (W m^-2 nm^-1) at 1 AU,
    modelled as a blackbody at ``T_sun`` seen under the solar solid angle:
    E(lambda) = pi B(lambda, T_sun) (R_sun / AU)^2.

    Integrated over all wavelengths this gives 1361 W m^-2 for T = 5772 K,
    i.e. the IAU nominal total solar irradiance. The blackbody model deviates
    from the measured spectrum (e.g. ASTM E490 / Gueymard 2004) by up to
    ~10 % in the SWIR; use a measured table through :func:`srf_dict` for
    higher fidelity.
    """
    return np.pi * planck(lambda_nm, T_sun) * SUN_GEOMETRY
