"""
Atmosphere helpers for infrared rendering: clear-sky emissivity, thermal sky
and air media, and line-by-line molecular transmittance from HITRAN.

See docs/ir/IR_TECHNICAL_REPORT.md, sections 6.4-6.6.
"""

import numpy as np

from .radiometry import band_range

#: Composition of dry air (volume mixing ratios) used for the HITRAN helper
#: (CO2 at 420 ppm, 2023-2024 global mean; CH4 1.9 ppm; N2O 0.33 ppm)
DRY_AIR_VMR = {'CO2': 420e-6, 'CH4': 1.9e-6, 'N2O': 0.33e-6, 'O3': 0.0}

#: HITRAN molecule ids and the isotopologue ids used (main isotopologue)
HITRAN_IDS = {'H2O': (1, 1), 'CO2': (2, 1), 'O3': (3, 1), 'N2O': (4, 1),
              'CO': (5, 1), 'CH4': (6, 1), 'O2': (7, 1)}


# ---------------------------------------------------------------------------
# Clear-sky emissivity
# ---------------------------------------------------------------------------

def sky_emissivity_idso(T_air, e_a, band='full'):
    """
    Effective clear-sky emissivity of Idso (1981), Water Resources Research
    17(2), 295-304, as a function of screen-level air temperature ``T_air``
    [K] and water vapour pressure ``e_a`` [hPa]:

    * ``'full'``      : eps = 0.70 + 5.95e-5 e_a exp(1500 / T)
    * ``'8-14'``      : eps = 0.24 + 2.98e-8 e_a^2 exp(3000 / T)
    * ``'10.5-12.5'`` : eps = 0.10 + 3.53e-8 e_a^2 exp(3000 / T)

    The band values are the emissivities of the atmospheric window regions
    and therefore much lower than the full-spectrum value (the H2O and CO2
    bands outside the window are nearly opaque). Use the ``'8-14'`` value
    as ``emissivity_zenith`` of a LWIR ``thermalsky``.
    """
    T = np.asarray(T_air, dtype=np.float64)
    e = np.asarray(e_a, dtype=np.float64)
    if band == 'full':
        eps = 0.70 + 5.95e-5 * e * np.exp(1500.0 / T)
    elif band in ('8-14', 'lwir'):
        eps = 0.24 + 2.98e-8 * e**2 * np.exp(3000.0 / T)
    elif band == '10.5-12.5':
        eps = 0.10 + 3.53e-8 * e**2 * np.exp(3000.0 / T)
    else:
        raise ValueError("band must be 'full', '8-14' or '10.5-12.5'")
    eps = np.clip(eps, 0.0, 1.0)
    return float(eps) if eps.ndim == 0 else eps


def saturation_vapour_pressure(T_air):
    """Saturation water vapour pressure over water [hPa] (Magnus formula, WMO 2008)."""
    t = np.asarray(T_air, dtype=np.float64) - 273.15
    es = 6.112 * np.exp(17.62 * t / (243.12 + t))
    return float(es) if es.ndim == 0 else es


def vapour_pressure(T_air, relative_humidity):
    """Water vapour pressure [hPa] from temperature [K] and relative humidity (0-1)."""
    e = float(relative_humidity) * saturation_vapour_pressure(T_air)
    return float(e) if np.ndim(e) == 0 else e


def thermalsky(T_air, emissivity_zenith=None, relative_humidity=0.5, band='8-14',
               model='plane_parallel', up=(0, 0, 1), sampling_weight=1.0, **extra):
    """
    ``thermalsky`` emitter dictionary. If ``emissivity_zenith`` is None it is
    computed with :func:`sky_emissivity_idso` for the given band from
    ``T_air`` and ``relative_humidity``. ``emissivity_zenith`` may also be a
    ``(wavelengths_nm, values)`` table.
    """
    if emissivity_zenith is None:
        emissivity_zenith = sky_emissivity_idso(T_air, vapour_pressure(T_air, relative_humidity), band)
    if isinstance(emissivity_zenith, (tuple, list)) and len(emissivity_zenith) == 2 \
            and np.ndim(emissivity_zenith[0]) == 1:
        lam, val = (np.asarray(v, dtype=np.float64) for v in emissivity_zenith)
        emissivity_zenith = {'type': 'irregular',
                             'wavelengths': ', '.join(f'{x:.6g}' for x in lam),
                             'values': ', '.join(f'{x:.6g}' for x in np.clip(val, 0, 1))}
    else:
        emissivity_zenith = float(emissivity_zenith)
    d = {'type': 'thermalsky', 'temperature': float(T_air),
         'emissivity_zenith': emissivity_zenith, 'model': model,
         'up': [float(v) for v in up], 'sampling_weight': float(sampling_weight)}
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# Air as an emissive medium
# ---------------------------------------------------------------------------

def air_medium(T_air, absorption, albedo=0.0, medium_id=None):
    """
    Homogeneous emissive medium for air.

    ``absorption`` is either a scalar absorption coefficient [1/m] or a
    ``(wavelengths_nm, sigma_a_per_m)`` table (e.g. from
    :func:`hitran_absorption`). The medium radiates sigma_a B(lambda, T_air)
    per unit length (volpath_ir). Scattering (aerosols) can be added through
    ``albedo``, in which case ``absorption`` is interpreted as the total
    extinction sigma_t.
    """
    if np.isscalar(absorption):
        sigma_t = float(absorption)
    else:
        lam, sig = (np.asarray(v, dtype=np.float64) for v in absorption)
        sigma_t = {'type': 'constvolume', 'value': {
            'type': 'irregular',
            'wavelengths': ', '.join(f'{x:.7g}' for x in lam),
            'values': ', '.join(f'{x:.6g}' for x in np.maximum(sig, 1e-12))}}
    d = {'type': 'homogeneous', 'sigma_t': sigma_t, 'albedo': float(albedo),
         'temperature': float(T_air)}
    if medium_id is not None:
        d['id'] = medium_id
    return d


def medium_box(medium_ref, p_min, p_max):
    """Axis-aligned box with a null BSDF filled with the referenced medium."""
    import mitsuba as mi
    p_min = np.asarray(p_min, dtype=np.float64)
    p_max = np.asarray(p_max, dtype=np.float64)
    c = 0.5 * (p_min + p_max)
    s = 0.5 * (p_max - p_min)
    return {'type': 'cube',
            'to_world': mi.ScalarTransform4f().translate(c.tolist()).scale(s.tolist()),
            'bsdf': {'type': 'null'},
            'interior': {'type': 'ref', 'id': medium_ref}}


# ---------------------------------------------------------------------------
# Water vapour continuum (empirical, 8-13 um window)
# ---------------------------------------------------------------------------

def rsb_continuum(lambda_nm, T_air=288.15, e_hPa=8.5, pressure_hPa=1013.25):
    """
    Water-vapour continuum absorption coefficient [1/m] in the 8-13 um
    window after Roberts, Selby & Biberman (1976), Appl. Opt. 15, 2085-2090
    (the LOWTRAN 3-5 continuum):

        k(nu) = C_s(nu, T) * rho_w * (e + gamma (P - e)),
        C_s(nu, T) = [a + b exp(-beta nu)] exp[T0 (1/T - 1/296)],

    with a = 4.18, b = 5578 cm^2 g^-1 atm^-1, beta = 7.87e-3 cm,
    T0 = 1800 K, gamma = 0.002 (foreign-to-self broadening ratio), rho_w the
    absolute humidity [g cm^-3], e the vapour pressure and P the total
    pressure [atm], nu in cm^-1. The fit is valid for 700-1250 cm^-1
    (8-14 um); outside that range the result is held at the boundary value
    and should not be trusted. The model has been superseded by MT_CKD
    (Mlawer et al. 2012) but reproduces the e_a^2-scaled, red-sloping window
    continuum to within ~20 % for typical conditions.
    """
    lam = np.asarray(lambda_nm, dtype=np.float64)
    nu = np.clip(1e7 / lam, 700.0, 1250.0)
    T = float(T_air)
    e_atm = float(e_hPa) / 1013.25
    P_atm = float(pressure_hPa) / 1013.25
    rho_w = float(e_hPa) * 100.0 / (461.5 * T) * 1e-3     # kg m^-3 -> g cm^-3 (x1e3 g/kg x1e-6 m^3/cm^3)
    C_s = (4.18 + 5578.0 * np.exp(-7.87e-3 * nu)) * np.exp(1800.0 * (1.0 / T - 1.0 / 296.0))
    k_cm = C_s * rho_w * (e_atm + 0.002 * (P_atm - e_atm))  # cm^-1
    return k_cm * 100.0                                      # m^-1


# ---------------------------------------------------------------------------
# HITRAN line-by-line absorption (optional dependency: hitran-api / hapi)
# ---------------------------------------------------------------------------

def hitran_absorption(band, T_air=288.15, pressure_hPa=1013.25, relative_humidity=0.5,
                      molecules=('H2O', 'CO2'), resolution_cm=0.05, line_profile='Voigt',
                      db_dir='hitran_db', vmr=None, continuum='rsb', verbose=False):
    """
    Molecular absorption coefficient sigma_a(lambda) [1/m] of air at sea
    level from HITRAN line-by-line data, via the HITRAN Application
    Programming Interface (``hapi``; Kochanov et al. 2016, JQSRT 177, 15-30).
    Line data are downloaded from hitran.org on first use and cached in
    ``db_dir``.

    Parameters
    ----------
    band : str or (lmin, lmax) in nm
    T_air, pressure_hPa, relative_humidity : ambient conditions
    molecules : HITRAN molecules to include (main isotopologues)
    resolution_cm : wavenumber grid step [cm^-1]. 0.05 resolves the ~0.07
          cm^-1 pressure-broadened half-width at 1 atm; band averages over
          a 1 km path change by < 0.2 % between 0.25 and 0.02.
    line_profile : 'Voigt' (default), 'Lorentz' or 'HT'
    vmr : dict of volume mixing ratios overriding :data:`DRY_AIR_VMR`
          (H2O is always taken from the humidity)
    continuum : 'rsb' (default) adds the empirical water-vapour continuum of
          Roberts, Selby & Biberman (1976) for wavelengths in 8-14 um
          (:func:`rsb_continuum`); None gives line absorption only.

    Returns
    -------
    (wavelengths_nm, sigma_a_per_m) sorted by increasing wavelength.

    **Limitations stated in the report (assumption A16):** line absorption
    is included within +-25 cm^-1 of each line; the water-vapour continuum
    is the legacy RSB fit (valid 8-14 um, ~20 % accuracy) rather than
    MT_CKD (Mlawer et al. 2012); aerosol extinction, O3 and the minor gases
    beyond those in :data:`DRY_AIR_VMR` are not included. For radiometric
    work at long range, pass a MODTRAN-type table to :func:`air_medium` /
    :func:`mitsuba.ir.sun` instead.
    """
    try:
        import hapi
    except ImportError as e:
        raise ImportError("hitran_absorption requires the 'hitran-api' package "
                          "(pip install hitran-api)") from e
    import os
    import contextlib
    import io

    lo, hi = band_range(band)
    numin, numax = 1e7 / hi, 1e7 / lo            # cm^-1
    vmr_all = dict(DRY_AIR_VMR)
    if vmr:
        vmr_all.update(vmr)
    e_a = vapour_pressure(T_air, relative_humidity)
    vmr_all['H2O'] = e_a / pressure_hPa
    p_atm = pressure_hPa / 1013.25

    os.makedirs(db_dir, exist_ok=True)
    silence = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with silence:
        hapi.db_begin(db_dir)
        nu_grid = None
        total = None
        for mol in molecules:
            mol_id, iso_id = HITRAN_IDS[mol]
            table = f'{mol}_{int(numin)}_{int(numax)}'
            if table not in hapi.tableList():
                hapi.fetch(table, mol_id, iso_id, numin - 25.0, numax + 25.0)
            x = vmr_all.get(mol, 0.0)
            if x <= 0.0:
                continue
            func = {'Voigt': hapi.absorptionCoefficient_Voigt,
                    'Lorentz': hapi.absorptionCoefficient_Lorentz,
                    'HT': hapi.absorptionCoefficient_HT}[line_profile]
            # absorption coefficient in cm^-1 for the pure gas scaled by its
            # partial pressure (Diluent: self + air broadening)
            nu, coef = func(SourceTables=table,
                            Environment={'T': float(T_air), 'p': float(p_atm)},
                            Diluent={'self': x, 'air': 1.0 - x},
                            OmegaStep=resolution_cm, OmegaRange=(numin, numax),
                            HITRAN_units=True)
            # HITRAN_units=True gives cm^2/molecule; multiply by number density
            n_mol = x * p_atm * 1.01325e5 / (1.380649e-23 * float(T_air)) * 1e-6   # molecules/cm^3
            coef = coef * n_mol          # cm^-1
            if total is None:
                nu_grid, total = nu, coef
            else:
                total = total + np.interp(nu_grid, nu, coef)
    if total is None:
        raise ValueError("no molecule with non-zero mixing ratio")
    lam_nm = 1e7 / nu_grid
    sigma_m = total * 100.0              # cm^-1 -> m^-1
    if continuum == 'rsb':
        in_window = (lam_nm >= 8000.0) & (lam_nm <= 14000.0)
        sigma_m = sigma_m + np.where(in_window, rsb_continuum(lam_nm, T_air, e_a, pressure_hPa), 0.0)
    elif continuum is not None:
        raise ValueError("continuum must be 'rsb' or None")
    order = np.argsort(lam_nm)
    return lam_nm[order], sigma_m[order]


def transmittance(sigma_table, path_length_m):
    """Beer-Lambert transmittance table (wavelengths_nm, tau) over ``path_length_m``."""
    lam, sig = (np.asarray(v, dtype=np.float64) for v in sigma_table)
    return lam, np.exp(-sig * float(path_length_m))


def band_average(table, band, weight=None):
    """Average of a (wavelengths_nm, values) table over a band (optionally weighted)."""
    lo, hi = band_range(band)
    lam, val = (np.asarray(v, dtype=np.float64) for v in table)
    grid = np.linspace(lo, hi, 20001)
    v = np.interp(grid, lam, val)
    w = np.ones_like(grid) if weight is None else np.interp(grid, *weight)
    return float(np.trapezoid(v * w, grid) / np.trapezoid(w, grid))
