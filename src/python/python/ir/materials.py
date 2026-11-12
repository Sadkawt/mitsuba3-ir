"""
Infrared material models and data.

Two kinds of material descriptions are supported, matching the two kinds of
data available in the literature (report, section 3.4 and Phase 2):

1. **Optical constants (n, k)** for smooth or rough interfaces: metals via
   the Lorentz-Drude model of Rakic et al. (1998), and tabulated/dispersion
   data converted from the refractiveindex.info database (CC0). Files are
   shipped in ``resources/ir_data/ior`` with the suffix ``_ir`` and can be used
   with every conductor BSDF through ``'material': 'Al_ir'``.

2. **Hemispherical emissivity / reflectance spectra** (ASTER / ECOSTRESS
   spectral libraries) for Lambertian surfaces: a ``diffuse`` BSDF with
   reflectance 1 - epsilon(lambda), from which Kirchhoff's law recovers the
   measured emissivity exactly.
"""

import os
import re
import numpy as np

EV_TO_UM = 1.239841984                     # h c / e in eV um (exact SI: 1.239841984...)

# ---------------------------------------------------------------------------
# Lorentz-Drude model of Rakic, Djurisic, Elazar & Majewski (1998)
# ---------------------------------------------------------------------------

#: Lorentz-Drude parameters from Rakic et al., Appl. Opt. 37(22), 5271 (1998),
#: Table 1 ("LD" model). For each metal: plasma frequency omega_p [eV] and a
#: list of oscillators (f_j, Gamma_j [eV], omega_j [eV]); the first oscillator
#: (omega_0 = 0) is the free-electron (Drude) term.
RAKIC_LD = {
    'Ag': (9.01, [(0.845, 0.048, 0.0), (0.065, 3.886, 0.816), (0.124, 0.452, 4.481),
                  (0.011, 0.065, 8.185), (0.840, 0.916, 9.083), (5.646, 2.419, 20.29)]),
    'Au': (9.03, [(0.760, 0.053, 0.0), (0.024, 0.241, 0.415), (0.010, 0.345, 0.830),
                  (0.071, 0.870, 2.969), (0.601, 2.494, 4.304), (4.384, 2.214, 13.32)]),
    'Cu': (10.83, [(0.575, 0.030, 0.0), (0.061, 0.378, 0.291), (0.104, 1.056, 2.957),
                   (0.723, 3.213, 5.300), (0.638, 4.305, 11.18)]),
    'Al': (14.98, [(0.523, 0.047, 0.0), (0.227, 0.333, 0.162), (0.050, 0.312, 1.544),
                   (0.166, 1.351, 1.808), (0.030, 3.382, 3.473)]),
    'Be': (18.51, [(0.084, 0.035, 0.0), (0.031, 1.664, 0.100), (0.140, 3.395, 1.032),
                   (0.530, 4.454, 3.183), (0.130, 1.802, 4.604)]),
    'Cr': (10.75, [(0.168, 0.047, 0.0), (0.151, 3.175, 0.121), (0.150, 1.305, 0.543),
                   (1.149, 2.676, 1.970), (0.825, 1.335, 8.775)]),
    'Ni': (15.92, [(0.096, 0.048, 0.0), (0.100, 4.511, 0.174), (0.135, 1.334, 0.582),
                   (0.106, 2.178, 1.597), (0.729, 6.292, 6.089)]),
    'Pd': (9.72, [(0.330, 0.008, 0.0), (0.649, 2.950, 0.336), (0.121, 0.555, 0.501),
                  (0.638, 4.621, 1.659), (0.453, 3.236, 5.715)]),
    'Pt': (9.59, [(0.333, 0.080, 0.0), (0.191, 0.517, 0.780), (0.659, 1.838, 1.314),
                  (0.547, 3.668, 3.141), (3.576, 8.517, 9.249)]),
    'Ti': (7.29, [(0.148, 0.082, 0.0), (0.899, 2.276, 0.777), (0.393, 2.518, 1.545),
                  (0.187, 1.663, 2.509), (0.001, 1.762, 19.43)]),
    'W': (13.22, [(0.206, 0.064, 0.0), (0.054, 0.530, 1.004), (0.166, 1.281, 1.917),
                  (0.706, 3.332, 3.580), (2.590, 5.836, 7.498)]),
}


def drude_lorentz_eps(metal, lambda_nm):
    """Complex relative permittivity eps(omega) of the Lorentz-Drude model."""
    omega_p, oscillators = RAKIC_LD[metal]
    lam_um = np.asarray(lambda_nm, dtype=np.float64) * 1e-3
    omega = EV_TO_UM / lam_um                       # photon energy in eV
    eps = np.ones_like(omega, dtype=np.complex128)
    for j, (f, gamma, omega_j) in enumerate(oscillators):
        if j == 0:
            # Free-electron (Drude) term: 1 - Omega_p^2 / (omega^2 + i Gamma0 omega)
            eps -= f * omega_p**2 / (omega**2 + 1j * gamma * omega)
        else:
            # Bound-electron (Lorentz) terms
            eps += f * omega_p**2 / ((omega_j**2 - omega**2) - 1j * gamma * omega)
    return eps


def drude_lorentz_nk(metal, lambda_nm):
    """
    Refractive index n and extinction coefficient k of a metal from the
    Lorentz-Drude model (Rakic et al. 1998). Convention: n + i k = sqrt(eps).

    The model is an analytic fit valid from the far UV to ~12 um (the fit
    range of the paper) and extrapolates smoothly as a Drude metal beyond.
    """
    m = np.sqrt(drude_lorentz_eps(metal, lambda_nm))
    return m.real, m.imag


# ---------------------------------------------------------------------------
# Fresnel / emissivity utilities
# ---------------------------------------------------------------------------

def fresnel_conductor(cos_theta, n, k):
    """Unpolarized Fresnel reflectance of an interface air -> (n + i k)."""
    cos_theta = np.asarray(cos_theta, dtype=np.float64)
    m2 = (np.asarray(n, dtype=np.float64) + 1j * np.asarray(k, dtype=np.float64))**2
    sin2 = 1.0 - cos_theta**2
    t = np.sqrt(m2 - sin2 + 0j)
    rs = (cos_theta - t) / (cos_theta + t)
    rp = (m2 * cos_theta - t) / (m2 * cos_theta + t)
    return 0.5 * (np.abs(rs)**2 + np.abs(rp)**2)


def normal_emissivity(n, k):
    """Normal spectral emissivity of a smooth opaque surface, 1 - F(0) = 4n / ((n+1)^2 + k^2)."""
    n = np.asarray(n, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    return 4.0 * n / ((n + 1.0)**2 + k**2)


def hemispherical_emissivity(n, k, n_theta=256):
    """Hemispherical emissivity of a smooth opaque surface: 1 - 2 int F(theta) cos sin dtheta."""
    mu, w = np.polynomial.legendre.leggauss(int(n_theta))
    mu = 0.5 * (mu + 1.0)
    w = 0.5 * w
    n = np.atleast_1d(np.asarray(n, dtype=np.float64))
    k = np.atleast_1d(np.asarray(k, dtype=np.float64))
    F = fresnel_conductor(mu[None, :], n[:, None], k[:, None])
    eps = 1.0 - 2.0 * np.sum(F * mu[None, :] * w[None, :], axis=1)
    return eps[0] if eps.size == 1 else eps


# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------

#: IR materials shipped in resources/ir_data/ior (suffix "_ir"). Values: short
#: description, usable range (nm), data source. The "opaque" flag indicates
#: that the material should be modelled as an opaque interface
#: (conductor/roughconductor with complex IOR) rather than as a window.
IR_MATERIALS = {
    'Ag_ir':  dict(desc='Silver (Lorentz-Drude fit)', range=(300, 20000), opaque=True, src='Rakic 1998'),
    'Al_ir':  dict(desc='Aluminium (Lorentz-Drude fit)', range=(300, 20000), opaque=True, src='Rakic 1998'),
    'Au_ir':  dict(desc='Gold (Lorentz-Drude fit)', range=(300, 20000), opaque=True, src='Rakic 1998'),
    'Be_ir':  dict(desc='Beryllium (Lorentz-Drude fit)', range=(300, 20000), opaque=True, src='Rakic 1998'),
    'Cr_ir':  dict(desc='Chromium (Lorentz-Drude fit)', range=(300, 20000), opaque=True, src='Rakic 1998'),
    'Cu_ir':  dict(desc='Copper (Lorentz-Drude fit)', range=(300, 20000), opaque=True, src='Rakic 1998'),
    'Ni_ir':  dict(desc='Nickel (Lorentz-Drude fit)', range=(300, 20000), opaque=True, src='Rakic 1998'),
    'Pd_ir':  dict(desc='Palladium (Lorentz-Drude fit)', range=(300, 20000), opaque=True, src='Rakic 1998'),
    'Pt_ir':  dict(desc='Platinum (Lorentz-Drude fit)', range=(300, 20000), opaque=True, src='Rakic 1998'),
    'Ti_ir':  dict(desc='Titanium (Lorentz-Drude fit)', range=(300, 20000), opaque=True, src='Rakic 1998'),
    'W_ir':   dict(desc='Tungsten (Lorentz-Drude fit)', range=(300, 20000), opaque=True, src='Rakic 1998'),
    'Fe_ir':  dict(desc='Iron (polished)', range=(300, 20000), opaque=True, src='Querry 1985'),
    'Si_ir':  dict(desc='Crystalline silicon (float zone), 300 K', range=(300, 20000), opaque=False, src='Franta 2017'),
    'Ge_ir':  dict(desc='Germanium', range=(400, 14000), opaque=False, src='Amotchkina 2020 (<2 um), Burnett 2016 (2-14 um)'),
    'ZnSe_ir': dict(desc='Zinc selenide', range=(500, 20000), opaque=False, src='Querry 1987'),
    'ZnS_ir': dict(desc='Zinc sulfide', range=(300, 20000), opaque=False, src='Querry 1987'),
    'Sapphire_ir': dict(desc='Sapphire (alpha-Al2O3, ordinary ray)', range=(300, 20000), opaque=False, src='Querry 1985'),
    'CaF2_ir': dict(desc='Calcium fluoride', range=(300, 12000), opaque=False, src='Li 1980'),
    'BaF2_ir': dict(desc='Barium fluoride', range=(300, 20000), opaque=False, src='Querry 1987'),
    'NaCl_ir': dict(desc='Sodium chloride', range=(300, 20000), opaque=False, src='Querry 1987'),
    'SiO2glass_ir': dict(desc='Fused silica glass, 25 C', range=(300, 20000), opaque=False, src='Franta 2016'),
    'H2O_ir': dict(desc='Liquid water, 25 C', range=(300, 20000), opaque=True, src='Hale & Querry 1973'),
}


def _data_dir():
    """Directory holding the ior data (build tree or installed package)."""
    try:
        import mitsuba as mi
        p = mi.file_resolver().resolve('data/ior')
        if os.path.isdir(str(p)):
            return str(p)
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, '..', '..', 'data', 'ior'),
                 os.path.join(here, '..', '..', '..', '..', 'resources', 'ir_data', 'ior')):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError("Could not locate the 'data/ior' directory")


def load_spd(path):
    """Read a Mitsuba ``.spd`` file; returns (wavelengths_nm, values)."""
    lam, val = [], []
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            a, b = line.split()[:2]
            lam.append(float(a))
            val.append(float(b))
    return np.asarray(lam), np.asarray(val)


def load_ior(name):
    """Load (wavelengths_nm, n, k) of a material preset from data/ior."""
    d = _data_dir()
    lam_n, n = load_spd(os.path.join(d, f'{name}.eta.spd'))
    lam_k, k = load_spd(os.path.join(d, f'{name}.k.spd'))
    if lam_n.shape != lam_k.shape or not np.allclose(lam_n, lam_k):
        k = np.interp(lam_n, lam_k, k)
    return lam_n, n, k


def write_spd(path, wavelengths_nm, values, header_lines=()):
    """Write a Mitsuba ``.spd`` file with comment header lines."""
    with open(path, 'w') as f:
        for h in header_lines:
            f.write(f'# {h}\n')
        for w, v in zip(wavelengths_nm, values):
            f.write(f'{w:.6g} {v:.6g}\n')


# ---------------------------------------------------------------------------
# refractiveindex.info YAML conversion
# ---------------------------------------------------------------------------

def _rii_formula(kind, coeffs, lam_um):
    """Dispersion formulas of refractiveindex.info (lambda in micrometres)."""
    c = list(coeffs) + [0.0] * 17
    x = lam_um
    if kind == 1:      # Sellmeier
        n2 = 1.0 + c[0]
        for i in range(1, 17, 2):
            n2 = n2 + c[i] * x**2 / (x**2 - c[i + 1]**2)
        return np.sqrt(n2)
    if kind == 2:      # Sellmeier-2
        n2 = 1.0 + c[0]
        for i in range(1, 17, 2):
            n2 = n2 + c[i] * x**2 / (x**2 - c[i + 1])
        return np.sqrt(n2)
    if kind == 3:      # Polynomial
        n2 = c[0]
        for i in range(1, 17, 2):
            n2 = n2 + c[i] * x**c[i + 1]
        return np.sqrt(n2)
    if kind == 4:      # RefractiveIndex.INFO
        n2 = (c[0] + c[1] * x**c[2] / (x**2 - c[3]**c[4])
              + c[5] * x**c[6] / (x**2 - c[7]**c[8]))
        for i in range(9, 17, 2):
            n2 = n2 + c[i] * x**c[i + 1]
        return np.sqrt(n2)
    if kind == 5:      # Cauchy
        n = c[0]
        for i in range(1, 11, 2):
            n = n + c[i] * x**c[i + 1]
        return n
    if kind == 6:      # Gases
        n = 1.0 + c[0]
        for i in range(1, 11, 2):
            n = n + c[i] / (c[i + 1] - x**-2)
        return n
    if kind == 7:      # Herzberger
        n = c[0] + c[1] / (x**2 - 0.028) + c[2] * (1.0 / (x**2 - 0.028))**2
        n = n + c[3] * x**2 + c[4] * x**4 + c[5] * x**6
        return n
    raise NotImplementedError(f"refractiveindex.info formula {kind} is not supported")


def rii_to_nk(yaml_path, lambda_nm=None):
    """
    Convert a refractiveindex.info database page (YAML) to (wavelengths_nm, n, k).

    Supports 'tabulated nk', 'tabulated n', 'tabulated k' and 'formula 1-7'
    entries. When ``lambda_nm`` is given, the data is interpolated onto it
    (formulas are evaluated only inside their stated range; outside, the
    nearest value is held). Returns also the reference text.
    """
    import yaml
    with open(yaml_path) as f:
        doc = yaml.safe_load(f)
    ref = re.sub(r'<[^>]+>', '', doc.get('REFERENCES', '')).strip()
    ref = ' '.join(ref.split())

    n_lam = n_val = k_lam = k_val = None
    formulas = []
    for entry in doc['DATA']:
        t = entry['type']
        if t.startswith('tabulated'):
            arr = np.array([[float(v) for v in line.split()]
                            for line in entry['data'].strip().splitlines()])
            lam = arr[:, 0] * 1e3
            if t == 'tabulated nk':
                n_lam, n_val, k_lam, k_val = lam, arr[:, 1], lam, arr[:, 2]
            elif t == 'tabulated n':
                n_lam, n_val = lam, arr[:, 1]
            elif t == 'tabulated k':
                k_lam, k_val = lam, arr[:, 1]
        elif t.startswith('formula'):
            kind = int(t.split()[1])
            lo, hi = [float(v) for v in str(entry['wavelength_range']).split()]
            coeffs = [float(v) for v in str(entry['coefficients']).split()]
            formulas.append((kind, coeffs, lo * 1e3, hi * 1e3))

    if n_lam is None and formulas:
        kind, coeffs, lo, hi = formulas[0]
        n_lam = np.geomspace(lo, hi, 400)
        n_val = _rii_formula(kind, coeffs, n_lam * 1e-3)
    if n_lam is None:
        raise ValueError(f"{yaml_path}: no refractive index data found")

    if lambda_nm is None:
        lambda_nm = n_lam
    lambda_nm = np.asarray(lambda_nm, dtype=np.float64)
    n = np.interp(lambda_nm, n_lam, n_val)
    k = np.interp(lambda_nm, k_lam, k_val) if k_lam is not None else np.zeros_like(lambda_nm)
    return lambda_nm, n, k, ref


# ---------------------------------------------------------------------------
# Spectral library (ECOSTRESS / ASTER) emissivity data
# ---------------------------------------------------------------------------

def load_ecostress(path):
    """
    Read an ECOSTRESS / ASTER spectral library ASCII file.

    Returns ``(wavelengths_nm, reflectance, metadata)`` with reflectance as a
    fraction (0-1), wavelengths ascending. The files report directional
    hemispherical reflectance in percent versus wavelength in micrometres;
    for TIR files the emissivity follows from Kirchhoff's law as
    1 - reflectance (Meerdink et al. 2019; Baldridge et al. 2009).
    """
    meta = {}
    data = []
    with open(path, errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) == 2:
                try:
                    data.append((float(parts[0]), float(parts[1])))
                    continue
                except ValueError:
                    pass
            if ':' in s:
                key, _, value = s.partition(':')
                meta[key.strip()] = value.strip()
    if not data:
        raise ValueError(f"{path}: no spectral data found")
    arr = np.array(data)
    order = np.argsort(arr[:, 0])
    lam_um, val = arr[order, 0], arr[order, 1]
    x_units = meta.get('X Units', 'micrometers').lower()
    y_units = meta.get('Y Units', 'percent').lower()
    lam_nm = lam_um * (1e3 if 'micro' in x_units else 1.0)
    refl = val / (100.0 if 'percent' in y_units else 1.0)
    # Remove duplicate wavelengths (some files contain them)
    lam_nm, idx = np.unique(lam_nm, return_index=True)
    refl = np.clip(refl[idx], 0.0, 1.0)
    return lam_nm, refl, meta


def _irregular(lam_nm, values, max_points=2048):
    lam_nm = np.asarray(lam_nm, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if lam_nm.size > max_points:
        idx = np.linspace(0, lam_nm.size - 1, max_points).round().astype(int)
        lam_nm, values = lam_nm[idx], values[idx]
    return {'type': 'irregular',
            'wavelengths': ', '.join(f'{x:.6g}' for x in lam_nm),
            'values': ', '.join(f'{x:.6g}' for x in values)}


def diffuse_from_reflectance(lam_nm, reflectance):
    """Lambertian BSDF dict from a hemispherical reflectance spectrum."""
    return {'type': 'diffuse', 'reflectance': _irregular(lam_nm, np.clip(reflectance, 0.0, 1.0))}


def diffuse_from_emissivity(lam_nm, emissivity):
    """
    Lambertian BSDF dict whose Kirchhoff emissivity equals the given spectrum:
    reflectance = 1 - emissivity (report, section 3.4).
    """
    return diffuse_from_reflectance(lam_nm, 1.0 - np.asarray(emissivity, dtype=np.float64))


# ---------------------------------------------------------------------------
# BSDF builders
# ---------------------------------------------------------------------------

def conductor_ir(material, alpha=None, alpha_u=None, alpha_v=None, distribution='ggx'):
    """
    Smooth (``alpha`` is None) or rough conductor BSDF using the shipped IR
    optical constants (``'Al_ir'``, ``'Fe_ir'``, ...). Any ``name`` for which
    ``data/ior/name.eta.spd`` and ``.k.spd`` exist is accepted.
    """
    if alpha is None and alpha_u is None:
        return {'type': 'conductor', 'material': material}
    d = {'type': 'roughconductor', 'material': material, 'distribution': distribution}
    if alpha_u is not None:
        d['alpha_u'] = float(alpha_u)
        d['alpha_v'] = float(alpha_v if alpha_v is not None else alpha_u)
    else:
        d['alpha'] = float(alpha)
    return d


def opaque_ir(material, alpha=None):
    """
    BSDF for an *opaque absorbing* material described by complex optical
    constants (water, glass in the LWIR, sapphire beyond 6 um, ...): the
    interface is modelled with the conductor BSDFs, which implement the exact
    Fresnel equations for a complex index. Valid when the absorption depth is
    small compared with the object thickness (k lambda / 4 pi << thickness).
    """
    return conductor_ir(material, alpha=alpha)


def window_ior(material, band, weight='planck', T=300.0):
    """
    Band-averaged real refractive index of a transparent IR material, for use
    as ``int_ior`` of Mitsuba's (scalar-IOR) ``dielectric`` /
    ``roughdielectric`` BSDFs. The average is weighted by B(lambda, T)
    (``weight='planck'``) or uniform (``weight=None``).

    Dispersion inside one IR band is small for the usual window materials
    (e.g. Ge: 4.03-4.00 over 8-14 um), so a scalar IOR is adequate; the
    returned value together with the band-averaged k allows the caller to
    judge transparency.
    Returns (n_mean, k_mean).
    """
    from .radiometry import band_range, planck
    lo, hi = band_range(band)
    lam, n, k = load_ior(material)
    m = (lam >= lo) & (lam <= hi)
    if m.sum() < 2:
        grid = np.linspace(lo, hi, 64)
        n_b, k_b = np.interp(grid, lam, n), np.interp(grid, lam, k)
    else:
        grid, n_b, k_b = lam[m], n[m], k[m]
    w = planck(grid, T) if weight == 'planck' else np.ones_like(grid)
    w = w / np.trapezoid(w, grid)
    return float(np.trapezoid(n_b * w, grid)), float(np.trapezoid(k_b * w, grid))
