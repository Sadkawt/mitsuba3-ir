"""
mitsubaIR fork validation: blackbody emission across visible + thermal IR.

The mitsubaIR fork extends Mitsuba 3's spectral sampling range from the
visible band (360-830 nm) up to 12 micrometers (12000 nm) so we can
render thermal-IR scenes.

This script:
  1. Loads a `blackbody` spectrum at temperature T spanning the new range
  2. Evaluates Mitsuba's spectral radiance L_lambda at a dense grid of
     wavelengths from 0.36 um -> 12 um
  3. Compares against the analytical Planck spectral radiance
     B_lambda(T) = (2 h c^2 / lambda^5) / (exp(h c / (lambda k T)) - 1)
     in W * m^-2 * sr^-1 * nm^-1
  4. Computes max relative error and the integrated radiant exitance
     (M = pi * integral L dlambda) versus Stefan-Boltzmann sigma * T^4.
"""
import math
import sys
import numpy as np

import mitsuba as mi
mi.set_variant("scalar_spectral")

import drjit as dr


# Physical constants (SI)
H = 6.62607015e-34      # Planck's constant [J*s]
C = 2.99792458e8        # speed of light [m/s]
KB = 1.380649e-23       # Boltzmann constant [J/K]
SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant [W * m^-2 * K^-4]


def planck_spectral_radiance(wavelength_nm: np.ndarray, T: float) -> np.ndarray:
    """Planck spectral radiance B_lambda(T), units W * m^-2 * sr^-1 * nm^-1.

    The 1e-9 factor converts the per-meter wavelength density returned by
    the textbook formula to a per-nanometer density (matching how Mitsuba's
    blackbody plugin defines it).
    """
    lam = wavelength_nm * 1e-9  # [m]
    c0 = 2.0 * H * C * C
    c1 = H * C / KB
    return 1e-9 * c0 / (lam ** 5 * (np.exp(c1 / (lam * T)) - 1.0))


def mi_eval_blackbody(temperature: float, wavelengths_nm: np.ndarray):
    """Evaluate Mitsuba's blackbody spectrum at the given wavelengths.

    Returns one Mitsuba radiance value per wavelength. Mitsuba's spectral
    variant evaluates 4 wavelengths simultaneously, so we splat the input
    wavelength into all 4 lanes and average the result."""
    bb = mi.load_dict({
        "type": "blackbody",
        "temperature": temperature,
    })

    out = np.empty_like(wavelengths_nm, dtype=np.float64)
    for i, lam in enumerate(wavelengths_nm):
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wavelengths = mi.Spectrum(float(lam))
        spec = bb.eval(si)
        # All four lanes contain the same value; take the first.
        out[i] = float(spec[0])
    return out


def main():
    T = 1500.0  # Kelvin -- Wien peak ~1932 nm, well into NIR/IR

    # Cover the full extended range with extra density near the Wien peak.
    coarse = np.linspace(400.0, 11800.0, 80)
    fine = np.linspace(1000.0, 4000.0, 40)
    wavelengths = np.unique(np.concatenate([coarse, fine]))

    print(f"\n=== mitsubaIR blackbody validation @ T = {T} K ===")
    print(f"Wien displacement peak: {2.897771955e-3 / T * 1e9:.1f} nm")
    print(f"Sampling {len(wavelengths)} wavelengths from "
          f"{wavelengths.min():.0f} nm to {wavelengths.max():.0f} nm "
          f"({wavelengths.min()/1000:.2f}-{wavelengths.max()/1000:.2f} um)")

    mi_vals = mi_eval_blackbody(T, wavelengths)
    analytic_vals = planck_spectral_radiance(wavelengths, T)

    rel_err = np.abs(mi_vals - analytic_vals) / np.maximum(analytic_vals, 1e-30)

    print(f"\n  lambda [nm] | Mitsuba L_lambda  | Analytic B_lambda  | rel err")
    print(f"  ------------+-------------------+--------------------+--------")
    sample_idx = [0, len(wavelengths)//8, len(wavelengths)//4,
                  len(wavelengths)//2, 3*len(wavelengths)//4, len(wavelengths)-1]
    # Always include the wavelengths nearest 1, 5, 10 um
    for target_um in (1.0, 5.0, 10.0):
        idx = int(np.argmin(np.abs(wavelengths - target_um * 1000.0)))
        sample_idx.append(idx)
    for i in sorted(set(sample_idx)):
        print(f"   {wavelengths[i]:8.1f}  |  {mi_vals[i]:.6e}   "
              f"|  {analytic_vals[i]:.6e}    | {rel_err[i]:.2e}")

    # Stefan-Boltzmann sanity check via numerical integration of L_lambda.
    # The hemispherical radiant exitance from a Lambertian blackbody is
    # M = pi * integral_0^inf L_lambda dlambda = sigma * T^4.
    # We integrate over our covered range (360 nm .. 12 um) which captures
    # >99% of emitted energy at 1500 K (peak ~1932 nm).
    M_mi = math.pi * np.trapezoid(mi_vals, wavelengths)
    M_an = math.pi * np.trapezoid(analytic_vals, wavelengths)
    M_sb = SIGMA * T**4

    print(f"\n  Integrated radiant exitance (W/m^2):")
    print(f"    Mitsuba (range-truncated):    {M_mi:.4e}")
    print(f"    Analytic (range-truncated):   {M_an:.4e}")
    print(f"    Stefan-Boltzmann sigma*T^4:   {M_sb:.4e}")
    print(f"    Coverage of total: {M_an / M_sb * 100:.2f}%")

    max_rel_err = float(rel_err.max())
    print(f"\n  Max pointwise relative error: {max_rel_err:.3e}")

    tol = 5e-4
    if max_rel_err < tol:
        print(f"\n  PASS: pointwise error within {tol:.0e}.")
        return 0
    else:
        print(f"\n  FAIL: pointwise error {max_rel_err:.3e} exceeds {tol:.0e}.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
