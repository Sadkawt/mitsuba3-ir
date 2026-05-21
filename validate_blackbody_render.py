"""
End-to-end render validation for the mitsubaIR fork.

Renders a tiny scene where a `radiancemeter` looks at a flat blackbody-
emitting surface that fills its entire field of view, and uses a `specfilm`
film with a stack of narrow rectangular spectral response functions
covering 400 nm -> 11800 nm. Each channel of the rendered image is the
radiance integrated over its narrow band (effectively the spectral
radiance at the band center).

This exercises the full path:
    Sensor::sample_wavelengths -> uniform sampling over [MI_WAVELENGTH_MIN,
    MI_WAVELENGTH_MAX] (mitsubaIR change)
    -> ray traced into scene
    -> hits area emitter wrapping a blackbody spectrum
    -> emitted radiance evaluated at the sampled wavelengths
    -> projected through specfilm bandpasses
"""
import math
import sys
import numpy as np

import mitsuba as mi
mi.set_variant("scalar_spectral")

import drjit as dr


H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23


def planck(wavelength_nm, T):
    lam = np.asarray(wavelength_nm) * 1e-9
    c0 = 2.0 * H * C * C
    c1 = H * C / KB
    return 1e-9 * c0 / (lam ** 5 * (np.exp(c1 / (lam * T)) - 1.0))


def make_band(center_nm, half_width_nm, name):
    """A regular-spectrum bandpass: 1.0 inside [c-hw, c+hw], 0 outside."""
    lo = center_nm - half_width_nm
    hi = center_nm + half_width_nm
    return {
        "type": "regular",
        "wavelength_min": float(lo),
        "wavelength_max": float(hi),
        "values": "1, 1",
    }, name


def main():
    T = 1500.0

    # Choose band centers across visible + IR.
    band_centers = [500.0, 1000.0, 1932.0, 3000.0, 5000.0, 8000.0, 11000.0]
    half_width = 50.0  # +/- 50 nm narrow band

    bands = {}
    band_names = []
    for c in band_centers:
        spec_dict, name = make_band(c, half_width, f"b_{int(c)}")
        bands[name] = spec_dict
        band_names.append(name)

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 2},
        "sensor": {
            "type": "radiancemeter",
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[0, 0, 0], target=[0, 0, 1], up=[0, 1, 0]),
            "film": {
                "type": "specfilm",
                "width": 1,
                "height": 1,
                "component_format": "float32",
                **{name: spec for name, spec in bands.items()},
            },
            "sampler": {"type": "independent", "sample_count": 262144},
        },
        "emitter": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[0, 0, 1.0], target=[0, 0, 0], up=[0, 1, 0]
            ).scale([100, 100, 1]),  # huge so it fills FOV
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "blackbody",
                    "temperature": T,
                },
            },
        },
    }

    scene = mi.load_dict(scene_dict)
    img = mi.render(scene, spp=262144)
    arr = np.array(img)
    # specfilm output: shape (1, 1, n_bands)
    measured = arr.reshape(-1).astype(np.float64)

    print(f"\n=== End-to-end blackbody render validation @ T = {T} K ===")
    print(f"specfilm bands: {band_centers}")
    # specfilm channel value is the integral L_lambda * SRF(lambda) dlambda.
    # Our SRF is a rectangle of height 1 and width 2*half_width, so divide
    # by that width to recover the (band-averaged) spectral radiance.
    band_width = 2.0 * half_width
    spectral_radiance = measured / band_width

    # Compare to the band-AVERAGED Planck spectrum, not to the value at
    # the band center. At 1500 K the spectrum varies steeply across a
    # 100 nm window in the visible, so a point comparison would
    # systematically bias short-wavelength bands by ~30%.
    print(f"\n  band [nm] | rendered L_avg | analytic L_avg  | rel err  (vs band-avg)")
    print(f"  ----------+----------------+-----------------+--------")
    err_vals = []
    for c, val in zip(band_centers, spectral_radiance):
        lam_grid = np.linspace(c - half_width, c + half_width, 257)
        analytic_avg = float(np.trapezoid(planck(lam_grid, T), lam_grid)
                             / band_width)
        rel = abs(val - analytic_avg) / max(analytic_avg, 1e-30)
        err_vals.append(rel)
        print(f"   {c:7.1f}  |  {val:.4e}   |  {analytic_avg:.4e}    | {rel:.2e}")

    max_err = max(err_vals)
    print(f"\n  Max relative error across bands: {max_err:.3e}")
    tol = 0.05  # 5% — Monte Carlo noise dominates
    print(f"  Tolerance (Monte Carlo): {tol:.0%}")
    print("  PASS" if max_err < tol else "  FAIL")
    return 0 if max_err < tol else 1


if __name__ == "__main__":
    sys.exit(main())
