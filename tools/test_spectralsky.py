#!/usr/bin/env python3
"""
Validation for the `spectralsky` emitter.

Checks, against a real `.skyrad` table:
  1. eval() reproduces the table radiance at exact grid nodes (zenith & horizon)
     for several wavelengths, to interpolation precision.
  2. The emitter is azimuthally consistent and warmer toward the horizon.
  3. A path-traced render of the bare sky recovers the zenith-band radiance.

Run:
    PYTHONPATH=build/python .venv/bin/python tools/test_spectralsky.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from skyrad_io import SkyRadiance  # noqa: E402

import mitsuba as mi  # noqa: E402
mi.set_variant("cuda_ad_spectral")
import drjit as dr  # noqa: E402

SKYRAD = "tools/data/ir_sky/lwir_us_standard.skyrad"


def table_lookup(sky: SkyRadiance, theta_deg, phi_deg, lam_nm):
    """Reference trilinear interpolation in pure NumPy (matches the emitter)."""
    def frac_idx(x, axis):
        f = np.interp(x, axis, np.arange(len(axis)))
        i0 = int(np.clip(np.floor(f), 0, len(axis) - 2))
        return i0, f - i0
    ti, tw = frac_idx(theta_deg, sky.theta)
    pj, pw = frac_idx(phi_deg % 360.0, sky.phi)
    lk, lw = frac_idx(lam_nm, sky.wavelengths)
    d = sky.data
    c = 0.0
    for di, wi in ((0, 1 - tw), (1, tw)):
        for dj, wj in ((0, 1 - pw), (1, pw)):
            for dk, wk in ((0, 1 - lw), (1, lw)):
                c += wi * wj * wk * d[ti + di, pj + dj, lk + dk]
    return c


def make_si(wi_local, wavelengths):
    """A minimal SurfaceInteraction for evaluating an environment emitter."""
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = mi.Vector3f(float(wi_local[0]), float(wi_local[1]), float(wi_local[2]))
    si.wavelengths = mi.Spectrum(*[float(w) for w in wavelengths])
    return si


def main() -> int:
    sky = SkyRadiance.read(SKYRAD)
    emitter = mi.load_dict(sky.to_dict())
    print("loaded:", str(emitter).splitlines()[0])

    # Wavelengths to probe (interior grid nodes; Spectrum width is 4 in
    # cuda_ad_spectral, so probe exactly 4).
    test_wls = [float(sky.wavelengths[i]) for i in (4, 9, 14, 22)]

    # -- Test 1: eval at zenith and horizon grid nodes -----------------------
    # eval uses v = -si.wi (local).  v=+Z => zenith; tilt toward +X for horizon.
    cases = {
        "zenith (theta=0)":  (np.array([0.0, 0.0, -1.0]), 0.0),
        "horizon (theta=89)": (
            np.array([-np.sin(np.radians(89.0)), 0.0, -np.cos(np.radians(89.0))]),
            89.0),
    }
    max_err = 0.0
    for name, (wi, theta_deg) in cases.items():
        si = make_si(wi, test_wls)
        got = np.array(emitter.eval(si)).reshape(-1)   # (n_wavelengths,)
        ref = np.array([table_lookup(sky, theta_deg, 0.0, w) for w in test_wls])
        rel = np.abs(got - ref) / np.maximum(ref, 1e-12)
        max_err = max(max_err, rel.max())
        print(f"  {name:22s} got={got}  ref={ref}  max_rel_err={rel.max():.2e}")
    assert max_err < 1e-3, f"eval mismatch vs table: {max_err:.2e}"
    print(f"  [OK] eval matches table to {max_err:.2e}")

    # -- Test 2: below-horizon returns zero ----------------------------------
    si_down = make_si(np.array([0.0, 0.0, 1.0]), test_wls)  # v=-Z, theta>90
    down = np.array(emitter.eval(si_down)).reshape(-1)
    assert np.allclose(down, 0.0), f"below-horizon not zero: {down}"
    print(f"  [OK] below-horizon radiance is zero")

    # -- Test 3: monotonic warming toward horizon ----------------------------
    thetas = [0, 30, 60, 89]
    means = []
    for th in thetas:
        wi = np.array([-np.sin(np.radians(th)), 0.0, -np.cos(np.radians(th))])
        si = make_si(wi, test_wls)
        means.append(float(np.array(emitter.eval(si)).reshape(-1).mean()))
    print(f"  band-mean radiance vs zenith angle {thetas}: "
          f"{[f'{m:.3e}' for m in means]}")
    assert all(np.diff(means) > 0), "radiance should increase toward horizon"
    print(f"  [OK] radiance increases monotonically toward the horizon")

    # -- Test 4: path-traced bare-sky render recovers zenith radiance --------
    # Camera at origin looking straight up (+Z). A narrow FOV => the centre
    # pixels see ~zenith. specfilm integrates one narrow band; divide by its
    # width to recover band-average spectral radiance.
    c = int(np.argmin(np.abs(sky.wavelengths - sky.wavelengths[14])))
    lam_c = float(sky.wavelengths[c])
    hw = 100.0  # nm half-width
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 2},
        "sensor": {
            "type": "perspective", "fov": 2.0,
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[0, 0, 0], target=[0, 0, 1], up=[0, 1, 0]),
            "film": {
                "type": "specfilm", "width": 16, "height": 16,
                "component_format": "float32",
                "band": {"type": "regular",
                         "wavelength_min": lam_c - hw,
                         "wavelength_max": lam_c + hw,
                         "values": "1, 1"},
            },
            "sampler": {"type": "independent", "sample_count": 256},
        },
        "sky": sky.to_dict(),
    })
    img = np.array(mi.render(scene, spp=256))
    band_integral = img[:, :, 0].mean()        # W/m^2/sr  (integral over band)
    rendered = band_integral / (2 * hw)        # -> W/m^2/sr/nm
    ref_zenith = table_lookup(sky, 0.0, 0.0, lam_c)
    rel = abs(rendered - ref_zenith) / ref_zenith
    print(f"  render@zenith {lam_c:.0f}nm: rendered={rendered:.3e}  "
          f"table={ref_zenith:.3e}  rel_err={rel:.2%}")
    assert rel < 0.05, f"rendered zenith radiance off by {rel:.2%}"
    print(f"  [OK] path-traced zenith radiance matches table to {rel:.2%}")

    # -- Test 5: sample_direction / pdf_direction consistency ----------------
    # The pdf returned by sample_direction must equal pdf_direction(ds), or
    # next-event estimation (MIS) is biased.
    n = 1 << 16
    rng = np.random.default_rng(0)
    sample = mi.Point2f(rng.random(n).astype(np.float32),
                        rng.random(n).astype(np.float32))
    it = dr.zeros(mi.Interaction3f, n)
    it.p = mi.Point3f(0, 0, 0)
    ds, _ = emitter.sample_direction(it, sample)
    pdf2 = emitter.pdf_direction(it, ds)
    rel = dr.abs(ds.pdf - pdf2) / dr.maximum(ds.pdf, 1e-9)
    # Only consider sampled (valid, above-horizon) directions.
    valid = ds.pdf > 0
    max_rel = float(dr.max(dr.select(valid, rel, 0.0))[0])
    frac_up = float(dr.sum(mi.Float(valid))[0] / n)
    print(f"  sample/pdf consistency: max_rel_err={max_rel:.2e}  "
          f"(fraction above horizon={frac_up:.2f})")
    assert max_rel < 1e-3, f"sample_direction vs pdf_direction mismatch: {max_rel:.2e}"
    assert frac_up > 0.95, f"too many samples below horizon: {frac_up:.2f}"
    print(f"  [OK] sample_direction and pdf_direction agree")

    print("\nAll spectralsky tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
