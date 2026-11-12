"""Unit tests for the ``thermalsky`` emitter."""

import numpy as np
import pytest
import drjit as dr
import mitsuba as mi

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
K_BOLTZ = 1.380649e-23


def planck_nm(lambda_nm, T):
    lam = np.asarray(lambda_nm, dtype=np.float64) * 1e-9
    return (2 * H_PLANCK * C_LIGHT**2 / lam**5
            / np.expm1(H_PLANCK * C_LIGHT / (lam * K_BOLTZ * T))) * 1e-9


WAVELENGTHS = [8000.0, 10000.0, 12000.0, 14000.0]


def make(**kw):
    d = {'type': 'thermalsky', 'temperature': 290.0, 'emissivity_zenith': 0.4}
    d.update(kw)
    return mi.load_dict(d)


def eval_dir(emitter, direction):
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wavelengths = mi.Spectrum(WAVELENGTHS)
    si.wi = -mi.Vector3f(*[float(v) for v in direction])   # si.wi = -ray.d for escaped rays
    return np.array(emitter.eval(si)).ravel()


def test01_zenith_and_horizon(variants_all_spectral):
    e = make()
    B = planck_nm(WAVELENGTHS, 290.0)
    assert np.allclose(eval_dir(e, [0, 0, 1]), 0.4 * B, rtol=2e-4)
    # Plane-parallel model: eps(theta) = 1 - (1 - eps_z)^(sec theta)
    theta = np.deg2rad(60.0)
    eps = 1 - (1 - 0.4) ** (1 / np.cos(theta))
    assert np.allclose(eval_dir(e, [np.sin(theta), 0, np.cos(theta)]), eps * B, rtol=2e-4)
    # Horizon: limited by cos_min
    eps_h = 1 - (1 - 0.4) ** (1 / 0.05)
    assert np.allclose(eval_dir(e, [1, 0, 1e-4]), eps_h * B, rtol=2e-3)
    # Below the horizon: opaque air
    assert np.allclose(eval_dir(e, [0, 0, -1]), B, rtol=2e-4)


def test02_isotropic_and_up_vector(variants_all_spectral):
    e = make(model='isotropic')
    B = planck_nm(WAVELENGTHS, 290.0)
    for d in ([0, 0, 1], [1, 0, 0.1], [0, 1, 0.5]):
        assert np.allclose(eval_dir(e, d), 0.4 * B, rtol=2e-4)
    e = make(up=[0, 1, 0])
    assert np.allclose(eval_dir(e, [0, 1, 0]), 0.4 * B, rtol=2e-4)
    assert np.allclose(eval_dir(e, [0, -1, 0]), B, rtol=2e-4)


def test03_spectral_zenith_emissivity(variants_all_spectral):
    e = make(emissivity_zenith={'type': 'regular', 'wavelength_min': 8000.0,
                                'wavelength_max': 14000.0, 'values': '0.2, 0.8'})
    B = planck_nm(WAVELENGTHS, 290.0)
    eps = np.interp(WAVELENGTHS, [8000.0, 14000.0], [0.2, 0.8])
    assert np.allclose(eval_dir(e, [0, 0, 1]), eps * B, rtol=1e-3)


def test04_sample_direction_consistency(variants_all_spectral):
    e = make()
    it = dr.zeros(mi.Interaction3f)
    it.wavelengths = mi.Spectrum(WAVELENGTHS)
    for u in [(0.1, 0.2), (0.5, 0.5), (0.9, 0.3), (0.3, 0.95)]:
        ds, spec = e.sample_direction(it, mi.Point2f(*u))
        pdf = e.pdf_direction(it, ds)
        value = e.eval_direction(it, ds)
        assert dr.allclose(ds.pdf, pdf, rtol=1e-4)
        assert dr.allclose(spec * ds.pdf, value, rtol=1e-4)
        assert np.allclose(np.array(value).ravel(), eval_dir(e, np.array(ds.d).ravel()), rtol=1e-4)


def test05_render_metal_plate_reflecting_sky(variants_all_spectral):
    # A cold smooth conductor plate (0 K) facing up reflects the zenith sky:
    # L = F(0) * eps_z * B(T_air)
    n, k = 3.0, 4.0
    F0 = ((n - 1) ** 2 + k ** 2) / ((n + 1) ** 2 + k ** 2)
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'path_ir', 'max_depth': 4},
        'sensor': {
            'type': 'radiancemeter',
            'to_world': mi.ScalarTransform4f().look_at(origin=[0, 0, 1], target=[0, 0, 0], up=[0, 1, 0]),
            'sampler': {'type': 'independent', 'sample_count': 16},
            'film': {'type': 'specfilm', 'width': 1, 'height': 1, 'rfilter': {'type': 'box'},
                     'band': {'type': 'regular', 'wavelength_min': 9990.0,
                              'wavelength_max': 10010.0, 'values': '1, 1'}},
        },
        'plate': {'type': 'rectangle', 'bsdf': {'type': 'conductor', 'eta': n, 'k': k}},
        'sky': {'type': 'thermalsky', 'temperature': 290.0, 'emissivity_zenith': 0.4},
    })
    value = float(np.array(mi.render(scene, spp=16)).ravel()[0])
    lam = np.linspace(9990.0, 10010.0, 201)
    ref = F0 * 0.4 * np.trapezoid(planck_nm(lam, 290.0), lam)
    assert np.isclose(value, ref, rtol=3e-3), (value, ref)
