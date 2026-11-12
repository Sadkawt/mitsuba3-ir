"""
Validation tests for the ``volpath_ir`` integrator: thermal emission of
participating media in the difference formulation (report, section 6.5).
"""

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


def band_radiance(lmin, lmax, T, n=4001):
    lam = np.linspace(lmin, lmax, n)
    return np.trapezoid(planck_nm(lam, T), lam)


NARROW = (9990.0, 10010.0)


def sensor_dict(origin, target, band, spp, medium_ref=None):
    d = {
        'type': 'radiancemeter',
        'to_world': mi.ScalarTransform4f().look_at(origin=origin, target=target, up=[0, 1, 0]),
        'sampler': {'type': 'independent', 'sample_count': spp},
        'film': {
            'type': 'specfilm', 'width': 1, 'height': 1, 'rfilter': {'type': 'box'},
            'band': {'type': 'regular', 'wavelength_min': band[0],
                     'wavelength_max': band[1], 'values': '1, 1'},
        }
    }
    if medium_ref is not None:
        d['medium'] = {'type': 'ref', 'id': medium_ref}
    return d


def medium_dict(sigma_t, albedo, T, spectral=True):
    d = {'type': 'homogeneous', 'sigma_t': sigma_t, 'albedo': albedo,
         'has_spectral_extinction': spectral}
    if T is not None:
        d['temperature'] = T
    return d


def slab_dict(medium_id, z0, z1, half=50.0):
    # Axis-aligned box [-half, half]^2 x [z0, z1] filled with the medium
    c = 0.5 * (z0 + z1)
    s = 0.5 * (z1 - z0)
    return {
        'type': 'cube',
        'to_world': mi.ScalarTransform4f().translate([0, 0, c]).scale([half, half, s]),
        'bsdf': {'type': 'null'},
        'interior': {'type': 'ref', 'id': medium_id},
    }


def render_value(scene_dict, spp):
    scene = mi.load_dict(scene_dict)
    return float(np.array(mi.render(scene, spp=spp)).ravel()[0])


def test01_medium_temperature_property(variants_all_spectral):
    m = mi.load_dict(medium_dict(1.0, 0.5, 300.0))
    assert m.is_thermal()
    mei = dr.zeros(mi.MediumInteraction3f)
    assert dr.allclose(m.temperature(mei), 300.0)
    m = mi.load_dict(medium_dict(1.0, 0.5, None))
    assert not m.is_thermal()


@pytest.mark.parametrize('spectral', [True, False])
def test02_cold_absorbing_slab(variants_all_spectral, spectral):
    # A purely absorbing slab (albedo 0, thickness d) at T_m in front of a
    # blackbody plate at T1:  L = exp(-sigma d) B(T1) + (1 - exp(-sigma d)) B(T_m)
    sigma, d, T1, Tm = 0.7, 1.0, 320.0, 280.0
    scene = {
        'type': 'scene',
        'integrator': {'type': 'volpath_ir', 'max_depth': 16},
        'air': medium_dict(sigma, 0.0, Tm, spectral),
        'sensor': sensor_dict([0, 0, 3], [0, 0, 0], NARROW, 4096),
        'slab': slab_dict('air', 1.0, 1.0 + d),
        'plate': {'type': 'rectangle', 'bsdf': {'type': 'diffuse', 'reflectance': 0.0},
                  'emitter': {'type': 'thermal', 'temperature': T1}},
    }
    value = render_value(scene, 4096)
    tr = np.exp(-sigma * d)
    ref = tr * band_radiance(*NARROW, T1) + (1 - tr) * band_radiance(*NARROW, Tm)
    assert np.isclose(value, ref, rtol=1.5e-2), (value, ref)


def test03_isothermal_slab_zero_variance(variants_all_spectral):
    # Absorbing slab and blackbody plate at the same temperature: every sample
    # equals B(T) (collision -> B(T_m); no collision -> B(T_plate))
    T = 300.0
    scene = {
        'type': 'scene',
        'integrator': {'type': 'volpath_ir', 'max_depth': 16},
        'air': medium_dict(1.3, 0.0, T),
        'sensor': sensor_dict([0, 0, 3], [0, 0, 0], NARROW, 8),
        'slab': slab_dict('air', 1.0, 2.0),
        'plate': {'type': 'rectangle', 'bsdf': {'type': 'diffuse', 'reflectance': 0.0},
                  'emitter': {'type': 'thermal', 'temperature': T}},
    }
    value = render_value(scene, 8)
    ref = band_radiance(*NARROW, T)
    assert np.isclose(value, ref, rtol=2e-3), (value, ref)


def test03b_isothermal_slab_reflecting_plate(variants_all_spectral):
    # Same slab over a plate with albedo rho under a cold (black) sky. The
    # plate reflects the downwelling slab emission:
    #   L_down = (1 - tau) B,  L_up = (1 - rho) B + rho L_down,
    #   L_cam  = tau L_up + (1 - tau) B,   tau = exp(-sigma d)
    T, rho, sigma, d = 300.0, 0.3, 1.3, 1.0
    scene = {
        'type': 'scene',
        'integrator': {'type': 'volpath_ir', 'max_depth': 16},
        'air': medium_dict(sigma, 0.0, T),
        'sensor': sensor_dict([0, 0, 3], [0, 0, 0], NARROW, 4096),
        'slab': slab_dict('air', 1.0, 1.0 + d),
        'plate': {'type': 'rectangle', 'bsdf': {'type': 'diffuse', 'reflectance': rho},
                  'emitter': {'type': 'thermal', 'temperature': T}},
    }
    value = render_value(scene, 4096)
    B = band_radiance(*NARROW, T)
    tau = np.exp(-sigma * d)
    L_up = (1 - rho) * B + rho * (1 - tau) * B
    ref = tau * L_up + (1 - tau) * B
    assert np.isclose(value, ref, rtol=1.5e-2), (value, ref)


def test04_emissive_slab_cold_background(variants_all_spectral):
    # Hot absorbing slab against nothing:  L = (1 - exp(-sigma d)) B(T_m)
    sigma, d, Tm = 0.5, 2.0, 350.0
    scene = {
        'type': 'scene',
        'integrator': {'type': 'volpath_ir', 'max_depth': 16},
        'gas': medium_dict(sigma, 0.0, Tm),
        'sensor': sensor_dict([0, 0, 5], [0, 0, 0], NARROW, 4096),
        'slab': slab_dict('gas', 0.0, d),
    }
    value = render_value(scene, 4096)
    ref = (1 - np.exp(-sigma * d)) * band_radiance(*NARROW, Tm)
    assert np.isclose(value, ref, rtol=1.5e-2), (value, ref)


def test05_scattering_only_slab_does_not_emit(variants_all_spectral):
    # Albedo 1 (no absorption) at any temperature: no emission; against a cold
    # background the radiance is zero in expectation (Kirchhoff for volumes)
    scene = {
        'type': 'scene',
        'integrator': {'type': 'volpath_ir', 'max_depth': 64, 'rr_depth': 64},
        'fog': medium_dict(0.5, 1.0, 350.0),
        'sensor': sensor_dict([0, 0, 5], [0, 0, 0], NARROW, 4096),
        'slab': slab_dict('fog', 0.0, 2.0, half=3.0),
    }
    value = render_value(scene, 4096)
    ref_scale = band_radiance(*NARROW, 350.0)
    assert abs(value) < 2e-2 * ref_scale, (value, ref_scale)


def test06_isothermal_cavity_with_medium(variants_all_spectral):
    # Camera inside a diffuse isothermal cavity filled with a scattering and
    # absorbing medium at the same temperature: L = B(T) everywhere
    T = 300.0
    scene = {
        'type': 'scene',
        'integrator': {'type': 'volpath_ir', 'max_depth': 64, 'rr_depth': 64},
        'air': medium_dict(1.0, 0.6, T),
        'sensor': sensor_dict([0, 0, 0], [0.3, 0.2, 1], NARROW, 1024, medium_ref='air'),
        'cavity': {
            'type': 'sphere', 'radius': 1.5,
            'bsdf': {'type': 'twosided', 'bsdf': {'type': 'diffuse', 'reflectance': 0.6}},
            'interior': {'type': 'ref', 'id': 'air'},
            'emitter': {'type': 'thermal', 'temperature': T},
        }
    }
    value = render_value(scene, 1024)
    ref = band_radiance(*NARROW, T)
    assert np.isclose(value, ref, rtol=2e-2), (value, ref)


def test07_surfaces_match_path_ir(variants_all_spectral):
    # Without media, volpath_ir must agree with path_ir (plate in cavity)
    rho, T1, T2 = 0.3, 320.0, 280.0
    scene = {
        'type': 'scene',
        'integrator': {'type': 'volpath_ir', 'max_depth': 8},
        'sensor': sensor_dict([0, 0, 1], [0, 0, 0], NARROW, 1024),
        'plate': {'type': 'rectangle', 'bsdf': {'type': 'diffuse', 'reflectance': rho},
                  'emitter': {'type': 'thermal', 'temperature': T1}},
        'cavity': {'type': 'sphere', 'radius': 5.0,
                   'bsdf': {'type': 'diffuse', 'reflectance': 0.0},
                   'emitter': {'type': 'thermal', 'temperature': T2}},
    }
    value = render_value(scene, 1024)
    ref = (1 - rho) * band_radiance(*NARROW, T1) + rho * band_radiance(*NARROW, T2)
    assert np.isclose(value, ref, rtol=1.5e-2), (value, ref)


def test08_spectral_absorption(variants_all_spectral):
    # Wavelength-dependent absorption (irregular spectrum): transmittance of
    # a cold slab must follow exp(-sigma(lambda) d) per wavelength
    d, T1 = 1.0, 320.0
    band = (9000.0, 11000.0)
    lam = np.array([8000.0, 9000.0, 10000.0, 11000.0, 12000.0])
    sig = np.array([0.1, 0.2, 1.5, 0.3, 0.1])
    scene = {
        'type': 'scene',
        'integrator': {'type': 'volpath_ir', 'max_depth': 16},
        'air': {'type': 'homogeneous', 'albedo': 0.0,
                'sigma_t': {'type': 'constvolume', 'value': {
                    'type': 'irregular',
                    'wavelengths': ', '.join(str(x) for x in lam),
                    'values': ', '.join(str(x) for x in sig)}}},
        'sensor': sensor_dict([0, 0, 3], [0, 0, 0], band, 8192),
        'slab': slab_dict('air', 1.0, 1.0 + d),
        'plate': {'type': 'rectangle', 'bsdf': {'type': 'diffuse', 'reflectance': 0.0},
                  'emitter': {'type': 'thermal', 'temperature': T1}},
    }
    value = render_value(scene, 8192)
    grid = np.linspace(*band, 2001)
    ref = np.trapezoid(np.exp(-np.interp(grid, lam, sig) * d) * planck_nm(grid, T1), grid)
    assert np.isclose(value, ref, rtol=2e-2), (value, ref)
