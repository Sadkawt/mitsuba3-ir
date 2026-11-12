"""
Validation tests for the ``path_ir`` integrator (Kirchhoff emission through the
difference formulation, see docs/ir/IR_TECHNICAL_REPORT.md).

All tests compare band-integrated radiance (W m^-2 sr^-1) produced by a
``specfilm`` against closed-form references.
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


def band_radiance(lmin, lmax, T, n=20001):
    lam = np.linspace(lmin, lmax, n)
    return np.trapezoid(planck_nm(lam, T), lam)


def fresnel_conductor_unpolarized(cos_theta, n, k):
    """Unpolarized Fresnel reflectance of a conductor (air -> n + ik)."""
    cos_theta = np.asarray(cos_theta, dtype=np.float64)
    sin2 = 1 - cos_theta**2
    m = complex(n, k)
    t = np.sqrt(m**2 - sin2 + 0j)
    rs = (cos_theta - t) / (cos_theta + t)
    rp = (m**2 * cos_theta - t) / (m**2 * cos_theta + t)
    return 0.5 * (abs(rs)**2 + abs(rp)**2)


def sensor_dict(origin, target, band, spp):
    return {
        'type': 'radiancemeter',
        'to_world': mi.ScalarTransform4f().look_at(origin=origin, target=target, up=[0, 1, 0]),
        'sampler': {'type': 'independent', 'sample_count': spp},
        'film': {
            'type': 'specfilm',
            'width': 1, 'height': 1,
            'rfilter': {'type': 'box'},
            'band': {
                'type': 'regular',
                'wavelength_min': band[0],
                'wavelength_max': band[1],
                'values': '1, 1',
            },
        }
    }


def render_value(scene_dict, spp=64):
    scene = mi.load_dict(scene_dict)
    img = mi.render(scene, spp=spp)
    return float(np.array(img).ravel()[0])


NARROW = (9990.0, 10010.0)   # B(lambda) is constant to <0.1% over this band
LWIR = (8000.0, 14000.0)


def test01_diffuse_plate_cold_background(variants_all_spectral):
    # A diffuse plate (albedo rho) at temperature T seen against a black (0 K)
    # background radiates (1 - rho) B(T). The estimator is exact for this case.
    rho, T = 0.3, 300.0
    scene = {
        'type': 'scene',
        'integrator': {'type': 'path_ir', 'max_depth': 8},
        'sensor': sensor_dict([0, 0, 1], [0, 0, 0], NARROW, 16),
        'plate': {
            'type': 'rectangle',
            'bsdf': {'type': 'diffuse', 'reflectance': rho},
            'emitter': {'type': 'thermal', 'temperature': T},
        }
    }
    value = render_value(scene, spp=16)
    ref = (1 - rho) * band_radiance(*NARROW, T)
    assert np.isclose(value, ref, rtol=2e-3), (value, ref)

    # The standard path tracer is not thermal-aware: it renders the plate as
    # a blackbody (a warning is logged when the scene is created).
    scene['integrator'] = {'type': 'path', 'max_depth': 8}
    value = render_value(scene, spp=16)
    ref = band_radiance(*NARROW, T)
    assert np.isclose(value, ref, rtol=2e-3), (value, ref)


def test02_isothermal_enclosure_specular(variants_all_spectral):
    # Inside a closed isothermal cavity every direction carries exactly B(T),
    # regardless of the (energy conserving) BSDF. With a smooth conductor the
    # difference formulation has zero variance.
    T = 300.0
    scene = {
        'type': 'scene',
        'integrator': {'type': 'path_ir', 'max_depth': 32},
        'sensor': sensor_dict([0, 0, 0], [0.3, 0.2, 1], NARROW, 8),
        'cavity': {
            'type': 'sphere',
            'radius': 1.0,
            'bsdf': {'type': 'twosided', 'bsdf': {
                'type': 'conductor', 'eta': 3.0, 'k': 4.0}},
            'emitter': {'type': 'thermal', 'temperature': T},
        }
    }
    value = render_value(scene, spp=8)
    ref = band_radiance(*NARROW, T)
    assert np.isclose(value, ref, rtol=2e-3), (value, ref)


def test03_isothermal_enclosure_diffuse(variants_all_spectral):
    # Same as above with a diffuse cavity (albedo 0.7): the expectation is
    # exactly B(T); the estimate is noisy because of MIS between emitter and
    # BSDF sampling.
    T = 300.0
    scene = {
        'type': 'scene',
        'integrator': {'type': 'path_ir', 'max_depth': 64, 'rr_depth': 64},
        'sensor': sensor_dict([0, 0, 0], [0.3, 0.2, 1], NARROW, 512),
        'cavity': {
            'type': 'sphere',
            'radius': 1.0,
            'bsdf': {'type': 'twosided', 'bsdf': {
                'type': 'diffuse', 'reflectance': 0.7}},
            'emitter': {'type': 'thermal', 'temperature': T},
        }
    }
    value = render_value(scene, spp=512)
    ref = band_radiance(*NARROW, T)
    assert np.isclose(value, ref, rtol=1.5e-2), (value, ref)


def test04_plate_in_blackbody_enclosure(variants_all_spectral):
    # Diffuse plate (albedo rho, T1) inside a large blackbody cavity at T2:
    # L = (1 - rho) B(T1) + rho B(T2)
    rho, T1, T2 = 0.3, 320.0, 280.0
    scene = {
        'type': 'scene',
        'integrator': {'type': 'path_ir', 'max_depth': 8},
        'sensor': sensor_dict([0, 0, 1], [0, 0, 0], NARROW, 512),
        'plate': {
            'type': 'rectangle',
            'bsdf': {'type': 'diffuse', 'reflectance': rho},
            'emitter': {'type': 'thermal', 'temperature': T1},
        },
        'cavity': {
            'type': 'sphere',
            'radius': 5.0,
            'bsdf': {'type': 'diffuse', 'reflectance': 0.0},
            'emitter': {'type': 'thermal', 'temperature': T2},
        }
    }
    value = render_value(scene, spp=512)
    ref = (1 - rho) * band_radiance(*NARROW, T1) + rho * band_radiance(*NARROW, T2)
    assert np.isclose(value, ref, rtol=1.5e-2), (value, ref)


@pytest.mark.parametrize('theta_deg', [0.0, 45.0, 75.0])
def test05_conductor_directional_emissivity(variants_all_spectral, theta_deg):
    # A smooth conductor plate at T against a cold background radiates
    # (1 - F(theta)) B(T): Kirchhoff's law in its directional form.
    n, k, T = 3.0, 4.0, 300.0
    theta = np.deg2rad(theta_deg)
    origin = [np.sin(theta), 0.0, np.cos(theta)]
    scene = {
        'type': 'scene',
        'integrator': {'type': 'path_ir', 'max_depth': 8},
        'sensor': sensor_dict(origin, [0, 0, 0], NARROW, 16),
        'plate': {
            'type': 'rectangle',
            'bsdf': {'type': 'conductor', 'eta': n, 'k': k},
            'emitter': {'type': 'thermal', 'temperature': T},
        }
    }
    value = render_value(scene, spp=16)
    F = fresnel_conductor_unpolarized(np.cos(theta), n, k)
    ref = (1 - F) * band_radiance(*NARROW, T)
    assert np.isclose(value, ref, rtol=3e-3), (value, ref, F)


def test06_lwir_band_integration(variants_all_spectral):
    # Band-integrated radiance of a blackbody plate over 8-14 um
    T = 300.0
    scene = {
        'type': 'scene',
        'integrator': {'type': 'path_ir', 'max_depth': 2},
        'sensor': sensor_dict([0, 0, 1], [0, 0, 0], LWIR, 1024),
        'plate': {
            'type': 'rectangle',
            'bsdf': {'type': 'diffuse', 'reflectance': 0.0},
            'emitter': {'type': 'thermal', 'temperature': T},
        }
    }
    value = render_value(scene, spp=1024)
    ref = band_radiance(*LWIR, T)
    assert np.isclose(value, ref, rtol=1e-2), (value, ref)
    assert np.isclose(ref, 54.93, rtol=1e-3)  # W m^-2 sr^-1, sanity check


def test07_thermal_plus_solar_mixed(variants_all_spectral):
    # MWIR-style mixed scene: a thermal diffuse plate illuminated by a
    # non-thermal directional source of irradiance E. Expected radiance:
    # (1 - rho) B(T) + rho E / pi.
    #
    # Next-event estimation selects emitters proportionally to their
    # 'sampling_weight'. Since every thermal surface is an emitter, the sun
    # is given a large weight so that it is selected (almost) always; the
    # plate sampling itself contributes zero (in-plane directions).
    rho, T, E = 0.4, 300.0, 5e-3   # E in W m^-2 nm^-1
    band = (3990.0, 4010.0)
    scene = {
        'type': 'scene',
        'integrator': {'type': 'path_ir', 'max_depth': 8},
        'sensor': sensor_dict([0, 0, 1], [0, 0, 0], band, 64),
        'plate': {
            'type': 'rectangle',
            'bsdf': {'type': 'diffuse', 'reflectance': rho},
            'emitter': {'type': 'thermal', 'temperature': T},
        },
        'sun': {
            'type': 'directional',
            'direction': [0, 0, -1],
            'sampling_weight': 1e4,
            'irradiance': {'type': 'uniform', 'value': E,
                           'wavelength_min': 3000.0, 'wavelength_max': 5000.0},
        }
    }
    value = render_value(scene, spp=256)
    width = band[1] - band[0]
    ref = (1 - rho) * band_radiance(*band, T) + rho * E * width / np.pi
    assert np.isclose(value, ref, rtol=5e-3), (value, ref)
