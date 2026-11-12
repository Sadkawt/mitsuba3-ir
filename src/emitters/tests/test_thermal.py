"""
Unit tests for the ``thermal`` emitter plugin.

The emitter reports the *blackbody* spectral radiance B(lambda, T) of the
surface it is attached to; see docs/ir/IR_TECHNICAL_REPORT.md.
"""

import numpy as np
import pytest
import drjit as dr
import mitsuba as mi

# Exact SI (2019) constants
H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
K_BOLTZ = 1.380649e-23


def planck_nm(lambda_nm, T):
    """Planck spectral radiance in W m^-2 sr^-1 nm^-1."""
    lam = np.asarray(lambda_nm, dtype=np.float64) * 1e-9
    return (2 * H_PLANCK * C_LIGHT**2 / lam**5
            / np.expm1(H_PLANCK * C_LIGHT / (lam * K_BOLTZ * T))) * 1e-9


def make_shape(T=300.0, twosided=True, **extra):
    return mi.load_dict({
        'type': 'sphere',
        'radius': 1.0,
        'bsdf': {'type': 'diffuse', 'reflectance': 0.5},
        'emitter': {
            'type': 'thermal',
            'temperature': T,
            'twosided': twosided,
            **extra,
        }
    })


def test01_constructor_and_flags(variant_scalar_spectral):
    shape = make_shape()
    emitter = shape.emitter()
    assert emitter.bbox() == shape.bbox()
    assert emitter.flags() & int(mi.EmitterFlags.Thermal)
    assert emitter.flags() & int(mi.EmitterFlags.Surface)

    # A to_world transformation is inherited from the shape and must be rejected
    with pytest.raises(RuntimeError):
        mi.load_dict({
            'type': 'thermal',
            'to_world': mi.ScalarTransform4f().translate([5, 0, 0])
        })

    # Invalid wavelength range
    with pytest.raises(RuntimeError):
        mi.load_dict({'type': 'thermal', 'wavelength_min': 5000, 'wavelength_max': 100})


def test02_thermal_requires_spectral_variant(variant_scalar_rgb):
    with pytest.raises(RuntimeError, match='spectral'):
        make_shape()


@pytest.mark.parametrize('T', [250.0, 300.0, 600.0, 1500.0])
def test03_eval_matches_planck(variants_all_spectral, T):
    # eval() must return Planck's law at the queried wavelengths
    shape = make_shape(T=T)
    emitter = shape.emitter()

    wavelengths = [1000.0, 4000.0, 10000.0, 14000.0]
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wavelengths = mi.Spectrum(wavelengths)
    si.wi = mi.Vector3f(0, 0, 1)

    value = np.array(emitter.eval(si)).ravel()
    ref = planck_nm(wavelengths, T)
    assert np.allclose(value, ref, rtol=2e-4), (value, ref)


def test04_twosided(variants_all_spectral):
    wavelengths = [8000.0, 10000.0, 12000.0, 14000.0]
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wavelengths = mi.Spectrum(wavelengths)
    ref = planck_nm(wavelengths, 300.0)

    # Default: emits from both sides
    emitter = make_shape(twosided=True).emitter()
    si.wi = mi.Vector3f(0, 0, -1)
    assert np.allclose(np.array(emitter.eval(si)).ravel(), ref, rtol=2e-4)

    # One-sided: nothing from the back
    emitter = make_shape(twosided=False).emitter()
    si.wi = mi.Vector3f(0, 0, 1)
    assert np.allclose(np.array(emitter.eval(si)).ravel(), ref, rtol=2e-4)
    si.wi = mi.Vector3f(0, 0, -1)
    assert np.allclose(np.array(emitter.eval(si)).ravel(), 0.0)


def test05_matches_blackbody_spectrum(variants_all_spectral):
    # The thermal emitter must agree with an 'area' emitter driven by the
    # existing 'blackbody' spectrum plugin (up to the slightly different
    # values of the physical constants used there).
    T = 350.0
    thermal = make_shape(T=T).emitter()
    area = mi.load_dict({
        'type': 'sphere',
        'emitter': {
            'type': 'area',
            'twosided': True,
            'radiance': {
                'type': 'blackbody',
                'temperature': T,
                'wavelength_min': 360.0,
                'wavelength_max': 20000.0,
            }
        }
    }).emitter()

    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wavelengths = mi.Spectrum([900.0, 3000.0, 5000.0, 12000.0])
    si.wi = mi.Vector3f(0, 0, 1)
    a = np.array(thermal.eval(si)).ravel()
    b = np.array(area.eval(si)).ravel()
    assert np.allclose(a, b, rtol=1e-4)


def test06_temperature_texture(variants_all_spectral):
    # Spatially varying temperature through a texture (checkerboard: 280/320 K)
    shape = mi.load_dict({
        'type': 'rectangle',
        'emitter': {
            'type': 'thermal',
            'temperature': {
                'type': 'checkerboard',
                'color0': 280.0,
                'color1': 320.0,
                'to_uv': mi.ScalarTransform4f().scale([1, 1, 1]),
            }
        }
    })
    emitter = shape.emitter()
    assert emitter.flags() & int(mi.EmitterFlags.SpatiallyVarying)

    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wavelengths = mi.Spectrum([10000.0] * 4)
    si.wi = mi.Vector3f(0, 0, 1)
    si.uv = mi.Point2f(0.25, 0.25)
    v0 = np.array(emitter.eval(si)).ravel()[0]
    si.uv = mi.Point2f(0.75, 0.25)
    v1 = np.array(emitter.eval(si)).ravel()[0]
    assert np.isclose(v0, planck_nm(10000.0, 280.0), rtol=2e-4)
    assert np.isclose(v1, planck_nm(10000.0, 320.0), rtol=2e-4)


def test07_sample_direction_consistency(variants_all_spectral):
    # sample_direction(), eval_direction() and pdf_direction() must agree
    shape = make_shape(T=300.0)
    emitter = shape.emitter()

    it = dr.zeros(mi.Interaction3f)
    it.p = mi.Point3f(0, 0, 5)
    it.wavelengths = mi.Spectrum([8000.0, 10000.0, 12000.0, 14000.0])

    for u in [(0.1, 0.2), (0.5, 0.5), (0.9, 0.3)]:
        ds, spec = emitter.sample_direction(it, mi.Point2f(*u))
        pdf = emitter.pdf_direction(it, ds)
        value = emitter.eval_direction(it, ds)
        assert dr.allclose(ds.pdf, pdf, rtol=1e-4)
        assert dr.allclose(spec * ds.pdf, value, rtol=1e-4)
        ref = planck_nm([8000.0, 10000.0, 12000.0, 14000.0], 300.0)
        assert np.allclose(np.array(value).ravel(), ref, rtol=2e-4)
