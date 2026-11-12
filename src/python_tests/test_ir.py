"""
Tests of the mitsuba.ir helper package (radiometry, materials, scene helpers)
and of the shipped infrared optical constants.
"""

import numpy as np
import pytest
import mitsuba as mi


def test01_planck_and_band_radiance(variant_scalar_spectral):
    from mitsuba import ir
    # Stefan-Boltzmann: int_0^inf B dlambda = sigma T^4 / pi
    T = 300.0
    L = ir.band_radiance(50.0, 2e6, T, n=400001)
    assert np.isclose(L, ir.radiometry.SIGMA_SB * T**4 / np.pi, rtol=1e-3)
    # Wien's displacement law: peak of B_lambda at b / T
    lam = np.linspace(1000.0, 30000.0, 200001)
    peak = lam[np.argmax(ir.planck(lam, T))]
    assert np.isclose(peak, 2.897771955e-3 / T * 1e9, rtol=1e-3)
    # Known band value (report, section 2.2)
    assert np.isclose(ir.band_radiance(*ir.BANDS['lwir'], 300.0), 54.93, rtol=2e-3)
    # Derivative
    h = 1e-3
    num = (ir.planck(10000.0, T + h) - ir.planck(10000.0, T - h)) / (2 * h)
    assert np.isclose(ir.planck_dT(10000.0, T), num, rtol=1e-5)
    # Cold / short wavelength combinations do not produce NaN/inf
    assert ir.planck(300.0, 50.0) == 0.0


def test02_apparent_temperature_roundtrip(variant_scalar_spectral):
    from mitsuba import ir
    for band in ['swir', 'mwir', 'lwir']:
        lo, hi = ir.BANDS[band]
        T = np.array([250.0, 300.0, 350.0, 800.0])
        L = ir.band_radiance(lo, hi, T)
        T2 = ir.apparent_temperature(L, lo, hi)
        assert np.allclose(T, T2, rtol=2e-3), (band, T, T2)
    # With a non-ideal response function
    lam = np.linspace(7500.0, 13500.0, 61)
    srf = (lam, np.exp(-0.5 * ((lam - 10500.0) / 2000.0) ** 2))
    L = ir.band_radiance(7500.0, 13500.0, 310.0, srf=srf)
    assert np.isclose(ir.apparent_temperature(L, 7500.0, 13500.0, srf=srf), 310.0, rtol=2e-3)


def test03_solar_irradiance(variant_scalar_spectral):
    from mitsuba import ir
    lam = np.geomspace(100.0, 1e6, 200001)
    E = np.trapezoid(ir.solar_irradiance_toa(lam), lam)
    # IAU nominal total solar irradiance 1361 W m^-2 (5772 K blackbody definition)
    assert np.isclose(E, 1361.0, rtol=5e-3)


def test04_drude_lorentz_sanity(variant_scalar_spectral):
    from mitsuba import ir
    # Handbook-level checks: polished metals have ~1-3 % normal emissivity at 10 um
    for metal, eps_max in [('Al', 0.02), ('Au', 0.02), ('Ag', 0.02), ('Cu', 0.015), ('W', 0.03)]:
        n, k = ir.drude_lorentz_nk(metal, 10000.0)
        eps = ir.normal_emissivity(n, k)
        assert 0.003 < eps < eps_max, (metal, eps)
    # Visible-range comparison with Mitsuba's shipped Au data (Johnson & Christy):
    # the LD fit reproduces n,k within ~15 % at 600-800 nm
    lam, n_ref = ir.materials.load_spd(str(mi.file_resolver().resolve('data/ior/Au.eta.spd')))
    _, k_ref = ir.materials.load_spd(str(mi.file_resolver().resolve('data/ior/Au.k.spd')))
    sel = (lam >= 600) & (lam <= 800)
    n, k = ir.drude_lorentz_nk('Au', lam[sel])
    assert np.all(np.abs(k - k_ref[sel]) / k_ref[sel] < 0.15)
    # Hemispherical emissivity is larger than the normal one for metals
    n, k = ir.drude_lorentz_nk('Al', 10000.0)
    assert ir.hemispherical_emissivity(n, k) > ir.normal_emissivity(n, k)


def test05_ir_presets_load_in_conductor(variants_all_spectral):
    from mitsuba import ir
    # Every shipped preset must load through the conductor BSDF
    for name in ir.IR_MATERIALS:
        bsdf = mi.load_dict({'type': 'conductor', 'material': name})
        assert bsdf is not None
    # The Fresnel reflectance of the loaded Al_ir must match the data at 10 um
    bsdf = mi.load_dict(ir.conductor_ir('Al_ir'))
    si = mi.SurfaceInteraction3f()
    si.wi = mi.Vector3f(0, 0, 1)
    si.wavelengths = mi.Spectrum([10000.0] * 4)
    si.sh_frame = mi.Frame3f(mi.Vector3f(0, 0, 1))
    si.n = mi.Normal3f(0, 0, 1)
    ctx = mi.BSDFContext()
    bs, weight = bsdf.sample(ctx, si, 0.5, mi.Point2f(0.5, 0.5))
    lam, n, k = ir.materials.load_ior('Al_ir')
    F = ir.materials.fresnel_conductor(1.0, np.interp(10000.0, lam, n), np.interp(10000.0, lam, k))
    assert np.allclose(np.array(weight).ravel()[0], F, rtol=2e-3)


def test06_window_ior(variant_scalar_spectral):
    from mitsuba import ir
    n, k = ir.window_ior('Ge_ir', 'lwir')
    assert 3.95 < n < 4.05 and k < 1e-3
    n, k = ir.window_ior('ZnSe_ir', 'lwir')
    assert 2.35 < n < 2.45
    n, k = ir.window_ior('H2O_ir', 'lwir')
    assert 1.1 < n < 1.3 and k > 0.03   # water is opaque in the LWIR


def test07_thermalize_and_validate(variant_scalar_spectral):
    from mitsuba import ir
    scene = {
        'type': 'scene',
        'integrator': {'type': 'path'},
        'sensor': {'type': 'perspective', 'film': {'type': 'hdrfilm'}},
        'a': {'type': 'sphere', 'bsdf': {'type': 'diffuse', 'reflectance': [0.5, 0.5, 0.5]}},
        'b': {'type': 'cube', 'emitter': {'type': 'area', 'radiance': 1.0}},
        'c': {'type': 'rectangle', 'bsdf': {'type': 'conductor', 'material': 'Au'}},
    }
    warn = ir.validate(scene, band='lwir')
    text = '\n'.join(warn)
    assert "not thermal-aware" in text
    assert "hdrfilm" in text
    assert "RGB triple" in text
    assert "'Au'" in text
    assert "has no emitter" in text

    out = ir.thermalize(scene, temperature=290.0, overrides={'c': 320.0})
    assert out['a']['emitter'] == {'type': 'thermal', 'temperature': 290.0, 'twosided': True}
    assert out['c']['emitter']['temperature'] == 320.0
    assert out['b']['emitter']['type'] == 'area'          # existing emitters are kept
    assert 'emitter' not in scene['a']                      # input not modified

    out = ir.thermalize(scene, temperature=lambda key, d: None if key == 'a' else 300.0)
    assert 'emitter' not in out['a'] and out['c']['emitter']['temperature'] == 300.0


def test08_sun_and_film_helpers(variants_all_spectral):
    from mitsuba import ir
    # A sun at normal incidence on a diffuse plate (rho) must give rho E / pi
    rho = 0.5
    band = (1490.0, 1510.0)
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'path_ir', 'max_depth': 4},
        'sensor': {
            'type': 'radiancemeter',
            'to_world': mi.ScalarTransform4f().look_at(origin=[0, 0, 1], target=[0, 0, 0], up=[0, 1, 0]),
            'sampler': {'type': 'independent', 'sample_count': 64},
            'film': ir.band_film(band, 1, 1, rfilter='box'),
        },
        'plate': {'type': 'rectangle', 'bsdf': {'type': 'diffuse', 'reflectance': rho},
                  'emitter': {'type': 'thermal', 'temperature': 300.0}},
        # Large sampling weight: next-event estimation otherwise picks the
        # (thermal) plate itself part of the time, which only adds noise
        'sun': ir.sun([0, 0, -1], band='swir', sampling_weight=1e4),
    })
    value = float(np.array(mi.render(scene, spp=64)).ravel()[0])
    lam = np.linspace(*band, 201)
    ref = rho * np.trapezoid(ir.solar_irradiance_toa(lam), lam) / np.pi
    assert np.isclose(value, ref, rtol=5e-3), (value, ref)


def test09_ecostress_loader(variant_scalar_spectral, tmp_path):
    from mitsuba import ir
    text = """Name: Test material
Type: manmade
Measurement: Directional (10 Degree) Hemispherical Reflectance
X Units: Wavelength (micrometers)
Y Units: Reflectance (percent)
First X Value: 2.0
Last X Value: 14.0
Number of X Values: 4

14.0  10.0
8.0  5.0
2.0  40.0
5.0  20.0
"""
    p = tmp_path / 'test.txt'
    p.write_text(text)
    lam, refl, meta = ir.load_ecostress(str(p))
    assert meta['Name'] == 'Test material'
    assert np.allclose(lam, [2000.0, 5000.0, 8000.0, 14000.0])
    assert np.allclose(refl, [0.4, 0.2, 0.05, 0.1])
    bsdf_dict = ir.diffuse_from_emissivity(lam, 1.0 - refl)
    bsdf = mi.load_dict(bsdf_dict)
    assert bsdf is not None


def test10_atmosphere_helpers(variant_scalar_spectral):
    from mitsuba import ir
    # Idso (1981) clear-sky emissivities: full spectrum ~0.8, window ~0.3-0.5
    e_a = ir.vapour_pressure(293.15, 0.6)
    assert 13.0 < e_a < 15.0                      # Magnus: 14.0 hPa
    assert 0.78 < ir.sky_emissivity_idso(293.15, e_a) < 0.88
    assert 0.3 < ir.sky_emissivity_idso(293.15, e_a, '8-14') < 0.5
    assert ir.sky_emissivity_idso(293.15, e_a, '10.5-12.5') < ir.sky_emissivity_idso(293.15, e_a, '8-14')
    # RSB continuum: a few 1e-2 per km in the window at 6-7 g/m^3, increasing
    # with wavelength and quadratically with vapour pressure
    k = ir.rsb_continuum(np.array([8000.0, 10000.0, 12000.0]), 288.15, 8.5)
    assert np.all(np.diff(k) > 0) and 2e-5 < k[1] < 1e-4
    # k ~ e (e + gamma (P - e)): doubling e gives x3.6 at 1 atm (x4 without the foreign term)
    ratio = ir.rsb_continuum(10000.0, 288.15, 17.0) / ir.rsb_continuum(10000.0, 288.15, 8.5)
    assert 3.5 < ratio < 4.0
    # thermalsky / air_medium dictionaries load
    sky = mi.load_dict(ir.thermalsky(290.0, relative_humidity=0.5))
    assert sky is not None
    med = mi.load_dict(ir.air_medium(288.15, ([8000.0, 10000.0, 14000.0], [1e-4, 2e-4, 5e-4])))
    assert med.is_thermal()
    ir.transmittance(([1.0, 2.0], [0.1, 0.2]), 10.0)
