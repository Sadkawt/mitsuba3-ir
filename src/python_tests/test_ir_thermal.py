"""
Tests of mitsuba.ir.thermal: solar geometry, clear-sky irradiance, the
weather model, the 1-D conduction solver and the scene coupling (shadows,
sky-view factors, temperature textures).
"""

import datetime as dt
import numpy as np
import pytest
import mitsuba as mi


def test01_solar_position(variant_scalar_spectral):
    from mitsuba.ir import thermal as th
    # Summer solstice, Stockholm, at solar noon (~10:49 UTC): elevation = 90 - lat + dec
    pos = th.solar_position(59.33, 18.07, dt.datetime(2026, 6, 21, 10, 49))
    assert abs(pos['elevation'] - (90.0 - 59.33 + 23.44)) < 0.15
    assert abs(pos['azimuth'] - 180.0) < 1.0
    assert abs(pos['declination'] - 23.44) < 0.05
    # Winter solstice, same place, solar noon
    pos = th.solar_position(59.33, 18.07, dt.datetime(2026, 12, 21, 10, 48))
    assert abs(pos['elevation'] - (90.0 - 59.33 - 23.44)) < 0.15
    # Southern hemisphere: sun in the north at noon
    pos = th.solar_position(-33.9, 151.2, dt.datetime(2026, 6, 21, 2, 0))   # Sydney ~12:00 AEST
    assert abs(pos['azimuth']) < 5.0 or abs(pos['azimuth'] - 360.0) < 5.0
    assert abs(pos['elevation'] - (90.0 - 33.9 - 23.44)) < 0.3
    # Night
    pos = th.solar_position(59.33, 18.07, dt.datetime(2026, 12, 21, 0, 0))
    assert pos['elevation'] < -40.0
    # Vectorised call and sun_direction conventions
    many = th.solar_position(59.33, 18.07, [dt.datetime(2026, 6, 21, h) for h in range(0, 24, 6)])
    assert many['elevation'].shape == (4,)
    d = th.sun_direction(90.0, 0.0, up=(0, 1, 0), north=(0, 0, -1))
    assert np.allclose(d, [0, 1, 0])
    d = th.sun_direction(0.0, 90.0, up=(0, 1, 0), north=(0, 0, -1))   # east
    assert np.allclose(d, [1, 0, 0], atol=1e-12)


def test02_bird_clear_sky(variant_scalar_spectral):
    from mitsuba.ir import thermal as th
    dni, dhi, ghi = th.bird_clear_sky(0.0, 172, 1013.25, 0.1, None, 1.5, 0.3, 0.2)
    assert 880 < dni < 980 and 80 < dhi < 150 and ghi > dni     # textbook clear-sky noon
    dni2, dhi2, ghi2 = th.bird_clear_sky(60.0, 172)
    assert dni2 < dni and ghi2 < 0.55 * ghi
    assert np.all(np.array(th.bird_clear_sky(95.0, 172)) == 0.0)   # night
    # Hazier atmosphere: less direct, more diffuse
    dni3, dhi3, _ = th.bird_clear_sky(30.0, 172, aod500=0.4)
    dni4, dhi4, _ = th.bird_clear_sky(30.0, 172, aod500=0.05)
    assert dni3 < dni4 and dhi3 > dhi4
    # Clouds reduce global and raise the diffuse fraction
    dni_c, dhi_c, ghi_c = th.cloud_correction(dni, dhi, ghi, 0.0, 0.8)
    assert ghi_c < ghi and dhi_c / ghi_c > dhi / ghi and dni_c < dni


def test03_weather_model(variant_scalar_spectral):
    from mitsuba.ir import thermal as th
    w = th.Weather('2026-07-15', 59.33, 18.07, utc_offset=2, T_min=th.celsius(14), T_max=th.celsius(26),
                   relative_humidity=0.55, wind_speed=2.0, cloud_cover=0.1, aod500=0.15)
    rise, set_ = w.sunrise_sunset()
    assert 3.5 < rise < 4.5 and 21.3 < set_ < 22.3           # Stockholm mid-July, UTC+2
    t = np.linspace(0, 24, 241)
    T = w.air_temperature(t)
    assert abs(T.min() - w.T_min) < 0.3 and abs(T.max() - w.T_max) < 0.3
    assert 14.0 < t[np.argmax(T)] < 16.5                     # maximum in the afternoon
    irr = w.irradiance(12.0)
    assert 600 < float(irr['dni']) < 800 and 600 < float(irr['ghi']) < 900
    assert float(w.irradiance(1.0)['ghi']) == 0.0
    L = w.sky_longwave(t)
    assert np.all((L > 280) & (L < 420))
    assert 10 < w.convection_coefficient() < 16
    assert 2.0 < w.precipitable_water() < 4.0


def test04_column_steady_state(variant_scalar_spectral):
    from mitsuba.ir import thermal as th
    from mitsuba.ir.thermal.heat import Column, equilibrium_surface_temperature
    m = th.MATERIALS['steel_sheet_insulated']
    col = Column(m, 3, 290.0)
    E = np.array([0.0, 300.0, 800.0])
    L = np.full(3, 350.0)
    for _ in range(3000):
        col.step(60.0, E, L, 295.0, 13.3)
    ref = equilibrium_surface_temperature(E, L, 295.0, 13.3, m.emissivity)
    assert np.allclose(col.surface, ref, atol=1e-3)
    # Internal heat flux raises the equilibrium accordingly
    m2 = m.replace(q_internal=200.0)
    col2 = Column(m2, 1, 290.0)
    for _ in range(3000):
        col2.step(60.0, np.zeros(1), np.full(1, 350.0), 295.0, 13.3)
    ref2 = equilibrium_surface_temperature(np.zeros(1), 350.0, 295.0, 13.3, m.emissivity, q=200.0)
    assert np.allclose(col2.surface, ref2, atol=1e-3)
    # Interior boundary: with a very large conductance the slab follows T_internal
    m3 = m.replace(back='interior', T_internal=330.0, U_internal=1e6)
    col3 = Column(m3, 1, 290.0)
    for _ in range(3000):
        col3.step(60.0, np.zeros(1), np.full(1, 350.0), 295.0, 13.3)
    assert abs(col3.T[-1, 0] - 330.0) < 0.05


def test05_column_periodic_penetration(variant_scalar_spectral):
    # Semi-infinite solid with sinusoidal surface temperature: amplitude at
    # depth z decays as exp(-z / delta), delta = sqrt(2 a / omega)
    # (Carslaw & Jaeger 1959, sec. 2.6)
    from mitsuba.ir import thermal as th
    from mitsuba.ir.thermal.heat import Column
    mat = th.Material(k=0.75, rho=2110.0, c=920.0, thickness=None, depth=1.5, n_nodes=40)
    col = Column(mat, 1, 290.0)
    col.T_deep = 290.0
    omega = 2 * np.pi / 86400.0
    for day in range(6):
        hist = []
        for s in range(1440):
            col.step(60.0, np.zeros(1), np.full(1, 5.670374419e-8 * 290.0 ** 4),
                     290.0 + 10.0 * np.sin(omega * s * 60.0), 1e5)
            hist.append(col.T[:, 0].copy())
    hist = np.array(hist)
    z = np.cumsum(mat.node_thicknesses()) - mat.node_thicknesses() / 2
    delta = np.sqrt(2 * mat.diffusivity / omega)
    amp = 0.5 * (hist.max(0) - hist.min(0))
    for zi in [0.05, 0.1, 0.2]:
        i = int(np.argmin(np.abs(z - zi)))
        assert np.isclose(amp[i] / amp[0], np.exp(-z[i] / delta), rtol=0.03)


def test06_thermal_scene_shadows_and_textures(variants_vec_backends_once_spectral):
    from mitsuba.ir import thermal as th
    from mitsuba import ir
    w = th.Weather('2026-07-15', 59.33, 18.07, utc_offset=2, T_min=th.celsius(14), T_max=th.celsius(26),
                   relative_humidity=0.55, wind_speed=2.0, cloud_cover=0.0, aod500=0.15)
    scene = {
        'type': 'scene',
        'ground': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(10)},
        'roof': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().translate([0, 2, 0]).rotate([1, 0, 0], -90).scale(3)},
    }
    # exchange=False: this test checks shadows and the texture round trip;
    # with exchange on, the ground under the roof also sees the warm roof
    # underside, which reduces the shadow contrast (see test09/test10)
    ts = th.ThermalScene(scene, w, {'ground': th.MATERIALS['asphalt'], 'roof': th.MATERIALS['steel_sheet_painted']},
                         resolution=32, up=(0, 1, 0), north=(0, 0, -1), sky_rays=32, exchange=False)
    assert ts.shapes['ground'].mode == 'uv'        # rectangle: a mesh with a unique UV map
    F = ts.shapes['ground'].F_sky
    assert F.min() < 0.3 and F.max() > 0.98        # under the roof vs. open ground
    assert ts.shapes['roof'].F_sky.min() > 0.98
    res = ts.run(days=3, dt=600.0, output_times=[6, 12, 15, 21])
    T12 = res.temperature('ground', 12)
    assert T12.shape == (32, 32)
    centre, corner = T12[16, 16], T12[1, 1]
    assert corner - centre > 10.0                   # thermal shadow under the roof
    T15, T21 = res.temperature('ground', 15), res.temperature('ground', 21)
    assert T15[1, 1] > T12[1, 1]                    # thermal lag: warmer at 15:00 than at noon
    assert T21[1, 1] > w.air_temperature(21.0)      # still warmer than air in the evening
    # Textures and the thermalised scene load and render
    d = res.apply(scene, 15)
    assert d['ground']['emitter']['type'] == 'thermal'
    d['integrator'] = {'type': 'path_ir', 'max_depth': 4}
    d['sensor'] = {'type': 'radiancemeter',
                   'to_world': mi.ScalarTransform4f().look_at(origin=[9, 5, 9], target=[9.2, 0, 9.2], up=[0, 1, 0]),
                   'sampler': {'type': 'independent', 'sample_count': 16},
                   'film': ir.band_film('lwir', 1, 1, rfilter='box')}
    L = float(np.array(mi.render(mi.load_dict(d), spp=16)).ravel()[0])
    T_app = ir.apparent_temperature(L, *ir.BANDS['lwir'])
    # Asphalt (eps 0.92 - diffuse 0.08 reflectance of a black sky) near the sampled corner temperature
    T_corner = res.temperature('ground', 15)[31, 31]
    assert abs(T_app - 0.98 * T_corner) < 8.0


def test07_mesh_vertices_and_ply_export(variants_vec_backends_once_spectral, tmp_path):
    from mitsuba.ir import thermal as th
    from mitsuba import ir
    w = th.Weather('2026-07-15', 45.0, 0.0, utc_offset=0, T_min=th.celsius(15), T_max=th.celsius(25),
                   cloud_cover=0.0, aod500=0.1)
    scene = {
        'type': 'scene',
        'ground': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(20)},
        # Note: a shape with an emitter and *no* BSDF gets a black BSDF in Mitsuba
        # (emissivity 1); give it an explicit one
        'box': {'type': 'cube', 'to_world': mi.ScalarTransform4f().translate([0, 1, 0]),
                'bsdf': {'type': 'diffuse', 'reflectance': 0.5}},
    }
    ts = th.ThermalScene(scene, w, {'box': th.MATERIALS['steel_sheet_insulated']},
                         up=(0, 1, 0), north=(0, 0, -1), sky_rays=32)
    sh = ts.shapes['box']
    assert sh.mode == 'vertex' and sh.E == 24                # cube: overlapping UVs -> vertices (24 split vertices)
    assert ts.shapes['box'].mode == 'vertex'
    res = ts.run(days=2, dt=600.0, output_times=[12.0])
    T = res.temperature('box', 12.0)
    n = sh.n
    top = T[n[:, 1] > 0.5].mean()
    bottom = T[n[:, 1] < -0.5].mean()
    north_face = T[n[:, 2] < -0.5].mean()                    # faces away from the sun (north = -z)
    south_face = T[n[:, 2] > 0.5].mean()
    assert top > south_face > north_face                      # sun from the south at noon, high in the sky
    assert bottom < south_face
    d = res.apply(scene, 12.0, mesh_dir=str(tmp_path))
    assert d['box']['type'] == 'ply' and d['box']['emitter']['temperature']['type'] == 'mesh_attribute'
    d['integrator'] = {'type': 'path_ir', 'max_depth': 2}
    d['sensor'] = {'type': 'radiancemeter',
                   'to_world': mi.ScalarTransform4f().look_at(origin=[0, 6, 0], target=[0, 1, 0], up=[0, 0, 1]),
                   'sampler': {'type': 'independent', 'sample_count': 8},
                   'film': ir.band_film((9990.0, 10010.0), 1, 1, rfilter='box')}
    L = float(np.array(mi.render(mi.load_dict(d), spp=8)).ravel()[0])
    T_app = ir.apparent_temperature(L, 9990.0, 10010.0)
    # top face: diffuse reflectance 0.5 -> emissivity 0.5 -> 0.5 B(T_top)
    assert abs(T_app - ir.apparent_temperature(0.5 * ir.band_radiance(9990.0, 10010.0, top), 9990.0, 10010.0)) < 1.0


def test08_material_from_bsdf(variants_vec_backends_once_spectral):
    import warnings
    from mitsuba import ir
    from mitsuba.ir import thermal as th
    # Constant diffuse: alpha_s = eps = 1 - rho
    a, e = ir.bsdf_thermal_properties({'type': 'diffuse', 'reflectance': 0.3}, n_samples=1024)
    assert abs(a - 0.7) < 1e-3 and abs(e - 0.7) < 1e-3
    # Two-band diffuse: bright in the solar, dark in the LWIR (concrete-like)
    a, e = ir.bsdf_thermal_properties(ir.two_band_diffuse(0.35, 0.08), n_samples=1024)
    assert abs(a - 0.65) < 0.02 and abs(e - 0.92) < 0.01
    # Smooth aluminium: low solar absorptance, very low emissivity
    a, e = ir.bsdf_thermal_properties(ir.conductor_ir('Al_ir'), n_samples=2048)
    assert 0.05 < a < 0.15 and 0.01 < e < 0.04
    # Fixed-incidence albedo at normal incidence equals 1 - F(0) for a smooth conductor
    lam, n, k = ir.materials.load_ior('Al_ir')
    rho_n = ir.albedo_spectrum(ir.conductor_ir('Al_ir'), [10000.0], n_samples=256, incidence=1.0)[0]
    F0 = ir.materials.fresnel_conductor(1.0, np.interp(10000.0, lam, n), np.interp(10000.0, lam, k))
    assert abs(rho_n - F0) < 1e-3
    # ThermalScene adopts the BSDF-derived values and warns about the table values
    w = th.Weather('2026-07-15', 59.33, 18.07, utc_offset=2)
    scene = {'type': 'scene',
             'ground': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(5),
                        'bsdf': {'type': 'diffuse', 'reflectance': 0.08}}}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        ts = th.ThermalScene(scene, w, {'ground': th.MATERIALS['concrete']}, resolution=4, up=(0, 1, 0), north=(0, 0, -1), sky_rays=4)
    assert abs(ts.shapes['ground'].material.alpha_solar - 0.92) < 0.01
    assert any('solar absorptance' in str(r.message) for r in rec)
    ts2 = th.ThermalScene(scene, w, {'ground': th.MATERIALS['concrete']}, resolution=4, up=(0, 1, 0), north=(0, 0, -1),
                          sky_rays=4, material_from_bsdf=False)
    assert ts2.shapes['ground'].material.alpha_solar == th.MATERIALS['concrete'].alpha_solar


def test09_radiative_exchange_two_plates(variants_vec_backends_once_spectral):
    # Plate B (thin, adiabatic back, facing down) 0.5 m above a large plate A
    # held at T_A. Polar night, constant air temperature: B's equilibrium is
    #   eps sigma T_B^4 = eps [f_A J_A + f_other sigma T_air^4] + h (T_air - T_B)
    from mitsuba.ir import thermal as th
    from mitsuba.ir.thermal.heat import SIGMA_SB
    T_A, T_air = 330.0, 265.0
    w = th.Weather('2026-12-21', 80.0, 0.0, utc_offset=0, T_min=T_air, T_max=T_air, wind_speed=2.0)
    assert float(w.irradiance(12.0)['ghi']) == 0.0
    scene = {
        'type': 'scene',
        'A': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(50),
              'bsdf': {'type': 'diffuse', 'reflectance': 0.1}},
        'B': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().translate([0, 0.5, 0]).rotate([1, 0, 0], 90).scale(1),
              'bsdf': {'type': 'diffuse', 'reflectance': 0.1}},
    }
    # Black plates (eps = 1): no reflected longwave, so the balance below is exact
    mats = {'A': th.MATERIALS['steel_sheet_insulated'].replace(back='interior', T_internal=T_A, U_internal=1e6,
                                                               emissivity=1.0),
            'B': th.MATERIALS['steel_sheet_insulated'].replace(emissivity=1.0)}
    for exchange in (True, False):
        ts = th.ThermalScene(scene, w, mats, resolution=8, up=(0, 1, 0), north=(0, 0, -1), sky_rays=64,
                             exchange=exchange, material_from_bsdf=False)
        shB = ts.shapes['B']
        assert shB.F_sky.max() < 1e-6                       # B faces down: no sky
        f_A = shB.f_sim.mean()
        assert f_A > 0.95
        res = ts.run(days=2, dt=600.0, output_times=[12.0])
        T_B = res.temperature('B', 12.0).mean()
        eps, h = mats['B'].emissivity, w.convection_coefficient()
        f_other = 1.0 - f_A
        J_A = eps * SIGMA_SB * T_A ** 4 if exchange else SIGMA_SB * T_air ** 4
        fA = f_A if exchange else 0.0
        fo = f_other if exchange else 1.0
        T = 300.0
        for _ in range(60):
            f = eps * (fA * J_A + fo * SIGMA_SB * T_air ** 4 - SIGMA_SB * T ** 4) + h * (T_air - T)
            T -= f / (-4 * eps * SIGMA_SB * T ** 3 - h)
        assert abs(T_B - T) < 0.3, (exchange, T_B, T)


def test10_wall_sees_hot_ground(variants_vec_backends_once_spectral):
    # A vertical plate over sunlit asphalt is warmer when it exchanges
    # radiation with the (hot) ground than with air-temperature surroundings
    from mitsuba.ir import thermal as th
    w = th.Weather('2026-07-15', 45.0, 0.0, utc_offset=0, T_min=th.celsius(18), T_max=th.celsius(30), cloud_cover=0.0)
    scene = {
        'type': 'scene',
        'ground': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(30),
                   'bsdf': {'type': 'diffuse', 'reflectance': 0.08}},
        'wall': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().translate([0, 1.0, 0]).rotate([0, 1, 0], 180).scale([4, 1, 1]),
                 'bsdf': {'type': 'diffuse', 'reflectance': 0.3}},   # faces north (-z) -> no direct sun
    }
    T = {}
    for exchange in (True, False):
        ts = th.ThermalScene(scene, w, {'ground': th.MATERIALS['asphalt'], 'wall': th.MATERIALS['steel_sheet_insulated']},
                             resolution={'ground': 32, 'wall': 8}, up=(0, 1, 0), north=(0, 0, -1), sky_rays=64,
                             exchange=exchange, material_from_bsdf=False)
        res = ts.run(days=3, dt=600.0, output_times=[15.0])
        T[exchange] = res.temperature('wall', 15.0).mean()
    assert T[True] - T[False] > 1.0, T
