"""
Tests of the shading-time (linear-response) thermal model: DynamicModel,
the thermal_dynamic emitter and its evaluation inside path_ir/volpath_ir.
"""

import numpy as np
import pytest
import mitsuba as mi


def _weather():
    from mitsuba.ir import thermal as th
    return th.Weather('2026-07-15', 59.33, 18.07, utc_offset=2, T_min=th.celsius(14),
                      T_max=th.celsius(26), relative_humidity=0.55, wind_speed=2.0,
                      cloud_cover=0.0, aod500=0.15)


def test01_impulse_response_properties(variant_scalar_spectral):
    from mitsuba.ir import thermal as th
    from mitsuba.ir.thermal.dynamic import impulse_response, periodise
    dt = 600.0
    G = impulse_response(th.MATERIALS['asphalt'], 13.3, dt, 6 * 144, 295.0)
    assert G[0] > 0 and np.all(np.diff(G) < 0)               # positive, monotonically decaying
    # Step response (cumulative) approaches the steady-state gain 1/h_total
    h_total = 13.3 + 4 * 0.95 * 5.670374419e-8 * 295.0 ** 3
    G_thin = impulse_response(th.MATERIALS['steel_sheet_insulated'], 13.3, dt, 2000, 295.0)
    assert np.isclose(G_thin.sum(), 1.0 / h_total, rtol=0.02)
    Gp = periodise(G, 144)
    assert len(Gp) == 144 and Gp[0] >= G[0] and np.isclose(Gp.sum(), G.sum(), rtol=1e-9)


def test02_dynamic_matches_texture_solver(variants_vec_backends_once_spectral):
    # The linear-response temperature of texels (NumPy reference formula with
    # the texel's own shadow history) must agree with the full per-texel
    # solver, including under the roof and in the roof's moving shadow
    from mitsuba.ir import thermal as th
    w = _weather()
    scene = {
        'type': 'scene',
        'ground': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(10)},
        'roof': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().translate([0, 2, 0]).rotate([1, 0, 0], -90).scale(3)},
    }
    ts = th.ThermalScene(scene, w, {'ground': th.MATERIALS['asphalt']}, resolution=24,
                         up=(0, 1, 0), north=(0, 0, -1), sky_rays=64, exchange=False)
    dt = 600.0
    times = [6.0, 11.0, 14.0, 17.0, 22.0]
    res = ts.run(days=4, dt=dt, output_times=times)
    model = ts.dynamic_models(dt=dt)['ground']
    sh = ts.shapes['ground']
    errs = []
    for t in times:
        T_tex = res.temperature('ground', t).ravel()[sh.idx]
        hist = np.array([v['ground'] if v is not None else np.zeros(sh.E) for v in ts._vis_cache])  # (steps, E)
        T_dyn = np.array([model.temperature(t, hist[:, e], F_sky=sh.F_sky[e]) for e in range(sh.E)])
        errs.append(np.abs(T_dyn - T_tex))
    errs = np.array(errs)
    # Shadow contrast present (>10 K at 14:00) and reproduced to < 1 K
    T14 = res.temperature('ground', 14.0).ravel()[sh.idx]
    assert T14.max() - T14.min() > 10.0
    assert errs.max() < 1.0, errs.max()
    assert errs.mean() < 0.3


def test03_dynamic_emitter_constant(variants_all_spectral):
    # Zero forcing tables: the emitter is a plain thermal surface at t_ref,
    # for both BSDF-sampled hits and next-event estimation (isothermal cavity)
    from mitsuba import ir
    T = 300.0
    em = {'type': 'thermal_dynamic', 't_ref': T, 'alpha_solar': 0.9, 'weights': '', 'sun_directions': '',
          'c_ref': 0.0, 'c_sky': 0.0, 'h_total': 0.0, 'h_conv': 13.3, 'emissivity': 0.9}
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'path_ir', 'max_depth': 32, 'rr_depth': 32, 'thermal_sky_rays': 0},
        'sensor': {'type': 'radiancemeter',
                   'to_world': mi.ScalarTransform4f().look_at(origin=[0, 0, 0], target=[0.3, 0.2, 1], up=[0, 1, 0]),
                   'sampler': {'type': 'independent', 'sample_count': 64},
                   'film': ir.band_film((9990.0, 10010.0), 1, 1, rfilter='box')},
        'cavity': {'type': 'sphere', 'radius': 1.0,
                   'bsdf': {'type': 'twosided', 'bsdf': {'type': 'diffuse', 'reflectance': 0.6}},
                   'emitter': em},
    })
    e = scene.emitters()[0]
    assert e.flags() & int(mi.EmitterFlags.ThermalDynamic)
    value = float(np.array(mi.render(scene, spp=64)).ravel()[0])
    ref = ir.band_radiance(9990.0, 10010.0, T)
    assert np.isclose(value, ref, rtol=1.5e-2), (value, ref)


def test04_dynamic_render_shadow_and_resolution(variants_vec_backends_once_spectral):
    # A plate half covered by a cover lying 2 cm above it, rendered with the
    # dynamic emitter: the covered half is colder, the uncovered temperature
    # equals the reference formula, and the edge is sharp at every image
    # resolution (the optical shadow of a 2 cm cover moves < 2 cm over the
    # day, unlike that of a high roof whose thermal shadow is physically soft)
    from mitsuba.ir import thermal as th
    from mitsuba import ir
    w = _weather()
    geom = {
        'type': 'scene',
        'ground': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(4),
                   'bsdf': {'type': 'diffuse', 'reflectance': 0.0}},
        'cover': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().translate([-50, 0.02, 0]).rotate([1, 0, 0], -90).scale(50)},
    }
    ts = th.ThermalScene(geom, w, {'ground': th.MATERIALS['asphalt']}, resolution=8,
                         up=(0, 1, 0), north=(0, 0, -1), sky_rays=16)
    t = 13.0
    scene = ts.dynamic(geom, t)
    lo, hi = 9990.0, 10010.0
    widths = []
    for res in (32, 128):
        d = dict(scene)
        d['integrator'] = {'type': 'path_ir', 'max_depth': 2, 'thermal_up': [0, 1, 0], 'thermal_sky_rays': 4}
        d['sensor'] = {'type': 'orthographic',
                       'to_world': mi.ScalarTransform4f().look_at(origin=[0, 10, 0], target=[0, 0, 0], up=[0, 0, -1]).scale([4, 4, 1]),
                       'sampler': {'type': 'independent', 'sample_count': 8},
                       'film': ir.band_film((lo, hi), res, res, rfilter='box')}
        img = np.array(mi.render(mi.load_dict(d), spp=8))[..., 0]
        T = ir.apparent_temperature(img, lo, hi)
        row = T[res // 2]
        # sunlit side vs shaded side
        assert row[-2] - row[1] > 8.0, (row[1], row[-2])
        # 10-90 % transition width in pixels
        lo_v, hi_v = row[1], row[-2]
        inside = (row > lo_v + 0.1 * (hi_v - lo_v)) & (row < lo_v + 0.9 * (hi_v - lo_v))
        widths.append(int(inside.sum()))
    # The edge occupies at most ~2 pixels at both resolutions (25 cm and 6 cm per
    # pixel): it is resolved by the renderer, not by a texture
    assert widths[0] <= 2 and widths[1] <= 3, widths
    # Unshaded temperature equals the reference element's (same normal, fully exposed)
    model = ts.dynamic_models()['ground']
    from mitsuba.ir.thermal.dynamic import step_index
    assert abs(T[res // 2, -2] - model.T_ref[step_index(t, model.dt, model.n_steps)]) < 0.3


def test05_volpath_ir_evaluates_dynamic(variants_vec_backends_once_spectral):
    from mitsuba.ir import thermal as th
    from mitsuba import ir
    w = _weather()
    geom = {'type': 'scene',
            'ground': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(4),
                       'bsdf': {'type': 'diffuse', 'reflectance': 0.0}}}
    ts = th.ThermalScene(geom, w, {'ground': th.MATERIALS['asphalt']}, resolution=4, up=(0, 1, 0), north=(0, 0, -1), sky_rays=8)
    d = ts.dynamic(geom, 13.0)
    lo, hi = 9990.0, 10010.0
    vals = []
    for integ in ['path_ir', 'volpath_ir']:
        dd = dict(d)
        dd['integrator'] = {'type': integ, 'max_depth': 2, 'thermal_up': [0, 1, 0], 'thermal_sky_rays': 4}
        dd['sensor'] = {'type': 'radiancemeter',
                        'to_world': mi.ScalarTransform4f().look_at(origin=[1, 3, 1], target=[1, 0, 1], up=[0, 0, -1]),
                        'sampler': {'type': 'independent', 'sample_count': 16},
                        'film': ir.band_film((lo, hi), 1, 1, rfilter='box')}
        vals.append(ir.apparent_temperature(float(np.array(mi.render(mi.load_dict(dd), spp=16)).ravel()[0]), lo, hi))
    assert abs(vals[0] - vals[1]) < 0.2
    model = ts.dynamic_models()['ground']
    from mitsuba.ir.thermal.dynamic import step_index
    assert abs(vals[0] - model.T_ref[step_index(13.0, model.dt, model.n_steps)]) < 0.3


def test06_exchange_dynamic_matches_texture(variants_vec_backends_once_spectral):
    # Roof over asphalt with radiative exchange: the ground under the roof
    # sees the (warm) roof underside. The shading-time model with exchange
    # fields must reproduce the per-element solver.
    from mitsuba.ir import thermal as th
    w = _weather()
    scene = {
        'type': 'scene',
        'ground': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(10),
                   'bsdf': {'type': 'diffuse', 'reflectance': 0.08}},
        'roof': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().translate([0, 2, 0]).rotate([1, 0, 0], -90).scale(3),
                 'bsdf': {'type': 'diffuse', 'reflectance': 0.25}},
    }
    mats = {'ground': th.MATERIALS['asphalt'], 'roof': th.MATERIALS['steel_sheet_painted']}
    ts = th.ThermalScene(scene, w, mats, resolution=24, up=(0, 1, 0), north=(0, 0, -1), sky_rays=64, exchange=True)
    dt = 600.0
    t = 14.0
    res = ts.run(days=4, dt=dt, output_times=[t])
    d = ts.dynamic(scene, t, dt=dt, days=4)
    assert 'exchange_lw_0' in d['ground']['emitter'] and 'exchange_lw_1' in d['ground']['emitter']
    # Compare the NumPy reference formula (with the texel's own shadow history,
    # sky view and exchange view factors) with the solver, texel by texel
    sh = ts.shapes['ground']
    model = ts.dynamic_models(dt=dt)['ground']
    models = ts.dynamic_models(dt=dt)
    hist = np.array([v['ground'] if v is not None else np.zeros(sh.E) for v in ts._vis_cache])
    T_tex = res.temperature('ground', t).ravel()[sh.idx]
    # exchange term: sum over view-factor entries of the ground rows
    base = ts._offsets['ground']
    rows, cols, wv = ts._vf_rows, ts._vf_cols, ts._vf_w
    sel = (rows >= base) & (rows < base + sh.E)
    H_lw = model.convolve_history(t, ts._J_hist)       # per element, receiver = ground material
    H_sw = model.convolve_history(t, ts._Eout_hist)
    h_air, s_ref = model.baselines(t)
    m = sh.material
    exch = np.bincount(rows[sel] - base, weights=wv[sel] * (m.emissivity * (H_lw[cols[sel]] - h_air)
                                                           + m.alpha_solar * (H_sw[cols[sel]] - s_ref)), minlength=sh.E)
    T_dyn = np.array([model.temperature(t, hist[:, e], F_sky=sh.F_sky[e]) for e in range(sh.E)])
    # add the exchange deviation inside the correction (same as the integrator): recompute with it
    from mitsuba.ir.thermal.dynamic import step_index
    s_now = step_index(t, model.dt, model.n_steps)
    T_dyn2 = []
    for e in range(sh.E):
        g = model.G_per[model._lags(t)]
        wk = g * model.dni
        q_sky = (m.alpha_solar * (model.dhi - w.ground_albedo * model.ghi) + m.emissivity * (model.L_sky - 5.670374419e-8 * model.T_air ** 4))
        dT = m.alpha_solar * (np.sum(wk * hist[:, e]) - np.sum(wk * model.cos_ref)) + (sh.F_sky[e] - 1) * np.sum(g * q_sky) + exch[e]
        T_mid = model.T_ref[s_now] + 0.5 * dT
        h_act = model.h_conv + 4 * m.emissivity * 5.670374419e-8 * T_mid ** 3
        T_dyn2.append(model.T_ref[s_now] + dT * model.h_total / max(h_act, 0.25 * model.h_total))
    T_dyn2 = np.array(T_dyn2)
    under = sh.F_sky < 0.5
    assert under.any()
    assert np.abs(T_dyn2 - T_tex).max() < 1.2, np.abs(T_dyn2 - T_tex).max()
    assert np.abs(T_dyn2 - T_tex)[under].mean() < 0.6
    # Rendering the dynamic scene reproduces the texture solver under the roof
    # (radiancemeter looking at the centre of the ground from below the roof)
    from mitsuba import ir
    d['integrator'] = {'type': 'path_ir', 'max_depth': 2, 'thermal_up': [0, 1, 0], 'thermal_sky_rays': 64}
    d['sensor'] = {'type': 'radiancemeter',
                   'to_world': mi.ScalarTransform4f().look_at(origin=[0, 1.0, 0], target=[0, 0, 0], up=[0, 0, -1]),
                   'sampler': {'type': 'independent', 'sample_count': 16},
                   'film': ir.band_film((9990.0, 10010.0), 1, 1, rfilter='box')}
    L = float(np.array(mi.render(mi.load_dict(d), spp=16)).ravel()[0])
    # asphalt diffuse 0.08 -> emissivity 0.92 at 10 um; it reflects 8 % of the roof underside
    T_app = ir.apparent_temperature(L, 9990.0, 10010.0)
    T_centre = res.temperature('ground', t)[12, 12]
    T_roof = res.temperature('roof', t).mean()
    L_ref = 0.92 * ir.band_radiance(9990.0, 10010.0, T_centre) + 0.08 * (0.75 * ir.band_radiance(9990.0, 10010.0, T_roof) + 0.25 * 0)
    assert abs(T_app - ir.apparent_temperature(L_ref, 9990.0, 10010.0)) < 2.0, (T_app, T_centre, T_roof)


# ---------------------------------------------------------------------------
# Regression tests for the code review of 2026-11-15
# ---------------------------------------------------------------------------

def _plate_geom():
    return {'type': 'scene',
            'ground': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(4),
                       'bsdf': {'type': 'diffuse', 'reflectance': 0.0}}}


def _plate_scene(ts_geom_extra=None, twosided=True, t=13.0, sky_rays=4):
    # The thermal tables are built under a JIT variant (ThermalScene requires
    # one); the emitter dictionary holds plain values and renders in any
    # spectral variant, including scalar ones
    from mitsuba.ir import thermal as th
    variant = mi.variant()
    with mi.scoped_set_variant('llvm_ad_spectral'):
        geom = _plate_geom()
        ts = th.ThermalScene(geom, _weather(), {'ground': th.MATERIALS['asphalt']}, resolution=4,
                             up=(0, 1, 0), north=(0, 0, -1), sky_rays=8, exchange=False)
        em = ts.dynamic(geom, t, twosided=twosided, exchange=False)['ground']['emitter']
    d = _plate_geom()                       # rebuilt in the caller's variant
    d['ground']['emitter'] = em
    if ts_geom_extra:
        d.update(ts_geom_extra)
    return ts, d


def _radiance(d, origin, target, up, integ='path_ir', sky_rays=4, spp=16):
    from mitsuba import ir
    dd = dict(d)
    dd['integrator'] = {'type': integ, 'max_depth': 2, 'thermal_up': [0, 1, 0], 'thermal_sky_rays': sky_rays}
    dd['sensor'] = {'type': 'radiancemeter',
                    'to_world': mi.ScalarTransform4f().look_at(origin=origin, target=target, up=up),
                    'sampler': {'type': 'independent', 'sample_count': spp},
                    'film': ir.band_film((9990.0, 10010.0), 1, 1, rfilter='box')}
    return float(np.array(mi.render(mi.load_dict(dd), spp=spp)).ravel()[0])


def test07_scalar_sky_rays_hit_non_emitter(variant_scalar_spectral):
    from mitsuba.ir import thermal as th
    with pytest.raises(RuntimeError, match='JIT variant'):
        th.ThermalScene(_plate_geom(), _weather(), {'ground': th.MATERIALS['asphalt']})
    # Sky rays hitting geometry without an emitter (and escaping into an empty
    # background) used to dereference a null EmitterPtr in scalar variants
    from mitsuba import ir
    extra = {'post': {'type': 'cube', 'to_world': mi.ScalarTransform4f().translate([2.5, 0.5, 0]).scale(0.5)}}
    ts, d = _plate_scene(extra)
    for integ in ('path_ir', 'volpath_ir'):
        L = _radiance(d, [0, 3, 0], [0, 0, 0], [0, 0, -1], integ=integ, sky_rays=8)
        assert np.isfinite(L) and L > 0


def test08_one_sided_dynamic_back_face(variants_all_spectral):
    # A one-sided thermal_dynamic emitter must not emit towards its back
    # (BSDF hits and NEE agree), for both IR integrators
    ts, d = _plate_scene(twosided=False)
    for integ in ('path_ir', 'volpath_ir'):
        front = _radiance(d, [0, 3, 0], [0, 0, 0], [0, 0, -1], integ=integ)
        back = _radiance(d, [0, -3, 0], [0, 0, 0], [0, 0, 1], integ=integ)
        assert front > 0 and back == 0.0, (integ, front, back)


def test09_null_bsdf_box_is_transparent(variants_vec_backends_once_spectral):
    # A null-BSDF box (e.g. a medium boundary from ir.medium_box) above the
    # plate must not shadow it or hide the sky in the shading-time model
    from mitsuba import ir
    ts, d = _plate_scene()
    L0 = _radiance(d, [0.5, 3, 0.5], [0.5, 0, 0.5], [0, 0, -1], sky_rays=8, spp=64)
    d2 = dict(d)
    d2['box'] = {'type': 'cube', 'to_world': mi.ScalarTransform4f().translate([0, 1.0, 0]).scale([50, 0.5, 50]),
                 'bsdf': {'type': 'null'}}
    L1 = _radiance(d2, [0.5, 3, 0.5], [0.5, 0, 0.5], [0, 0, -1], sky_rays=8, spp=64)
    T0 = ir.apparent_temperature(L0, 9990.0, 10010.0)
    T1 = ir.apparent_temperature(L1, 9990.0, 10010.0)
    assert abs(T0 - T1) < 0.3, (T0, T1)


def test10_dynamic_does_not_change_exchange_setting(variant_llvm_ad_spectral):
    from mitsuba.ir import thermal as th
    w = _weather()
    geom = {'type': 'scene',
            'ground': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(5),
                       'bsdf': {'type': 'diffuse', 'reflectance': 0.08}},
            'roof': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().translate([0, 2, 0]).rotate([1, 0, 0], -90).scale(2),
                     'bsdf': {'type': 'diffuse', 'reflectance': 0.25}}}
    mats = {'ground': th.MATERIALS['asphalt'], 'roof': th.MATERIALS['steel_sheet_painted']}
    ts = th.ThermalScene(geom, w, mats, resolution=8, up=(0, 1, 0), north=(0, 0, -1), sky_rays=32, exchange=False)
    T_before = ts.run(days=2, dt=600.0, output_times=[22.0]).temperature('ground', 22.0)
    ts.dynamic(geom, 22.0, exchange=True, days=2)
    assert ts.exchange is False
    T_after = ts.run(days=2, dt=600.0, output_times=[22.0]).temperature('ground', 22.0)
    assert np.allclose(T_before, T_after)
    d = ts.dynamic(geom, 22.0)                      # default: the scene's setting (no exchange)
    assert 'exchange_lw_0' not in d['ground']['emitter']


def test11_weather_change_invalidates_caches(variant_llvm_ad_spectral):
    import dataclasses
    from mitsuba.ir import thermal as th
    w = _weather()
    geom = {'type': 'scene',
            'ground': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(5),
                       'bsdf': {'type': 'diffuse', 'reflectance': 0.08}}}
    ts = th.ThermalScene(geom, w, {'ground': th.MATERIALS['asphalt']}, resolution=4, up=(0, 1, 0),
                         north=(0, 0, -1), sky_rays=8, exchange=False)
    t_summer = ts.dynamic(geom, 13.0)['ground']['emitter']['t_ref']
    ts.weather = dataclasses.replace(w, date='2026-12-21')
    t_winter = ts.dynamic(geom, 13.0)['ground']['emitter']['t_ref']
    assert t_winter < t_summer - 10.0
    # Weather objects derived with replace() / mutated do not reuse cached sunrise times
    w2 = dataclasses.replace(w, date='2026-12-21')
    assert w2.sunrise_sunset() != w.sunrise_sunset()
    w.date = '2026-12-21'
    assert w.sunrise_sunset() == w2.sunrise_sunset()


def test12_medium_temperature_update(variants_vec_backends_once_spectral):
    m = mi.load_dict({'type': 'homogeneous', 'sigma_t': 1.0, 'albedo': 0.0})
    assert not m.is_thermal()
    params = mi.traverse(m)
    key = [k for k in params.keys() if k.startswith('temperature')][0]
    params[key] = 300.0
    params.update()
    assert m.is_thermal()
