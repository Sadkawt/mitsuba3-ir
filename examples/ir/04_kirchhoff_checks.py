#!/usr/bin/env python3
"""
Example 4 - Analytic checks of the thermal emission model.

Renders a few configurations with closed-form solutions and prints rendered
versus analytic band radiance (report, section 6):

1. diffuse plate (albedo rho, T) against a cold background:  (1 - rho) B(T)
2. the same plate inside a blackbody cavity at T2:  (1 - rho) B(T) + rho B(T2)
3. camera inside an isothermal cavity with a rough-conductor wall:  B(T)
4. smooth conductor plate seen at several angles:  (1 - F(theta)) B(T)

Run:  python 04_kirchhoff_checks.py [variant]
"""

import sys
import numpy as np
import mitsuba as mi

mi.set_variant(sys.argv[1] if len(sys.argv) > 1 else 'scalar_spectral')
from mitsuba import ir

BAND = (9500.0, 10500.0)


def sensor(origin, target, spp):
    return {
        'type': 'radiancemeter',
        'to_world': mi.ScalarTransform4f().look_at(origin=origin, target=target, up=[0, 1, 0]),
        'sampler': {'type': 'independent', 'sample_count': spp},
        'film': ir.band_film(BAND, 1, 1, rfilter='box'),
    }


def render(scene, spp):
    return float(np.array(mi.render(mi.load_dict(scene), spp=spp)).ravel()[0])


def report(name, value, ref):
    print(f'{name:48s} rendered {value:9.4f}   analytic {ref:9.4f}   rel. error {abs(value - ref) / ref:8.2e}')


B = lambda T: ir.band_radiance(*BAND, T)
rho, T, T2 = 0.3, 300.0, 280.0

# 1. plate against cold background (exact estimator: zero variance)
scene = {'type': 'scene', 'integrator': {'type': 'path_ir', 'max_depth': 8},
         'sensor': sensor([0, 0, 1], [0, 0, 0], 16),
         'plate': {'type': 'rectangle', 'bsdf': {'type': 'diffuse', 'reflectance': rho},
                   'emitter': {'type': 'thermal', 'temperature': T}}}
report('1. diffuse plate, cold background', render(scene, 16), (1 - rho) * B(T))

# 2. plate inside a blackbody cavity
scene['cavity'] = {'type': 'sphere', 'radius': 5.0, 'bsdf': {'type': 'diffuse', 'reflectance': 0.0},
                   'emitter': {'type': 'thermal', 'temperature': T2}}
scene['sensor'] = sensor([0, 0, 1], [0, 0, 0], 1024)
report('2. diffuse plate in blackbody cavity (280 K)', render(scene, 1024), (1 - rho) * B(T) + rho * B(T2))

# 3. isothermal cavity, rough conductor wall
scene = {'type': 'scene', 'integrator': {'type': 'path_ir', 'max_depth': 64, 'rr_depth': 64},
         'sensor': sensor([0, 0, 0], [0.3, 0.2, 1], 4096),
         'cavity': {'type': 'sphere', 'radius': 1.0,
                    'bsdf': {'type': 'twosided', 'bsdf': ir.conductor_ir('Al_ir', alpha=0.3)},
                    'emitter': {'type': 'thermal', 'temperature': T}}}
# L = B is the exact solution inside an isothermal cavity for *any* BSDF. For a
# diffuse wall the estimator has zero variance; for a glossy wall the MIS
# between emitter and BSDF sampling leaves ~1 % noise at 256 spp (0.2 % at 4096).
report('3. isothermal cavity, rough Al wall', render(scene, 4096), B(T))
scene['cavity']['bsdf'] = {'type': 'twosided', 'bsdf': {'type': 'diffuse', 'reflectance': 0.7}}
scene['sensor'] = sensor([0, 0, 0], [0.3, 0.2, 1], 16)
report('3b. isothermal cavity, diffuse wall (zero variance)', render(scene, 16), B(T))

# 4. smooth conductor plate at several angles: directional Kirchhoff law
lam, n, k = ir.materials.load_ior('Fe_ir')
grid = np.linspace(*BAND, 201)
for theta_deg in [0, 30, 60, 80]:
    th = np.deg2rad(theta_deg)
    scene = {'type': 'scene', 'integrator': {'type': 'path_ir', 'max_depth': 8},
             'sensor': sensor([np.sin(th), 0, np.cos(th)], [0, 0, 0], 16),
             'plate': {'type': 'rectangle', 'bsdf': ir.conductor_ir('Fe_ir'),
                       'emitter': {'type': 'thermal', 'temperature': T}}}
    F = ir.materials.fresnel_conductor(np.cos(th), np.interp(grid, lam, n), np.interp(grid, lam, k))
    ref = np.trapezoid((1 - F) * ir.planck(grid, T), grid)
    report(f'4. smooth Fe plate, theta = {theta_deg:2d} deg', render(scene, 16), ref)
