#!/usr/bin/env python3
"""
Example 8 - The thermal sky.

Demonstrates the ``thermalsky`` emitter: a clear sky whose LWIR radiance
increases from the zenith (low effective emissivity, cold apparent
temperature) to the horizon (opaque air at the air temperature). The zenith
emissivity is taken from Idso's (1981) empirical clear-sky formula for the
8-14 um window as a function of air temperature and humidity.

Two things are rendered:
1. a fisheye-like sweep of apparent sky temperature versus zenith angle,
2. a polished aluminium plate tilted at several angles: as a near-perfect
   mirror it "sees" the sky and reads much colder than its true temperature -
   the classic error source of thermography on metal surfaces.

Run:  python 08_sky.py [variant]
"""

import sys
import numpy as np
import mitsuba as mi

mi.set_variant(sys.argv[1] if len(sys.argv) > 1 else 'llvm_ad_spectral')
from mitsuba import ir
from _util import output_dir

BAND = 'lwir'
lo, hi = ir.BANDS[BAND]
T_AIR, RH = 293.15, 0.6
e_a = ir.vapour_pressure(T_AIR, RH)
eps_z = ir.sky_emissivity_idso(T_AIR, e_a, '8-14')
print(f'air {T_AIR:.2f} K, RH {RH:.0%} -> vapour pressure {e_a:.1f} hPa, '
      f'Idso zenith emissivity (8-14 um) = {eps_z:.3f}, full spectrum = {ir.sky_emissivity_idso(T_AIR, e_a):.3f}')

sky = ir.thermalsky(T_AIR, relative_humidity=RH, band='8-14')


def radiancemeter(origin, target, spp=64):
    return {'type': 'radiancemeter',
            'to_world': mi.ScalarTransform4f().look_at(origin=origin, target=target, up=[0, 1, 0]),
            'sampler': {'type': 'independent', 'sample_count': spp},
            'film': ir.band_film(BAND, 1, 1, rfilter='box')}


# 1. Sky radiance versus zenith angle (no geometry at all)
print('\nSky apparent temperature vs zenith angle:')
for theta in [0, 30, 60, 75, 85, 89]:
    th = np.deg2rad(theta)
    d = [np.sin(th), 0.0, np.cos(th)]
    scene = {'type': 'scene', 'integrator': {'type': 'path_ir'}, 'sensor': radiancemeter([0, 0, 0], d, 4), 'sky': sky}
    L = float(np.array(mi.render(mi.load_dict(scene), spp=4)).ravel()[0])
    print(f'  theta = {theta:2d} deg: L = {L:6.2f} W m^-2 sr^-1, T_app = {ir.apparent_temperature(L, lo, hi):6.1f} K')

# 2. Polished aluminium plate at 300 K, tilted; camera looks at it along the
#    mirror direction of the zenith angle it reflects
print('\nPolished Al plate at 300 K (true), viewed at angle theta -> mirrors the sky at zenith angle theta:')
for theta in [0, 45, 70, 85]:
    th = np.deg2rad(theta)
    scene = {
        'type': 'scene',
        'integrator': {'type': 'path_ir', 'max_depth': 4},
        'sensor': radiancemeter([np.sin(th), 0, np.cos(th)], [0, 0, 0], 16),
        'plate': {'type': 'rectangle', 'bsdf': ir.conductor_ir('Al_ir'),
                  'emitter': {'type': 'thermal', 'temperature': 300.0}},
        'sky': sky,
    }
    L = float(np.array(mi.render(mi.load_dict(scene), spp=16)).ravel()[0])
    print(f'  theta = {theta:2d} deg: T_app = {ir.apparent_temperature(L, lo, hi):6.1f} K')

# 3. Image: hemispherical view of the sky with a thermal ground plane
scene = {
    'type': 'scene',
    'integrator': {'type': 'path_ir', 'max_depth': 6},
    'sensor': {'type': 'perspective', 'fov': 120,
               'to_world': mi.ScalarTransform4f().look_at(origin=[0, 1.5, 0], target=[0, 1.5, 1], up=[0, 1, 0]),
               'sampler': {'type': 'independent', 'sample_count': 64},
               'film': ir.band_film(BAND, 640, 400)},
    'ground': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(500),
               'bsdf': {'type': 'diffuse', 'reflectance': 0.08},
               'emitter': {'type': 'thermal', 'temperature': 295.0}},
    'sky': ir.thermalsky(T_AIR, relative_humidity=RH, band='8-14', up=[0, 1, 0]),
}
img = np.array(mi.render(mi.load_dict(scene), spp=64))[..., 0]
T_app = ir.apparent_temperature(img, lo, hi)
try:
    from _util import save_temperature_png
    save_temperature_png(T_app, f'{output_dir()}/08_sky.png', title='clear LWIR sky over a 295 K ground (apparent temperature)')
except Exception as e:
    print('image export skipped:', e)
