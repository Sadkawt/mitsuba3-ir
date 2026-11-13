#!/usr/bin/env python3
"""
Example 11 - Shading-time thermal shadows (no temperature texture).

The scene of example 10 is rendered with ``thermal_dynamic`` emitters: the
integrator evaluates the surface temperature at every shading point from
the linear-response model (reference temperature + convolution of the
point's own shadow history with the column's response function), casting
shadow rays towards the sun positions of the day and a few sky rays. Shadows
are therefore as sharp as the geometry at any image resolution, and the cost
scales with pixels x samples instead of with a texture resolution.

The script renders a close-up of the wall/tree shadows with the texture
solver (160 x 160 texels on a 40 m ground, i.e. 25 cm) and with the dynamic
model, and reports timings.

Run:  python 11_thermal_dynamic.py [variant]
"""

import sys
import time
import numpy as np
import mitsuba as mi

mi.set_variant(sys.argv[1] if len(sys.argv) > 1 else 'llvm_ad_spectral')
from mitsuba import ir
from mitsuba.ir import thermal as th
from _util import output_dir, save_grid_png

UP, NORTH = (0, 1, 0), (0, 0, -1)
BAND = 'lwir'
lo, hi = ir.BANDS[BAND]
T = mi.ScalarTransform4f

weather = th.Weather('2026-07-15', latitude=59.33, longitude=18.07, utc_offset=2,
                     T_min=th.celsius(14), T_max=th.celsius(26), relative_humidity=0.55,
                     wind_speed=2.0, cloud_cover=0.1, aod500=0.15, ground_albedo=0.15)
geometry = {
    'ground': {'type': 'rectangle', 'to_world': T().rotate([1, 0, 0], -90).scale(20),
               'bsdf': {'type': 'diffuse', 'reflectance': 0.07}},
    'wall': {'type': 'cube', 'to_world': T().translate([-6, 1.5, 0]).scale([0.15, 1.5, 4.0]),
             'bsdf': {'type': 'diffuse', 'reflectance': 0.08}},
    'box': {'type': 'cube', 'to_world': T().translate([3, 0.9, 2]).rotate([0, 1, 0], 20).scale([2.2, 0.9, 1.1]),
            'bsdf': {'type': 'roughplastic', 'int_ior': 1.5, 'alpha': 0.3, 'diffuse_reflectance': 0.05}},
    'trunk': {'type': 'cylinder', 'p0': [6, 0, -5], 'p1': [6, 3.5, -5], 'radius': 0.25,
              'bsdf': {'type': 'diffuse', 'reflectance': 0.1}},
    'canopy': {'type': 'sphere', 'center': [6, 5.0, -5], 'radius': 2.2,
               'bsdf': {'type': 'diffuse', 'reflectance': 0.05}},
}
materials = {
    'ground': th.MATERIALS['asphalt'],
    'wall': th.MATERIALS['concrete_wall'].replace(T_internal=th.celsius(20)),
    'box': th.MATERIALS['steel_sheet_painted'].replace(back='interior', T_internal=th.celsius(60), U_internal=8.0),
    'trunk': th.MATERIALS['wood'].replace(back='adiabatic'),
    'canopy': th.MATERIALS['vegetation_canopy'],
}
HOUR = 14.0

sim = th.ThermalScene({'type': 'scene', **geometry}, weather, materials,
                      resolution={'ground': 160, 'wall': 48, 'box': 48, 'trunk': 24, 'canopy': 48},
                      up=UP, north=NORTH, sky_rays=48)


def camera(w, h, spp):
    # Close-up on the east side of the wall and the box: at 14:00 the sun is in
    # the south-west, so both cast their shadows towards the north-east
    return {'type': 'perspective', 'fov': 42,
            'to_world': T().look_at(origin=[1.5, 5.0, 3.0], target=[-3.5, 0.0, -1.5], up=[0, 1, 0]),
            'sampler': {'type': 'independent', 'sample_count': spp},
            'film': ir.band_film(BAND, w, h)}


def render(scene, w=720, h=480, spp=32):
    scene['sensor'] = camera(w, h, spp)
    scene['sky'] = ir.thermalsky(weather.air_temperature(HOUR), relative_humidity=weather.relative_humidity,
                                 band='8-14', up=UP)
    t0 = time.time()
    img = np.array(mi.render(mi.load_dict(scene), spp=spp))[..., 0]
    return ir.apparent_temperature(img, lo, hi) - 273.15, time.time() - t0


# 1. Texture solver (per-texel columns), as in example 10
t0 = time.time()
result = sim.run(days=3, dt=300.0, output_times=[HOUR])
t_solve = time.time() - t0
scene_tex = result.apply({'type': 'scene', **geometry}, HOUR)
scene_tex['integrator'] = {'type': 'path_ir', 'max_depth': 6}
img_tex, t_tex = render(scene_tex)
print(f'texture solver: {t_solve:.1f} s precomputation + {t_tex:.1f} s render')

# 2. Shading-time model: no precomputed field, shadow rays at render time
t0 = time.time()
scene_dyn = sim.dynamic({'type': 'scene', **geometry}, HOUR)
t_prep = time.time() - t0
scene_dyn['integrator'] = {'type': 'path_ir', 'max_depth': 6, 'thermal_up': list(UP), 'thermal_sky_rays': 8}
img_dyn, t_dyn = render(scene_dyn)
n_steps = len(scene_dyn['ground']['emitter']['weights'].split(','))
print(f'dynamic model: {t_prep:.1f} s response functions + {t_dyn:.1f} s render '
      f'({n_steps} sun positions per shading point)')
print(f'ground temperature range  texture: {np.nanmin(img_tex):.1f}..{np.nanmax(img_tex):.1f} C   '
      f'dynamic: {np.nanmin(img_dyn):.1f}..{np.nanmax(img_dyn):.1f} C')

save_grid_png([img_tex, img_dyn],
              [f'{HOUR:02.0f}:00, texture solver (25 cm texels)', f'{HOUR:02.0f}:00, shading-time evaluation'],
              f'{output_dir()}/11_thermal_dynamic.png', cmap='inferno', clabel='apparent temperature [C]',
              vmin=20, vmax=50)
