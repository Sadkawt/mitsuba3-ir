#!/usr/bin/env python3
"""
Example 12 - Radiative exchange between scene surfaces.

Without exchange, every direction that does not see the sky is assumed to
see "surroundings" at air temperature. In a street canyon the walls and the
road see each other: during the day the hot road heats the wall bases by
longwave radiation and reflected sunlight; at night the canyon stays warmer
than open ground because its surfaces exchange radiation with each other
instead of with the cold sky. The ground under a vehicle sees the warm
underbody instead of the sky.

The scene (asphalt road between two concrete walls, a heated vehicle-like
box) is simulated with and without ``exchange`` and rendered at 22:00 with the
shading-time model (whose exchange fields come from the per-element
radiosity). Probe temperatures and a difference image are produced.

Run:  python 12_radiative_exchange.py [variant]
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
HOUR = 22.0

weather = th.Weather('2026-07-15', latitude=59.33, longitude=18.07, utc_offset=2,
                     T_min=th.celsius(14), T_max=th.celsius(26), relative_humidity=0.55,
                     wind_speed=1.5, cloud_cover=0.0, aod500=0.15, ground_albedo=0.15)

geometry = {
    'road': {'type': 'rectangle', 'to_world': T().rotate([1, 0, 0], -90).scale(20),
             'bsdf': ir.two_band_diffuse(0.10, 0.07)},
    # two concrete walls along z (north-south), 4 m tall, 6 m apart
    'wall_w': {'type': 'cube', 'to_world': T().translate([-3.15, 2.0, 0]).scale([0.15, 2.0, 10.0]),
               'bsdf': ir.two_band_diffuse(0.35, 0.08)},
    'wall_e': {'type': 'cube', 'to_world': T().translate([3.15, 2.0, 0]).scale([0.15, 2.0, 10.0]),
               'bsdf': ir.two_band_diffuse(0.35, 0.08)},
    'box': {'type': 'cube', 'to_world': T().translate([0.5, 0.75, 3]).scale([2.0, 0.55, 1.0]),
            'bsdf': {'type': 'roughplastic', 'int_ior': 1.5, 'alpha': 0.3,
                     'diffuse_reflectance': ir.two_band_diffuse(0.25, 0.04)['reflectance']}},
}
materials = {
    'road': th.MATERIALS['asphalt'],
    'wall_w': th.MATERIALS['concrete_wall'].replace(T_internal=th.celsius(21)),
    'wall_e': th.MATERIALS['concrete_wall'].replace(T_internal=th.celsius(21)),
    'box': th.MATERIALS['steel_sheet_painted'].replace(back='interior', T_internal=th.celsius(45), U_internal=6.0),
}
resolution = {'road': 128, 'wall_w': 48, 'wall_e': 48, 'box': 32}

# Road probes (u, v) -> x = (u - 0.5) 40, z = (0.5 - v) 40
PROBES = {'road, canyon centre': (0.5, 0.5), 'road, 0.5 m from W wall': (0.5 - 2.5 / 40, 0.5),
          'road, open (x = 15 m)': (0.875, 0.5), 'road under the box': (0.5 + 0.5 / 40, 0.5 - 3.0 / 40)}

results, images = {}, {}
for exchange in (False, True):
    t0 = time.time()
    sim = th.ThermalScene({'type': 'scene', **geometry}, weather, materials, resolution=resolution,
                          up=UP, north=NORTH, sky_rays=96, exchange=exchange)
    res = sim.run(days=3, dt=600.0, output_times=[HOUR, 14.0])
    results[exchange] = res
    scene = sim.dynamic({'type': 'scene', **geometry}, HOUR)
    scene['integrator'] = {'type': 'path_ir', 'max_depth': 6, 'thermal_up': list(UP), 'thermal_sky_rays': 12}
    scene['sensor'] = {'type': 'perspective', 'fov': 50,
                       'to_world': T().look_at(origin=[0.0, 7.0, 14.0], target=[0.0, 1.0, 0.0], up=[0, 1, 0]),
                       'sampler': {'type': 'independent', 'sample_count': 48},
                       'film': ir.band_film(BAND, 640, 420)}
    scene['sky'] = ir.thermalsky(weather.air_temperature(HOUR), relative_humidity=weather.relative_humidity,
                                 band='8-14', up=UP)
    img = np.array(mi.render(mi.load_dict(scene), spp=48))[..., 0]
    images[exchange] = ir.apparent_temperature(img, lo, hi) - 273.15
    print(f'exchange={exchange!s:5s}: solver + render {time.time() - t0:.1f} s')

print(f'\nSurface temperatures [C], air {weather.air_temperature(HOUR) - 273.15:.1f} C at {HOUR:.0f}:00, '
      f'{weather.air_temperature(14.0) - 273.15:.1f} C at 14:00')
print(f"{'probe':>28s} {'14h off':>8s} {'14h on':>8s} {'22h off':>8s} {'22h on':>8s}")
for name, (u, v) in PROBES.items():
    vals = []
    for t in (14.0, HOUR):
        for ex in (False, True):
            Tm = results[ex].temperature('road', t)
            vals.append(Tm[int(v * Tm.shape[0]), int(u * Tm.shape[1])] - 273.15)
    print(f'{name:>28s}' + ''.join(f'{x:9.1f}' for x in vals))
for key in ('wall_w', 'wall_e', 'box'):
    vals = [results[ex].statistics(key, t)['mean'] - 273.15 for t in (14.0, HOUR) for ex in (False, True)]
    print(f'{key + " (mean)":>28s}' + ''.join(f'{x:9.1f}' for x in vals))

save_grid_png([images[False], images[True], images[True] - images[False]],
              ['22:00, surroundings at air temperature', '22:00, radiative exchange', 'difference [K]'],
              f'{output_dir()}/12_radiative_exchange.png', cmap='inferno', clabel='apparent temperature [C]')
