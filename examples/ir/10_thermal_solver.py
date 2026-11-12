#!/usr/bin/env python3
"""
Example 10 - Scene temperatures from the weather: thermal shadows and lag.

``mitsuba.ir.thermal`` computes the surface temperature of every object from
a one-dimensional transient energy balance driven by one parametrised
weather model (date, location, air temperature range, humidity, wind,
clouds, aerosol). Shadows and sky-view factors are obtained by ray casting
through the Mitsuba scene itself, so

* sunlit and shaded surfaces develop different temperatures,
* the shadow pattern moves with the sun while the ground remembers where
  the shadow *was* (thermal lag), i.e. "thermal shadows",
* thin metal heats and cools fast, thick concrete/asphalt slowly,
* internal heat sources (here: an engine compartment) show through.

The scene: asphalt ground, a concrete wall running north-south, a painted
steel box with a 60 C interior (vehicle-like), a tree (leaf canopy on a
trunk), in Stockholm on 2026-07-15. The script prints temperatures of a few
probes through the day and renders the LWIR view at several times.

Run:  python 10_thermal_solver.py [variant]
"""

import sys
import time
import numpy as np
import mitsuba as mi

mi.set_variant(sys.argv[1] if len(sys.argv) > 1 else 'llvm_ad_spectral')
from mitsuba import ir
from mitsuba.ir import thermal as th
from _util import output_dir, save_grid_png

UP, NORTH = (0, 1, 0), (0, 0, -1)         # Y up, north = -Z (east = +X)
BAND = 'lwir'
lo, hi = ir.BANDS[BAND]

weather = th.Weather('2026-07-15', latitude=59.33, longitude=18.07, utc_offset=2,
                     T_min=th.celsius(14), T_max=th.celsius(26), relative_humidity=0.55,
                     wind_speed=2.0, cloud_cover=0.1, aod500=0.15, ground_albedo=0.15)
print(weather.summary())

# --- geometry (metres) -------------------------------------------------------
T = mi.ScalarTransform4f
geometry = {
    'ground': {'type': 'rectangle', 'to_world': T().rotate([1, 0, 0], -90).scale(20),
               'bsdf': {'type': 'diffuse', 'reflectance': 0.07}},                      # asphalt, eps 0.93
    'wall': {'type': 'cube', 'to_world': T().translate([-6, 1.5, 0]).scale([0.15, 1.5, 4.0]),
             'bsdf': {'type': 'diffuse', 'reflectance': 0.08}},                        # concrete
    # painted steel: optically a rough dielectric paint layer (LWIR emissivity ~0.9),
    # thermally a 2 mm steel sheet (see 'materials' below)
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
    # 2 mm painted steel over an engine compartment at 60 C (U = 8 W/m^2K)
    'box': th.MATERIALS['steel_sheet_painted'].replace(back='interior', T_internal=th.celsius(60), U_internal=8.0),
    'trunk': th.MATERIALS['wood'].replace(back='adiabatic'),
    'canopy': th.MATERIALS['vegetation_canopy'],
}

t0 = time.time()
sim = th.ThermalScene({'type': 'scene', **geometry}, weather, materials,
                      resolution={'ground': 160, 'wall': 48, 'box': 48, 'trunk': 24, 'canopy': 48},
                      up=UP, north=NORTH, sky_rays=48)
times = [5, 8, 11, 14, 17, 20, 23]
result = sim.run(days=3, dt=300.0, output_times=times, progress=False)
print(f'thermal simulation: {sum(s.E for s in sim.shapes.values())} elements, {time.time() - t0:.1f} s')

# --- probes -----------------------------------------------------------------
def probe(key, u, v):
    Tm = result.temperature
    return [Tm(key, t)[int(v * Tm(key, t).shape[0]), int(u * Tm(key, t).shape[1])] - 273.15 for t in times]

print(f"\n{'local time':>28s}" + ''.join(f'{t:7d}h' for t in times))
print(f"{'air':>28s}" + ''.join(f'{weather.air_temperature(t) - 273.15:8.1f}' for t in times))
# Ground UV -> world: x = (u - 0.5) * 40, z = (0.5 - v) * 40 (rectangle scaled by 20, rotated to y-up)
PROBES = {'asphalt, open': ('ground', 0.85, 0.85),              # x = +14, z = -14
          'asphalt 0.8 m W of wall': ('ground', 0.33, 0.5),     # shaded in the morning
          'asphalt 3.2 m W of wall': ('ground', 0.27, 0.5),     # shaded only early
          'asphalt 1.0 m E of wall': ('ground', 0.325 + 0.05, 0.5),  # shaded in the afternoon
          'asphalt in 14 h tree shadow': ('ground', 0.715, 0.7175)}  # x = 8.6, z = -8.7
for name, (key, u, v) in PROBES.items():
    print(f'{name:>28s}' + ''.join(f'{x:8.1f}' for x in probe(key, u, v)))
for key in ['wall', 'box', 'canopy']:
    print(f'{key + " (mean)":>28s}' + ''.join(f'{result.statistics(key, t)["mean"] - 273.15:8.1f}' for t in times))

# --- LWIR renders through the day --------------------------------------------
images, titles = [], []
for t in [8, 14, 20]:
    scene = result.apply({'type': 'scene', **geometry}, t)
    scene['integrator'] = {'type': 'path_ir', 'max_depth': 8}
    scene['sensor'] = {'type': 'perspective', 'fov': 55,
                       'to_world': T().look_at(origin=[4, 9, 16], target=[0, 0.5, -1], up=[0, 1, 0]),
                       'sampler': {'type': 'independent', 'sample_count': 64},
                       'film': ir.band_film(BAND, 640, 400)}
    T_air = weather.air_temperature(t)
    scene['sky'] = ir.thermalsky(T_air, relative_humidity=weather.relative_humidity, band='8-14', up=UP)
    img = np.array(mi.render(mi.load_dict(scene), spp=64))[..., 0]
    images.append(ir.apparent_temperature(img, lo, hi) - 273.15)
    titles.append(f'{t:02d}:00 local, air {T_air - 273.15:.1f} C')
save_grid_png(images, titles, f'{output_dir()}/10_thermal_solver.png', cmap='inferno',
              clabel='apparent temperature [C]', vmin=10, vmax=50)

# --- diurnal curves figure ----------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fine = list(range(0, 24))
    res2 = sim.run(days=3, dt=300.0, output_times=fine)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    ax.plot(fine, [weather.air_temperature(t) - 273.15 for t in fine], 'k--', label='air')
    for name, (key, u, v) in {k: PROBES[k] for k in ['asphalt, open', 'asphalt 0.8 m W of wall',
                                                       'asphalt 1.0 m E of wall', 'asphalt in 14 h tree shadow']}.items():
        ax.plot(fine, [res2.temperature(key, t)[int(v * res2.temperature(key, t).shape[0]), int(u * res2.temperature(key, t).shape[1])] - 273.15 for t in fine], label=name)
    for key in ['wall', 'box', 'canopy']:
        ax.plot(fine, [res2.statistics(key, t)['mean'] - 273.15 for t in fine], label=f'{key} (mean)')
    ax.set_xlabel('local time [h]')
    ax.set_ylabel('surface temperature [C]')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('Stockholm, 2026-07-15: surface temperatures from the energy balance')
    fig.tight_layout()
    fig.savefig(f'{output_dir()}/10_thermal_curves.png')
    print(f'wrote {output_dir()}/10_thermal_curves.png')
except ImportError:
    pass
