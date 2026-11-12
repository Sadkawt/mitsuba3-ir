#!/usr/bin/env python3
"""
Example 2 - The same scene in the SWIR, MWIR and LWIR, with sunlight.

Demonstrates
* ``mitsuba.ir.sun``: a directional source with a physically scaled solar
  spectrum (5772 K blackbody, 1361 W m^-2 total; the atmosphere is ignored
  here, so this is "sunlight at the top of the atmosphere"),
* how the balance between reflected sunlight and thermal self-emission
  changes with the band (report, Table 1): the SWIR image is a reflectance
  image, the LWIR image a temperature image, the MWIR a mixture,
* separating the two components by rendering with and without the sun
  (both components are linear in the sources).

Run:  python 02_bands.py [variant]
"""

import sys
import numpy as np
import mitsuba as mi

mi.set_variant(sys.argv[1] if len(sys.argv) > 1 else 'llvm_ad_spectral')
from mitsuba import ir
from _util import output_dir, save_grid_png

SUN_DIR = [-0.4, -0.7, -0.5]   # direction of propagation of the sunlight


def make_scene(band, with_sun):
    scene = {
        'type': 'scene',
        'integrator': {'type': 'path_ir', 'max_depth': 12},
        'sensor': {
            'type': 'perspective', 'fov': 45,
            'to_world': mi.ScalarTransform4f().look_at(origin=[0, 2.2, 6], target=[0, 0.5, 0], up=[0, 1, 0]),
            'sampler': {'type': 'independent', 'sample_count': 96},
            'film': ir.band_film(band, 480, 320),
        },
        'ground': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(8),
            'bsdf': {'type': 'diffuse', 'reflectance': 0.25},     # soil-like
            'emitter': {'type': 'thermal', 'temperature': 295.0},
        },
        'sphere': {
            'type': 'sphere', 'center': [-1.3, 1.0, 0], 'radius': 1.0,
            'bsdf': {'type': 'diffuse', 'reflectance': 0.6},      # bright paint
            'emitter': {'type': 'thermal', 'temperature': 305.0},
        },
        'box': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f().translate([1.4, 0.75, 0]).rotate([0, 1, 0], 25).scale(0.75),
            'bsdf': ir.conductor_ir('Fe_ir', alpha=0.25),         # steel-like
            'emitter': {'type': 'thermal', 'temperature': 310.0},
        },
        'sky': ir.sky_blackbody(250.0),
    }
    if with_sun:
        scene['sun'] = ir.sun(SUN_DIR, band=band)
    return scene


out = output_dir()
images, titles = [], []
for band in ['swir', 'mwir', 'lwir']:
    total = np.array(mi.render(mi.load_dict(make_scene(band, True)), spp=96))[..., 0]
    thermal = np.array(mi.render(mi.load_dict(make_scene(band, False)), spp=96))[..., 0]
    solar = total - thermal
    lo, hi = ir.BANDS[band]
    print(f'{band.upper():5s} {lo/1e3:.1f}-{hi/1e3:.1f} um: mean band radiance '
          f'{total.mean():.3e} W m^-2 sr^-1; thermal share {thermal.sum() / max(total.sum(), 1e-30):6.1%}, '
          f'solar share {solar.sum() / max(total.sum(), 1e-30):6.1%}')
    images.append(total)
    titles.append(f'{band.upper()} {lo/1e3:.1f}-{hi/1e3:.1f} um, total')

save_grid_png(images, titles, f'{out}/02_bands_radiance.png', cmap='gray', clabel='band radiance [W m^-2 sr^-1]')
