#!/usr/bin/env python3
"""
Example 1 - A first LWIR (8-14 um) image.

Demonstrates
* assigning temperatures to objects with the ``thermal`` emitter
  (explicitly, and for everything else with ``mitsuba.ir.thermalize``),
* Kirchhoff coupling: no emissivity is specified anywhere - it follows from
  the BSDF of each object (diffuse reflectance 0.05 -> emissivity 0.95; a
  rough aluminium box with Drude-Lorentz optical constants -> emissivity of
  a few percent, which makes it *mirror the cold sky*),
* the ``specfilm`` band film (``mitsuba.ir.band_film``) whose channel holds
  the band radiance in W m^-2 sr^-1,
* conversion of band radiance to apparent temperature.

Run:  python 01_lwir_basic.py [variant]
"""

import sys
import numpy as np
import mitsuba as mi

mi.set_variant(sys.argv[1] if len(sys.argv) > 1 else 'llvm_ad_spectral')
from mitsuba import ir
from _util import output_dir, save_exr, save_temperature_png

BAND = 'lwir'
T_GROUND, T_SPHERE, T_BOX, T_SKY = 290.0, 315.0, 300.0, 250.0

scene = {
    'type': 'scene',
    'integrator': {'type': 'path_ir', 'max_depth': 12},
    'sensor': {
        'type': 'perspective', 'fov': 45,
        'to_world': mi.ScalarTransform4f().look_at(origin=[0, 2.2, 6], target=[0, 0.5, 0], up=[0, 1, 0]),
        'sampler': {'type': 'independent', 'sample_count': 128},
        'film': ir.band_film(BAND, 640, 427),
    },
    # Ground: asphalt-like Lambertian surface, emissivity 0.92 (typical values
    # in Vollmer & Moellmann 2018, Table 1.x: asphalt 0.90-0.96)
    'ground': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(8),
        'bsdf': {'type': 'diffuse', 'reflectance': 0.08},
        'emitter': {'type': 'thermal', 'temperature': T_GROUND},
    },
    # Painted sphere (matte paint, emissivity ~0.95), warmer than the ground
    'sphere': {
        'type': 'sphere', 'center': [-1.3, 1.0, 0], 'radius': 1.0,
        'bsdf': {'type': 'diffuse', 'reflectance': 0.05},
        'emitter': {'type': 'thermal', 'temperature': T_SPHERE},
    },
    # Brushed aluminium box: optical constants from the Lorentz-Drude model
    # (Rakic et al. 1998); no temperature given here -> assigned by thermalize()
    'box': {
        'type': 'cube',
        'to_world': mi.ScalarTransform4f().translate([1.4, 0.75, 0]).rotate([0, 1, 0], 25).scale(0.75),
        'bsdf': ir.conductor_ir('Al_ir', alpha=0.15),
    },
    # Clear sky approximated as a blackbody at an effective temperature
    'sky': ir.sky_blackbody(T_SKY),
}

# Every shape without an emitter gets the default temperature
scene = ir.thermalize(scene, temperature=T_BOX)

for w in ir.validate(scene, band=BAND):
    print('validate:', w)

img = np.array(mi.render(mi.load_dict(scene), spp=128))[..., 0]
lo, hi = ir.BANDS[BAND]
T_app = ir.apparent_temperature(img, lo, hi)

print(f'band radiance: {img.min():.2f} .. {img.max():.2f} W m^-2 sr^-1')
print(f'apparent temperature: {T_app.min():.1f} .. {T_app.max():.1f} K')
for name, (y, x) in {'ground': (400, 320), 'sphere': (200, 160), 'box side': (260, 440), 'box top': (165, 560), 'sky': (30, 320)}.items():
    print(f'  {name:9s} L = {img[y, x]:6.2f} W m^-2 sr^-1   T_app = {T_app[y, x]:6.1f} K')

out = output_dir()
save_exr(img, f'{out}/01_lwir_radiance.exr')
save_temperature_png(T_app, f'{out}/01_lwir_apparent_temperature.png', vmin=245, vmax=320,
                     title='LWIR 8-14 um, apparent temperature')
