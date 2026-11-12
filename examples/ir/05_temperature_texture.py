#!/usr/bin/env python3
"""
Example 5 - Spatially varying temperature.

Demonstrates a temperature field given as a texture: a plate with a hot
spot (e.g. a heated component behind a panel), stored as a float bitmap in
kelvin and attached to the ``thermal`` emitter with ``raw=True`` so that no
colour processing is applied. Any Mitsuba texture works (bitmap, checkerboard,
mesh attribute from a thermal solver, ...).

Run:  python 05_temperature_texture.py [variant]
"""

import sys
import numpy as np
import mitsuba as mi

mi.set_variant(sys.argv[1] if len(sys.argv) > 1 else 'llvm_ad_spectral')
from mitsuba import ir
from _util import output_dir, save_temperature_png

BAND = 'lwir'
RES = 256
T_BASE, T_PEAK = 295.0, 340.0

# Temperature map: Gaussian hot spot plus a cooler stripe
v, u = np.mgrid[0:RES, 0:RES] / (RES - 1)
T_map = T_BASE + (T_PEAK - T_BASE) * np.exp(-((u - 0.35) ** 2 + (v - 0.5) ** 2) / (2 * 0.08 ** 2))
T_map[(u > 0.7) & (u < 0.8)] -= 10.0
temperature_bitmap = mi.Bitmap(np.ascontiguousarray(T_map[..., None].astype(np.float32)))

scene = {
    'type': 'scene',
    'integrator': {'type': 'path_ir', 'max_depth': 6},
    'sensor': {
        'type': 'perspective', 'fov': 35,
        'to_world': mi.ScalarTransform4f().look_at(origin=[0, 0, 3.2], target=[0, 0, 0], up=[0, 1, 0]),
        'sampler': {'type': 'independent', 'sample_count': 64},
        'film': ir.band_film(BAND, 512, 512),
    },
    'panel': {
        'type': 'rectangle',
        'bsdf': {'type': 'diffuse', 'reflectance': 0.1},      # painted panel, emissivity 0.9
        'emitter': {
            'type': 'thermal',
            'temperature': {'type': 'bitmap', 'bitmap': temperature_bitmap, 'raw': True,
                            'filter_type': 'bilinear', 'wrap_mode': 'clamp'},
        },
    },
    'sky': ir.sky_blackbody(260.0),   # the panel reflects 10 % of this
}

img = np.array(mi.render(mi.load_dict(scene), spp=64))[..., 0]
lo, hi = ir.BANDS[BAND]
T_app = ir.apparent_temperature(img, lo, hi)

# Expected apparent temperature at the hot spot: 0.9 B(T_peak) + 0.1 B(260 K)
L_expected = 0.9 * ir.band_radiance(lo, hi, T_PEAK) + 0.1 * ir.band_radiance(lo, hi, 260.0)
print(f'hot spot: rendered T_app max = {T_app.max():.1f} K, expected '
      f'{ir.apparent_temperature(L_expected, lo, hi):.1f} K (true surface temperature {T_PEAK:.0f} K)')

out = output_dir()
save_temperature_png(T_app, f'{out}/05_temperature_texture.png', title='textured temperature field (LWIR)')
