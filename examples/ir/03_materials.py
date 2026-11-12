#!/usr/bin/env python3
"""
Example 3 - Infrared materials.

Demonstrates
* the shipped IR optical constants (``*_ir`` presets, see
  resources/ir_data/ior/README_ir.md): metals from the Lorentz-Drude model,
  dielectrics / windows / water from the refractiveindex.info database,
* how emissivity emerges from the BSDF: spheres of identical temperature but
  different materials have very different apparent temperatures,
* the helpers ``normal_emissivity`` / ``hemispherical_emissivity`` and a plot
  of spectral normal emissivity for a few materials (3-14 um).

Run:  python 03_materials.py [variant]
"""

import sys
import numpy as np
import mitsuba as mi

mi.set_variant(sys.argv[1] if len(sys.argv) > 1 else 'llvm_ad_spectral')
from mitsuba import ir
from _util import output_dir, save_temperature_png, project

CAM = dict(origin=[0, 3.2, 10.5], target=[0, 0.7, 0], up=[0, 1, 0])
FOV, W, H = 62, 960, 360

BAND = 'lwir'
T_OBJ, T_GROUND, T_SKY = 320.0, 290.0, 250.0

# (label, bsdf dictionary)
materials = [
    ('matte paint\n(diffuse 0.05)', {'type': 'diffuse', 'reflectance': 0.05}),
    ('polished Al', ir.conductor_ir('Al_ir')),
    ('rough Al\n(alpha 0.3)', ir.conductor_ir('Al_ir', alpha=0.3)),
    ('polished Fe', ir.conductor_ir('Fe_ir')),
    ('water\n(smooth)', ir.opaque_ir('H2O_ir')),
    ('glass\n(SiO2, smooth)', ir.opaque_ir('SiO2glass_ir')),
]

scene = {
    'type': 'scene',
    'integrator': {'type': 'path_ir', 'max_depth': 12},
    'sensor': {
        'type': 'perspective', 'fov': FOV,
        'to_world': mi.ScalarTransform4f().look_at(**CAM),
        'sampler': {'type': 'independent', 'sample_count': 128},
        'film': ir.band_film(BAND, W, H),
    },
    'ground': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f().rotate([1, 0, 0], -90).scale(12),
        'bsdf': {'type': 'diffuse', 'reflectance': 0.08},
        'emitter': {'type': 'thermal', 'temperature': T_GROUND},
    },
    'sky': ir.sky_blackbody(T_SKY),
}
n = len(materials)
for i, (label, bsdf) in enumerate(materials):
    x = (i - (n - 1) / 2) * 2.1
    scene[f'sphere_{i}'] = {
        'type': 'sphere', 'center': [x, 0.9, 0], 'radius': 0.9,
        'bsdf': bsdf, 'emitter': {'type': 'thermal', 'temperature': T_OBJ},
    }

img = np.array(mi.render(mi.load_dict(scene), spp=128))[..., 0]
lo, hi = ir.BANDS[BAND]
T_app = ir.apparent_temperature(img, lo, hi)

print(f'All spheres are at {T_OBJ} K; ground {T_GROUND} K; sky {T_SKY} K. Apparent temperature at sphere centres:')
centres = [[(i - (n - 1) / 2) * 2.1, 0.9, 0.9] for i in range(n)]   # front-most point of each sphere
pix = project(centres, CAM['origin'], CAM['target'], CAM['up'], FOV, W, H)
for (label, _), (x, y) in zip(materials, pix):
    print(f'  {label.replace(chr(10), " "):24s} T_app = {T_app[int(y), int(x)]:6.1f} K')

out = output_dir()
save_temperature_png(T_app, f'{out}/03_materials_apparent_temperature.png', vmin=250, vmax=325,
                     title=f'LWIR: all spheres at {T_OBJ:.0f} K - emissivity from the BSDF')

# Spectral normal emissivity of smooth surfaces from the optical constants
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lam = np.geomspace(1000.0, 14000.0, 400)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
    for name in ['Al_ir', 'Au_ir', 'Fe_ir', 'Ti_ir', 'H2O_ir', 'SiO2glass_ir', 'Sapphire_ir']:
        l, nn, kk = ir.materials.load_ior(name)
        eps = ir.normal_emissivity(np.interp(lam, l, nn), np.interp(lam, l, kk))
        ax.plot(lam / 1e3, eps, label=name.replace('_ir', ''))
    ax.set_xscale('log')
    ax.set_xlabel('wavelength [um]')
    ax.set_ylabel('normal emissivity (smooth surface)')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title('1 - F(0) from the shipped optical constants')
    fig.tight_layout()
    fig.savefig(f'{out}/03_materials_normal_emissivity.png')
    print(f'wrote {out}/03_materials_normal_emissivity.png')
except ImportError:
    pass
