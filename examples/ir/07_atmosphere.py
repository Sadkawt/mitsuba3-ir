#!/usr/bin/env python3
"""
Example 7 - Atmospheric path radiance and transmission.

Demonstrates the ``volpath_ir`` integrator with an *emissive* air medium:
air both attenuates the radiance of a distant object and adds its own
thermal emission (path radiance), so a hot target seen through a long path
converges towards the apparent temperature of the air. The example renders
a hot plate (350 K) behind air layers of increasing thickness and compares
the result with the closed-form solution for a homogeneous non-scattering
medium,  L = tau B(T_target) + (1 - tau) B(T_air),  tau = exp(-sigma_a d).

Molecular absorption: a gray coefficient is used by default; pass ``--hitran``
to compute sigma_a(lambda) line-by-line from HITRAN (requires the
``hitran-api`` package and network access; see ``09_hitran_transmittance.py``).

Run:  python 07_atmosphere.py [--hitran] [variant]
"""

import sys
import numpy as np
import mitsuba as mi

args = [a for a in sys.argv[1:] if not a.startswith('--')]
mi.set_variant(args[0] if args else 'llvm_ad_spectral')
from mitsuba import ir

BAND = 'lwir'
T_AIR, T_TARGET = 288.15, 350.0
lo, hi = ir.BANDS[BAND]

if '--hitran' in sys.argv:
    print('computing HITRAN absorption (H2O + CO2) ...')
    absorption = ir.hitran_absorption(BAND, T_air=T_AIR, relative_humidity=0.5, resolution_cm=0.25)
    label = 'HITRAN H2O+CO2 lines, RH 50 %'
else:
    absorption = 1.5e-4   # 1/m: gray stand-in, ~0.15 km^-1 (clear LWIR window incl. continuum)
    label = f'gray sigma_a = {absorption:g} 1/m'


def scene_for(depth_m):
    return {
        'type': 'scene',
        'integrator': {'type': 'volpath_ir', 'max_depth': 32},
        'air': ir.air_medium(T_AIR, absorption, medium_id='air'),
        'sensor': {
            'type': 'radiancemeter',
            'to_world': mi.ScalarTransform4f().look_at(origin=[0, 0, depth_m + 1.0], target=[0, 0, 0], up=[0, 1, 0]),
            'sampler': {'type': 'independent', 'sample_count': 1024},
            'film': ir.band_film(BAND, 1, 1, rfilter='box'),
        },
        'air_volume': ir.medium_box('air', [-1e4, -1e4, 0.5], [1e4, 1e4, depth_m + 0.5]),
        'target': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().scale(1e3),
                   'bsdf': {'type': 'diffuse', 'reflectance': 0.05},
                   'emitter': {'type': 'thermal', 'temperature': T_TARGET}},
    }


print(f'Air at {T_AIR:.2f} K ({label}); target plate at {T_TARGET:.0f} K, emissivity 0.95')
print(f"{'path [m]':>10s} {'rendered L':>12s} {'analytic L':>12s} {'T_app [K]':>10s}")
grid = np.linspace(lo, hi, 4001)
if np.isscalar(absorption):
    sig = np.full_like(grid, absorption)
else:
    sig = np.interp(grid, absorption[0], absorption[1])
# Hemispherical downwelling emission of a slab of optical depth t onto the
# plate: L_down / B = 1 - 2 E_3(t) = 2 int_0^1 (1 - exp(-t / mu)) mu dmu
mu, w = np.polynomial.legendre.leggauss(64)
mu, w = 0.5 * (mu + 1), 0.5 * w
for depth in [10.0, 300.0, 1000.0, 3000.0, 10000.0]:
    L = float(np.array(mi.render(mi.load_dict(scene_for(depth)), spp=1024)).ravel()[0])
    t = sig * depth
    tau = np.exp(-t)
    hemi = 2 * np.sum((1 - np.exp(-t[:, None] / mu[None, :])) * (mu * w)[None, :], axis=1)
    L_down = hemi * ir.planck(grid, T_AIR)
    L_up = 0.95 * ir.planck(grid, T_TARGET) + 0.05 * L_down
    L_ref = np.trapezoid(tau * L_up + (1 - tau) * ir.planck(grid, T_AIR), grid)
    print(f'{depth:10.0f} {L:12.3f} {L_ref:12.3f} {ir.apparent_temperature(L, lo, hi):10.1f}')
print('(analytic: tau L_up + (1 - tau) B_air with L_up = 0.95 B_target + 0.05 L_down, L_down = (1 - 2 E_3(sigma d)) B_air)')
