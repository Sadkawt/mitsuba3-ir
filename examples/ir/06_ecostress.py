#!/usr/bin/env python3
"""
Example 6 - Measured emissivity spectra (ECOSTRESS / ASTER spectral library).

The ECOSTRESS spectral library (Meerdink et al. 2019, https://speclib.jpl.nasa.gov)
provides directional-hemispherical reflectance spectra of natural and
man-made materials from 0.35 to 15.4 um. For Lambertian surfaces the
emissivity is 1 - reflectance (Kirchhoff), and the matching Mitsuba material
is a ``diffuse`` BSDF with reflectance 1 - epsilon(lambda) - which is exactly
what ``mitsuba.ir.diffuse_from_emissivity`` / ``diffuse_from_reflectance``
build. The ``thermal`` emitter then radiates with the measured emissivity.

Download one or more ASCII spectra from the library and pass them on the
command line:

    python 06_ecostress.py path/to/manmade.concrete.pavingconcrete.solid.all.0092uuucnc.jhu.becknic.spectrum.txt ...

Run without arguments to see the workflow on a synthetic spectrum.
"""

import os
import sys
import numpy as np
import mitsuba as mi

mi.set_variant('scalar_spectral')
from mitsuba import ir

files = [a for a in sys.argv[1:] if os.path.isfile(a)]
spectra = []
if files:
    for f in files:
        lam, refl, meta = ir.load_ecostress(f)
        spectra.append((meta.get('Name', os.path.basename(f)), lam, refl))
        print(f"{meta.get('Name', '?')}: {lam.min()/1e3:.2f}-{lam.max()/1e3:.2f} um, "
              f"{len(lam)} samples, {meta.get('Measurement', '')}")
else:
    print('No ECOSTRESS files given; using a synthetic reflectance spectrum with a '
          'reststrahlen feature (quartz-like) for illustration.')
    lam = np.linspace(2000.0, 15000.0, 500)
    refl = 0.05 + 0.5 * np.exp(-0.5 * ((lam - 8800.0) / 500.0) ** 2) + 0.2 * np.exp(-0.5 * ((lam - 12500.0) / 400.0) ** 2)
    spectra.append(('synthetic quartz-like', lam, refl))

BAND = 'lwir'
lo, hi = ir.BANDS[BAND]
T = 310.0
for name, lam, refl in spectra:
    eps = 1.0 - refl
    # Band-averaged emissivity (Planck-weighted) and the radiance a Mitsuba
    # plate with this material would produce against a cold background
    grid = np.linspace(lo, hi, 2001)
    w = ir.planck(grid, T)
    eps_band = np.trapezoid(np.interp(grid, lam, eps) * w, grid) / np.trapezoid(w, grid)

    scene = {
        'type': 'scene',
        'integrator': {'type': 'path_ir', 'max_depth': 4},
        'sensor': {'type': 'radiancemeter',
                   'to_world': mi.ScalarTransform4f().look_at(origin=[0, 0, 1], target=[0, 0, 0], up=[0, 1, 0]),
                   'sampler': {'type': 'independent', 'sample_count': 256},
                   'film': ir.band_film(BAND, 1, 1, rfilter='box')},
        'plate': {'type': 'rectangle', 'bsdf': ir.diffuse_from_reflectance(lam, refl),
                  'emitter': {'type': 'thermal', 'temperature': T}},
    }
    L = float(np.array(mi.render(mi.load_dict(scene), spp=256)).ravel()[0])
    print(f'{name}: band emissivity {eps_band:.3f}; rendered radiance {L:.3f} vs '
          f'eps_band * B = {eps_band * ir.band_radiance(lo, hi, T):.3f} W m^-2 sr^-1; '
          f'apparent temperature {ir.apparent_temperature(L, lo, hi):.1f} K for a true {T:.0f} K')
