#!/usr/bin/env python3
"""
Example 9 - Line-by-line atmospheric transmittance from HITRAN.

Uses ``mitsuba.ir.hitran_absorption`` (HITRAN Application Programming
Interface, Kochanov et al. 2016) to compute the molecular absorption
coefficient of humid air for H2O and CO2 in the MWIR and LWIR, plots the
transmittance over a 1 km horizontal path, and builds (a) an emissive air
medium and (b) an attenuated sun from it. Line data are downloaded from
hitran.org on first use (cached in ./hitran_db).

The figure extends beyond the nominal bands so that the band edges are
visible: the H2O nu2 band wing closes the window below ~7.5 um, the CO2 nu2
band above ~14 um; in the MWIR the CO2 nu3 band at 4.2-4.4 um splits the
window. The empirical Roberts-Selby-Biberman water-vapour continuum (on by
default) adds the smooth, red-sloping absorption of the 8-13 um window;
the grey curve shows line absorption alone.

Limitation (report, assumption A16): no aerosols, no O3, legacy continuum
model; for radiometric work at long ranges use MODTRAN-type data.

Run:  python 09_hitran_transmittance.py
"""

import numpy as np
import mitsuba as mi

mi.set_variant('scalar_spectral')
from mitsuba import ir
from _util import output_dir

T_AIR, RH, PATH_M = 288.15, 0.5, 1000.0
RANGES = {'mwir': (2500.0, 5500.0), 'lwir': (6500.0, 15500.0)}   # computed range (wider than the band)
tables, lines_only = {}, {}
for band, rng in RANGES.items():
    print(f'computing {band.upper()} absorption (H2O + CO2) over {rng[0]/1e3:.1f}-{rng[1]/1e3:.1f} um ...')
    tables[band] = ir.hitran_absorption(rng, T_air=T_AIR, relative_humidity=RH)
    lines_only[band] = ir.hitran_absorption(rng, T_air=T_AIR, relative_humidity=RH, continuum=None)
    lam, tau = ir.transmittance(tables[band], PATH_M)
    lam0, tau0 = ir.transmittance(lines_only[band], PATH_M)
    print(f'  band-averaged transmittance over {PATH_M/1e3:.0f} km ({band}): '
          f'{ir.atmosphere.band_average((lam, tau), band):.3f} with continuum, '
          f'{ir.atmosphere.band_average((lam0, tau0), band):.3f} lines only')

# Build scene ingredients from the LWIR table
air = ir.air_medium(T_AIR, tables['lwir'], medium_id='air')
sun = ir.sun([0, 0, -1], band='mwir', transmittance=ir.transmittance(tables['mwir'], PATH_M))
print('air medium and attenuated MWIR sun dictionaries built; e.g. sun irradiance spectrum has',
      len(sun['irradiance']['wavelengths'].split(',')), 'samples')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), dpi=110)
    for ax, band in zip(axes, ['mwir', 'lwir']):
        lam0, tau0 = ir.transmittance(lines_only[band], PATH_M)
        lam, tau = ir.transmittance(tables[band], PATH_M)
        ax.plot(lam0 / 1e3, tau0, lw=0.5, color='0.6', label='H2O + CO2 lines only')
        ax.plot(lam / 1e3, tau, lw=0.5, color='C0', label='+ RSB H2O continuum (8-14 um)')
        lo, hi = ir.BANDS[band]
        ax.axvspan(lo / 1e3, hi / 1e3, color='C1', alpha=0.08, label=f'{band.upper()} band')
        ax.set_xlabel('wavelength [um]')
        ax.set_ylabel(f'transmittance, {PATH_M/1e3:.0f} km')
        ax.set_ylim(0, 1.02)
        ax.set_title(f'{band.upper()}: sea-level horizontal path, T = {T_AIR:.1f} K, RH = {RH:.0%}')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    path = f'{output_dir()}/09_hitran_transmittance.png'
    fig.savefig(path)
    print('wrote', path)
except ImportError:
    pass
