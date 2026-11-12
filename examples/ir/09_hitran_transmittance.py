#!/usr/bin/env python3
"""
Example 9 - Line-by-line atmospheric transmittance from HITRAN.

Uses ``mitsuba.ir.hitran_absorption`` (HITRAN Application Programming
Interface, Kochanov et al. 2016) to compute the molecular absorption
coefficient of humid air for H2O and CO2 in the MWIR and LWIR, plots the
transmittance over a 1 km horizontal path, and builds (a) an emissive air
medium and (b) an attenuated sun from it. Line data are downloaded from
hitran.org on first use (cached in ./hitran_db).

Limitation (report, assumption A15): the water-vapour continuum and aerosol
extinction are not included, so the 8-13 um window transmittance is an
upper bound; for radiometric work at long ranges use MODTRAN-type data.

Run:  python 09_hitran_transmittance.py
"""

import numpy as np
import mitsuba as mi

mi.set_variant('scalar_spectral')
from mitsuba import ir
from _util import output_dir

T_AIR, RH, PATH_M = 288.15, 0.5, 1000.0
tables = {}
for band in ['mwir', 'lwir']:
    print(f'computing {band.upper()} absorption (H2O + CO2) ...')
    tables[band] = ir.hitran_absorption(band, T_air=T_AIR, relative_humidity=RH, resolution_cm=0.25)
    lam, tau = ir.transmittance(tables[band], PATH_M)
    print(f'  band-averaged transmittance over {PATH_M/1e3:.0f} km: {ir.atmosphere.band_average((lam, tau), band):.3f}')

# Build scene ingredients from the LWIR table
air = ir.air_medium(T_AIR, tables['lwir'], medium_id='air')
sun = ir.sun([0, 0, -1], band='mwir', transmittance=ir.transmittance(tables['mwir'], PATH_M))
print('air medium and attenuated MWIR sun dictionaries built; e.g. sun irradiance spectrum has',
      len(sun['irradiance']['wavelengths'].split(',')), 'samples')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=110)
    for ax, band in zip(axes, ['mwir', 'lwir']):
        lam, tau = ir.transmittance(tables[band], PATH_M)
        ax.plot(lam / 1e3, tau, lw=0.6)
        ax.set_xlabel('wavelength [um]')
        ax.set_ylabel(f'transmittance, {PATH_M/1e3:.0f} km')
        ax.set_ylim(0, 1.02)
        ax.set_title(f'{band.upper()}: H2O + CO2 line absorption, T = {T_AIR:.1f} K, RH = {RH:.0%} (no continuum)')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f'{output_dir()}/09_hitran_transmittance.png'
    fig.savefig(path)
    print('wrote', path)
except ImportError:
    pass
