"""
mitsuba.ir — helpers for physically based infrared rendering with Mitsuba 3.

Submodules
----------
radiometry   Planck's law, band radiance, apparent temperature, band/film
             definitions, solar spectrum.
materials    Infrared optical constants (Drude-Lorentz metals, data from the
             refractiveindex.info database), BSDF builders, loaders for
             spectral-library emissivity files (ECOSTRESS/ASTER).
scene        Scene-level helpers: assign temperatures to all shapes
             ("thermalize"), validate a scene dictionary for IR rendering,
             sun and sky sources.
atmosphere   Clear-sky emissivity (Idso 1981), thermal sky and emissive air
             media, HITRAN line-by-line molecular absorption (via hapi).

See docs/ir/IR_TECHNICAL_REPORT.md for the physical background and the
assumptions behind every function.
"""

from . import radiometry, materials, scene, atmosphere

from .radiometry import (
    planck, planck_dT, band_radiance, apparent_temperature,
    BANDS, band_range, band_film, boxcar_srf, solar_irradiance_toa,
)
from .materials import (
    drude_lorentz_nk, RAKIC_LD, conductor_ir, opaque_ir, window_ior,
    normal_emissivity, hemispherical_emissivity, load_spd,
    load_ecostress, diffuse_from_emissivity, diffuse_from_reflectance,
    IR_MATERIALS,
)
from .scene import thermalize, validate, sun, sky_blackbody
from .atmosphere import (
    sky_emissivity_idso, thermalsky, air_medium, medium_box, hitran_absorption,
    transmittance, vapour_pressure, rsb_continuum,
)

__all__ = [
    'radiometry', 'materials', 'scene', 'atmosphere',
    'planck', 'planck_dT', 'band_radiance', 'apparent_temperature',
    'BANDS', 'band_range', 'band_film', 'boxcar_srf', 'solar_irradiance_toa',
    'drude_lorentz_nk', 'RAKIC_LD', 'conductor_ir', 'opaque_ir', 'window_ior',
    'normal_emissivity', 'hemispherical_emissivity', 'load_spd',
    'load_ecostress', 'diffuse_from_emissivity', 'diffuse_from_reflectance',
    'IR_MATERIALS',
    'thermalize', 'validate', 'sun', 'sky_blackbody',
    'sky_emissivity_idso', 'thermalsky', 'air_medium', 'medium_box', 'hitran_absorption',
    'transmittance', 'vapour_pressure', 'rsb_continuum',
]
