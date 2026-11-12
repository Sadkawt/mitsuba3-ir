"""
mitsuba.ir.thermal — surface temperatures of outdoor scenes from a
surface energy balance.

Every sample point of a surface owns a one-dimensional transient heat
conduction column (semi-infinite ground or a slab of given thickness) whose
surface is driven by absorbed solar irradiance (direct, with shadows cast by
the scene geometry; diffuse and ground-reflected, with the sky-view factor),
longwave exchange with the sky and the surroundings, convection, and optional
internal heating. The forcing comes from one parametrised weather model
(date, location, air temperature range, humidity, wind, cloud cover,
aerosol load). The result is a temperature texture per shape for the
``thermal`` emitter.

Modules
-------
solar      Solar position (Michalsky 1988), Bird & Hulstrom (1981) clear-sky
           irradiance, Kasten & Czeplak (1980) cloud correction.
weather    ``Weather``: the parametrised diurnal forcing (air temperature,
           humidity, wind, sky longwave, irradiance).
heat       ``Material`` properties with a table of common outdoor materials,
           and the vectorised implicit 1-D conduction solver.
scene      ``ThermalScene``: samples shapes of a Mitsuba scene, computes
           shadows and sky-view factors by ray casting, runs the solver and
           produces temperature textures / thermalised scene dictionaries.

See docs/ir/IR_TECHNICAL_REPORT.md, section 6.8.
"""

from . import solar, weather, heat, scene
from .solar import solar_position, sun_direction, bird_clear_sky, cloud_correction, day_of_year
from .weather import Weather, celsius
from .heat import Material, MATERIALS, Column
from .scene import ThermalScene, ThermalResult

__all__ = [
    'solar', 'weather', 'heat', 'scene',
    'solar_position', 'sun_direction', 'bird_clear_sky', 'cloud_correction', 'day_of_year',
    'Weather', 'celsius', 'Material', 'MATERIALS', 'Column',
    'ThermalScene', 'ThermalResult',
]
