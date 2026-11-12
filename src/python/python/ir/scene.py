"""
Scene-level helpers for infrared rendering.
"""

import numpy as np

from .radiometry import band_range, solar_irradiance_toa, planck, T_SUN

#: Plugin names of Mitsuba shapes that can carry an emitter
SHAPE_TYPES = {
    'obj', 'ply', 'serialized', 'rectangle', 'cube', 'sphere', 'disk',
    'cylinder', 'bsplinecurve', 'linearcurve', 'sdfgrid', 'ellipsoids',
    'ellipsoidsmesh', 'blender', 'merge',
}
#: Shapes that cannot carry an emitter
SHAPE_NO_EMITTER = {'instance', 'shapegroup'}


def _is_shape(d):
    return isinstance(d, dict) and d.get('type') in SHAPE_TYPES


def _copy_tree(node):
    """Copy the dict/list structure of a scene; leaves (numbers, strings,
    Mitsuba objects such as transforms or bitmaps) are shared, not copied."""
    if isinstance(node, dict):
        return {k: _copy_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_copy_tree(v) for v in node]
    return node


def thermalize(scene, temperature=300.0, overrides=None, twosided=True, in_place=False):
    """
    Attach a ``thermal`` emitter to every shape of a scene dictionary that
    has no emitter yet (report, assumption A6: untagged surfaces are at 0 K).

    Parameters
    ----------
    scene : dict
        Scene dictionary (``'type': 'scene'``). Nested shapes are handled.
    temperature : float or callable
        Default temperature in kelvin, or ``f(key, shape_dict) -> float | None``
        (``None`` leaves the shape untagged).
    overrides : dict, optional
        ``{shape_key: temperature}`` taking precedence over ``temperature``.
    twosided : bool
        Passed to the thermal emitters.
    in_place : bool
        Modify ``scene`` instead of returning a deep copy.
    """
    overrides = overrides or {}
    out = scene if in_place else _copy_tree(scene)

    def visit(key, node):
        if not isinstance(node, dict):
            return
        if _is_shape(node) and 'emitter' not in node:
            if key in overrides:
                T = overrides[key]
            elif callable(temperature):
                T = temperature(key, node)
            else:
                T = temperature
            if T is not None:
                node['emitter'] = {'type': 'thermal', 'temperature': float(T),
                                   'twosided': bool(twosided)}
        for k, v in node.items():
            if isinstance(v, dict):
                visit(k, v)

    visit('', out)
    return out


_RGB_TYPES = {'rgb', 'srgb', 'srgb_d65'}
_VISIBLE_ONLY_SPECTRA = {'d65'}
#: Parameter names under which a 3-vector is geometry, not a colour
_GEOMETRIC_KEYS = {
    'center', 'direction', 'position', 'origin', 'target', 'up', 'scale',
    'translate', 'rotate', 'axis', 'normal', 'size', 'p0', 'p1', 'p2', 'p3',
    'min', 'max', 'offset', 'to_world', 'to_uv', 'wavelengths', 'values',
    'shift', 'resolution', 'crop_offset', 'crop_size', 'eye', 'look_at',
}


def validate(scene, band=None):
    """
    Check a scene dictionary for constructs that are wrong or meaningless in
    infrared rendering. Returns a list of human-readable warnings (empty when
    the scene looks fine). Checks:

    * integrator is thermal-aware (``path_ir``)
    * film is a ``specfilm`` whose bands lie in the IR
    * no RGB / sRGB colour inputs, no D65 spectra (defined on 360-830 nm only)
    * conductor presets use IR optical constants (``*_ir``)
    * ``uniform``/``regular``/``blackbody`` spectra cover the requested band
    * every shape carries an emitter (temperature)
    """
    warnings = []
    lo, hi = band_range(band) if band is not None else (None, None)

    def covers(node, where):
        wmin = node.get('wavelength_min')
        wmax = node.get('wavelength_max')
        if lo is not None and wmin is not None and wmax is not None:
            if float(wmin) > lo or float(wmax) < hi:
                warnings.append(f"{where}: spectrum range [{wmin}, {wmax}] nm does not cover "
                                f"the band [{lo:g}, {hi:g}] nm (values outside are zero)")
        if node.get('type') == 'blackbody' and (wmin is None or wmax is None):
            warnings.append(f"{where}: 'blackbody' spectrum defaults to 360-830 nm; set "
                            f"wavelength_min/max or use the 'thermal' emitter")

    def visit(path, node):
        if isinstance(node, (list, tuple)) and len(node) == 3 and all(
                isinstance(v, (int, float)) for v in node):
            key = path.rsplit('.', 1)[-1]
            if key not in _GEOMETRIC_KEYS:
                warnings.append(f"{path}: RGB triple is spectrally upsampled on 360-830 nm only; "
                                f"use a 'uniform', 'regular' or 'irregular' spectrum")
            return
        if not isinstance(node, dict):
            return
        t = node.get('type')
        where = path or 'scene'
        if t in _RGB_TYPES:
            warnings.append(f"{where}: '{t}' colour is defined on 360-830 nm only")
        if t in _VISIBLE_ONLY_SPECTRA:
            warnings.append(f"{where}: '{t}' spectrum is visible-light only")
        if t == 'hdrfilm':
            warnings.append(f"{where}: 'hdrfilm' converts to XYZ/RGB and is zero in the IR; "
                            f"use 'specfilm' (mitsuba.ir.band_film)")
        if t in ('uniform', 'regular', 'blackbody'):
            covers(node, where)
        if t in ('conductor', 'roughconductor') and 'material' in node \
                and not str(node['material']).endswith('_ir'):
            warnings.append(f"{where}: conductor preset '{node['material']}' has visible-light "
                            f"optical constants (360-830 nm); use an '*_ir' preset")
        if t in ('conductor', 'roughconductor') and 'material' not in node \
                and 'eta' not in node and 'k' not in node:
            warnings.append(f"{where}: conductor without 'material'/'eta'/'k' defaults to a "
                            f"visible-light preset")
        if t in ('path', 'volpath', 'volpathmis', 'direct', 'prb', 'prb_basic', 'ptracer'):
            warnings.append(f"{where}: integrator '{t}' is not thermal-aware; thermal surfaces "
                            f"render as blackbodies. Use 'path_ir'")
        if _is_shape(node) and 'emitter' not in node:
            warnings.append(f"{where}: shape '{t}' has no emitter, i.e. it is at 0 K "
                            f"(use mitsuba.ir.thermalize)")
        if t == 'specfilm' and lo is not None:
            srfs = [v for k, v in node.items() if isinstance(v, dict) and 'type' in v
                    and v['type'] not in ('gaussian', 'box', 'tent', 'mitchell', 'lanczos',
                                          'catmullrom')]
            if not srfs:
                warnings.append(f"{where}: specfilm has no spectral response function")
        for k, v in node.items():
            if k == 'type':
                continue
            visit(f"{path}.{k}" if path else k, v)

    visit('', scene)
    return warnings


def sun(direction, band=None, scale=1.0, transmittance=None, T_sun=T_SUN,
        n_points=512, sampling_weight=100.0, lambda_range=None):
    """
    Directional sun emitter with a physically scaled solar spectrum.

    The irradiance is E(lambda) = scale * tau(lambda) * pi B(lambda, T_sun) (R_sun/AU)^2
    (W m^-2 nm^-1, normal incidence) where ``transmittance`` is an optional
    ``(wavelengths_nm, tau)`` table (e.g. an atmospheric transmission
    spectrum from MODTRAN/HITRAN) and ``scale`` a global factor (e.g. for
    ``cos`` of the zenith angle when projecting onto a horizontal surface is
    *not* desired, keep 1: Mitsuba applies the cosine automatically).

    ``direction`` is the direction *in which the light travels* (Mitsuba
    convention). ``sampling_weight`` is large by default because in an IR
    scene every surface is an emitter competing for next-event estimation
    (report, section 4.4).
    """
    if lambda_range is None:
        lambda_range = band_range(band) if band is not None else (300.0, 20000.0)
    lo, hi = lambda_range
    lam = np.geomspace(lo, hi, int(n_points))
    E = scale * solar_irradiance_toa(lam, T_sun)
    if transmittance is not None:
        E = E * np.interp(lam, np.asarray(transmittance[0], float),
                          np.asarray(transmittance[1], float), left=0.0, right=0.0)
    return {
        'type': 'directional',
        'direction': [float(v) for v in direction],
        'sampling_weight': float(sampling_weight),
        'irradiance': {'type': 'irregular',
                       'wavelengths': ', '.join(f'{x:.6g}' for x in lam),
                       'values': ', '.join(f'{x:.6g}' for x in E)},
    }


def sky_blackbody(T_sky, lambda_range=(300.0, 20000.0), sampling_weight=1.0):
    """
    Isotropic sky as a blackbody at an effective (apparent) temperature. A
    crude but common first-order model for the LWIR downwelling radiance of
    a clear sky (effective zenith temperatures of 230-260 K are typical;
    Idso 1981). A zenith-angle dependent sky model is planned (Phase 3).
    """
    return {'type': 'constant', 'sampling_weight': float(sampling_weight),
            'radiance': {'type': 'blackbody', 'temperature': float(T_sky),
                         'wavelength_min': float(lambda_range[0]),
                         'wavelength_max': float(lambda_range[1])}}
