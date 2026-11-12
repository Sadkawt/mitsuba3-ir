"""
Coupling of the heat solver to a Mitsuba scene: sampling of surface
elements, shadow rays and sky-view factors, time integration, and
construction of temperature textures for the ``thermal`` emitter.
"""

import copy
import numpy as np

from . import solar as _solar
from .heat import Column, Material, SIGMA_SB


def _copy_tree(node):
    if isinstance(node, dict):
        return {k: _copy_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_copy_tree(v) for v in node]
    return node


def _as_numpy(arr):
    a = np.array(arr)
    return a


class _ShapeElements:
    """Surface sample points of one shape (UV grid or vertices)."""

    def __init__(self, key, shape, material, resolution):
        import mitsuba as mi
        import drjit as dr
        self.key = key
        self.shape = shape
        self.material = material
        self.mode = 'uv'
        res = (resolution, resolution) if np.isscalar(resolution) else tuple(resolution)
        self.res = res
        H, W = res
        u = (np.arange(W) + 0.5) / W
        v = (np.arange(H) + 0.5) / H
        uu, vv = np.meshgrid(u, v)
        try:
            uv = mi.Point2f(mi.Float(uu.ravel()), mi.Float(vv.ravel()))
            si = shape.eval_parameterization(uv, +mi.RayFlags.Default, dr.full(mi.Bool, True, H * W))
            valid = _as_numpy(si.is_valid()).ravel()
            if valid.size == 1:
                valid = np.full(H * W, bool(valid[0]))
            p = _as_numpy(si.p).reshape(3, -1).T
            n = _as_numpy(si.n).reshape(3, -1).T
            # Constant quantities (e.g. the normal of a flat rectangle) come
            # back as width-1 literals: broadcast them
            if p.shape[0] == 1 and H * W > 1:
                p = np.broadcast_to(p, (H * W, 3)).copy()
            if n.shape[0] == 1 and H * W > 1:
                n = np.broadcast_to(n, (H * W, 3)).copy()
            if not valid.all():
                # Meshes with partial UV coverage: keep valid texels only
                pass
            self.valid = valid
            self.p = p
            self.n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        except Exception:
            # Fallback: vertices of a mesh without UVs
            if not hasattr(shape, 'vertex_positions_buffer'):
                raise
            self.mode = 'vertex'
            vp = _as_numpy(shape.vertex_positions_buffer()).reshape(-1, 3)
            if shape.has_vertex_normals():
                vn = _as_numpy(shape.vertex_normals_buffer()).reshape(-1, 3)
            else:
                faces = _as_numpy(shape.faces_buffer()).reshape(-1, 3)
                vn = np.zeros_like(vp)
                fn = np.cross(vp[faces[:, 1]] - vp[faces[:, 0]], vp[faces[:, 2]] - vp[faces[:, 0]])
                for j in range(3):
                    np.add.at(vn, faces[:, j], fn)
            self.p = vp
            self.n = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
            self.valid = np.ones(len(vp), dtype=bool)
            self.res = (len(vp), 1)
        self.E = int(self.valid.sum())
        self.idx = np.where(self.valid)[0]


class ThermalResult:
    """Temperatures of all sampled shapes at the requested output times."""

    def __init__(self, times, shapes, temps, weather):
        self.times = list(times)                 # local hours
        self._shapes = shapes                    # key -> _ShapeElements
        self._temps = temps                      # key -> array (n_times, E)
        self.weather = weather

    def keys(self):
        return list(self._shapes)

    def _time_index(self, t):
        if isinstance(t, str):
            h, m = t.split(':')
            t = float(h) + float(m) / 60.0
        if isinstance(t, (int, np.integer)) and not isinstance(t, bool) and t < len(self.times) and float(t) not in self.times:
            return int(t)
        arr = np.asarray(self.times)
        i = int(np.argmin(np.abs(arr - float(t))))
        if abs(arr[i] - float(t)) > 1e-6:
            raise KeyError(f"time {t} h not in output times {self.times}")
        return i

    def temperature(self, key, t):
        """Temperature array of a shape (H, W) for UV sampling or (V,) for vertices; NaN where unsampled."""
        sh = self._shapes[key]
        T = self._temps[key][self._time_index(t)]
        if sh.mode == 'uv':
            out = np.full(sh.res[0] * sh.res[1], np.nan)
            out[sh.idx] = T
            return out.reshape(sh.res)
        return T

    def texture(self, key, t, fill=None):
        """Mitsuba texture dictionary holding the temperature field of a shape [K]."""
        import mitsuba as mi
        sh = self._shapes[key]
        T = self.temperature(key, t)
        if sh.mode == 'uv':
            if fill is None:
                fill = float(np.nanmean(T))
            T = np.where(np.isfinite(T), T, fill).astype(np.float32)
            return {'type': 'bitmap', 'bitmap': mi.Bitmap(np.ascontiguousarray(T[..., None])),
                    'raw': True, 'filter_type': 'bilinear', 'wrap_mode': 'clamp'}
        # Vertex mode: attach a mesh attribute
        sh.shape.add_attribute('vertex_temperature', 1, mi.Float(T.astype(np.float32)))
        return {'type': 'mesh_attribute', 'name': 'vertex_temperature'}

    def apply(self, scene_dict, t, twosided=True, default_temperature=None):
        """
        Copy of ``scene_dict`` in which every simulated shape carries a
        ``thermal`` emitter with its temperature texture at time ``t``.
        Shapes that were not simulated get ``default_temperature`` (K) if
        given (e.g. the air temperature), otherwise they are left untouched.
        """
        out = _copy_tree(scene_dict)
        for key in self.keys():
            node = out.get(key)
            if node is None:
                continue
            if self._shapes[key].mode == 'vertex':
                # The mesh attribute lives on the loaded shape; the dictionary
                # scene must reference the same mesh, so we store the array
                # for the caller instead of a texture (see mesh_attribute docs)
                node['emitter'] = {'type': 'thermal', 'twosided': twosided,
                                   'temperature': float(np.nanmean(self.temperature(key, t)))}
                node['_vertex_temperature'] = self.temperature(key, t)
                continue
            node['emitter'] = {'type': 'thermal', 'twosided': twosided,
                               'temperature': self.texture(key, t)}
        if default_temperature is not None:
            from ..scene import thermalize
            out = thermalize(out, temperature=float(default_temperature), twosided=twosided, in_place=True)
        return out

    def statistics(self, key, t):
        T = self.temperature(key, t)
        return dict(min=float(np.nanmin(T)), max=float(np.nanmax(T)), mean=float(np.nanmean(T)))


class ThermalScene:
    """
    Surface-temperature simulation of the shapes of a Mitsuba scene.

    Parameters
    ----------
    scene_dict : dict
        Mitsuba scene dictionary (loaded internally for ray casting; a
        current JIT variant such as ``llvm_ad_spectral`` is recommended).
    weather : Weather
    materials : dict
        ``{shape_key: Material}``; shapes not listed are not simulated but
        still cast shadows.
    resolution : int or dict
        Samples per UV axis (default 64), optionally per shape key.
    up, north : world-space zenith and geographic north directions.
    sky_rays : hemisphere rays per element for the sky-view factor.
    T_surroundings : temperature of the unresolved surroundings seen by an
        element in directions blocked from the sky; None = air temperature.
    """

    def __init__(self, scene_dict, weather, materials, resolution=64, up=(0, 0, 1),
                 north=(0, 1, 0), sky_rays=128, T_surroundings=None, seed=0):
        import mitsuba as mi
        self.weather = weather
        self.up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
        self.north = np.asarray(north, dtype=np.float64)
        self.T_surroundings = T_surroundings
        self.scene_dict = scene_dict
        self.scene = mi.load_dict(_strip_sensors(scene_dict))
        ids = {s.id(): s for s in self.scene.shapes()}
        self.shapes = {}
        for key, mat in materials.items():
            if key not in ids:
                raise KeyError(f"shape '{key}' not found in the scene (ids: {sorted(ids)})")
            res = resolution.get(key, 64) if isinstance(resolution, dict) else resolution
            self.shapes[key] = _ShapeElements(key, ids[key], mat, res)
        bbox = self.scene.bbox()
        self.eps = 1e-4 * float(np.linalg.norm(np.array(bbox.max) - np.array(bbox.min)))
        self._rng = np.random.default_rng(seed)
        self._compute_sky_view(sky_rays)

    # --- ray casting helpers ------------------------------------------------
    def _ray_test(self, o, d):
        import mitsuba as mi
        o = np.asarray(o, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        ray = mi.Ray3f(mi.Point3f(mi.Float(o[:, 0]), mi.Float(o[:, 1]), mi.Float(o[:, 2])),
                       mi.Vector3f(mi.Float(d[:, 0]), mi.Float(d[:, 1]), mi.Float(d[:, 2])))
        hit = _as_numpy(self.scene.ray_test(ray)).ravel()
        if hit.size == 1 and o.shape[0] > 1:
            hit = np.full(o.shape[0], bool(hit[0]))
        return hit.astype(bool)

    def _compute_sky_view(self, n_rays):
        """Fraction of the hemisphere above each element from which the sky is visible
        (cosine-weighted, i.e. the configuration factor to the sky). Uses a
        stratified (Hammersley) direction set rotated randomly per element."""
        # Cosine-weighted Hammersley directions in the local frame
        k = np.arange(n_rays)
        u1 = (k + 0.5) / n_rays
        bits = k.astype(np.uint32)
        u2 = np.zeros(n_rays)
        f = 0.5
        while bits.any():
            u2 += f * (bits & 1)
            bits >>= 1
            f *= 0.5
        r = np.sqrt(u1)
        for sh in self.shapes.values():
            p = sh.p[sh.idx]
            n = sh.n[sh.idx]
            E = len(p)
            F = np.zeros(E)
            t = np.cross(n, np.where(np.abs(n[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
            t /= np.linalg.norm(t, axis=1, keepdims=True)
            b = np.cross(n, t)
            rot = self._rng.random(E) * 2 * np.pi           # decorrelating rotation about the normal
            for j in range(n_rays):
                phi = 2 * np.pi * u2[j] + rot
                d = (r[j] * np.cos(phi))[:, None] * t + (r[j] * np.sin(phi))[:, None] * b \
                    + np.sqrt(1 - u1[j]) * n
                above = (d @ self.up) > 0.0
                hit = self._ray_test(p + n * self.eps, d)
                F += (~hit) & above
            sh.F_sky = F / n_rays

    def _solar_visibility(self, sun_dir):
        """cos(theta_i) * visibility for every element (0 when shadowed or facing away)."""
        out = {}
        for key, sh in self.shapes.items():
            p = sh.p[sh.idx]
            n = sh.n[sh.idx]
            cos_i = n @ sun_dir
            lit = cos_i > 1e-3
            vis = np.zeros(len(p))
            if lit.any():
                hit = self._ray_test(p[lit] + n[lit] * self.eps, np.broadcast_to(sun_dir, (lit.sum(), 3)))
                vis[lit] = np.where(hit, 0.0, cos_i[lit])
            out[key] = vis
        return out

    # --- simulation -----------------------------------------------------------
    def run(self, days=3, dt=300.0, output_times=(0, 3, 6, 9, 12, 15, 18, 21), T0=None,
            progress=False):
        """
        Integrate the energy balance over ``days`` repetitions of the diurnal
        cycle (spin-up) and return a :class:`ThermalResult` with the
        temperatures at ``output_times`` (local hours or 'HH:MM') of the last day.
        """
        w = self.weather
        h = w.convection_coefficient()
        T_deep = w.deep_temperature()
        out_times = [float(t) if not isinstance(t, str) else float(t.split(':')[0]) + float(t.split(':')[1]) / 60.0
                     for t in output_times]
        T_init = w.T_mean if T0 is None else T0

        cols = {}
        for key, sh in self.shapes.items():
            col = Column(sh.material, sh.E, T_init)
            col.T_deep = T_deep
            cols[key] = col

        n_steps = int(round(24 * 3600.0 / dt))
        dt = 24 * 3600.0 / n_steps
        # Precompute forcing on the diurnal grid (periodic)
        t_grid = (np.arange(n_steps) + 0.5) * dt / 3600.0
        irr = w.irradiance(t_grid)
        T_air = w.air_temperature(t_grid)
        L_sky = w.sky_longwave(t_grid)
        sun_dirs = _solar.sun_direction(irr['elevation'], irr['azimuth'], self.up, self.north)
        # Shadow/visibility factors per step (cached for all spin-up days)
        vis_cache = []
        for s in range(n_steps):
            if irr['elevation'][s] > 0.0 and float(irr['dni'][s]) > 0.0:
                vis_cache.append(self._solar_visibility(sun_dirs[s]))
            else:
                vis_cache.append(None)

        records = {key: [] for key in self.shapes}
        rec_steps = {int(np.argmin(np.abs(t_grid - t))): t for t in out_times}
        for day in range(int(days)):
            for s in range(n_steps):
                dni, dhi, ghi = float(irr['dni'][s]), float(irr['dhi'][s]), float(irr['ghi'][s])
                Ta, Lsky = float(T_air[s]), float(L_sky[s])
                T_sur = Ta if self.T_surroundings is None else self.T_surroundings
                vis = vis_cache[s]
                for key, sh in self.shapes.items():
                    m = sh.material
                    F = sh.F_sky
                    direct = dni * vis[key] if vis is not None else 0.0
                    E_abs = m.alpha_solar * (direct + F * dhi + (1.0 - F) * w.ground_albedo * ghi)
                    L_eff = F * Lsky + (1.0 - F) * SIGMA_SB * T_sur ** 4
                    cols[key].step(dt, E_abs, L_eff, Ta, h, T_air_back=Ta, L_back=None)
                if day == int(days) - 1 and s in rec_steps:
                    for key in self.shapes:
                        records[key].append(cols[key].surface.copy())
            if progress:
                print(f"  day {day + 1}/{days}: " + ', '.join(
                    f"{k} {cols[k].surface.mean() - 273.15:.1f} C" for k in self.shapes))

        times_sorted = [rec_steps[s] for s in sorted(rec_steps)]   # recorded in step order
        temps = {key: np.array(records[key]) for key in self.shapes}
        return ThermalResult(times_sorted, self.shapes, temps, w)


def _strip_sensors(scene_dict):
    """Scene without sensors/integrator/emitters (geometry only, for ray tests)."""
    out = {}
    for k, v in scene_dict.items():
        if isinstance(v, dict) and v.get('type') in ('perspective', 'thinlens', 'orthographic',
                                                     'radiancemeter', 'irradiancemeter', 'distant',
                                                     'path', 'path_ir', 'volpath', 'volpath_ir',
                                                     'constant', 'thermalsky', 'directional', 'envmap',
                                                     'sunsky', 'point', 'spot'):
            continue
        if isinstance(v, dict):
            v = _copy_tree(v)
            v.pop('emitter', None)
        out[k] = v
    return out
