"""
Coupling of the heat solver to a Mitsuba scene: sampling of surface
elements, shadow rays and sky-view factors, time integration, and
construction of temperature textures for the ``thermal`` emitter.
"""

import copy
import numpy as np

from . import solar as _solar
from .heat import Column, Material, SIGMA_SB


import contextlib as _contextlib
_nullcontext = _contextlib.nullcontext


def _copy_tree(node):
    if isinstance(node, dict):
        return {k: _copy_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_copy_tree(v) for v in node]
    return node


def _as_numpy(arr):
    a = np.array(arr)
    return a


def _mesh_has_unique_uvs(mesh, grid=128, max_faces=200000, max_overlap=0.02):
    """
    True when the mesh has texture coordinates that map its faces onto
    disjoint regions of UV space (rasterised overlap test), so that a UV
    texture can represent a per-point quantity. Mitsuba's ``cube`` (all
    faces on the same square) and dense models without an unwrap return False.
    """
    import mitsuba as mi
    import drjit as dr
    nf = mesh.face_count()
    nv = mesh.vertex_count()
    if nf == 0 or nf > max_faces:
        return False
    uv = _as_numpy(mesh.vertex_texcoord(dr.arange(mi.UInt32, nv))).reshape(2, -1).T
    if not np.any(uv != 0.0):
        return False
    fi = _as_numpy(mesh.face_indices(dr.arange(mi.UInt32, nf))).reshape(3, -1).T
    cover = np.zeros((grid, grid), dtype=np.int32)
    tri = uv[fi] * grid                                   # (F, 3, 2) in pixel units
    for a, b, c in tri:
        lo = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        hi = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        lo = np.clip(lo, 0, grid - 1)
        hi = np.clip(hi, 0, grid - 1)
        if np.any(hi < lo):
            continue
        xs = np.arange(lo[0], hi[0] + 1) + 0.5
        ys = np.arange(lo[1], hi[1] + 1) + 0.5
        X, Y = np.meshgrid(xs, ys)
        d = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(d) < 1e-12:
            continue
        w0 = ((b[0] - X) * (c[1] - Y) - (c[0] - X) * (b[1] - Y)) / d
        w1 = ((c[0] - X) * (a[1] - Y) - (a[0] - X) * (c[1] - Y)) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        cover[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1] += inside.astype(np.int32)
    covered = cover >= 1
    if not covered.any():
        return False
    overlap = np.count_nonzero(cover >= 2) / np.count_nonzero(covered)
    return overlap <= max_overlap


class _ShapeElements:
    """
    Surface sample points of one shape.

    * Analytic shapes (rectangle, sphere, disk, cylinder) and meshes with
      ``mesh_mode='uv'`` are sampled on a UV grid; the result is a bitmap
      temperature texture.
    * Meshes without a unique UV unwrap (``mesh_mode='auto'`` decides with a
      rasterised overlap test; Mitsuba's built-in ``cube`` and many imported
      models map several faces onto the same UV region) are sampled at their
      vertices. The result is a per-vertex temperature attribute, exported
      with the mesh to PLY by :meth:`ThermalResult.apply`. The vertex density
      of the mesh then sets the thermal resolution.
    """

    def __init__(self, key, shape, material, resolution, mesh_mode='auto'):
        self.key = key
        self.shape = shape
        self.material = material
        is_mesh = hasattr(shape, 'vertex_count')
        if is_mesh and mesh_mode == 'auto':
            mesh_mode = 'uv' if _mesh_has_unique_uvs(shape) else 'vertex'
        if is_mesh and mesh_mode == 'vertex':
            self._init_vertices(shape)
        else:
            self._init_uv(shape, resolution)
        self.E = int(self.valid.sum())
        self.idx = np.where(self.valid)[0]

    def _init_uv(self, shape, resolution):
        import mitsuba as mi
        import drjit as dr
        self.mode = 'uv'
        res = (resolution, resolution) if np.isscalar(resolution) else tuple(resolution)
        self.res = res
        H, W = res
        u = (np.arange(W) + 0.5) / W
        v = (np.arange(H) + 0.5) / H
        uu, vv = np.meshgrid(u, v)
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
        self.valid = valid
        self.p = p
        self.n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def _init_vertices(self, shape):
        import mitsuba as mi
        import drjit as dr
        self.mode = 'vertex'
        nv = shape.vertex_count()
        idx = dr.arange(mi.UInt32, nv)
        vp = _as_numpy(shape.vertex_position(idx)).reshape(3, -1).T
        vn = _as_numpy(shape.vertex_normal(idx)).reshape(3, -1).T
        if not np.all(np.isfinite(vn)) or np.any(np.linalg.norm(vn, axis=1) < 0.5):
            # Area-weighted normals from the faces
            nf = shape.face_count()
            fi = _as_numpy(shape.face_indices(dr.arange(mi.UInt32, nf))).reshape(3, -1).T
            fn = np.cross(vp[fi[:, 1]] - vp[fi[:, 0]], vp[fi[:, 2]] - vp[fi[:, 0]])
            vn = np.zeros_like(vp)
            for j in range(3):
                np.add.at(vn, fi[:, j], fn)
        self.p = vp
        self.n = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        self.valid = np.ones(nv, dtype=bool)
        self.res = (nv, 1)


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
        """Mitsuba texture dictionary holding the temperature field of a UV-sampled shape [K]."""
        import mitsuba as mi
        sh = self._shapes[key]
        if sh.mode != 'uv':
            raise ValueError(f"shape '{key}' is sampled per vertex; use apply() or write_mesh()")
        T = self.temperature(key, t)
        if fill is None:
            fill = float(np.nanmean(T))
        T = np.where(np.isfinite(T), T, fill).astype(np.float32)
        return {'type': 'bitmap', 'bitmap': mi.Bitmap(np.ascontiguousarray(T[..., None])),
                'raw': True, 'filter_type': 'bilinear', 'wrap_mode': 'clamp'}

    def write_mesh(self, key, t, filename):
        """
        Write a vertex-sampled mesh to PLY (world coordinates) with its
        temperature as the vertex attribute ``vertex_temperature``.
        """
        import mitsuba as mi
        sh = self._shapes[key]
        if sh.mode != 'vertex':
            raise ValueError(f"shape '{key}' is sampled on a UV grid; use texture()")
        T = self.temperature(key, t).astype(np.float32)
        mesh = sh.shape
        if mesh.has_attribute('vertex_temperature'):
            mesh.remove_attribute('vertex_temperature')
        mesh.add_attribute('vertex_temperature', mi.TensorXf(T.reshape(-1, 1)))
        with mi.ScopedSetLogLevel(mi.LogLevel.Error) if hasattr(mi, 'ScopedSetLogLevel') else _nullcontext():
            mesh.write_ply(str(filename))
        return str(filename)

    def apply(self, scene_dict, t, twosided=True, default_temperature=None, mesh_dir=None):
        """
        Copy of ``scene_dict`` in which every simulated shape carries a
        ``thermal`` emitter with its temperature field at time ``t``:
        UV-sampled shapes get a bitmap texture; vertex-sampled meshes are
        written to ``mesh_dir`` (default: a temporary directory) as PLY files
        in world coordinates with a ``vertex_temperature`` attribute and
        referenced from the dictionary. Shapes that were not simulated get
        ``default_temperature`` (K) if given, otherwise they are left untouched.
        """
        import os
        import tempfile
        out = _copy_tree(scene_dict)
        for key in self.keys():
            node = out.get(key)
            if node is None:
                continue
            sh = self._shapes[key]
            if sh.mode == 'vertex':
                if mesh_dir is None:
                    mesh_dir = tempfile.mkdtemp(prefix='mitsuba_ir_thermal_')
                os.makedirs(mesh_dir, exist_ok=True)
                tag = f'{t:05.2f}'.replace('.', '_') if not isinstance(t, str) else t.replace(':', '_')
                path = os.path.join(mesh_dir, f'{key}_{tag}.ply')
                self.write_mesh(key, t, path)
                new_node = {k: v for k, v in node.items() if k not in ('type', 'to_world', 'filename',
                                                                      'face_normals', 'flip_normals')}
                new_node.update({'type': 'ply', 'filename': path})
                new_node['emitter'] = {'type': 'thermal', 'twosided': twosided,
                                       'temperature': {'type': 'mesh_attribute', 'name': 'vertex_temperature'}}
                out[key] = new_node
            else:
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
        Samples per UV axis for UV-sampled shapes (default 64), optionally
        per shape key. Meshes are sampled per vertex unless ``mesh_mode='uv'``.
    mesh_mode : 'auto' (default), 'uv' or 'vertex'
        How meshes are sampled: 'auto' uses a UV grid when the mesh has a
        unique UV unwrap (rasterised overlap test) and vertices otherwise.
    up, north : world-space zenith and geographic north directions.
    sky_rays : hemisphere rays per element for the sky-view factor.
    T_surroundings : temperature of the unresolved surroundings seen by an
        element in directions blocked from the sky; None = air temperature.
    """

    def __init__(self, scene_dict, weather, materials, resolution=64, up=(0, 0, 1),
                 north=(0, 1, 0), sky_rays=128, T_surroundings=None, seed=0, mesh_mode='auto'):
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
            self.shapes[key] = _ShapeElements(key, ids[key], mat, res, mesh_mode=mesh_mode)
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

    # --- shading-time (linear response) model -----------------------------------
    def dynamic_models(self, dt=600.0, days_memory=6, ref_days=4):
        """
        :class:`DynamicModel` per simulated shape (cached). The reference
        element of a shape has the shape's mean normal when it is well
        defined (plates, roofs) and the zenith otherwise (closed objects).
        """
        from .dynamic import DynamicModel
        key_dt = (float(dt), int(days_memory), int(ref_days))
        cache = getattr(self, '_dyn_cache', {})
        if cache.get('params') != key_dt:
            cache = {'params': key_dt, 'models': {}}
        models = {}
        for key, sh in self.shapes.items():
            n_ref = _mean_normal(sh)
            n_ref = self.up if n_ref is None else n_ref
            ck = (id(sh.material), tuple(np.round(n_ref, 3)))
            if ck not in cache['models']:
                cache['models'][ck] = DynamicModel(self.weather, sh.material, dt=dt, days_memory=days_memory,
                                                   reference_normal=n_ref, up=self.up, north=self.north,
                                                   ref_days=ref_days)
            models[key] = cache['models'][ck]
        self._dyn_cache = cache
        return models

    def dynamic(self, scene_dict, t_hours, twosided=True, default_temperature=None, dt=600.0,
                days_memory=6):
        """
        Copy of ``scene_dict`` in which every simulated shape carries a
        ``thermal_dynamic`` emitter for local time ``t_hours``: the surface
        temperature is then evaluated by ``path_ir``/``volpath_ir`` at every
        shading point from shadow and sky rays (no texture, no resolution
        limit; report section 6.8.3). Set the integrator's ``thermal_up`` to
        this scene's zenith direction.
        """
        if isinstance(t_hours, str):
            h, m = t_hours.split(':')
            t_hours = float(h) + float(m) / 60.0
        models = self.dynamic_models(dt=dt, days_memory=days_memory)
        out = _copy_tree(scene_dict)
        for key, model in models.items():
            node = out.get(key)
            if node is None:
                continue
            node['emitter'] = model.emitter_dict(float(t_hours), twosided=twosided)
        if default_temperature is not None:
            from ..scene import thermalize
            out = thermalize(out, temperature=float(default_temperature), twosided=twosided, in_place=True)
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

        self._vis_cache = vis_cache
        self._t_grid = t_grid
        records = {key: [] for key in self.shapes}
        from .dynamic import step_index
        rec_steps = {step_index(t, dt, n_steps): t for t in out_times}
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


def _mean_normal(sh):
    n = sh.n[sh.idx].mean(axis=0)
    norm = np.linalg.norm(n)
    return n / norm if norm > 0.5 else None


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
