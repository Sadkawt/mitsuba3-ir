#!/usr/bin/env python3
"""
Example 14 - Thermal time lapse of a landscape, from the ground or the air.

A larger and more realistic version of example 13: a meadow with instanced
grass, an asphalt road, a gravel parking pad with two cars (one with a warm
engine), a dirt path, a concrete slab with a shed (metal roof), a pond,
trees and rocks. Every surface type has its own thermal material, so the
scene develops the contrasts of a real thermal image over the day: the road
and gravel heat up fast, the pond hardly moves, metal roofs mirror the cold
sky, shadows of the trees leave cool trails.

Temperatures come from the shading-time thermal model (report 6.8.3): one
per-element solve for the day, then every frame is rendered at its own local
time and each visible point gets its temperature from its own shadow history.

Vegetation: about 10^5 instanced grass tufts (three tuft meshes) and tree
crowns made of leaf clusters. Mitsuba cannot attach emitters to instances, so
all vegetation is emitter-less and the integrator evaluates its temperature
with the nested ``ambient`` leaf model (report A24) at every hit point, using
shadow rays: sunlit blade tips are warm, blades shaded by their neighbours
are cooler, and the trees' shadows show on the grass. The blades also shade
the soil below them.

Views (``--view``): ``ground`` (eye level by the road), ``aerial`` (oblique
from 38 m up), ``top`` (straight down from 110 m), or ``custom`` with
``--origin`` / ``--target``.

The output imitates an uncooled LWIR camera: white-hot grey scale with a gain
fixed over the sequence, a slight optical blur and NETD noise. Use
``--palette inferno --netd 0 --blur 0`` for a clean false-colour image.

Usage
-----
    python 14_timelapse_landscape.py --view ground --start 06:00 --stop 21:00 --step 30
    python 14_timelapse_landscape.py --view aerial --start 14:00 --stop 14:00      # single frame
    python 14_timelapse_landscape.py --view top --start 20:00 --stop 05:00 --step 45

Output: examples/ir/output/14_landscape_<view>/frame_NNN.png and
examples/ir/output/14_landscape_<view>.mp4 (GIF without ffmpeg).
"""

import argparse
import os
import shutil
import subprocess
import time
import warnings

import numpy as np


def parse_time(text):
    text = str(text).strip()
    if ':' in text:
        h, m = text.split(':')
        return float(h) + float(m) / 60.0
    return float(text)


def fmt_time(t):
    t = t % 24.0
    h = int(t)
    m = int(round((t - h) * 60.0))
    if m == 60:
        h, m = (h + 1) % 24, 0
    return f'{h:02d}:{m:02d}'


ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument('--start', default='06:00', help='start time (local, HH:MM or hours)')
ap.add_argument('--stop', default='21:00', help='stop time (inclusive; earlier than start = across midnight)')
ap.add_argument('--step', type=float, default=30.0, help='time delta between frames [minutes]')
ap.add_argument('--view', default='ground', choices=['ground', 'aerial', 'top', 'custom'])
ap.add_argument('--origin', type=float, nargs=3, default=None, help='camera position for --view custom (x east, y up, z south)')
ap.add_argument('--target', type=float, nargs=3, default=None, help='camera target for --view custom')
ap.add_argument('--fov', type=float, default=None, help='field of view [deg] (default depends on the view)')
ap.add_argument('--date', default='2026-07-15')
ap.add_argument('--lat', type=float, default=59.33)
ap.add_argument('--lon', type=float, default=18.07)
ap.add_argument('--utc-offset', type=float, default=2.0)
ap.add_argument('--tmin', type=float, default=14.0, help='daily minimum air temperature [C]')
ap.add_argument('--tmax', type=float, default=26.0, help='daily maximum air temperature [C]')
ap.add_argument('--humidity', type=float, default=0.55)
ap.add_argument('--wind', type=float, default=2.0, help='wind speed [m/s]')
ap.add_argument('--clouds', type=float, default=0.1)
ap.add_argument('--grass-density', type=float, default=1.0, help='scale factor of the number of grass tufts (0 = no grass)')
ap.add_argument('--exchange', action='store_true', help='radiative exchange between surfaces (slower scene loading)')
ap.add_argument('--width', type=int, default=960)
ap.add_argument('--height', type=int, default=540)
ap.add_argument('--spp', type=int, default=32)
ap.add_argument('--fps', type=float, default=6.0)
ap.add_argument('--palette', default='whitehot', choices=['whitehot', 'blackhot', 'inferno'])
ap.add_argument('--netd', type=float, default=0.05, help='sensor noise (NETD) [K]')
ap.add_argument('--blur', type=float, default=0.6, help='optical blur, Gaussian sigma [pixels]')
ap.add_argument('--vmin', type=float, default=None, help='display range minimum [C] (default: from all frames)')
ap.add_argument('--vmax', type=float, default=None, help='display range maximum [C]')
ap.add_argument('--seed', type=int, default=7, help='seed of the scene layout')
ap.add_argument('--variant', default='llvm_ad_spectral', choices=['llvm_ad_spectral', 'cuda_ad_spectral'])
args = ap.parse_args()

t_start, t_stop = parse_time(args.start), parse_time(args.stop)
if t_stop < t_start:
    t_stop += 24.0
if args.step <= 0:
    ap.error('--step must be positive')
times = np.arange(t_start, t_stop + 1e-9, args.step / 60.0)
if args.view == 'custom' and (args.origin is None or args.target is None):
    ap.error('--view custom needs --origin and --target')

import mitsuba as mi
mi.set_variant(args.variant)
from mitsuba import ir
from mitsuba.ir import thermal as th
from mitsuba.ir.thermal.dynamic import step_index
from _util import output_dir

UP, NORTH = (0, 1, 0), (0, 0, -1)            # Y up, north = -Z, east = +X
T = mi.ScalarTransform4f
BAND = 'lwir'
lo, hi = ir.BANDS[BAND]
rng = np.random.default_rng(args.seed)

weather = th.Weather(args.date, latitude=args.lat, longitude=args.lon, utc_offset=args.utc_offset,
                     T_min=th.celsius(args.tmin), T_max=th.celsius(args.tmax),
                     relative_humidity=args.humidity, wind_speed=args.wind,
                     cloud_cover=args.clouds, aod500=0.15, ground_albedo=0.2)
print(weather.summary())

out_root = output_dir()
frame_dir = os.path.join(out_root, f'14_landscape_{args.view}')
asset_dir = os.path.join(out_root, '14_landscape_assets')
os.makedirs(frame_dir, exist_ok=True)
os.makedirs(asset_dir, exist_ok=True)

# =============================================================================
# Scene. Reflectances are two-band (solar / LWIR): the thermal solver derives
# the solar absorptance and the emissivity from the same BSDFs that are rendered.
# =============================================================================
geometry, materials, resolution = {}, {}, {}


def add(key, shape, material, res=16):
    geometry[key] = shape
    materials[key] = material
    resolution[key] = res


def flat(cx, cz, half_x, half_z, y):
    """Horizontal rectangle centred at (cx, cz), facing up."""
    return T().translate([cx, y, cz]).rotate([1, 0, 0], -90).scale([half_x, half_z, 1])


# --- ground types -------------------------------------------------------------
add('meadow', {'type': 'rectangle', 'to_world': flat(0, 0, 200, 200, 0.0),
               'bsdf': ir.two_band_diffuse(0.22, 0.04)},                       # soil / thatch under the grass
    th.MATERIALS['grass'], 160)
add('road', {'type': 'rectangle', 'to_world': flat(0, 17, 200, 3.0, 0.012),
             'bsdf': ir.two_band_diffuse(0.09, 0.06)},                          # asphalt
    th.MATERIALS['asphalt'], (8, 256))
add('gravel_pad', {'type': 'rectangle', 'to_world': flat(0, 9, 7.0, 5.0, 0.010),
                   'bsdf': ir.two_band_diffuse(0.32, 0.07)},                    # light gravel
    th.MATERIALS['soil_dry'], 48)
add('dirt_path', {'type': 'rectangle', 'to_world': flat(0, -5, 1.0, 9.0, 0.008),
                  'bsdf': ir.two_band_diffuse(0.25, 0.06)},                     # trodden dry soil
    th.MATERIALS['soil_dry'], (64, 8))
add('slab', {'type': 'rectangle', 'to_world': flat(0, -19, 5.0, 5.0, 0.02),
             'bsdf': ir.two_band_diffuse(0.38, 0.08)},                          # concrete
    th.MATERIALS['concrete'], 32)
POND = (22.0, -12.0, 7.0)
add('pond', {'type': 'disk', 'to_world': T().translate([POND[0], 0.015, POND[1]]).rotate([1, 0, 0], -90).scale(POND[2]),
             'bsdf': ir.opaque_ir('H2O_ir')},                                   # smooth water: mirrors the sky
    # Wind and convection keep a pond's water mixed: modelled as a 1 m layer with
    # a high effective (eddy) conductivity, i.e. nearly isothermal in depth.
    # Evaporative cooling is not modelled (report A18), so the pond still ends
    # up somewhat warm on a sunny afternoon.
    th.MATERIALS['water_shallow'].replace(k=200.0, thickness=1.0, n_nodes=4, name='mixed pond water 1 m'), 32)

# --- shed on the slab: concrete walls, sheet-metal roof ---------------------------
add('shed_walls', {'type': 'cube', 'to_world': T().translate([0, 1.4, -19]).scale([3.2, 1.4, 2.6]),
                   'bsdf': ir.two_band_diffuse(0.40, 0.08)},
    th.MATERIALS['concrete_wall'].replace(T_internal=th.celsius(18), U_internal=2.0))
add('shed_roof', {'type': 'cube', 'to_world': T().translate([0, 2.86, -19]).rotate([0, 0, 1], 4).scale([3.6, 0.04, 3.0]),
                  'bsdf': ir.conductor_ir('Al_ir', alpha=0.25)},                # weathered sheet metal
    th.MATERIALS['aluminium_sheet_bare'])


# --- cars -----------------------------------------------------------------------
def add_car(prefix, x, z, yaw, rho_solar, warm_engine):
    paint = {'type': 'roughplastic', 'int_ior': 1.5, 'alpha': 0.25,
             'diffuse_reflectance': ir.two_band_diffuse(rho_solar, 0.04)['reflectance']}
    rubber = ir.two_band_diffuse(0.06, 0.05)
    glass = ir.opaque_ir('SiO2glass_ir')                 # glass is opaque in the LWIR
    base = T().translate([x, 0, z]).rotate([0, 1, 0], yaw)
    skin = th.MATERIALS['steel_sheet_painted']           # 2 mm steel, cabin air behind
    add(f'{prefix}_body', {'type': 'cube', 'to_world': base @ T().translate([0, 0.62, 0]).scale([2.15, 0.32, 0.88]),
                           'bsdf': paint}, skin)
    add(f'{prefix}_cabin', {'type': 'cube', 'to_world': base @ T().translate([-0.3, 1.20, 0]).scale([1.15, 0.26, 0.80]),
                            'bsdf': glass}, th.Material(k=1.0, rho=2500.0, c=800.0, thickness=0.004, back='air',
                                                        n_nodes=2, name='glazing 4 mm'))
    add(f'{prefix}_roof', {'type': 'cube', 'to_world': base @ T().translate([-0.3, 1.475, 0]).scale([1.05, 0.015, 0.74]),
                           'bsdf': paint}, skin)
    hood_mat = skin.replace(back='interior', T_internal=th.celsius(75), U_internal=10.0) if warm_engine else skin
    add(f'{prefix}_hood', {'type': 'cube', 'to_world': base @ T().translate([1.45, 0.955, 0]).scale([0.68, 0.015, 0.84]),
                           'bsdf': paint}, hood_mat)
    tyre = th.Material(k=0.2, rho=1150.0, c=1900.0, thickness=0.015, back='adiabatic', n_nodes=4, name='tyre rubber')
    for i, (wx, wz) in enumerate([(1.35, 0.92), (1.35, -0.92), (-1.35, 0.92), (-1.35, -0.92)]):
        p0 = base @ mi.ScalarPoint3f(wx, 0.33, wz - 0.11)
        p1 = base @ mi.ScalarPoint3f(wx, 0.33, wz + 0.11)
        add(f'{prefix}_wheel{i}', {'type': 'cylinder', 'p0': list(p0), 'p1': list(p1), 'radius': 0.33, 'bsdf': rubber}, tyre, 12)
        add(f'{prefix}_wheel{i}_a', {'type': 'disk', 'bsdf': rubber,
                                     'to_world': base @ T().translate([wx, 0.33, wz + 0.11]).scale(0.33)}, tyre, 8)
        add(f'{prefix}_wheel{i}_b', {'type': 'disk', 'bsdf': rubber,
                                     'to_world': base @ T().translate([wx, 0.33, wz - 0.11]).rotate([0, 1, 0], 180).scale(0.33)}, tyre, 8)


add_car('car_a', -3.0, 8.5, 90.0, rho_solar=0.10, warm_engine=False)   # dark car, parked all day
add_car('car_b', 2.6, 9.3, 84.0, rho_solar=0.55, warm_engine=True)     # light car with a warm engine

# --- trees: trunk + a crown of ~30 leaf clusters ------------------------------------
# The crowns carry no emitter: like the grass, their leaves take the shading-time
# temperature of the ambient leaf model at every hit point, so the sunlit side of
# a crown is warm and its interior and shaded side are cool. They still cast
# shadows in the thermal solver (geometry without a material).
leaf_bsdf = {'type': 'twosided', 'bsdf': ir.two_band_diffuse(0.30, 0.03)}
TREES = [(-11.0, 11.0), (14.0, 7.0), (-20.0, -6.0), (9.0, -9.0), (-9.0, -23.0), (28.0, 3.0),
         (-30.0, 10.0), (17.0, -30.0), (33.0, -28.0), (-34.0, -27.0), (-24.0, 27.0), (22.0, 30.0)]
for i, (x, z) in enumerate(TREES):
    h = rng.uniform(2.4, 3.6)                      # height of the crown base
    r = rng.uniform(2.2, 3.4)                      # crown radius
    add(f'tree{i}_trunk', {'type': 'cylinder', 'p0': [x, 0, z], 'p1': [x, h + 0.6 * r, z], 'radius': rng.uniform(0.18, 0.30),
                           'bsdf': ir.two_band_diffuse(0.40, 0.10)}, th.MATERIALS['wood'].replace(back='adiabatic'), 12)
    centre = np.array([x, h + r, z])
    for j in range(30):
        # clusters on an ellipsoidal shell (slightly flattened crown), sizes varying
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        c = centre + v * r * rng.uniform(0.45, 0.95) * np.array([1.0, 0.8, 1.0])
        geometry[f'tree{i}_leaves{j}'] = {'type': 'sphere', 'center': [float(c[0]), float(c[1]), float(c[2])],
                                          'radius': float(r * rng.uniform(0.20, 0.36)), 'bsdf': leaf_bsdf}

# --- rocks by the pond --------------------------------------------------------------
ROCKS = [(14.2, -15.5, 0.7), (15.6, -8.2, 0.5), (29.8, -9.0, 0.9), (26.5, -19.6, 0.6), (13.0, -11.0, 0.35)]
for i, (x, z, r) in enumerate(ROCKS):
    add(f'rock{i}', {'type': 'sphere', 'center': [x, 0.25 * r, z], 'radius': r,
                     'bsdf': ir.two_band_diffuse(0.30, 0.07)}, th.MATERIALS['concrete'], 12)

# =============================================================================
# Instanced grass: three tuft meshes, ~1e5 instances, denser near the cars
# =============================================================================
def write_tuft(path, seed, n_blades=60, radius=0.24):
    r_ = np.random.default_rng(seed)
    V, F = [], []
    for _ in range(n_blades):
        rr, a = radius * np.sqrt(r_.random()), r_.random() * 2 * np.pi
        base = np.array([rr * np.cos(a), 0.0, rr * np.sin(a)])
        h, w, d = r_.uniform(0.18, 0.42), r_.uniform(0.015, 0.026), r_.random() * 2 * np.pi
        lean = r_.uniform(0.08, 0.45) * h
        side = np.array([np.cos(d + np.pi / 2), 0.0, np.sin(d + np.pi / 2)]) * w / 2
        fwd = np.array([np.cos(d), 0.0, np.sin(d)])
        i = len(V)
        V += [base - side, base + side,
              base - 0.6 * side + fwd * lean * 0.3 + [0, 0.55 * h, 0], base + 0.6 * side + fwd * lean * 0.3 + [0, 0.55 * h, 0],
              base + fwd * lean + [0, h, 0]]
        F += [[i, i + 1, i + 2], [i + 1, i + 3, i + 2], [i + 2, i + 3, i + 4]]
    with open(path, 'w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\n'
                f'property float z\nelement face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n')
        f.writelines('%.5f %.5f %.5f\n' % tuple(v) for v in V)
        f.writelines('3 %d %d %d\n' % tuple(t) for t in F)


def on_bare_ground(x, z):
    """True where no grass grows (road, pad, path, slab, pond, trunks, rocks)."""
    bare = (np.abs(z - 17.0) < 3.4)                                   # road and verge
    bare |= (np.abs(x) < 7.3) & (np.abs(z - 9.0) < 5.3)               # gravel pad
    bare |= (np.abs(x) < 1.2) & (z > -14.2) & (z < 4.2)               # dirt path
    bare |= (np.abs(x) < 5.2) & (np.abs(z + 19.0) < 5.2)              # slab
    bare |= (x - POND[0]) ** 2 + (z - POND[1]) ** 2 < (POND[2] + 0.4) ** 2
    for tx, tz in TREES:
        bare |= (x - tx) ** 2 + (z - tz) ** 2 < 0.45 ** 2
    for rx, rz, rr in ROCKS:
        bare |= (x - rx) ** 2 + (z - rz) ** 2 < rr ** 2
    return bare


grass = {}
n_tufts = 0
if args.grass_density > 0:
    grass_bsdf = {'type': 'twosided', 'bsdf': ir.two_band_diffuse(0.30, 0.03)}
    for k in range(3):
        path = os.path.join(asset_dir, f'tuft{k}.ply')
        write_tuft(path, 100 + k)
        grass[f'tuft{k}'] = {'type': 'shapegroup', 'blades': {'type': 'ply', 'filename': path, 'bsdf': grass_bsdf}}
    # One uniform layer (about 18 tufts per m^2) over an 80 m disc: a density that
    # varied with distance showed up as artificial rings in the aerial views
    layers = [(int(360000 * args.grass_density), 80.0, (0.0, 0.0), (0.9, 1.5))]
    for n, radius, (cx, cz), (s0, s1) in layers:
        rr = radius * np.sqrt(rng.random(n))
        aa = rng.random(n) * 2 * np.pi
        x, z = cx + rr * np.cos(aa), cz + rr * np.sin(aa)
        keep = ~on_bare_ground(x, z)
        for xi, zi, ang, sc, kind in zip(x[keep], z[keep], rng.random(n)[keep] * 360.0,
                                         rng.uniform(s0, s1, n)[keep], rng.integers(0, 3, n)[keep]):
            grass[f'g{n_tufts}'] = {'type': 'instance', 'shapegroup': {'type': 'ref', 'id': f'tuft{kind}'},
                                    'to_world': T().translate([float(xi), 0.0, float(zi)]).rotate([0, 1, 0], float(ang)).scale(float(sc))}
            n_tufts += 1
print(f'scene: {len(geometry)} thermal shapes, {n_tufts} grass tufts ({n_tufts * 60} blades)')

# =============================================================================
# Thermal model (one solve for the day; the grass is not part of the per-element
# solver: it enters at render time through the shadow rays)
# =============================================================================
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)       # table alpha/eps differ from the BSDF-derived ones
    sim = th.ThermalScene({'type': 'scene', **geometry}, weather, materials, resolution=resolution,
                          up=UP, north=NORTH, sky_rays=64, exchange=args.exchange, reference_normal='up')
    # Ambient leaf model for all emitter-less vegetation (grass blades, crowns):
    # a thin leaf layer, alpha_s / eps of the leaf BSDF (two-band 0.30 / 0.03)
    leaf = th.DynamicModel(weather, th.MATERIALS['vegetation_canopy'].replace(alpha_solar=0.70, emissivity=0.97),
                           up=UP, north=NORTH)
print(f'thermal scene: {sim.n_elements} elements, {len(set(id(m) for m in sim.dynamic_models().values()))} '
      f'response models, set up in {time.time() - t0:.1f} s')

CAMERAS = {
    'ground': dict(origin=[15.0, 1.7, 2.0], target=[-2.0, 0.6, 10.5], fov=60.0, up=[0, 1, 0]),
    'aerial': dict(origin=[46.0, 38.0, 52.0], target=[1.0, 0.0, -3.0], fov=40.0, up=[0, 1, 0]),
    'top': dict(origin=[4.0, 110.0, -2.0], target=[4.0, 0.0, -2.0], fov=38.0, up=[0, 0, -1]),
}
cam = CAMERAS.get(args.view) or dict(origin=args.origin, target=args.target, fov=50.0, up=[0, 1, 0])
camera = {'type': 'perspective', 'fov': args.fov or cam['fov'],
          'to_world': T().look_at(origin=cam['origin'], target=cam['target'], up=cam['up']),
          'sampler': {'type': 'independent', 'sample_count': args.spp},
          'film': ir.band_film(BAND, args.width, args.height)}

# =============================================================================
# Render
# =============================================================================
for f in os.listdir(frame_dir):
    if f.startswith('frame_') and f.endswith('.png'):
        os.remove(os.path.join(frame_dir, f))
mesh_dir = os.path.join(asset_dir, 'meshes')

frames = []
t_all = time.time()
for i, t in enumerate(times):
    tf = time.time()
    t = float(t)
    scene = sim.dynamic({'type': 'scene', **geometry}, t, mesh_dir=mesh_dir)
    scene.update(grass)
    T_air = weather.air_temperature(t)
    T_leaf = float(leaf.T_ref[step_index(t, leaf.dt, leaf.n_steps)])       # exposed leaf, for the log
    scene['integrator'] = {'type': 'path_ir', 'max_depth': 5, 'thermal_up': list(UP), 'thermal_sky_rays': 6,
                           'ambient': leaf.emitter_dict(t)}
    scene['sensor'] = camera
    scene['sky'] = ir.thermalsky(T_air, relative_humidity=weather.relative_humidity, band='8-14', up=UP)
    img = np.array(mi.render(mi.load_dict(scene, optimize=False), spp=args.spp))[..., 0]
    T_app = ir.apparent_temperature(img, lo, hi) - 273.15
    frames.append((t, T_air - 273.15, T_app.astype(np.float32)))
    print(f'  frame {i + 1:3d}/{len(times)}  {fmt_time(t)}  air {T_air - 273.15:5.1f} C  grass {T_leaf - 273.15:5.1f} C  '
          f'scene {np.percentile(T_app, 2):5.1f}..{np.percentile(T_app, 98):5.1f} C  ({time.time() - tf:.1f} s)')
print(f'rendered {len(frames)} frames in {time.time() - t_all:.1f} s')

# =============================================================================
# Camera look: blur, NETD noise, fixed gain, palette, on-screen display
# =============================================================================
from PIL import Image, ImageDraw


def gaussian_blur(a, sigma):
    if sigma <= 0:
        return a
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode='edge'), k, mode='valid'), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode='edge'), k, mode='valid'), 1, a)


noise_rng = np.random.default_rng(1234)
processed = []
for t, T_air, T_app in frames:
    img = gaussian_blur(T_app.astype(np.float64), args.blur)
    if args.netd > 0:
        img = img + noise_rng.normal(0.0, args.netd, img.shape)
    processed.append(img)
all_T = np.concatenate([p.ravel() for p in processed])
# Robust display span (as a camera's automatic gain): small extremes such as a
# metal roof mirroring the cold zenith sky saturate instead of compressing the scene
vmin = args.vmin if args.vmin is not None else float(np.percentile(all_T, 2.0))
vmax = args.vmax if args.vmax is not None else float(np.percentile(all_T, 99.5))

if args.palette == 'inferno':
    import matplotlib
    lut = (matplotlib.colormaps['inferno'](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
else:
    g = np.arange(256, dtype=np.uint8) if args.palette == 'whitehot' else np.arange(255, -1, -1, dtype=np.uint8)
    lut = np.stack([g, g, g], 1)

for i, ((t, T_air, _), img) in enumerate(zip(frames, processed)):
    idx = np.clip((img - vmin) / max(vmax - vmin, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(lut[idx])
    draw = ImageDraw.Draw(im)
    osd = f'{args.date} {fmt_time(t)}   LWIR 8-14 um   air {T_air:4.1f} C   span {vmin:.0f}..{vmax:.0f} C'
    draw.rectangle([6, 6, 12 + 6.2 * len(osd), 22], fill=(0, 0, 0))
    draw.text((10, 8), osd, fill=(255, 255, 255))
    im.save(os.path.join(frame_dir, f'frame_{i:03d}.png'))

pattern = os.path.join(frame_dir, 'frame_%03d.png')
if len(frames) > 1 and shutil.which('ffmpeg'):
    video = os.path.join(out_root, f'14_landscape_{args.view}.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps), '-i', pattern,
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', video],
                   check=True)
    print(f'wrote {video}')
elif len(frames) > 1:
    video = os.path.join(out_root, f'14_landscape_{args.view}.gif')
    imgs = [Image.open(os.path.join(frame_dir, f'frame_{i:03d}.png')).convert('P', palette=Image.ADAPTIVE)
            for i in range(len(frames))]
    imgs[0].save(video, save_all=True, append_images=imgs[1:], duration=int(1000 / args.fps), loop=0)
    print(f'wrote {video}')
print(f'display span {vmin:.1f}..{vmax:.1f} C; wrote {len(frames)} frame(s) to {frame_dir}')
