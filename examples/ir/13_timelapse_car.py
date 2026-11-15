#!/usr/bin/env python3
"""
Example 13 - Thermal time lapse: a car parked on grass next to a tree.

The surface temperatures come from the shading-time thermal model
(``thermal_dynamic`` emitters, report section 6.8.3): the per-element solver
runs once for the day (radiative exchange included), and each video frame is
rendered at its own local time. The integrator then evaluates every visible
surface point's temperature from its own shadow history. The tree's shadow
sweeps across the grass and the car with sharp edges, and the warm/cool
"memory" it leaves behind (thermal lag) is visible frame by frame.

The car is built from primitives (body, cabin, four wheels): a dark painted
2 mm steel skin over cabin air at ambient temperature, with rubber tyres. The
grass is modelled without evapotranspiration (report A18), so it runs a few
kelvin warmer in the afternoon than real transpiring grass.

Usage
-----
    python 13_timelapse_car.py --start 05:00 --stop 22:00 --step 20
    python 13_timelapse_car.py --start 18:00 --stop 06:00 --step 30 --date 2026-04-10

Times are local (HH:MM or decimal hours). A stop time earlier than the start
spans midnight; the diurnal forcing is periodic, so this is the night
between two identical days. Run ``--help`` for weather, camera and quality
options.

Output: examples/ir/output/13_timelapse/frame_NNN.png and
examples/ir/output/13_timelapse.mp4 (or .gif when ffmpeg is not available).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np


def parse_time(text):
    """'HH:MM' or decimal hours -> hours (float)."""
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
ap.add_argument('--start', default='05:00', help='start time (local, HH:MM or hours)')
ap.add_argument('--stop', default='22:00', help='stop time (local, inclusive; earlier than start = across midnight)')
ap.add_argument('--step', type=float, default=30.0, help='time delta between frames [minutes]')
ap.add_argument('--date', default='2026-07-15', help='date (YYYY-MM-DD)')
ap.add_argument('--lat', type=float, default=59.33, help='latitude [deg, north positive]')
ap.add_argument('--lon', type=float, default=18.07, help='longitude [deg, east positive]')
ap.add_argument('--utc-offset', type=float, default=2.0, help='local time minus UTC [h]')
ap.add_argument('--tmin', type=float, default=14.0, help='daily minimum air temperature [C]')
ap.add_argument('--tmax', type=float, default=26.0, help='daily maximum air temperature [C]')
ap.add_argument('--humidity', type=float, default=0.55, help='relative humidity (0-1)')
ap.add_argument('--wind', type=float, default=2.0, help='wind speed [m/s]')
ap.add_argument('--clouds', type=float, default=0.1, help='cloud cover (0-1)')
ap.add_argument('--band', default='lwir', choices=['lwir', 'lwir_bolometer', 'mwir'], help='spectral band')
ap.add_argument('--width', type=int, default=640)
ap.add_argument('--height', type=int, default=400)
ap.add_argument('--spp', type=int, default=32, help='samples per pixel')
ap.add_argument('--fps', type=float, default=6.0, help='video frame rate')
ap.add_argument('--vmin', type=float, default=None, help='colour scale minimum [C] (default: from all frames)')
ap.add_argument('--vmax', type=float, default=None, help='colour scale maximum [C] (default: from all frames)')
ap.add_argument('--no-exchange', action='store_true', help='disable radiative exchange between surfaces')
ap.add_argument('--camera-target', type=float, nargs=3, default=None, help=argparse.SUPPRESS)
ap.add_argument('--fov', type=float, default=48.0, help='camera field of view [deg]')
ap.add_argument('--variant', default='llvm_ad_spectral', choices=['llvm_ad_spectral', 'cuda_ad_spectral'])
args = ap.parse_args()

t_start, t_stop = parse_time(args.start), parse_time(args.stop)
if t_stop < t_start:
    t_stop += 24.0
if args.step <= 0:
    ap.error('--step must be positive')
times = np.arange(t_start, t_stop + 1e-9, args.step / 60.0)
if len(times) == 0:
    ap.error('no frames in the requested interval')

import mitsuba as mi
mi.set_variant(args.variant)
from mitsuba import ir
from mitsuba.ir import thermal as th
from _util import output_dir

UP, NORTH = (0, 1, 0), (0, 0, -1)           # Y up, north = -Z, east = +X
T = mi.ScalarTransform4f
lo, hi = ir.BANDS[args.band]

weather = th.Weather(args.date, latitude=args.lat, longitude=args.lon, utc_offset=args.utc_offset,
                     T_min=th.celsius(args.tmin), T_max=th.celsius(args.tmax),
                     relative_humidity=args.humidity, wind_speed=args.wind,
                     cloud_cover=args.clouds, aod500=0.15, ground_albedo=0.2)
print(weather.summary())

# --- geometry (metres). The car is parked with its long axis east-west; the
# tree stands to its south-west, so the afternoon sun casts the canopy's
# shadow across the car. -----------------------------------------------------
paint = {'type': 'roughplastic', 'int_ior': 1.5, 'alpha': 0.25,
         'diffuse_reflectance': ir.two_band_diffuse(0.12, 0.04)['reflectance']}    # dark blue-grey paint
rubber = ir.two_band_diffuse(0.06, 0.05)
geometry = {
    'grass': {'type': 'rectangle', 'to_world': T().rotate([1, 0, 0], -90).scale(15),
              'bsdf': ir.two_band_diffuse(0.25, 0.03)},
    'car_body': {'type': 'cube', 'to_world': T().translate([0, 0.65, 0]).scale([2.2, 0.35, 0.9]),
                 'bsdf': paint},
    'car_cabin': {'type': 'cube', 'to_world': T().translate([-0.25, 1.28, 0]).scale([1.25, 0.28, 0.8]),
                  'bsdf': paint},
    'trunk': {'type': 'cylinder', 'p0': [-4.0, 0, 4.0], 'p1': [-4.0, 3.2, 4.0], 'radius': 0.25,
              'bsdf': ir.two_band_diffuse(0.40, 0.10)},
    'canopy': {'type': 'sphere', 'center': [-4.0, 5.0, 4.0], 'radius': 2.4,
               'bsdf': ir.two_band_diffuse(0.30, 0.03)},
}
# Wheels: Mitsuba's cylinder is open-ended, so each tyre is closed with two disks
wheels = {}
for i, (x, z) in enumerate([(1.4, 0.95), (1.4, -0.95), (-1.4, 0.95), (-1.4, -0.95)]):
    wheels[f'wheel_{i}'] = {'type': 'cylinder', 'p0': [x, 0.33, z - 0.12], 'p1': [x, 0.33, z + 0.12],
                            'radius': 0.33, 'bsdf': rubber}
    # disk normal is +z: one cap facing +z at z + 0.12, one facing -z at z - 0.12
    wheels[f'wheel_{i}_cap_p'] = {'type': 'disk', 'bsdf': rubber,
                                  'to_world': T().translate([x, 0.33, z + 0.12]).scale(0.33)}
    wheels[f'wheel_{i}_cap_m'] = {'type': 'disk', 'bsdf': rubber,
                                  'to_world': T().translate([x, 0.33, z - 0.12]).rotate([0, 1, 0], 180).scale(0.33)}
geometry.update(wheels)

rubber_mat = th.Material(k=0.2, rho=1150.0, c=1900.0, thickness=0.015, back='adiabatic',
                         n_nodes=4, name='tyre rubber 15 mm')
materials = {
    'grass': th.MATERIALS['grass'],
    # 2 mm steel skin; the back faces the cabin air, taken at ambient temperature
    'car_body': th.MATERIALS['steel_sheet_painted'],
    'car_cabin': th.MATERIALS['steel_sheet_painted'],
    'trunk': th.MATERIALS['wood'].replace(back='adiabatic'),
    'canopy': th.MATERIALS['vegetation_canopy'],
    **{k: rubber_mat for k in wheels},
}
resolution = {'grass': 128, 'trunk': 24, 'canopy': 48, **{k: 12 for k in wheels}}  # cubes: per vertex

# --- thermal model: one solve for the day ------------------------------------
t0 = time.time()
with warnings.catch_warnings():
    # alpha_s / eps come from the BSDFs (report A22); the table values of the
    # generic materials differ, which ThermalScene would report for each shape
    warnings.simplefilter('ignore', UserWarning)
    sim = th.ThermalScene({'type': 'scene', **geometry}, weather, materials, resolution=resolution,
                          up=UP, north=NORTH, sky_rays=96, exchange=not args.no_exchange)
print(f'thermal scene: {sim.n_elements} elements, view factors in {time.time() - t0:.1f} s')
for key in ('grass', 'car_body', 'canopy', 'wheel_0'):
    m = sim.shapes[key].material
    print(f'  {key:9s} alpha_s = {m.alpha_solar:.2f}, eps = {m.emissivity:.2f} (from the BSDF)')

out_root = output_dir()
frame_dir = os.path.join(out_root, '13_timelapse')
mesh_dir = os.path.join(frame_dir, 'meshes')
os.makedirs(frame_dir, exist_ok=True)
for f in os.listdir(frame_dir):
    if f.startswith('frame_') and f.endswith('.png'):
        os.remove(os.path.join(frame_dir, f))

camera = {'type': 'perspective', 'fov': args.fov,
          'to_world': T().look_at(origin=[7.0, 4.0, 10.0], target=args.camera_target or [-1.2, 0.9, 0.8], up=[0, 1, 0]),
          'sampler': {'type': 'independent', 'sample_count': args.spp},
          'film': ir.band_film(args.band, args.width, args.height)}

# --- render all frames ---------------------------------------------------------
frames = []
t_render = time.time()
for i, t in enumerate(times):
    tf = time.time()
    scene = sim.dynamic({'type': 'scene', **geometry}, float(t), mesh_dir=mesh_dir)
    scene['integrator'] = {'type': 'path_ir', 'max_depth': 6, 'thermal_up': list(UP), 'thermal_sky_rays': 8}
    scene['sensor'] = camera
    T_air = weather.air_temperature(float(t))
    scene['sky'] = ir.thermalsky(T_air, relative_humidity=weather.relative_humidity, band='8-14', up=UP)
    img = np.array(mi.render(mi.load_dict(scene), spp=args.spp))[..., 0]
    T_app = ir.apparent_temperature(img, lo, hi) - 273.15
    el = weather.sun(float(t) % 24.0)['elevation']
    frames.append((float(t), T_air - 273.15, el, T_app))
    print(f'  frame {i + 1:3d}/{len(times)}  {fmt_time(t)}  air {T_air - 273.15:5.1f} C  '
          f'sun {el:5.1f} deg  scene {np.percentile(T_app, 2):5.1f}..{np.percentile(T_app, 98):5.1f} C  '
          f'({time.time() - tf:.1f} s)')
print(f'rendered {len(frames)} frames in {time.time() - t_render:.1f} s')

# --- fixed colour scale over the whole sequence, then write annotated frames ----
all_T = np.concatenate([f[3].ravel() for f in frames])
vmin = args.vmin if args.vmin is not None else float(np.floor(np.percentile(all_T, 1.0)))
vmax = args.vmax if args.vmax is not None else float(np.ceil(np.percentile(all_T, 99.5)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for i, (t, T_air, el, T_app) in enumerate(frames):
    fig, ax = plt.subplots(figsize=(args.width / 100 + 1.4, args.height / 100 + 0.6), dpi=100)
    im = ax.imshow(T_app, cmap='inferno', vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    sun = f'sun {el:4.1f}°' if el > 0 else 'sun below horizon'
    ax.set_title(f'{args.date}  {fmt_time(t)}   air {T_air:4.1f} °C   {sun}', fontsize=11, loc='left')
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(f'apparent temperature [°C], {args.band.upper()}')
    fig.tight_layout()
    fig.savefig(os.path.join(frame_dir, f'frame_{i:03d}.png'))
    plt.close(fig)

# --- assemble the video ---------------------------------------------------------
pattern = os.path.join(frame_dir, 'frame_%03d.png')
if shutil.which('ffmpeg'):
    video = os.path.join(out_root, '13_timelapse.mp4')
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps), '-i', pattern,
           '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video]
    subprocess.run(cmd, check=True)
else:
    from PIL import Image
    video = os.path.join(out_root, '13_timelapse.gif')
    imgs = [Image.open(os.path.join(frame_dir, f'frame_{i:03d}.png')).convert('P', palette=Image.ADAPTIVE)
            for i in range(len(frames))]
    imgs[0].save(video, save_all=True, append_images=imgs[1:], duration=int(1000 / args.fps), loop=0)
print(f'colour scale {vmin:.0f}..{vmax:.0f} C; wrote {len(frames)} frames to {frame_dir}')
print(f'wrote {video}')
