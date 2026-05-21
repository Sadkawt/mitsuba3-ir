#!/usr/bin/env python3
"""Validate the per-shape radiance field against stock Mitsuba area emitters.

A "pretty formation" of many small spheres is arranged on a Fibonacci sphere and
coloured as a rainbow. We build two scenes with identical geometry:

  * ``emitter``  : each sphere carries an ``area`` emitter   (stock Mitsuba; NEE)
  * ``radiance`` : each sphere carries the new ``radiance`` field + black BSDF
                   (no emitter registered, no NEE)

Both estimators are unbiased and must converge to the same image. We report the
scene build time, render time and the difference between the two images.

Run with e.g.:
  PYTHONPATH=build/python ./.venv-build/bin/python radiance_field_demo/many_lights.py cuda_ad_rgb 4000
"""

import sys, time, math
import numpy as np

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "llvm_ad_rgb"
N       = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
SPP     = int(sys.argv[3]) if len(sys.argv) > 3 else 256
WITH_FLOOR = "--floor" in sys.argv

import mitsuba as mi
mi.set_variant(VARIANT)

RES        = 512
FORM_R     = 3.0      # radius of the formation
LIGHT_R    = 0.05     # radius of each little sphere
INTENSITY  = 8.0      # peak radiance
MAX_DEPTH  = 6 if WITH_FLOOR else 2


def hsv_to_rgb(h, s, v):
    i = int(h * 6.0) % 6
    f = h * 6.0 - math.floor(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i]


def formation():
    """Return list of (position, rgb_radiance) for N lights on a Fibonacci sphere."""
    pts = []
    ga = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    for k in range(N):
        y = 1.0 - 2.0 * (k + 0.5) / N
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = ga * k
        p = [FORM_R * r * math.cos(theta), FORM_R * y, FORM_R * r * math.sin(theta)]
        hue = (k / N + 0.5 * (y * 0.5 + 0.5)) % 1.0
        rgb = [c * INTENSITY for c in hsv_to_rgb(hue, 0.85, 1.0)]
        pts.append((p, rgb))
    return pts


def base_scene():
    d = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": MAX_DEPTH},
        "sensor": {
            "type": "perspective", "fov": 45,
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[0, 0, 9], target=[0, 0, 0], up=[0, 1, 0]),
            "film": {"type": "hdrfilm", "width": RES, "height": RES,
                     "rfilter": {"type": "gaussian"}, "pixel_format": "rgb"},
            "sampler": {"type": "independent", "sample_count": SPP},
        },
    }
    if WITH_FLOOR:
        d["floor"] = {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f().translate([0, -FORM_R - 1.5, 0])
                         .rotate([1, 0, 0], -90).scale(12),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.6}},
        }
    return d


def build(mode, lights):
    d = base_scene()
    for i, (p, rgb) in enumerate(lights):
        to_world = mi.ScalarTransform4f().translate(p).scale(LIGHT_R)
        if mode == "emitter":
            d[f"l{i}"] = {
                "type": "sphere", "to_world": to_world,
                "emitter": {"type": "area",
                            "radiance": {"type": "rgb", "value": rgb}},
            }
        else:  # radiance field, explicit black BSDF to match the emitter case
            d[f"l{i}"] = {
                "type": "sphere", "to_world": to_world,
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.0}},
                "radiance": {"type": "rgb", "value": rgb},
            }
    return d


def render(mode, lights, seed=0):
    t0 = time.time()
    scene = mi.load_dict(build(mode, lights))
    t1 = time.time()
    img = mi.render(scene, spp=SPP, seed=seed)
    mi.util.write_bitmap(f"radiance_field_demo/out_{mode}_{VARIANT}.exr", img)
    arr = np.array(img, dtype=np.float64)
    t2 = time.time()
    n_emitters = len(scene.emitters())
    print(f"  [{mode:8s}] build={t1 - t0:6.2f}s  render={t2 - t1:6.2f}s  "
          f"scene_emitters={n_emitters}")
    return arr


def tonemap(a):
    return (np.clip(a, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def main():
    print(f"variant={VARIANT}  lights={N}  spp={SPP}  max_depth={MAX_DEPTH}  "
          f"floor={WITH_FLOOR}")
    lights = formation()

    ref = render("emitter", lights, seed=0)     # stock Mitsuba
    rf  = render("radiance", lights, seed=0)     # new feature

    diff = np.abs(ref - rf)
    mse  = float(np.mean((ref - rf) ** 2))
    mae  = float(np.mean(diff))
    denom = float(np.mean(ref)) + 1e-8
    print(f"\n  MSE={mse:.3e}   MAE={mae:.3e}   "
          f"relMAE={mae / denom:.3e}   ref_mean={np.mean(ref):.4f}")

    # Side-by-side composite: reference | radiance | 10x abs-diff
    h, w, _ = ref.shape
    gap = np.zeros((h, 8, 3), np.uint8)
    comp = np.concatenate(
        [tonemap(ref), gap, tonemap(rf), gap, tonemap(diff * 10.0)], axis=1)
    mi.Bitmap(comp).write(f"radiance_field_demo/compare_{VARIANT}.png")
    print(f"  wrote radiance_field_demo/compare_{VARIANT}.png "
          f"(reference | radiance-field | 10x diff)")


if __name__ == "__main__":
    main()
