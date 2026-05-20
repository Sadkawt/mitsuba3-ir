#!/usr/bin/env python3
"""Validate the `thermal` grey-body spectrum + its use as a shape radiance field.

Spectral variants only. Three checks:
  1. Equivalence: thermal(T, eps=1) radiance field == stock `blackbody` radiance
     field (same Planck law), and eps scales it linearly.
  2. Wien shift: rendered chromaticity of a uniform grey body marches red->white
     ->blue as temperature increases.
  3. Temperature texture (TAITherm-style): a per-texel Kelvin map drives spatially
     varying emission.

  PYTHONPATH=build/python ./.venv-build/bin/python radiance_field_demo/thermal_demo.py cuda_ad_spectral
"""
import sys, os
import numpy as np

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "cuda_ad_spectral"
RES, SPP = 256, 64
HERE = os.path.dirname(__file__)

import mitsuba as mi
mi.set_variant(VARIANT)


def single_sphere(radiance_dict):
    return mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 2},
        "sensor": {
            "type": "perspective", "fov": 35,
            "to_world": mi.ScalarTransform4f().look_at([0, 0, 4], [0, 0, 0], [0, 1, 0]),
            "film": {"type": "hdrfilm", "width": RES, "height": RES,
                     "rfilter": {"type": "box"}, "pixel_format": "rgb"},
            "sampler": {"type": "independent", "sample_count": SPP},
        },
        "s": {"type": "sphere", "radius": 1.0,
              "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.0}},
              "radiance": radiance_dict},
    })


def disc_mean(img):
    """Mean RGB over the bright (sphere) pixels."""
    a = np.array(img, dtype=np.float64)
    lum = a.sum(2)
    mask = lum > 1e-12
    return a[mask].mean(0) if mask.any() else np.zeros(3)


def check_equivalence():
    print("\n[1] Equivalence thermal(T,eps=1) vs blackbody(T), and eps scaling")
    for T in (800.0, 2400.0, 5500.0):
        bb = disc_mean(mi.render(single_sphere({"type": "blackbody", "temperature": T})))
        th = disc_mean(mi.render(single_sphere(
            {"type": "thermal", "temperature": T, "emissivity": 1.0})))
        th_half = disc_mean(mi.render(single_sphere(
            {"type": "thermal", "temperature": T, "emissivity": 0.5})))
        rel = np.linalg.norm(th - bb) / (np.linalg.norm(bb) + 1e-20)
        ratio = (th_half / np.maximum(th, 1e-30)).mean()
        print(f"  T={T:6.0f}K  rel|thermal-blackbody|={rel:.2e}  "
              f"eps=0.5 ratio={ratio:.4f} (expect 0.5)")


def check_wien():
    print("\n[2] Wien shift -- normalized chromaticity vs temperature")
    for T in (1000, 2000, 3500, 5500, 8000):
        rgb = disc_mean(mi.render(single_sphere({"type": "thermal", "temperature": float(T)})))
        chroma = rgb / (rgb.sum() + 1e-20)
        print(f"  T={T:5d}K  chroma R={chroma[0]:.3f} G={chroma[1]:.3f} B={chroma[2]:.3f}")


def make_temperature_exr(path, n=256):
    """A synthetic 'heat map': cool base with a hot spot, in Kelvin."""
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    base = 500.0 + 700.0 * xx                      # gradient 500K -> 1200K
    hot = 2200.0 * np.exp(-(((xx - 0.65) ** 2 + (yy - 0.4) ** 2) / 0.02))
    T = (base + hot).astype(np.float32)
    mi.Bitmap(T[..., None]).write(path)            # single-channel float EXR (Kelvin)
    return T


def check_texture():
    print("\n[3] Temperature texture (TAITherm-style heat map -> emission)")
    exr = os.path.join(HERE, "heat_K.exr")
    T = make_temperature_exr(exr)
    print(f"  wrote {exr}  (Kelvin range {T.min():.0f}..{T.max():.0f})")
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 2},
        "sensor": {
            "type": "perspective", "fov": 35,
            "to_world": mi.ScalarTransform4f().look_at([0, 0, 4.5], [0, 0, 0], [0, 1, 0]),
            "film": {"type": "hdrfilm", "width": RES, "height": RES,
                     "rfilter": {"type": "gaussian"}, "pixel_format": "rgb"},
            "sampler": {"type": "independent", "sample_count": SPP},
        },
        "panel": {
            "type": "rectangle", "to_world": mi.ScalarTransform4f().scale(1.5),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.0}},
            "radiance": {
                "type": "thermal",
                "temperature": {"type": "bitmap", "filename": exr, "raw": True,
                                "filter_type": "nearest"},
                "emissivity": 0.9,
            },
        },
    })
    img = mi.render(scene)
    a = np.array(img, dtype=np.float64)
    # exposure-normalize then gamma for viewing
    disp = (np.clip(a / (a.max() + 1e-12), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    out = os.path.join(HERE, f"thermal_texture_{VARIANT}.png")
    mi.Bitmap(disp).write(out)
    print(f"  rendered emission, max radiance={a.max():.3e}  ->  {out}")


def main():
    print(f"variant={VARIANT}  res={RES} spp={SPP}  spectral={mi.is_spectral}")
    check_equivalence()
    check_wien()
    check_texture()


if __name__ == "__main__":
    main()
