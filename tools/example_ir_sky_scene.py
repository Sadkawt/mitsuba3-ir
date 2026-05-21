#!/usr/bin/env python3
"""
Minimal example: render a scene lit by the `spectralsky` IR sky emitter.

A diffuse sphere on a diffuse ground plane, illuminated purely by the
libRadtran downwelling-sky radiance table. Renders three LWIR bands and saves
a brightness-temperature false-colour image.

Run:
    PYTHONPATH=build/python .venv/bin/python tools/example_ir_sky_scene.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from skyrad_io import SkyRadiance

import mitsuba as mi
mi.set_variant("cuda_ad_spectral")

SKYRAD = "tools/data/ir_sky/lwir_us_standard.skyrad"
OUT = "renders/ir_sky_example.png"

# Three LWIR display bands (nm) and their half-widths.
BANDS = [(9000, 250), (10000, 250), (11000, 250)]

H_PLANCK, C_LIGHT, K_BOLTZ = 6.62607015e-34, 2.99792458e8, 1.380649e-23


def brightness_temp(L_per_nm, lam_nm):
    """Invert Planck to a brightness temperature [K]."""
    lam = lam_nm * 1e-9
    L = np.maximum(L_per_nm * 1e9, 1e-30)   # per-nm -> per-m
    return (H_PLANCK * C_LIGHT / (lam * K_BOLTZ)) / np.log1p(
        2 * H_PLANCK * C_LIGHT**2 / (lam**5 * L))


def main() -> int:
    sky = SkyRadiance.read(SKYRAD)

    bands = {f"b{c}": {"type": "regular",
                       "wavelength_min": float(c - hw),
                       "wavelength_max": float(c + hw),
                       "values": "1, 1"} for c, hw in BANDS}

    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 4},
        "sensor": {
            "type": "perspective", "fov": 45,
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[0, -4, 2], target=[0, 0, 0.5], up=[0, 0, 1]),
            "film": {"type": "specfilm", "width": 256, "height": 256,
                     "component_format": "float32", **bands},
            "sampler": {"type": "independent", "sample_count": 128},
        },
        # The IR sky. Local +Z is the zenith, matching the scene's up vector.
        "sky": sky.to_dict(),
        "ground": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f().scale([6, 6, 1]),
            "bsdf": {"type": "diffuse",
                     "reflectance": {"type": "uniform", "value": 0.10}},
        },
        "sphere": {
            "type": "sphere", "center": [0, 0, 0.5], "radius": 0.5,
            "bsdf": {"type": "diffuse",
                     "reflectance": {"type": "uniform", "value": 0.35}},
        },
    })

    print("rendering ...")
    img = np.array(mi.render(scene, spp=128))   # (H, W, 3 bands), W/m^2/sr
    print(f"  film shape {img.shape}, range [{img.min():.3e}, {img.max():.3e}]")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for k, ((c, hw), ax) in enumerate(zip(BANDS, axes)):
        bt = brightness_temp(img[:, :, k] / (2 * hw), float(c))
        im = ax.imshow(bt, cmap="inferno")
        ax.set_title(f"{c/1000:.1f} um  (reflected sky only)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="T_B [K]")
    fig.suptitle("Scene lit by the libRadtran spectralsky IR emitter", fontsize=13)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
