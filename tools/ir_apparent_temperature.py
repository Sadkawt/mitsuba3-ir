#!/usr/bin/env python3
"""Render a sphere over a ground in the LWIR and report apparent temperature.

  sphere : 300 K, emissivity 0.5
  ground : 293 K, emissivity 0.4
  sky    : libRadtran LWIR sky (spectralsky)

The output is the apparent (brightness) temperature: the temperature a perfect
blackbody would need to match the radiance measured at the sensor. It sits below
the true temperature because each surface is a grey body (emissivity < 1) that
also reflects the cold downwelling sky.

Run:  PYTHONPATH=build/python python tools/ir_apparent_temperature.py
"""
import sys
from pathlib import Path

import numpy as np
import mitsuba as mi
mi.set_variant("cuda_ad_spectral")

# spectralsky's Python helper (skyrad_io) lives alongside this script in tools/
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from skyrad_io import SkyRadiance

# physical constants (SI)
H, C, K = 6.62607015e-34, 2.99792458e8, 1.380649e-23

LWIR_MIN, LWIR_MAX = 8000.0, 12000.0     # nm, the thermal band the grey bodies emit in
BAND_NM, BAND_HW = 10000.0, 250.0        # we measure apparent temperature at 10 µm ± 0.25 µm
SKYRAD = "tools/data/ir_sky/lwir_us_standard.skyrad"


def grey_body(temperature_K, emissivity):
    """A thermal radiance field: Planck emission scaled by a constant emissivity."""
    return {"type": "thermal", "temperature": temperature_K, "emissivity": emissivity,
            "wavelength_min": LWIR_MIN, "wavelength_max": LWIR_MAX}


def apparent_temperature(band_radiance):
    """Invert Planck's law: band radiance [W/m²/sr] -> brightness temperature [K].
    Pixels with no radiance (open sky beyond the ground) become NaN, i.e. no data."""
    lam = BAND_NM * 1e-9                              # wavelength in metres
    L = band_radiance / (2 * BAND_HW * 1e-9)         # band integral -> spectral radiance [W/m²/sr/m]
    L = np.where(L > 0, L, np.nan)
    return (H * C / (lam * K)) / np.log1p(2 * H * C**2 / (lam**5 * L))


scene = mi.load_dict({
    "type": "scene",
    "integrator": {"type": "path", "max_depth": 4},

    "sensor": {
        "type": "perspective",
        "fov": 40,
        "to_world": mi.ScalarTransform4f.look_at(origin=[4, -4, 2.5], target=[0, 0, 0.5], up=[0, 0, 1]),
        "film": {
            "type": "specfilm", "width": 256, "height": 256, "rfilter": {"type": "box"},
            "band_10um": {"type": "regular", "wavelength_min": BAND_NM - BAND_HW,
                          "wavelength_max": BAND_NM + BAND_HW, "values": "1, 1"},
        },
        "sampler": {"type": "independent", "sample_count": 256},
    },

    "sky": SkyRadiance.read(SKYRAD).to_dict(),

    # ground: 293 K, emissivity 0.4 (so it reflects the remaining 0.6 of the sky)
    "ground": {
        "type": "rectangle",
        "to_world": mi.ScalarTransform4f.scale([100000, 100000, 1]),
        "bsdf": {"type": "diffuse", "reflectance": 0.6},
        "radiance": grey_body(293.0, 0.4),
    },

    # sphere: 300 K, emissivity 0.5, resting on the ground
    "sphere": {
        "type": "sphere", "radius": 0.5, "center": [0, 0, 0.5],
        "bsdf": {"type": "diffuse", "reflectance": 0.5},
        "radiance": grey_body(300.0, 0.5),
    },
})

band = np.array(mi.render(scene, spp=256))[:, :, 0]      # measured radiance in the 10 µm band
appT = apparent_temperature(band)                        # NaN where there is no surface (open sky)

print(f"apparent temperature [K]:  min {np.nanmin(appT):.1f}   max {np.nanmax(appT):.1f}")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
Path("renders").mkdir(exist_ok=True)
plt.imshow(appT, cmap="inferno")
plt.colorbar(label="apparent temperature [K]")
plt.title("LWIR apparent temperature @ 10 µm")
plt.axis("off")
plt.savefig("renders/apparent_temperature.png", dpi=150, bbox_inches="tight")
print("saved renders/apparent_temperature.png")
