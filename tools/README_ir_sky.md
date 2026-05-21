# Hyperspectral IR sky (`spectralsky`) — tutorial

The `spectralsky` emitter is a physically based, spectrally resolved sky dome
for thermal-infrared (and other hyperspectral) rendering in the mitsubaIR fork.
Its downwelling radiance is precomputed with **libRadtran/uvspec** and stored as
a `.skyrad` table, then loaded as an infinite environment emitter.

Unlike `envmap` (which stores RGB and upsamples to a spectrum), `spectralsky`
stores *true spectral radiance* `L(θ, φ, λ)` in `W·m⁻²·sr⁻¹·nm⁻¹`, so it is
correct across the LWIR band where RGB upsampling is meaningless.

## Pieces

| File | Role |
|------|------|
| `tools/generate_ir_sky.py` | Drives uvspec, writes a `.skyrad` table |
| `tools/skyrad_io.py` | `.skyrad` read/write + `to_dict()` for scenes |
| `tools/example_ir_sky_scene.py` | Runnable example scene |
| `tools/test_spectralsky.py` | Validation suite |
| `src/emitters/spectralsky.cpp` | The emitter plugin |
| `tools/data/ir_sky/lwir_us_standard.skyrad` | Ready-made sample (8–12 µm, US-standard atmosphere) |

> All Python is run against the local build:
> `PYTHONPATH=build/python .venv/bin/python <script>`

## 1. (Optional) generate a sky table

A validated sample table ships in `tools/data/ir_sky/`. To make your own
(e.g. a different atmosphere or band), drive libRadtran:

```bash
PYTHONPATH=build/python .venv/bin/python tools/generate_ir_sky.py \
    --output tools/data/ir_sky/lwir_midlat.skyrad \
    --wl 8000 12000 --reptran coarse \
    --atmosphere midlatitude_summer \
    --n-theta 19 --n-phi 8
```

This runs `uvspec` with `source thermal`, `mol_abs_param reptran`, looking
upward from the surface (`zout 0`, negative `umu`), and converts the
per-wavenumber output to per-nm SI radiance. It prints a Planck-envelope sanity
check and confirms the horizon is warmer than the zenith.

Key options: `--reptran {coarse,medium,fine}` (spectral resolution),
`--atmosphere` (any of the built-in afgl profiles), `--n-theta/--n-phi`
(angular resolution), `--wl MIN MAX` (band in nm).

## 2. Add the sky to a scene

```python
import mitsuba as mi
mi.set_variant("cuda_ad_spectral")          # spectral variant required

import sys; sys.path.insert(0, "tools")
from skyrad_io import SkyRadiance

sky = SkyRadiance.read("tools/data/ir_sky/lwir_us_standard.skyrad")

scene = mi.load_dict({
    "type": "scene",
    "integrator": {"type": "path", "max_depth": 4},
    "sensor": {
        "type": "perspective", "fov": 45,
        "to_world": mi.ScalarTransform4f().look_at(
            origin=[0, -4, 2], target=[0, 0, 0.5], up=[0, 0, 1]),
        "film": {
            "type": "specfilm", "width": 256, "height": 256,
            "component_format": "float32",
            # one channel per LWIR band:
            "b10000": {"type": "regular", "wavelength_min": 9750,
                       "wavelength_max": 10250, "values": "1, 1"},
        },
        "sampler": {"type": "independent", "sample_count": 128},
    },
    "sky": sky.to_dict(),                    # <-- the IR sky emitter
    "ground": {
        "type": "rectangle",
        "to_world": mi.ScalarTransform4f().scale([6, 6, 1]),
        "bsdf": {"type": "diffuse",
                 "reflectance": {"type": "uniform", "value": 0.1}},
    },
})

img = mi.render(scene, spp=128)              # film: W·m⁻²·sr⁻¹ integrated per band
```

`sky.to_dict()` builds the emitter dict directly from the table (no file path
needed inside the scene). To orient the dome or align the solar azimuth, pass a
transform: `sky.to_dict(to_world=mi.ScalarTransform4f().rotate([0,0,1], 30))`.
You can also scale the radiance: `sky.to_dict(scale=1.5)`.

Run the full worked example (saves a brightness-temperature image to
`renders/ir_sky_example.png`):

```bash
PYTHONPATH=build/python .venv/bin/python tools/example_ir_sky_scene.py
```

## Conventions & gotchas

- **Up axis**: local **+Z is the zenith**. Use the emitter's `to_world` for any
  other orientation. (The sample scenes use `up=[0,0,1]`.)
- **Below the horizon**: directions pointing downward (and rays that escape
  below the horizon) emit **zero** — the table only covers the sky hemisphere.
  Provide your own ground geometry/emission for the lower hemisphere.
- **Spectral only**: the emitter returns zero in `scalar_rgb`/mono variants.
  Use `scalar_spectral` or `cuda_ad_spectral`.
- **Units**: the table is per-nm SI radiance. libRadtran's reptran thermal
  output is per *wavenumber*; `generate_ir_sky.py` already converts it
  (`L_λ = L_ν · 1e7/λ²`). Verified against uvspec brightness temperature.
- **Importance sampling**: directional sampling uses a hierarchical warp over
  the sky hemisphere weighted by band-integrated radiance × sinθ, so the sky
  combines correctly with BSDF sampling via MIS.

## Validate

```bash
PYTHONPATH=build/python .venv/bin/python tools/test_spectralsky.py
```

Checks table reproduction, below-horizon zero, horizon warming, a path-traced
zenith-radiance recovery, and `sample_direction`/`pdf_direction` consistency.
