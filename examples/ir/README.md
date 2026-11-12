# Infrared rendering examples

Small, self-contained demonstrations of the infrared extension of Mitsuba 3.
Each script prints what it checks and writes images to `examples/ir/output/`.
The physics behind every feature is described in
[`docs/ir/IR_TECHNICAL_REPORT.md`](../../docs/ir/IR_TECHNICAL_REPORT.md).

Run from the build environment (`source build/setpath.sh`), e.g.

```
cd examples/ir
python 01_lwir_basic.py              # llvm_ad_spectral by default
python 01_lwir_basic.py cuda_ad_spectral
```

| Script | Feature |
|---|---|
| `01_lwir_basic.py` | First LWIR image: `thermal` emitters, `mitsuba.ir.thermalize`, Kirchhoff coupling (a rough aluminium box mirrors the cold sky), `band_film`, apparent temperature |
| `02_bands.py` | Same scene in SWIR / MWIR / LWIR with a physically scaled sun (`mitsuba.ir.sun`); separates reflected and emitted components |
| `03_materials.py` | Shipped IR optical constants (`*_ir` presets): metals, water, glass; spheres at one temperature with different apparent temperatures; emissivity spectra plot |
| `04_kirchhoff_checks.py` | Rendered vs. analytic radiance for plates, cavities and angular emissivity of a conductor |
| `05_temperature_texture.py` | Temperature field as a raw float bitmap texture |
| `06_ecostress.py` | Using measured ECOSTRESS/ASTER reflectance spectra as Lambertian IR materials |

Requirements: a spectral variant of Mitsuba (`scalar_spectral`, `llvm_ad_spectral`
or `cuda_ad_spectral`), NumPy; matplotlib is used for the figures when available.
