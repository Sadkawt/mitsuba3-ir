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
| `07_atmosphere.py` | Emissive air (`volpath_ir`): attenuation and path radiance versus range, gray or `--hitran` absorption |
| `08_sky.py` | `thermalsky`: clear-sky radiance versus zenith angle; a polished metal plate mirroring the sky |
| `09_hitran_transmittance.py` | Line-by-line H₂O/CO₂ transmittance from HITRAN (`hapi`); attenuated sun and air medium |
| `10_thermal_solver.py` | Surface temperatures from the weather (`mitsuba.ir.thermal`): energy balance with shadows and sky-view factors from the scene, thermal lag, internal heat; LWIR renders through the day |

Requirements: a spectral variant of Mitsuba (`scalar_spectral`, `llvm_ad_spectral`
or `cuda_ad_spectral`), NumPy; matplotlib is used for the figures when available.
`09_hitran_transmittance.py` (and `07 --hitran`) need the `hitran-api` package and
network access to hitran.org on first run (line data are cached in `hitran_db/`).
