#!/usr/bin/env python3
"""
Integration test: thermal radiance field + PLY vertex-temperature + spectralsky.

Scene
-----
  ground  - rectangle at z=0, thermal radiance field at a fixed 290 K (emissivity 0.95)
  panel   - square plate loaded from PLY, hovering at z=1.2 m, per-vertex temperatures
            (280 K / 310 K / 330 K / 360 K at the four corners) driving a thermal
            radiance field via the mesh_attribute texture
  sky     - spectralsky emitter built from the lwir_us_standard.skyrad table

Camera & film
-------------
  Perspective view from above-and-side.
  specfilm with three LWIR bands (8500, 10000, 11500 nm) rendered as a false-colour PNG.

Run
---
  PYTHONPATH=build/python python test_ir_integration.py [scalar_spectral]
"""

import sys
import struct
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Mitsuba setup ────────────────────────────────────────────────────────────
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "scalar_spectral"
import mitsuba as mi
mi.set_variant(VARIANT)

# skyrad_io lives in tools/
sys.path.insert(0, str(Path(__file__).parent / "tools"))
from skyrad_io import SkyRadiance  # noqa: E402

# ── paths ─────────────────────────────────────────────────────────────────────
HERE    = Path(__file__).parent
OUT_DIR = HERE / "renders" / "test_ir_integration"
PLY_PATH = OUT_DIR / "panel.ply"
SKYRAD  = HERE / "tools" / "data" / "ir_sky" / "lwir_us_standard.skyrad"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── physical constants (for the brightness-temperature inverse) ───────────────
H_PLANCK = 6.62607015e-34
C_LIGHT  = 2.99792458e8
K_BOLTZ  = 1.380649e-23

LWIR_MIN, LWIR_MAX = 8000.0, 12000.0   # nm  (8-12 µm)


# ── PLY creation ─────────────────────────────────────────────────────────────

def write_panel_ply(path: Path) -> None:
    """
    Unit square in the XY plane (z=0) with per-vertex temperatures.
    Four corners, two triangles, ASCII PLY.

    Corner temperatures (Kelvin):
      bottom-left  (-0.5, -0.5)  : 280 K  (coldest)
      bottom-right ( 0.5, -0.5)  : 310 K
      top-left     (-0.5,  0.5)  : 330 K
      top-right    ( 0.5,  0.5)  : 360 K  (hottest)

    The attribute is called `temperature` in the PLY vertex element;
    Mitsuba exposes it as `vertex_temperature` via mesh_attribute.
    """
    vertices = [
        (-0.5, -0.5, 0.0, 280.0),   # idx 0
        ( 0.5, -0.5, 0.0, 310.0),   # idx 1
        (-0.5,  0.5, 0.0, 330.0),   # idx 2
        ( 0.5,  0.5, 0.0, 360.0),   # idx 3
    ]
    faces = [(0, 1, 3), (0, 3, 2)]

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        # Mitsuba PLY loader requires a single-letter postfix (_0, _x, _r …) to
        # detect and register vertex attributes.  A scalar attribute uses _0;
        # the registered name becomes "vertex_temperature" (prefix "vertex_" +
        # PLY field prefix "temperature").
        f.write("property float temperature_0\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for x, y, z, t in vertices:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {t:.2f}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")

    print(f"  wrote PLY: {path}")
    print(f"  corner temperatures: 280 K (BL), 310 K (BR), 330 K (TL), 360 K (TR)")


# ── scene construction ────────────────────────────────────────────────────────

def build_scene(spp: int = 256, resolution: int = 256) -> dict:
    sky  = SkyRadiance.read(SKYRAD)

    # Three LWIR specfilm bands used as false-colour R/G/B channels.
    BANDS = [(8500, 250), (10000, 250), (11500, 250)]
    band_srfs = {
        f"band_{c}": {
            "type": "regular",
            "wavelength_min": float(c - hw),
            "wavelength_max": float(c + hw),
            "values": "1, 1",
        }
        for c, hw in BANDS
    }

    # Camera: slightly above and to the side, looking down at the scene.
    cam_T = mi.ScalarTransform4f.look_at(
        origin=[3.5, -3.5, 3.0],
        target=[0.0,  0.0, 0.6],
        up=[0.0, 0.0, 1.0],
    )

    # Ground: large rectangle in the XY plane (Mitsuba default rectangle lies
    # in XY), centred at origin, z=0.
    ground_T = mi.ScalarTransform4f.scale([5, 5, 1])

    # Panel: the PLY is a unit square in the XY plane at z=0; we translate it
    # up to z=1.2 m so it hovers above the ground.
    panel_T = mi.ScalarTransform4f.translate([0, 0, 1.2])

    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 4},

        "sensor": {
            "type": "perspective",
            "fov": 45.0,
            "to_world": cam_T,
            "film": {
                "type": "specfilm",
                "width": resolution,
                "height": resolution,
                "component_format": "float32",
                "rfilter": {"type": "box"},
                **band_srfs,
            },
            "sampler": {"type": "independent", "sample_count": spp},
        },

        # ── spectralsky (libRadtran-driven LWIR sky) ─────────────────────────
        "sky": sky.to_dict(),

        # ── Ground plane (290 K, emissivity 0.95) via radiance field ─────────
        "ground": {
            "type": "rectangle",
            "to_world": ground_T,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "uniform", "value": 0.05},
            },
            "radiance": {
                "type": "thermal",
                "temperature": 290.0,
                "emissivity": 0.95,
                "wavelength_min": LWIR_MIN,
                "wavelength_max": LWIR_MAX,
            },
        },

        # ── Hovering panel (PLY, per-vertex temperature) via radiance field ───
        "panel": {
            "type": "ply",
            "filename": str(PLY_PATH),
            "to_world": panel_T,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "uniform", "value": 0.05},
            },
            "radiance": {
                "type": "thermal",
                # vertex_temperature is the PLY `temperature` field exposed by
                # Mitsuba's mesh_attribute texture (vertex_ prefix is automatic)
                "temperature": {
                    "type": "mesh_attribute",
                    "name": "vertex_temperature",
                },
                "emissivity": 0.90,
                "wavelength_min": LWIR_MIN,
                "wavelength_max": LWIR_MAX,
            },
        },
    }


# ── rendering & output ────────────────────────────────────────────────────────

def brightness_temp(L_per_nm: np.ndarray, lam_nm: float) -> np.ndarray:
    """Invert Planck: band-average spectral radiance -> brightness temperature [K].
    Returns NaN for zero or negative radiance pixels (sky, black background).
    """
    lam = lam_nm * 1e-9
    L = L_per_nm * 1e9   # W/m²/sr/nm -> W/m²/sr/m
    with np.errstate(divide="ignore", invalid="ignore"):
        bt = (H_PLANCK * C_LIGHT / (lam * K_BOLTZ)) / np.log1p(
            2 * H_PLANCK * C_LIGHT**2 / (lam**5 * np.where(L > 0, L, np.nan))
        )
    return bt


def norm01(x: np.ndarray) -> np.ndarray:
    finite = x[np.isfinite(x)]
    lo = finite.min() if len(finite) else 0.0
    hi = finite.max() if len(finite) else 1.0
    return np.clip((x - lo) / (hi - lo + 1e-30), 0.0, 1.0)


def main() -> int:
    SPP = 512
    RES = 256

    print(f"=== IR integration test  variant={VARIANT}  {RES}×{RES}  {SPP} spp ===\n")
    write_panel_ply(PLY_PATH)

    print("\nloading scene …")
    scene = mi.load_dict(build_scene(spp=SPP, resolution=RES))
    print("rendering …")
    img = np.array(mi.render(scene, spp=SPP))   # (H, W, 3 bands)
    print(f"  output shape: {img.shape}  range: [{img.min():.3e}, {img.max():.3e}]")

    # The three bands were ordered 8500, 10000, 11500 nm.
    BAND_CENTERS_NM = [8500.0, 10000.0, 11500.0]
    BAND_HW_NM      = [250.0,  250.0,   250.0]

    # Convert specfilm integral [W/m²/sr per band] -> spectral radiance [W/m²/sr/nm]
    spectral = np.stack([
        img[:, :, k] / (2 * hw)
        for k, hw in enumerate(BAND_HW_NM)
    ], axis=-1)

    # Brightness temperature per band (NaN where L=0)
    bt = np.stack([
        brightness_temp(spectral[:, :, k], float(c))
        for k, c in enumerate(BAND_CENTERS_NM)
    ], axis=-1)

    # ── save raw arrays ───────────────────────────────────────────────────────
    np.save(OUT_DIR / "bands.npy", img)
    print(f"  saved raw bands -> {OUT_DIR / 'bands.npy'}")

    # ── false-colour plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        "IR integration test — ground 290 K · panel 280–360 K · spectralsky\n"
        f"variant: {VARIANT}",
        fontsize=12, fontweight="bold",
    )

    cmap_bands = ["plasma", "inferno", "hot"]
    for k, (center, cmap_name) in enumerate(zip(BAND_CENTERS_NM, cmap_bands)):
        # Use raw radiance (normalised) to avoid NaN in colorbar
        im = axes[k].imshow(norm01(img[:, :, k]), cmap=cmap_name, vmin=0, vmax=1)
        axes[k].set_title(f"{center/1000:.1f} µm\n(normalised radiance)", fontsize=10)
        axes[k].axis("off")
        fig.colorbar(im, ax=axes[k], fraction=0.046, label="norm. L")

    # Composite false-colour: R=11.5µm, G=10µm, B=8.5µm
    composite = np.stack([
        norm01(img[:, :, 2]),   # R  = 11.5 µm
        norm01(img[:, :, 1]),   # G  = 10.0 µm
        norm01(img[:, :, 0]),   # B  =  8.5 µm
    ], axis=-1)
    axes[3].imshow(np.clip(composite, 0, 1))
    axes[3].set_title("False colour\nR=11.5 G=10 B=8.5 µm", fontsize=10)
    axes[3].axis("off")

    plt.tight_layout()
    out_png = OUT_DIR / f"ir_integration_{VARIANT}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved render -> {out_png}")

    # ── sanity checks ─────────────────────────────────────────────────────────
    print("\n=== sanity checks ===")
    ok = True

    # 1. The panel's hot corner (360 K) must be clearly brighter than the
    #    ground (290 K) in LWIR. We compare the panel's hottest pixels rather
    #    than its mean: the panel spans 280-360 K, so its *coldest* corner is
    #    genuinely colder than the ground, and the ground itself has a
    #    viewing-geometry brightness gradient. The hot corner is the global
    #    radiance maximum and is the unambiguous signature that the per-vertex
    #    temperature is driving emission.
    b10 = img[:, :, 1]                                   # 10 µm band
    H, W, _ = img.shape
    panel_max = b10.max()
    # Ground reference: median of the lower image (mostly ground). The median is
    # robust to the small, bright panel projected into it.
    ground_med = np.median(b10[int(0.55 * H):, :])
    ratio = panel_max / (ground_med + 1e-30)
    # Report the hot-corner cluster location for context.
    ys, xs = np.where(b10 > 0.7 * panel_max)
    print(f"  hot-corner cluster: rows {ys.min()}-{ys.max()}, cols {xs.min()}-{xs.max()}")
    print(f"  10 µm panel hot   : {panel_max:.4e} W/m²/sr (360 K corner)")
    print(f"  10 µm ground med  : {ground_med:.4e} W/m²/sr (290 K)")
    print(f"  hot-panel/ground  : {ratio:.2f}  (expect >1 — panel 360 K > ground 290 K)")
    if ratio <= 1.1:
        print("  WARNING: panel hot corner not brighter than ground — check radiance field")
        ok = False
    else:
        print("  OK")

    # 2. Image should have non-trivial signal
    if img.max() < 1e-10:
        print("  FAIL: rendered image is black — no radiance reaching sensor")
        ok = False
    else:
        print(f"  OK: max radiance {img.max():.3e} W/m²/sr (non-zero)")

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    rc = main()
    # Force immediate exit to bypass the CUDA/OptiX teardown crash that occurs
    # during normal Python GC on some driver versions.
    import os
    os._exit(rc)
