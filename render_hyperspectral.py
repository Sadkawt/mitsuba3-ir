#!/usr/bin/env python3
"""
Hyperspectral scene render — mitsubaIR fork.

Scene: hot iron cube (1000 K), vegetation sphere (295 K), concrete floor (295 K).
Illumination: physically calibrated sky spectrum (solar proxy + 260 K sky thermal)
+ self-emission from all objects.

Spectral bands (specfilm importance-samples from combined SRF):
  VIS  : 400 – 800 nm  (9 bands × 50 nm)
  SWIR : 1000 – 2300 nm (5 bands × 200 nm)
  LWIR : 8500 – 11500 nm (7 bands × 500 nm)

Outputs (renders/hyperspectral/):
  false_color_composites.png – VIS true-color | SWIR false-color | LWIR thermal
  all_bands_grid.png          – every band as a greyscale/inferno tile
  spectral_curves.png         – per-pixel spectral radiance vs analytical Planck
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import mitsuba as mi
mi.set_variant("scalar_spectral")

# ── physical constants ─────────────────────────────────────────────────────
H_PLANCK = 6.62607015e-34
C_LIGHT  = 2.99792458e+8
K_BOLTZ  = 1.380649e-23


def planck_nm(lam_nm, T):
    """Spectral radiance B_λ [W m⁻² sr⁻¹ nm⁻¹] for a blackbody at T [K]."""
    lam = np.asarray(lam_nm, float) * 1e-9
    c0 = 2.0 * H_PLANCK * C_LIGHT**2
    c1 = H_PLANCK * C_LIGHT / K_BOLTZ
    # clip exponent to avoid overflow in float64
    exponent = np.clip(c1 / (lam * T), None, 700.0)
    return 1e-9 * c0 / (lam**5 * (np.exp(exponent) - 1.0))


# ── spectral material data ─────────────────────────────────────────────────
# Wavelength grid shared by all reflectance/emissivity spectra [nm]
SPEC_WAV = np.array([
    350, 400, 450, 500, 550, 600, 650, 680, 700, 720, 750, 800, 900,
    1000, 1100, 1200, 1350, 1400, 1500, 1600, 1800, 1950, 2000, 2300, 2500,
    3000, 4000, 5000, 6000, 7000, 8000, 8500, 9000, 9500, 10000,
    10500, 11000, 11500, 12000,
], dtype=float)

# ---- Vegetation (broadleaf) -----------------------------------------------
# VIS: chlorophyll + carotenoid absorption except green bump ~550 nm
# Red edge ~720 nm: jump from 5 % to 40 %
# NIR plateau ~750-1350 nm: 40-48 %
# SWIR water-absorption dips: 1350 nm, 1800-1950 nm
# LWIR: very low (leaves absorb; emissivity ≈ 0.98)
VEGE_REFL = np.array([
    0.04, 0.05, 0.06, 0.07, 0.09, 0.07, 0.05, 0.05, 0.05, 0.12,
    0.41, 0.47, 0.49, 0.48, 0.46, 0.42, 0.13, 0.05, 0.38, 0.39,
    0.10, 0.05, 0.32, 0.28, 0.20,
    0.06, 0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02,
    0.02, 0.02, 0.02, 0.02,
])
VEGE_EMIS = 0.98   # leaves behave as near-blackbody in LWIR

# ---- Concrete / asphalt ---------------------------------------------------
# Broadband gray; slightly rising toward NIR; high emissivity in LWIR
CONC_REFL = np.array([
    0.10, 0.12, 0.15, 0.18, 0.22, 0.25, 0.27, 0.28, 0.29, 0.30,
    0.31, 0.33, 0.35, 0.36, 0.36, 0.35, 0.33, 0.31, 0.32, 0.32,
    0.29, 0.27, 0.30, 0.28, 0.24,
    0.10, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05,
    0.05, 0.05, 0.05, 0.05,
])
CONC_EMIS = 0.95

# ---- Oxidised iron / steel ------------------------------------------------
# High specular component in VIS–NIR; moderate emissivity in LWIR (~0.20)
IRON_REFL = np.array([
    0.55, 0.57, 0.58, 0.58, 0.59, 0.60, 0.60, 0.60, 0.61, 0.62,
    0.63, 0.65, 0.67, 0.69, 0.71, 0.72, 0.71, 0.70, 0.71, 0.71,
    0.70, 0.70, 0.70, 0.70, 0.69,
    0.70, 0.70, 0.71, 0.71, 0.71, 0.80, 0.80, 0.80, 0.80, 0.80,
    0.80, 0.80, 0.80, 0.80,
])
# LWIR reflectance ≈ 1 − emissivity_LWIR (Kirchhoff); 0.80 → ε ≈ 0.20
IRON_EMIS = 0.20

T_IRON  = 1000.0   # hot industrial metal [K]
T_SCENE = 295.0    # vegetation + concrete [K]
T_SKY   = 260.0    # clear-sky effective radiating temperature [K]

# Solar scaling factor: calibrated so the combined sky spectrum matches the
# measured clear-sky solar irradiance at 500 nm (~1.8 W m⁻² nm⁻¹), converted
# to Lambertian radiance L = E/π. The sun's surface blackbody at 5800 K gives
# B_5800(500 nm) ≈ 2.69e4 W m⁻² sr⁻¹ nm⁻¹. Scaling to L_sol(500 nm) = 0.57:
SOLAR_SCALE = 0.57 / 2.69e4   # ≈ 2.12e-5 sr (effective solid-angle × transmittance)


# ── helpers ────────────────────────────────────────────────────────────────

def irregular_spectrum(wavs, vals):
    return {
        "type": "irregular",
        "wavelengths": ", ".join(f"{w:.2f}" for w in wavs),
        "values":      ", ".join(f"{max(v, 0.0):.8e}" for v in vals),
    }


def emission_dict(wavs, T, emissivity):
    """Grey-body emitted radiance spectrum:  ε(λ) · B_λ(T)."""
    return irregular_spectrum(wavs, emissivity * planck_nm(wavs, T))


def combined_sky_spectrum(wavs):
    """
    Physically calibrated sky radiance spectrum [W m⁻² sr⁻¹ nm⁻¹]:
      L_sky(λ) = SOLAR_SCALE · B_5800(λ)   (solar scatter, dominates VIS)
              + B_260K(λ)                   (sky thermal, dominates LWIR)

    At 500 nm : ~0.57 W m⁻² sr⁻¹ nm⁻¹  (matches clear-sky solar)
    At 10 µm  : ~4.7e-3 W m⁻² sr⁻¹ nm⁻¹ (physical sky thermal)
    Iron cube (1000 K) at 10 µm: 0.074 → 16× above sky background ✓
    """
    vals = SOLAR_SCALE * planck_nm(wavs, 5800.0) + planck_nm(wavs, T_SKY)
    return irregular_spectrum(wavs, vals)


def rectangular_srf(center_nm, half_nm):
    """Rectangular SRF for one specfilm band."""
    return {
        "type": "regular",
        "wavelength_min": float(center_nm - half_nm),
        "wavelength_max": float(center_nm + half_nm),
        "values": "1, 1",
    }


# ── spectral bands ─────────────────────────────────────────────────────────
# Use zero-padded 5-digit names so alphabetical == wavelength order.
VIS_CENTERS  = [400, 450, 500, 550, 600, 650, 700, 750, 800]
SWIR_CENTERS = [1000, 1300, 1600, 2000, 2300]
LWIR_CENTERS = [8500, 9000, 9500, 10000, 10500, 11000, 11500]

VIS_HW  =  25   # ± nm
SWIR_HW = 100
LWIR_HW = 250


def all_band_defs():
    """Return {param_name: srf_dict} in wavelength order with 0-padded names."""
    d = {}
    for c in VIS_CENTERS:
        d[f"vis_{c:05d}"] = rectangular_srf(c, VIS_HW)
    for c in SWIR_CENTERS:
        d[f"swir_{c:05d}"] = rectangular_srf(c, SWIR_HW)
    for c in LWIR_CENTERS:
        d[f"lwir_{c:05d}"] = rectangular_srf(c, LWIR_HW)
    return d


def sorted_band_names():
    # specfilm outputs channels in insertion order (the order keys were added
    # to the scene dict), NOT alphabetically.  all_band_defs() inserts
    # VIS → SWIR → LWIR, so that is the channel order in the rendered array.
    names = []
    for c in VIS_CENTERS:
        names.append(f"vis_{c:05d}")
    for c in SWIR_CENTERS:
        names.append(f"swir_{c:05d}")
    for c in LWIR_CENTERS:
        names.append(f"lwir_{c:05d}")
    return names


BAND_NAMES  = sorted_band_names()
BAND_CENTERS = {
    n: int(n.split("_")[1])
    for n in BAND_NAMES
}


# ── scene construction ──────────────────────────────────────────────────────

def build_scene(resolution=(256, 256), spp=512):
    W, H = resolution

    iron_refl_sp = irregular_spectrum(SPEC_WAV, IRON_REFL)
    vege_refl_sp = irregular_spectrum(SPEC_WAV, VEGE_REFL)
    conc_refl_sp = irregular_spectrum(SPEC_WAV, CONC_REFL)

    iron_emit_sp = emission_dict(SPEC_WAV, T_IRON,  IRON_EMIS)
    vege_emit_sp = emission_dict(SPEC_WAV, T_SCENE, VEGE_EMIS)
    conc_emit_sp = emission_dict(SPEC_WAV, T_SCENE, CONC_EMIS)

    srfs = all_band_defs()

    # Camera: slightly above and to the side, looking toward the objects
    cam_T = mi.ScalarTransform4f.look_at(
        origin=[3.5, 2.5, 3.5],
        target=[0.0, 0.5, 0.0],
        up=[0, 1, 0],
    )

    # Ground: rectangle originally in XY plane, rotated to lie flat on XZ (y=0)
    ground_T = (
        mi.ScalarTransform4f()
        .rotate([1, 0, 0], -90)
        .scale([6, 6, 1])
    )

    # Cube: sitting on the floor at origin, 1.2 m tall
    cube_T = (
        mi.ScalarTransform4f()
        .translate([0.0, 0.6, 0.0])
        .scale([0.6, 0.6, 0.6])
    )

    return {
        "type": "scene",

        "integrator": {"type": "path", "max_depth": 3},

        "sensor": {
            "type": "perspective",
            "fov": 45.0,
            "to_world": cam_T,
            "film": {
                "type": "specfilm",
                "width": W,
                "height": H,
                "component_format": "float32",
                "rfilter": {"type": "box"},
                **srfs,
            },
            "sampler": {"type": "independent", "sample_count": spp},
        },

        # Physically calibrated sky: solar proxy (5800 K scaled to ground level)
        # + sky thermal emission at 260 K. This ensures:
        #   - VIS images are lit by realistic solar scatter (~0.57 W/m²/sr/nm)
        #   - LWIR thermal emission from objects (0.007-0.074 W/m²/sr/nm) clearly
        #     exceeds sky background (~4.7e-3 W/m²/sr/nm)
        "sky": {
            "type": "constant",
            "radiance": combined_sky_spectrum(SPEC_WAV),
        },

        # ── Concrete ground (295 K) ─────────────────────────────────────────
        "ground": {
            "type": "rectangle",
            "to_world": ground_T,
            "bsdf": {"type": "diffuse", "reflectance": conc_refl_sp},
            "emitter": {"type": "area", "radiance": conc_emit_sp},
        },

        # ── Hot iron cube (1000 K) ──────────────────────────────────────────
        "cube": {
            "type": "cube",
            "to_world": cube_T,
            "bsdf": {"type": "diffuse", "reflectance": iron_refl_sp},
            "emitter": {"type": "area", "radiance": iron_emit_sp},
        },

        # ── Vegetation sphere (295 K) ───────────────────────────────────────
        "sphere": {
            "type": "sphere",
            "center": [-2.0, 0.6, -0.5],
            "radius": 0.6,
            "bsdf": {"type": "diffuse", "reflectance": vege_refl_sp},
            "emitter": {"type": "area", "radiance": vege_emit_sp},
        },
    }


# ── render ──────────────────────────────────────────────────────────────────

def render(resolution=(256, 256), spp=512, cache_path=None):
    if cache_path and Path(cache_path).exists():
        print(f"  Loading cached render from {cache_path}")
        return np.load(cache_path)

    print(f"  Rendering {resolution[0]}×{resolution[1]} px, {spp} spp …")
    scene = mi.load_dict(build_scene(resolution=resolution, spp=spp))
    img   = mi.render(scene, spp=spp)
    arr   = np.array(img)   # (H, W, n_bands)

    if cache_path:
        np.save(cache_path, arr)
        print(f"  Saved render cache to {cache_path}")
    return arr


# ── plotting helpers ─────────────────────────────────────────────────────────

def norm01(x):
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-30 else (x - lo) / (hi - lo)


def false_color(arr, r_name, g_name, b_name):
    """RGB false-color image from three named specfilm bands."""
    r_idx = BAND_NAMES.index(r_name)
    g_idx = BAND_NAMES.index(g_name)
    b_idx = BAND_NAMES.index(b_name)
    return np.stack([
        norm01(arr[:, :, r_idx]),
        norm01(arr[:, :, g_idx]),
        norm01(arr[:, :, b_idx]),
    ], axis=-1)


def gamma(x, g=2.2):
    return np.clip(x, 0, 1) ** (1.0 / g)


def thermal_colormap(arr, name, cmap="inferno"):
    idx = BAND_NAMES.index(name)
    return plt.colormaps[cmap](norm01(arr[:, :, idx]))[:, :, :3]


def band_image(arr, name, cmap="gray"):
    idx = BAND_NAMES.index(name)
    return arr[:, :, idx]


# ── plots ────────────────────────────────────────────────────────────────────

def plot_false_color_composites(arr, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        "Hyperspectral False-Color Composites\n"
        "Iron cube @ 1000 K · Vegetation sphere @ 295 K · Concrete floor @ 295 K",
        fontsize=12, fontweight="bold",
    )

    # VIS true color: R=650 nm, G=550 nm, B=450 nm
    vis_img = gamma(false_color(arr, "vis_00650", "vis_00550", "vis_00450"))
    axes[0].imshow(vis_img)
    axes[0].set_title("VIS True Color\nR=650 G=550 B=450 nm", fontsize=10)
    axes[0].axis("off")

    # SWIR false color: R=2300 nm (water bands), G=1600 nm (NIR plateau), B=1000 nm
    swir_img = gamma(false_color(arr, "swir_02300", "swir_01600", "swir_01000"))
    axes[1].imshow(swir_img)
    axes[1].set_title("SWIR False Color\nR=2300 G=1600 B=1000 nm", fontsize=10)
    axes[1].axis("off")

    # LWIR thermal: inferno at 10000 nm
    lwir_img = thermal_colormap(arr, "lwir_10000", cmap="inferno")
    axes[2].imshow(lwir_img)
    axes[2].set_title("LWIR Thermal (10 µm)\ninferno colormap", fontsize=10)
    axes[2].axis("off")
    # Add a colorbar-like annotation
    axes[2].text(0.02, 0.02, "cool", transform=axes[2].transAxes,
                 color="white", fontsize=8, va="bottom")
    axes[2].text(0.98, 0.02, "hot", transform=axes[2].transAxes,
                 color="white", fontsize=8, va="bottom", ha="right")

    plt.tight_layout()
    p = out_dir / "false_color_composites.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def plot_all_bands(arr, out_dir):
    n  = len(BAND_NAMES)
    nc = 7
    nr = (n + nc - 1) // nc

    fig, axes = plt.subplots(nr, nc, figsize=(nc * 2.8, nr * 2.8))
    fig.suptitle("All Spectral Bands (normalised)", fontsize=13, fontweight="bold")

    for i, name in enumerate(BAND_NAMES):
        ax  = axes[i // nc, i % nc]
        ctr = BAND_CENTERS[name]
        reg = name.split("_")[0]
        bimg = norm01(arr[:, :, i])
        cmap = "inferno" if reg == "lwir" else ("hot" if reg == "swir" else "gray")
        ax.imshow(bimg, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{reg.upper()}\n{ctr/1000:.2f} µm", fontsize=7)
        ax.axis("off")

    # Hide surplus panels
    for j in range(n, nr * nc):
        axes[j // nc, j % nc].axis("off")

    plt.tight_layout()
    p = out_dir / "all_bands_grid.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def plot_spectral_curves(arr, out_dir):
    """Spectral radiance curves for representative pixels vs analytical emission."""
    H_img, W_img = arr.shape[:2]

    # Pixel positions calibrated to the rendered scene layout.
    # Camera at (3.5,2.5,3.5) looking toward (0,0.5,0):
    #   cube  ≈ right-centre of image, sphere ≈ upper-left, ground ≈ bottom-strip.
    px_cube   = (H_img // 2,          int(W_img * 0.60))
    px_sphere = (int(H_img * 0.38),   int(W_img * 0.22))
    px_ground = (int(H_img * 0.82),   W_img // 2)

    centers_nm = [BAND_CENTERS[n] for n in BAND_NAMES]

    # Band widths in nm — same order as BAND_NAMES
    _hw_map = {n: VIS_HW  for n in BAND_NAMES if n.startswith("vis")}
    _hw_map.update({n: SWIR_HW for n in BAND_NAMES if n.startswith("swir")})
    _hw_map.update({n: LWIR_HW for n in BAND_NAMES if n.startswith("lwir")})
    bw = np.array([2.0 * _hw_map[n] for n in BAND_NAMES])  # full bandwidth [nm]

    def get_spec(px):
        # specfilm output is ∫ L·SRF dλ (W/m²/sr); divide by bandwidth
        # to recover band-average spectral radiance (W/m²/sr/nm).
        raw = np.array([float(arr[px[0], px[1], i]) for i in range(len(BAND_NAMES))])
        return raw / bw

    spec_cube   = get_spec(px_cube)
    spec_sphere = get_spec(px_sphere)
    spec_ground = get_spec(px_ground)

    # Analytical Planck curves — clamp to a physically visible floor so they
    # don't drive the y-axis down to 10^-50 in the VIS for 295 K bodies.
    Y_FLOOR = 1e-8
    lam_fine = np.linspace(360, 12000, 2000)
    P_iron   = np.maximum(IRON_EMIS * planck_nm(lam_fine, T_IRON),  Y_FLOOR)
    P_vege   = np.maximum(VEGE_EMIS * planck_nm(lam_fine, T_SCENE), Y_FLOOR)
    P_conc   = np.maximum(CONC_EMIS * planck_nm(lam_fine, T_SCENE), Y_FLOOR)

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.semilogy(centers_nm, spec_cube,   "r-o", ms=5, lw=1.8,
                label=f"Rendered – iron cube (1000 K)  px{px_cube}")
    ax.semilogy(centers_nm, spec_sphere, "g-s", ms=5, lw=1.8,
                label=f"Rendered – veg sphere (295 K)   px{px_sphere}")
    ax.semilogy(centers_nm, spec_ground, "b-^", ms=5, lw=1.8,
                label=f"Rendered – ground (295 K)        px{px_ground}")

    ax.plot(lam_fine, P_iron, "r--", alpha=0.45, lw=1.5,
            label="Analytic  ε·B_λ iron @ 1000 K (thermal only)")
    ax.plot(lam_fine, P_vege, "g--", alpha=0.45, lw=1.5,
            label="Analytic  ε·B_λ veg  @ 295 K  (thermal only)")
    ax.plot(lam_fine, P_conc, "b--", alpha=0.45, lw=1.5,
            label="Analytic  ε·B_λ conc @ 295 K  (thermal only)")

    # Spectral region shading
    for span, label, colour in [
        ((380,  830), "VIS",  "violet"),
        ((830, 2500), "SWIR", "orange"),
        ((2500,5000), "MWIR", "salmon"),
        ((8000,12100),"LWIR", "red"),
    ]:
        ax.axvspan(*span, alpha=0.06, color=colour, label=label)

    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Spectral radiance  (W m⁻² sr⁻¹ nm⁻¹)", fontsize=12)
    ax.set_title(
        "Spectral Radiance at Selected Scene Points\n"
        "Solid: rendered (reflected sky + thermal emission)  ·  "
        "Dashed: pure thermal emission only",
        fontsize=11,
    )
    ax.set_xlim(350, 12100)
    ax.set_ylim(1e-6, 5.0)      # sensible range: avoid 295 K Planck in VIS (≈10^-50)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, which="both", alpha=0.25)

    plt.tight_layout()
    p = out_dir / "spectral_curves.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def plot_region_mosaics(arr, out_dir):
    """3-row plot: individual band images for VIS, SWIR, LWIR."""
    vis_names  = [n for n in BAND_NAMES if n.startswith("vis")]
    swir_names = [n for n in BAND_NAMES if n.startswith("swir")]
    lwir_names = [n for n in BAND_NAMES if n.startswith("lwir")]

    rows = [vis_names, swir_names, lwir_names]
    row_labels = ["VIS", "SWIR", "LWIR"]
    cmaps = ["gray", "hot", "inferno"]
    n_cols = max(len(r) for r in rows)

    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 2.4, 9))
    fig.suptitle("Spectral Band Mosaics", fontsize=13, fontweight="bold")

    for row_i, (names, label, cmap) in enumerate(zip(rows, row_labels, cmaps)):
        for col_i in range(n_cols):
            ax = axes[row_i, col_i]
            if col_i < len(names):
                nm = names[col_i]
                ctr = BAND_CENTERS[nm]
                bimg = norm01(arr[:, :, BAND_NAMES.index(nm)])
                ax.imshow(bimg, cmap=cmap, vmin=0, vmax=1)
                ax.set_title(f"{ctr/1000:.2f} µm", fontsize=8)
            ax.axis("off")

        # Row label on the left
        axes[row_i, 0].set_ylabel(label, fontsize=11, rotation=90, labelpad=4)

    plt.tight_layout()
    p = out_dir / "region_mosaics.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir    = Path("/home/sadcat/Code/mitsubaIR/renders/hyperspectral")
    cache_file = out_dir / "bands.npy"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== mitsubaIR hyperspectral render ===")
    print(f"Bands: {len(BAND_NAMES)}  "
          f"(VIS={len(VIS_CENTERS)}, SWIR={len(SWIR_CENTERS)}, LWIR={len(LWIR_CENTERS)})")

    arr = render(resolution=(256, 256), spp=1024000, cache_path=str(cache_file))
    print(f"Array shape: {arr.shape}   dtype: {arr.dtype}")
    print(f"Value range: [{arr.min():.3e}, {arr.max():.3e}]")

    print("Generating plots …")
    plot_false_color_composites(arr, out_dir)
    plot_all_bands(arr, out_dir)
    plot_spectral_curves(arr, out_dir)
    plot_region_mosaics(arr, out_dir)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
