"""
LWIR emissivity optimisation for thermal camouflage (8-12 um).

Optimises a spatially-varying emissivity TEXTURE on a sphere target so that
the LWIR rendered image matches a reference scene without the sphere.

Physics / Kirchhoff's law  (opaque diffuse body):
    rho(u,v) = 1 - eps(u,v)          reflectance
    L_emit(u,v,lam) ~ eps(u,v) * B(T_sphere, lam)    emission

Parameterisation:
    eps(u,v) = sigmoid(logit(u,v))  in (0, 1)

Emitter approximation: the Planck function varies < 10% across 8.5-11.5 um
at 310 K (we are near the spectral peak), so a flat-spectrum bitmap emitter
with value eps(u,v) * B_ref (where B_ref = mean Planck over the LWIR band)
is an accurate approximation.  The reflective part from the sky emitter
keeps the full spectral Planck shape.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mitsuba as mi
mi.set_variant("cuda_ad_spectral")
import drjit as dr

# ── Physical constants ────────────────────────────────────────────────────────
H_PLANCK = 6.62607015e-34
C_LIGHT  = 2.99792458e8
K_BOLTZ  = 1.380649e-23


def planck_nm(lam_nm, T):
    """Spectral radiance  [W m-2 sr-1 nm-1]."""
    lam = np.asarray(lam_nm, dtype=np.float64) * 1e-9
    return 1e-9 * 2*H_PLANCK*C_LIGHT**2 / (
        lam**5 * (np.exp(H_PLANCK*C_LIGHT / (lam*K_BOLTZ*T)) - 1.0))


# ── Scene / optimisation parameters ──────────────────────────────────────────
T_SPHERE   = 310.0   # K  warm target
T_GROUND   = 295.0   # K  ambient ground
T_SKY      = 270.0   # K  effective downwelling sky

EPS_GROUND = 0.90    # ground emissivity

# specfilm LWIR bands, 8-12 um (insertion order = channel order)
BAND_CENTERS = [8500, 9000, 9500, 10000, 10500, 11000, 11500]  # nm
BAND_HW      = 250   # nm half-width (500 nm wide bands)
N_BANDS      = len(BAND_CENTERS)

IMG_W  = 64
IMG_H  = 64
N_ITER = 150
LR     = 0.02        # Adam learning rate for texture
SPP    = 32          # SPP per gradient render (Adam handles the noise)
SPP_HQ = 512         # high-quality diagnostic renders

TEX_H  = 32          # emissivity texture height (sphere UV v-axis)
TEX_W  = 32          # emissivity texture width  (sphere UV u-axis)

# Emitter spectrum grid (for ground / sky emitters)
N_SPEC   = 60
spec_lam = np.linspace(8000.0, 12000.0, N_SPEC)  # nm

_PL_sph = planck_nm(spec_lam, T_SPHERE)
_PL_gnd = planck_nm(spec_lam, T_GROUND)
_PL_sky = planck_nm(spec_lam, T_SKY)

# Representative Planck value for sphere (flat-spectrum emitter approximation)
# At 310 K, Planck varies < 10% across 8.5-11.5 um - flat spectrum is fine
B_REF = float(np.mean(planck_nm(np.array(BAND_CENTERS, dtype=float), T_SPHERE)))


# ── Scene construction ────────────────────────────────────────────────────────
def _spec_str(arr: np.ndarray) -> str:
    return ", ".join(f"{v:.6e}" for v in arr)


def _lwir_bands() -> dict:
    """specfilm band definitions.  Insertion order = channel order in output."""
    return {
        f"b{c}": {
            "type": "regular",
            "wavelength_min": float(c - BAND_HW),
            "wavelength_max": float(c + BAND_HW),
            "values": "1, 1",
        }
        for c in BAND_CENTERS
    }


def _base_scene() -> dict:
    """Ground + sky + sensor (no sphere)."""
    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 4},
        "sensor": {
            "type": "perspective",
            "fov": 50.0,
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[0, -4, 3], target=[0, 0, 0.5], up=[0, 0, 1]
            ),
            "film": {
                "type": "specfilm",
                "width":  IMG_W,
                "height": IMG_H,
                "component_format": "float32",
                **_lwir_bands(),
            },
            "sampler": {"type": "independent", "sample_count": 1},
        },
        "ground": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f().scale([6, 6, 1]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "checkerboard",
                    "color0": {"type": "uniform", "value": 1.0 - EPS_GROUND},
                    "color1": {"type": "uniform", "value": (1.0 - EPS_GROUND) * 0.5},
                    "to_uv": mi.ScalarTransform4f().scale([5, 5, 1]),
                },
            },
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "regular",
                    "wavelength_min": 8000.0,
                    "wavelength_max": 12000.0,
                    "values": _spec_str(EPS_GROUND * _PL_gnd),
                },
            },
        },
        "sky": {
            "type": "constant",
            "radiance": {
                "type": "regular",
                "wavelength_min": 8000.0,
                "wavelength_max": 12000.0,
                "values": _spec_str(_PL_sky),
            },
        },
    }


def compute_sphere_mask() -> np.ndarray:
    """
    Render a depth map using Mitsuba's AOV integrator (depth AOV) into an
    hdrfilm, then find sphere pixels via bimodal histogram thresholding.

    The `depth` AOV stores the raw geometric ray distance as a float,
    bypassing any spectral-to-colour conversion — so it works correctly in
    cuda_ad_spectral mode (unlike `depth` integrator + specfilm, which
    corrupts depth values via the wavelength-sampling PDF).

    Output channel layout with pixel_format="rgb":
        arr[:, :, 0:3] = base RGB (not used for masking)
        arr[:, :, 3]   = depth AOV in metres (first intersection distance)

    Sphere pixels form a near cluster; ground pixels form a far cluster.
    We locate the valley between the two histogram modes and threshold there —
    fully data-driven, no hardcoded scene distances.

    Returns: (H, W) float32 mask, 1.0 = sphere pixel, 0.0 = background.
    """
    d = _base_scene()

    # Replace specfilm with hdrfilm — AOV depth channel is a plain float,
    # not integrated over a spectral band.  Box filter gives sharp pixel
    # boundaries for a clean binary mask.
    d["sensor"]["film"] = {
        "type": "hdrfilm",
        "width":  IMG_W,
        "height": IMG_H,
        "pixel_format": "rgb",
        "component_format": "float32",
        "filter": {"type": "box"},
    }

    # AOV integrator: wraps path (max_depth=1 for fast depth-only render)
    # and appends the depth AOV as channel 3 after base RGB.
    d["integrator"] = {
        "type": "aov",
        "aovs": "dd:depth",
        "my_integrator": {"type": "path", "max_depth": 1},
    }

    d["sphere"] = {
        "type": "sphere",
        "center": [0, 0, 0.5],
        "radius": 0.5,
        "bsdf": {"type": "diffuse",
                 "reflectance": {"type": "uniform", "value": 0.5}},
    }

    mask_scene = mi.load_dict(d)
    # 4 SPP is enough — depth is nearly deterministic (no spectral noise)
    img = mi.render(mask_scene, spp=4)
    arr = np.array(img, dtype=np.float64)   # (H, W, N_channels)

    # Depth AOV is the last channel (index 3 for rgb base + 1 aov)
    depth_m = arr[:, :, -1]                  # metres

    # Rays that hit only the sky constant env. return depth ≈ 0 or inf
    valid = (depth_m > 0.01) & np.isfinite(depth_m)

    if valid.sum() == 0:
        raise RuntimeError("AOV depth render returned no valid pixels — check scene geometry.")

    depths_valid = depth_m[valid].ravel()
    d_min, d_max = depths_valid.min(), depths_valid.max()

    # Bimodal histogram: sphere = near peak, ground = far peak.
    # Scan for the histogram minimum in the near half [d_min, midpoint].
    nbins = 256
    hist, edges = np.histogram(depths_valid, bins=nbins, range=(d_min, d_max))
    centers = 0.5 * (edges[:-1] + edges[1:])

    mid_idx = int(np.searchsorted(centers, 0.5 * (d_min + d_max)))
    mid_idx = max(mid_idx, 1)
    valley_idx = int(np.argmin(hist[:mid_idx]))
    threshold  = float(centers[valley_idx])

    mask = (valid & (depth_m < threshold)).astype(np.float32)

    n_sphere = int(mask.sum())
    total    = IMG_H * IMG_W
    print(f"\n  AOV depth sphere mask: {n_sphere}/{total} pixels "
          f"({100*n_sphere/total:.1f}%)  |  depth threshold = {threshold:.3f} m "
          f"(sphere near={d_min:.2f} m, ground far={d_max:.2f} m)")

    return mask


def make_scene_scalar(with_sphere: bool, eps: float = 0.50) -> dict:
    """Scalar-eps scene (for reference and diagnostic renders)."""
    d = _base_scene()
    if with_sphere:
        d["sphere"] = {
            "type": "sphere",
            "center": [0, 0, 0.5],
            "radius": 0.5,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "uniform", "value": float(max(0.0, 1.0 - eps))},
            },
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "regular",
                    "wavelength_min": 8000.0,
                    "wavelength_max": 12000.0,
                    "values": _spec_str(eps * _PL_sph),
                },
            },
        }
    return d


def make_texture_scene(init_eps: float = 0.50) -> dict:
    """
    Sphere scene with bitmap textures for spatially-varying emissivity.

    BSDF reflectance  = bitmap(1 - eps(u,v))     [raw float, no sRGB conversion]
    Emitter radiance  = bitmap(eps(u,v) * B_ref)  [flat-spectrum approximation]

    B_ref = mean Planck(T_sphere) over the 7 LWIR bands ~ 11.1e-3 W/m2/sr/nm.
    The Planck function varies < 10% at 310 K across 8.5-11.5 um, so this is
    accurate within the LWIR band.
    """
    d = _base_scene()

    refl_data = np.full((TEX_H, TEX_W, 1), 1.0 - init_eps, dtype=np.float32)
    emit_data = np.full((TEX_H, TEX_W, 1), init_eps * B_REF,  dtype=np.float32)

    d["sphere"] = {
        "type": "sphere",
        "center": [0, 0, 0.5],
        "radius": 0.5,
        "bsdf": {
            "type": "diffuse",
            "reflectance": {
                "type": "bitmap",
                "bitmap": mi.Bitmap(refl_data),
                "raw": True,
                "filter_type": "bilinear",
                "wrap_mode": "clamp",
            },
        },
        "emitter": {
            "type": "area",
            "radiance": {
                "type": "bitmap",
                "bitmap": mi.Bitmap(emit_data),
                "raw": True,
                "filter_type": "bilinear",
                "wrap_mode": "clamp",
            },
        },
    }
    return d


# ── Visualisation helpers ─────────────────────────────────────────────────────
def brightness_temp_map(arr: np.ndarray, band_idx: int) -> np.ndarray:
    """
    Convert specfilm band output to brightness temperature [K].

    specfilm output: channel_value = integral L(lam) dlam  [W m-2 sr-1]
    Divide by band width -> mean spectral radiance [W m-2 sr-1 nm-1], then
    invert the Planck function: T_B = c2 / (lam * ln(1 + c1/(lam^5 * L)))
    """
    bw    = 2.0 * BAND_HW
    lam_m = BAND_CENTERS[band_idx] * 1e-9
    L     = np.maximum(arr[:, :, band_idx] / bw, 1e-30)
    c2    = H_PLANCK * C_LIGHT / K_BOLTZ           # 0.014388 m K
    arg   = 1e-9 * 2.0 * H_PLANCK * C_LIGHT**2 / (lam_m**5 * L)
    return c2 / (lam_m * np.log1p(arg))


def _bt_lims(*arrays, band_idx: int, margin_K: float = 1.0):
    """Shared BT colour limits across multiple arrays."""
    bt_all = [brightness_temp_map(a, band_idx) for a in arrays]
    return (min(b.min() for b in bt_all) - margin_K,
            max(b.max() for b in bt_all) + margin_K)


def _add_bt_image(ax, arr, band_idx, title, vmin, vmax, fig, cmap="inferno"):
    lam_um = BAND_CENTERS[band_idx] // 1000
    bt = brightness_temp_map(arr, band_idx)
    im = ax.imshow(bt, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{title}\nBT @ {lam_um} um", fontsize=9)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("T_B [K]", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    return im


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    out_dir = "renders/lwir_opt"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print("  LWIR Emissivity TEXTURE Optimisation  (8-12 um)")
    print("=" * 64)
    print(f"  T_sphere={T_SPHERE} K  T_ground={T_GROUND} K  T_sky={T_SKY} K")
    print(f"  eps_ground={EPS_GROUND}  texture={TEX_H}x{TEX_W}  img={IMG_W}x{IMG_H}")
    print(f"  B_ref = {B_REF:.4e} W/m2/sr/nm  (mean Planck sphere over LWIR bands)")

    # Single-bounce analytical optimal eps at 10 um (for reference)
    B_s  = float(planck_nm(10000.0, T_SPHERE))
    B_g  = float(planck_nm(10000.0, T_GROUND))
    L_sk = float(planck_nm(10000.0, T_SKY))
    eps_analytic = EPS_GROUND * (B_g - L_sk) / (B_s - L_sk)
    print(f"\n  Analytical uniform eps* ~ {eps_analytic:.4f} @ 10 um (1-bounce approx)")

    # ── [1] Reference render (no sphere) ─────────────────────────────────────
    print("\n[1/5] Rendering reference (no sphere) ...")
    ref_scene = mi.load_dict(make_scene_scalar(with_sphere=False))
    ref_img   = mi.render(ref_scene, spp=SPP_HQ)
    ref_np    = np.array(ref_img, dtype=np.float32)
    print(f"      shape={ref_np.shape}  mean={ref_np.mean():.3e}  max={ref_np.max():.3e}")

    # ── [2] Initial render (scalar eps=0.5, for comparison) ──────────────────
    print("[2/5] Rendering initial scene (scalar eps=0.50) ...")
    init_np = np.array(
        mi.render(mi.load_dict(make_scene_scalar(with_sphere=True, eps=0.5)), spp=SPP_HQ),
        dtype=np.float32)

    # ── Sphere mask (depth integrator + bimodal threshold) ───────────────────
    # Masking the loss to sphere-visible pixels:
    #   (a) removes the Monte Carlo noise floor from background pixels
    #   (b) eliminates gradient variance from multi-bounce paths that
    #       incidentally connect background pixels to sphere texture cells
    # The expected gradient for eps_texture from background pixels is zero,
    # but the Monte Carlo estimate is not, so masking improves gradient SNR.
    mask_np = compute_sphere_mask()   # (H, W) float32

    # Broadcast (H,W) -> (H,W,N_BANDS) then flatten to a DrJIT Float
    mask_exp = np.broadcast_to(mask_np[:, :, np.newaxis],
                               (IMG_H, IMG_W, N_BANDS)).astype(np.float32)
    mask_dr  = mi.Float(list(mask_exp.reshape(-1)))

    # ── [3] Texture optimisation ──────────────────────────────────────────────
    print("[3/5] Running texture optimisation ...")
    opt_scene = mi.load_dict(make_texture_scene(init_eps=0.5))
    params    = mi.traverse(opt_scene)

    sphere_keys = sorted(k for k in params.keys() if "sphere" in k.lower())
    print(f"      Sphere params: {sphere_keys}")

    # Find the correct data keys for BSDF and emitter bitmap textures
    refl_key = next(k for k in sphere_keys if "reflectance" in k and "data" in k)
    emit_key = next(k for k in sphere_keys if "emitter" in k and "radiance" in k and "data" in k)
    print(f"      BSDF key:    {refl_key}")
    print(f"      Emitter key: {emit_key}")

    ref_dr = mi.TensorXf(ref_np)

    # Optimisation variable: logit-space emissivity texture
    opt = mi.ad.Adam(lr=LR)
    opt['eps_logit'] = mi.TensorXf(np.zeros((TEX_H, TEX_W, 1), dtype=np.float32))

    losses:    list[float] = []
    eps_means: list[float] = []
    tex_snaps: list[tuple] = []   # (iteration, eps_np copy)

    for it in range(N_ITER):
        # eps(u,v) = sigmoid(logit(u,v)) in (0,1) per texture pixel
        eps_tex = 1.0 / (1.0 + dr.exp(-opt['eps_logit']))

        # Kirchhoff: BSDF reflectance = 1 - eps(u,v)
        params[refl_key] = 1.0 - eps_tex

        # Thermal emitter: eps(u,v) * B_ref  (flat-spectrum approx, < 10% error)
        params[emit_key] = float(B_REF) * eps_tex

        params.update()

        # Differentiable render
        img = mi.render(opt_scene, params=params, spp=SPP)

        # Masked L2 loss: only sphere pixels contribute.
        # Zeroing background pixels removes their Monte Carlo noise from the
        # gradient estimate without changing the expected gradient direction.
        diff = (img.array - ref_dr.array) * mask_dr
        loss = dr.dot(diff, diff)

        dr.backward(loss)
        opt.step()

        lv = float(loss[0])
        eps_np = np.array(eps_tex)[:, :, 0]
        ev = float(eps_np.mean())
        losses.append(lv)
        eps_means.append(ev)

        # Save texture snapshots at 0 %, 25 %, 50 %, 75 %, 100 %
        if it in (0, N_ITER // 4, N_ITER // 2, 3 * N_ITER // 4, N_ITER - 1):
            tex_snaps.append((it, eps_np.copy()))

        if it % 15 == 0 or it == N_ITER - 1:
            print(f"      iter {it:4d} | eps_mean={ev:.4f} "
                  f"range=[{eps_np.min():.3f},{eps_np.max():.3f}] | loss={lv:.4e}")

    final_eps_np = np.array(1.0 / (1.0 + dr.exp(-opt['eps_logit'])))[:, :, 0]
    print(f"\n  Converged: eps_mean = {final_eps_np.mean():.4f}  "
          f"range=[{final_eps_np.min():.3f}, {final_eps_np.max():.3f}]")
    print(f"  Analytic uniform eps* = {eps_analytic:.4f}")

    # ── [4] Final high-quality render ─────────────────────────────────────────
    print("[4/5] Rendering final scene ...")
    final_t = mi.TensorXf(final_eps_np[:, :, np.newaxis].astype(np.float32))
    params[refl_key] = 1.0 - final_t
    params[emit_key] = float(B_REF) * final_t
    params.update()
    final_np = np.array(mi.render(opt_scene, spp=SPP_HQ), dtype=np.float32)

    # ── [5] Plots ─────────────────────────────────────────────────────────────
    print("[5/5] Saving plots ...")

    DISP = 3  # display band: 10000 nm = 10 um
    bt_vmin, bt_vmax = _bt_lims(ref_np, init_np, final_np, band_idx=DISP)

    dt_ref   = brightness_temp_map(ref_np,   DISP)
    dt_init  = brightness_temp_map(init_np,  DISP) - dt_ref
    dt_final = brightness_temp_map(final_np, DISP) - dt_ref
    dt_lim   = max(abs(dt_init).max(), abs(dt_final).max(), 0.1)

    n_snaps = len(tex_snaps)
    fig = plt.figure(figsize=(20, 15))
    # Layout: 3 rows
    #   row 0: ref BT | init BT | final BT | dT_init
    #   row 1: loss | eps_mean | dT_final | final eps texture
    #   row 2: texture snapshots (n_snaps panels)
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    # ── Row 0: BT renders ────────────────────────────────────────────────────
    for col, (arr, title) in enumerate(zip(
        [ref_np, init_np, final_np],
        ["Reference (no sphere)", "Initial  eps=0.50 (scalar)", "Optimised (texture)"]
    )):
        ax = fig.add_subplot(gs[0, col])
        _add_bt_image(ax, arr, DISP, title, bt_vmin, bt_vmax, fig)

    ax03 = fig.add_subplot(gs[0, 3])
    im03 = ax03.imshow(dt_init, cmap="RdBu_r", vmin=-dt_lim, vmax=dt_lim,
                       interpolation="nearest")
    # Overlay depth-based mask contour in white
    ax03.contour(mask_np, levels=[0.5], colors="white", linewidths=0.8)
    ax03.set_title(f"DT initial (sphere - ref)\n@ {BAND_CENTERS[DISP]//1000} um\n"
                   f"(white contour = depth mask)", fontsize=9)
    ax03.axis("off")
    cb = fig.colorbar(im03, ax=ax03, fraction=0.046, pad=0.03)
    cb.set_label("DT [K]", fontsize=7); cb.ax.tick_params(labelsize=7)

    # ── Row 1: analysis ──────────────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.semilogy(losses, color="steelblue", lw=1.5)
    ax10.set_xlabel("Iteration"); ax10.set_ylabel("L2 loss")
    ax10.set_title("Optimisation loss"); ax10.grid(alpha=0.3)

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(eps_means, color="darkorange", lw=1.5, label="eps_mean (texture)")
    ax11.axhline(eps_analytic, color="red", ls="--", lw=1.5,
                 label=f"eps* analytic = {eps_analytic:.4f}")
    ax11.set_xlabel("Iteration"); ax11.set_ylabel("Mean emissivity eps")
    ax11.set_title("Mean emissivity evolution\n(Kirchhoff: rho = 1 - eps)")
    ax11.set_ylim(0.0, 1.0); ax11.legend(fontsize=8); ax11.grid(alpha=0.3)

    ax12 = fig.add_subplot(gs[1, 2])
    im12 = ax12.imshow(dt_final, cmap="RdBu_r", vmin=-dt_lim, vmax=dt_lim,
                       interpolation="nearest")
    ax12.set_title(f"DT final (sphere - ref)\n@ {BAND_CENTERS[DISP]//1000} um", fontsize=9)
    ax12.axis("off")
    cb = fig.colorbar(im12, ax=ax12, fraction=0.046, pad=0.03)
    cb.set_label("DT [K]", fontsize=7); cb.ax.tick_params(labelsize=7)

    ax13 = fig.add_subplot(gs[1, 3])
    im13 = ax13.imshow(final_eps_np, cmap="inferno", vmin=0, vmax=1,
                       interpolation="nearest", origin="upper")
    ax13.set_title(f"Final eps texture ({TEX_H}x{TEX_W})\n(sphere UV: u=longitude, v=latitude)",
                   fontsize=9)
    ax13.set_xlabel("u  (longitude 0 -> 2pi)"); ax13.set_ylabel("v  (latitude 0 -> pi)")
    cb = fig.colorbar(im13, ax=ax13, fraction=0.046, pad=0.03)
    cb.set_label("Emissivity eps", fontsize=7); cb.ax.tick_params(labelsize=7)
    ax13.tick_params(labelsize=7)

    # ── Row 2: texture snapshots ──────────────────────────────────────────────
    snap_axes = [fig.add_subplot(gs[2, c]) for c in range(min(4, n_snaps))]
    for ax, (it, tex) in zip(snap_axes, tex_snaps[:4]):
        ax.imshow(tex, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"eps texture @ iter {it}", fontsize=9)
        ax.axis("off")
    # hide unused
    for c in range(len(snap_axes), 4):
        fig.add_subplot(gs[2, c]).axis("off")

    fig.suptitle(
        f"LWIR Emissivity Texture Optimisation  |  "
        f"sphere T={T_SPHERE} K  ground T={T_GROUND} K (eps={EPS_GROUND})  sky T={T_SKY} K\n"
        f"Kirchhoff: eps + rho = 1  |  texture {TEX_H}x{TEX_W}  |  "
        f"eps_mean = {final_eps_np.mean():.4f}  |  analytic eps* = {eps_analytic:.4f}",
        fontsize=11
    )

    out_path = os.path.join(out_dir, "optimization_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Save raw data
    for name, arr in [("ref", ref_np), ("init", init_np), ("final", final_np)]:
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)
    np.save(os.path.join(out_dir, "losses.npy"),    np.array(losses))
    np.save(os.path.join(out_dir, "eps_means.npy"), np.array(eps_means))
    np.save(os.path.join(out_dir, "eps_texture_final.npy"), final_eps_np)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
