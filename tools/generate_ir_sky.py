#!/usr/bin/env python3
"""
Generate a hyperspectral downwelling-sky radiance table with libRadtran/uvspec
and write it as a `.skyrad` file for the `spectralsky` Mitsuba emitter.

Physics
-------
We compute the *downwelling* spectral radiance reaching the surface from the
sky hemisphere.  In uvspec terms this is `zout 0` (surface level) with
*upward-looking* directions (negative `umu` = -cos(zenith)).  With
`source thermal` the result is the atmosphere's own LWIR emission; for the
8-12 um window the zenith is cold (transparent atmosphere -> deep space) while
the horizon is warm (long, opaque slant path near surface air temperature).

Output grid:  L[theta, phi, lambda]  in  W m^-2 sr^-1 nm^-1

    theta  : zenith angle, 0 (zenith) -> THETA_MAX (near horizon)
    phi    : azimuth, relative to the solar azimuth (irrelevant for pure
             thermal, which is azimuthally symmetric, but kept as a real axis
             so the same pipeline extends to solar-scattered VIS/SWIR later)
    lambda : reptran wavelength grid over [WL_MIN, WL_MAX]

Units
-----
With `mol_abs_param reptran` and `source thermal`, uvspec emits spectral
radiance *per wavenumber*: W m^-2 sr^-1 (cm^-1)^-1.  We convert to the per-nm
SI convention used by the rest of the mitsubaIR pipeline via

    L_lambda = L_nu * |dnu/dlambda| = L_nu * 1e7 / lambda_nm^2

(verified against uvspec's own `output_quantity brightness` to <0.5%: the
inverted Planck brightness temperature matches the converted radiance).  A
Planck-envelope sanity check warns if any sample exceeds the blackbody bound
at the warmest atmospheric temperature (a sign of a unit/parse error).

Usage
-----
    python tools/generate_ir_sky.py --output renders/sky/lwir_clear.skyrad
    python tools/generate_ir_sky.py --reptran fine --wl 8000 12000 \
        --n-theta 19 --n-phi 8 --atmosphere midlatitude_summer
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Import the format module whether run as a script or imported as a module.
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from skyrad_io import SkyRadiance  # noqa: E402

LIBRADTRAN_DIR = Path("~/Code/libRadtran-2.0.6").expanduser()
UVSPEC = LIBRADTRAN_DIR / "bin" / "uvspec"
DATA_DIR = LIBRADTRAN_DIR / "data"

# Built-in atmosphere profiles shipped with libRadtran (data/atmmod/).
ATMOSPHERES = {
    "tropical": "afglt.dat",
    "midlatitude_summer": "afglms.dat",
    "midlatitude_winter": "afglmw.dat",
    "subarctic_summer": "afglss.dat",
    "subarctic_winter": "afglsw.dat",
    "us_standard": "afglus.dat",
}

# Physical constants for the Planck sanity check.
H_PLANCK, C_LIGHT, K_BOLTZ = 6.62607015e-34, 2.99792458e8, 1.380649e-23


def planck_nm(lam_nm: np.ndarray, T: float) -> np.ndarray:
    """Blackbody spectral radiance [W m^-2 sr^-1 nm^-1]."""
    lam = np.asarray(lam_nm, dtype=np.float64) * 1e-9
    expo = np.clip(H_PLANCK * C_LIGHT / (lam * K_BOLTZ * T), None, 700.0)
    return 1e-9 * 2 * H_PLANCK * C_LIGHT**2 / (lam**5 * (np.exp(expo) - 1.0))


# ---------------------------------------------------------------------------
# uvspec I/O
# ---------------------------------------------------------------------------

def make_uvspec_input(theta_deg: np.ndarray, phi_deg: np.ndarray,
                      wl_min: float, wl_max: float, reptran: str,
                      atmosphere_file: str, albedo: float,
                      n_streams: int) -> str:
    # Upward-looking directions: negative cosine of the zenith angle.
    umu = -np.cos(np.radians(theta_deg))
    umu_str = " ".join(f"{u:.6f}" for u in umu)
    phi_str = " ".join(f"{p:.4f}" for p in phi_deg)
    return (
        f"data_files_path {DATA_DIR}/\n"
        f"atmosphere_file {atmosphere_file}\n"
        "source thermal\n"
        f"mol_abs_param reptran {reptran}\n"
        "rte_solver disort\n"
        f"number_of_streams {n_streams}\n"
        f"wavelength {wl_min:.1f} {wl_max:.1f}\n"
        "zout 0\n"
        f"albedo {albedo:.4f}\n"
        f"umu {umu_str}\n"
        f"phi {phi_str}\n"
        "output_user lambda uu\n"
        "quiet\n"
    )


def run_uvspec(inp: str, timeout: float = 1800.0) -> str:
    if not UVSPEC.exists():
        raise FileNotFoundError(f"uvspec binary not found at {UVSPEC}")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".inp", delete=False) as f:
        f.write(inp)
        fname = f.name
    try:
        with open(fname) as stdin:
            result = subprocess.run(
                [str(UVSPEC)], stdin=stdin, capture_output=True,
                text=True, timeout=timeout, cwd=str(LIBRADTRAN_DIR))
    finally:
        Path(fname).unlink(missing_ok=True)

    if result.returncode != 0:
        sys.stderr.write("uvspec stderr:\n" + result.stderr[:4000] + "\n")
        raise RuntimeError(f"uvspec failed (exit {result.returncode})")
    return result.stdout


def parse_output(output: str, n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse `output_user lambda uu` output.

    Each non-empty line is one wavelength:
        lambda  uu[t0,p0] uu[t0,p1] ... uu[t0,p_{P-1}]  uu[t1,p0] ...

    i.e. the radiance block runs umu (theta) in the outer loop and phi in the
    inner loop, matching uvspec's documented `uu` ordering.

    Returns
    -------
    wls  : (n_lambda,)                         [nm]
    data : (n_lambda, n_theta, n_phi)          [W m^-2 sr^-1 (cm^-1)^-1]
    """
    rows = [ln.split() for ln in output.splitlines() if ln.strip()]
    expected = 1 + n_theta * n_phi
    wls, blocks = [], []
    for r in rows:
        if len(r) != expected:
            raise RuntimeError(
                f"expected {expected} columns (1 + {n_theta}*{n_phi}), "
                f"got {len(r)}: {' '.join(r[:6])} ...")
        wls.append(float(r[0]))
        vals = np.asarray(r[1:], dtype=np.float64).reshape(n_theta, n_phi)
        blocks.append(vals)
    if not wls:
        raise RuntimeError("uvspec produced no radiance rows")
    return np.asarray(wls), np.stack(blocks, axis=0)  # (n_lambda, n_theta, n_phi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, help="Output .skyrad path")
    p.add_argument("--wl", type=float, nargs=2, default=[8000.0, 12000.0],
                   metavar=("MIN", "MAX"), help="Wavelength range [nm]")
    p.add_argument("--reptran", choices=["coarse", "medium", "fine"],
                   default="coarse", help="reptran spectral resolution")
    p.add_argument("--atmosphere", choices=sorted(ATMOSPHERES),
                   default="us_standard", help="Built-in atmosphere profile")
    p.add_argument("--atmosphere-file", default=None,
                   help="Explicit atmosphere_file path (overrides --atmosphere)")
    p.add_argument("--n-theta", type=int, default=19,
                   help="Number of zenith samples in [0, THETA_MAX]")
    p.add_argument("--theta-max", type=float, default=89.0,
                   help="Max zenith angle [deg] (< 90 to avoid grazing singularity)")
    p.add_argument("--n-phi", type=int, default=8,
                   help="Number of azimuth samples in [0, 360)")
    p.add_argument("--albedo", type=float, default=0.0,
                   help="Surface albedo (0 => emissivity 1, sky-only downwelling)")
    p.add_argument("--n-streams", type=int, default=16,
                   help="DISORT number_of_streams")
    args = p.parse_args()

    if args.atmosphere_file:
        atm = str(Path(args.atmosphere_file).expanduser())
    else:
        atm = str(DATA_DIR / "atmmod" / ATMOSPHERES[args.atmosphere])

    theta = np.linspace(0.0, args.theta_max, args.n_theta)
    phi = np.linspace(0.0, 360.0, args.n_phi, endpoint=False)

    print(f"libRadtran : {LIBRADTRAN_DIR}")
    print(f"atmosphere : {atm}")
    print(f"reptran    : {args.reptran}   wl {args.wl[0]:.0f}-{args.wl[1]:.0f} nm")
    print(f"grid       : {args.n_theta} theta x {args.n_phi} phi "
          f"(theta 0-{args.theta_max} deg)")
    print("running uvspec ...")

    inp = make_uvspec_input(theta, phi, args.wl[0], args.wl[1], args.reptran,
                            atm, args.albedo, args.n_streams)
    raw = run_uvspec(inp)
    wls, data_nu = parse_output(raw, args.n_theta, args.n_phi)
    print(f"  parsed {len(wls)} wavelengths "
          f"({wls[0]:.1f}-{wls[-1]:.1f} nm)")

    # Reorder to (theta, phi, lambda), then convert per-wavenumber radiance
    # [W m^-2 sr^-1 (cm^-1)^-1] to per-nm [W m^-2 sr^-1 nm^-1].
    data_nu = np.transpose(data_nu, (1, 2, 0))           # (theta, phi, lambda)
    nu_to_nm = (1e7 / wls**2)[None, None, :]             # |dnu/dlambda|
    data_w = data_nu * nu_to_nm

    # --- Planck-envelope sanity check ---------------------------------------
    # Downwelling sky radiance can never exceed a blackbody at the warmest
    # atmospheric temperature.  afgl profiles peak near ~300 K at the surface;
    # allow a small margin for profile/interp slack.
    T_max = 320.0
    bb = planck_nm(wls, T_max)[None, None, :]
    over = data_w > bb * 1.05
    if over.any():
        frac = 100.0 * over.mean()
        worst = float((data_w / bb).max())
        print(f"  WARNING: {frac:.2f}% of samples exceed the {T_max:.0f} K "
              f"Planck bound (max ratio {worst:.2f}). Check units/parse order.",
              file=sys.stderr)
    else:
        print(f"  Planck check OK (all <= {T_max:.0f} K blackbody)")

    zen = data_w[0].mean(axis=0)   # zenith spectrum (avg over phi)
    hor = data_w[-1].mean(axis=0)  # near-horizon spectrum
    print(f"  band-mean radiance [W/m2/sr/nm]: "
          f"zenith={zen.mean():.3e}  horizon={hor.mean():.3e}")
    if hor.mean() <= zen.mean():
        print("  WARNING: horizon not warmer than zenith - check umu sign.",
              file=sys.stderr)

    sky = SkyRadiance(theta=theta, phi=phi, wavelengths=wls, data=data_w)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sky.write(out)
    size_kb = out.stat().st_size / 1024.0
    print(f"wrote {out}  ({sky.data.shape}, {size_kb:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
