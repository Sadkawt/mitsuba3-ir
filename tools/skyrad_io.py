"""
`.skyrad` hyperspectral sky-radiance file format (mitsubaIR fork).

A `.skyrad` file stores a precomputed downwelling-sky spectral radiance table
produced by libRadtran/uvspec, indexed by (zenith, azimuth, wavelength):

    L[i_theta, j_phi, k_lambda]   in   W m^-2 sr^-1 nm^-1

It is consumed by the `spectralsky` environment emitter.  The axes are:

    theta  : zenith angle [degrees], 0 (zenith) -> 90 (horizon), ascending
    phi    : azimuth      [degrees], 0 -> 360 (periodic), relative to solar azimuth
    lambda : wavelength   [nm], ascending (libRadtran reptran spacing is irregular)

Only the sky hemisphere (theta in [0, 90]) is stored; the emitter returns zero
radiance for directions below the horizon.

Binary layout (little-endian)
-----------------------------
    magic    8 bytes   b"SKYRAD01"
    n_theta  uint32
    n_phi    uint32
    n_lambda uint32
    theta    float32[n_theta]    (degrees)
    phi      float32[n_phi]      (degrees)
    lambda   float32[n_lambda]   (nm)
    data     float32[n_theta * n_phi * n_lambda]
             C order: theta outermost, then phi, then lambda;
             units W m^-2 sr^-1 nm^-1

The format is intentionally simple and self-describing so the C++ emitter can
parse it with a plain stream reader, and so it round-trips losslessly through
NumPy here.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"SKYRAD01"


@dataclass
class SkyRadiance:
    """In-memory representation of a `.skyrad` table."""

    theta: np.ndarray    # (n_theta,) zenith angle [deg], ascending
    phi: np.ndarray      # (n_phi,)   azimuth [deg], ascending in [0, 360)
    wavelengths: np.ndarray  # (n_lambda,) [nm], ascending
    data: np.ndarray     # (n_theta, n_phi, n_lambda) [W m^-2 sr^-1 nm^-1]

    def __post_init__(self):
        self.theta = np.ascontiguousarray(self.theta, dtype=np.float32)
        self.phi = np.ascontiguousarray(self.phi, dtype=np.float32)
        self.wavelengths = np.ascontiguousarray(self.wavelengths, dtype=np.float32)
        self.data = np.ascontiguousarray(self.data, dtype=np.float32)
        self._validate()

    def _validate(self):
        n_t, n_p, n_l = len(self.theta), len(self.phi), len(self.wavelengths)
        if self.data.shape != (n_t, n_p, n_l):
            raise ValueError(
                f"data shape {self.data.shape} does not match axes "
                f"(theta={n_t}, phi={n_p}, lambda={n_l})"
            )
        for name, axis in (("theta", self.theta),
                           ("phi", self.phi),
                           ("wavelengths", self.wavelengths)):
            if axis.ndim != 1 or len(axis) < 2:
                raise ValueError(f"{name} must be a 1-D array with >= 2 samples")
            if not np.all(np.diff(axis) > 0):
                raise ValueError(f"{name} must be strictly ascending")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("data contains non-finite values")

    # -- (de)serialization ---------------------------------------------------

    def write(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<III", len(self.theta),
                                len(self.phi), len(self.wavelengths)))
            f.write(self.theta.tobytes())
            f.write(self.phi.tobytes())
            f.write(self.wavelengths.tobytes())
            f.write(self.data.tobytes())

    @classmethod
    def read(cls, path: str | Path) -> "SkyRadiance":
        path = Path(path)
        with path.open("rb") as f:
            magic = f.read(8)
            if magic != MAGIC:
                raise ValueError(
                    f"{path}: bad magic {magic!r}, expected {MAGIC!r}")
            n_t, n_p, n_l = struct.unpack("<III", f.read(12))
            theta = np.frombuffer(f.read(4 * n_t), dtype="<f4").copy()
            phi = np.frombuffer(f.read(4 * n_p), dtype="<f4").copy()
            lam = np.frombuffer(f.read(4 * n_l), dtype="<f4").copy()
            data = np.frombuffer(f.read(4 * n_t * n_p * n_l), dtype="<f4")
            data = data.reshape(n_t, n_p, n_l).copy()
        return cls(theta=theta, phi=phi, wavelengths=lam, data=data)

    # -- Mitsuba interop ------------------------------------------------------

    def to_dict(self, scale: float = 1.0, to_world=None) -> dict:
        """
        Build a `spectralsky` emitter dict for `mitsuba.load_dict`.

        The emitter receives the axis arrays plus the radiance tensor directly,
        so no `.skyrad` file is needed when constructing scenes from Python.
        Arrays are wrapped as the active variant's ``mi.TensorXf`` (the scene
        parser does not accept bare NumPy arrays); this is the only place
        ``skyrad_io`` touches Mitsuba, and the import is deferred to call time.
        """
        import mitsuba as mi

        d = {
            "type": "spectralsky",
            "theta": mi.TensorXf(self.theta.reshape(-1, 1)),
            "phi": mi.TensorXf(self.phi.reshape(-1, 1)),
            "wavelengths": mi.TensorXf(self.wavelengths.reshape(-1, 1)),
            "data": mi.TensorXf(self.data),     # (n_theta, n_phi, n_lambda)
            "scale": float(scale),
        }
        if to_world is not None:
            d["to_world"] = to_world
        return d
