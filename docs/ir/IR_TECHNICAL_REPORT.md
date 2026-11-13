# Physically Based Infrared Rendering in Mitsuba 3 — Technical Report

| | |
|---|---|
| **Version** | 0.5 — Phase 3b (scene temperatures; shading-time evaluation with resolution-free thermal shadows) |
| **Date** | 2026-11-12 |
| **Base** | Mitsuba 3, `master` @ `aa01500f` (Embree build fix), Dr.Jit as bundled |
| **Bands** | SWIR (0.9–2.5 µm), MWIR (3–5 µm), LWIR (8–14 µm) |
| **Status legend** | ✅ implemented and tested · 🔧 implemented, validation pending · 📝 planned |

This document is the living specification of the infrared (IR) extension of
Mitsuba 3. It records *what* was added, *why* each modelling choice was made,
and *which assumptions* it rests on. Every assumption is numbered (**A1**,
**A2**, …), stated explicitly, motivated, and referenced. The document is
updated with every change to the code; the changelog is in Appendix A.

---

## Contents

1. [Scope and goals](#1-scope-and-goals)
2. [Radiometric framework](#2-radiometric-framework)
3. [Surface emission model](#3-surface-emission-model)
4. [Implementation reference](#4-implementation-reference)
5. [Spectral sampling, bands and films](#5-spectral-sampling-bands-and-films)
6. [Infrared material data](#6-infrared-material-data)
7. [Helper library and examples](#7-helper-library-and-examples)
8. [Validation](#8-validation)
9. [Limitations and roadmap](#9-limitations-and-roadmap)
10. [References](#10-references)
- [Appendix A — Changelog](#appendix-a--changelog)
- [Appendix B — Build notes](#appendix-b--build-notes)

---

## 1. Scope and goals

### 1.1 Goal

Extend Mitsuba 3 [15, 16] so that it produces *radiometrically meaningful*
images in the three infrared bands used by electro-optical imaging systems:
short-wave (SWIR), mid-wave (MWIR) and long-wave (LWIR) infrared. "Physically
accurate" is understood in the sense of the rendering community: the renderer
solves the (spectral) radiative transfer equation without further
approximation than Monte Carlo noise, given a scene description whose
materials, temperatures and sources are physically specified. The accuracy
of a rendered image is then bounded by the accuracy of the inputs (surface
temperatures, spectral optical constants, atmosphere), and the extension
therefore also provides physically grounded *defaults* and *data* for these
inputs.

### 1.2 Spectral bands

ISO 20473 [14] divides the infrared into near (0.78–3 µm), mid (3–50 µm) and
far (50–1000 µm) infrared. Imaging-system engineering uses a finer split,
dictated by the atmospheric transmission windows and by detector technology
[11 §1.2, 12 ch. 4, 13, 40]:

| Band | Range used here | Typical detector | Dominant signal at ~300 K scene |
|---|---|---|---|
| SWIR | 0.9–2.5 µm (0.9–1.7 µm for InGaAs) | InGaAs, HgCdTe | Reflected sunlight/sky; emission negligible |
| MWIR | 3–5 µm | InSb, HgCdTe, T2SL | Mixed: thermal emission *and* reflected sunlight |
| LWIR | 8–14 µm (7.5–14 µm microbolometer) | HgCdTe, microbolometer | Thermal emission; reflected sunlight negligible |

Everything in this extension is wavelength-generic; the bands only enter
through the sensor response functions (Section 5).

### 1.3 Why visible-light rendering is not enough

Table 1 quantifies the two terms of the radiance leaving a surface at 300 K
with hemispherical reflectance ρ = 0.5 under direct sunlight: the thermal
emission B(λ, 300 K) (per Planck, Section 2.2) and the reflected solar
radiance ρ E_sun(λ)/π, with E_sun the top-of-atmosphere solar spectral
irradiance modelled as a 5772 K blackbody [27] seen under the solid angle
of the solar disc ((R_sun/1 AU)² = 2.163·10⁻⁵). The atmosphere would reduce
E_sun further (Section 9); the numbers are therefore an *upper bound* on the
reflected term.

**Table 1 — Thermal emission vs. reflected sunlight (W m⁻² sr⁻¹ nm⁻¹).**

| λ (µm) | B(300 K) | B(320 K) | B(600 K) | E_sun,TOA (W m⁻² nm⁻¹) | ρE/π (ρ=0.5) | emission / reflection |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9 | 1.5·10⁻¹⁸ | 4.1·10⁻¹⁷ | 5.4·10⁻⁷ | 0.92 | 0.15 | 1·10⁻¹⁷ |
| 1.5 | 2.0·10⁻¹⁰ | 1.5·10⁻⁹ | 1.8·10⁻³ | 0.25 | 4.0·10⁻² | 5·10⁻⁹ |
| 2.2 | 7.9·10⁻⁷ | 3.1·10⁻⁶ | 4.3·10⁻² | 7.5·10⁻² | 1.2·10⁻² | 7·10⁻⁵ |
| 3.0 | 5.6·10⁻⁵ | 1.5·10⁻⁴ | 0.17 | 2.6·10⁻² | 4.1·10⁻³ | 1.4·10⁻² |
| 4.0 | 7.2·10⁻⁴ | 1.5·10⁻³ | 0.29 | 9.1·10⁻³ | 1.5·10⁻³ | 0.50 |
| 5.0 | 2.6·10⁻³ | 4.7·10⁻³ | 0.32 | 4.0·10⁻³ | 6.4·10⁻⁴ | 4.1 |
| 8.0 | 9.1·10⁻³ | 1.3·10⁻² | 0.19 | 6.8·10⁻⁴ | 1.1·10⁻⁴ | 84 |
| 10.0 | 9.9·10⁻³ | 1.3·10⁻² | 0.12 | 2.9·10⁻⁴ | 4.6·10⁻⁵ | 218 |
| 12.0 | 9.0·10⁻³ | 1.2·10⁻² | 7.5·10⁻² | 1.4·10⁻⁴ | 2.2·10⁻⁵ | 400 |
| 14.0 | 7.4·10⁻³ | 9.3·10⁻³ | 4.9·10⁻² | 7.7·10⁻⁵ | 1.2·10⁻⁵ | 606 |

Consequences that drive the design:

1. **Every surface is a light source in the MWIR and LWIR.** A renderer that
   treats emission as a property of a few "light" objects cannot represent
   this; emission must be a property of *all* geometry (Section 3).
2. **Emission and reflection are coupled.** By Kirchhoff's law the emissivity
   is one minus the reflectance (Section 3.1). A material described by a
   BSDF therefore already determines its emissivity; specifying it
   separately would double-count or violate energy conservation. This
   motivates deriving emissivity from the BSDF rather than from a parameter.
3. **The MWIR is genuinely mixed** (ratio ≈ 0.5 at 4 µm): both terms must be
   simulated together, which rules out "thermal-only" shortcuts.
4. **Small temperature contrasts matter.** In the LWIR, dB/dT at 300 K is
   1.6 % per kelvin (4.0 % per kelvin in the MWIR); a camera with a noise
   equivalent temperature difference of 50 mK resolves ≈ 0.1 % radiance
   contrasts. Monte Carlo noise must be well below that, which motivates a
   low-variance estimator for emission (Section 3.2).
5. **Visible-light material data do not extrapolate.** The optical
   constants shipped with Mitsuba (`resources/data/ior`) cover 360–830 nm;
   spectral upsampling from RGB (`srgb` textures) is defined only on that
   range. IR renderings need IR optical constants and emissivity spectra
   (Section 6).

---

## 2. Radiometric framework

### 2.1 Units

> **A1 — Units.** Scene geometry is in metres, wavelengths in nanometres,
> temperatures in kelvin, spectral radiance in W m⁻² sr⁻¹ nm⁻¹.

*Motivation.* Mitsuba is unit-agnostic except where Planck's law introduces
absolute units (its `blackbody` plugin already uses W m⁻² sr⁻¹ nm⁻¹ and
nanometre wavelengths). Keeping nanometres (LWIR: 8 000–14 000 nm) keeps
every existing spectrum plugin, file format (`.spd`) and the spectral film
usable without change. The scene must be modelled in metres because the
emitted radiance is independent of scale only if geometry and radiometry
use the same length unit.

### 2.2 Planck's law

The spectral radiance of a blackbody at temperature T is [1]

$$
B_\lambda(\lambda, T) = \frac{2hc^2}{\lambda^5}\,
\frac{1}{\exp\!\left(\dfrac{hc}{\lambda k_B T}\right) - 1}
\quad [\mathrm{W\,m^{-2}\,sr^{-1}\,m^{-1}}],
$$

multiplied by 10⁻⁹ to obtain a density per nanometre. The implementation
(`src/emitters/thermal.cpp`) uses the exact SI 2019 values [20]
h = 6.626 070 15·10⁻³⁴ J s, c = 299 792 458 m s⁻¹, k_B = 1.380 649·10⁻²³ J K⁻¹,
giving c₁L = 2hc² = 1.191 042 972·10⁻¹⁶ W m² sr⁻¹ and
c₂ = hc/k_B = 1.438 776 877·10⁻² m K. (Mitsuba's `blackbody` spectrum plugin
uses CODATA 2014 values; the relative difference is < 3·10⁻⁶ and is checked
by `test05_matches_blackbody_spectrum`.)

Band-integrated blackbody radiance at 300 K (numerical integration, used as
test references):

| Band | ∫B dλ (W m⁻² sr⁻¹) | Fraction of σT⁴/π |
|---|---:|---:|
| SWIR 0.9–1.7 µm | 3.2·10⁻⁷ | ≈ 0 |
| SWIR 0.9–2.5 µm | 8.7·10⁻⁴ | ≈ 0 |
| MWIR 3–5 µm | 1.866 | 1.28 % |
| LWIR 8–14 µm | 54.93 | 37.6 % |
| LWIR 7.5–13.5 µm | 55.49 | 38.0 % |

Numerical note: the implementation evaluates Planck's law in single
precision. For λ ≥ 900 nm and T ≥ 100 K all intermediate quantities
(λ⁵ ≥ 6·10⁻³², exp(c₂/λT)) stay inside the normal range of `float`; for
colder/shorter combinations `exp` overflows to +∞ and the radiance correctly
evaluates to 0.

### 2.3 Band radiance and sensor response

A detector with spectral response R(λ) (dimensionless, 0–1) integrates

$$
L_{\mathrm{band}} = \int L_\lambda(\lambda)\, R(\lambda)\, d\lambda
\quad [\mathrm{W\,m^{-2}\,sr^{-1}}].
$$

Mitsuba's `specfilm` computes exactly this quantity per channel: wavelengths
are importance-sampled proportionally to the sum of all channel responses,
and each sample's contribution is weighted by R_j(λ)/p(λ) (Section 5). With
R ≡ 1 on a band the output is the band radiance; with the detector's actual
R it is the in-band radiance "seen" by the detector. Conversion to apparent
(brightness) temperature — inverting the band-integrated Planck function —
is a post-process (📝 Phase 4).

---

## 3. Surface emission model

### 3.1 Kirchhoff's law

For a surface element in local thermodynamic equilibrium (LTE) at
temperature T, the directional spectral emissivity equals the directional
spectral absorptivity [2, 3, 5 §3, 6 §3]. For an *opaque* surface,
absorptivity is one minus the directional-hemispherical reflectance ρ_dh,
which is the integral of the BRDF f over the hemisphere [4]:

$$
\varepsilon(\lambda, \omega_o) = 1 - \rho_{dh}(\lambda, \omega_o)
= 1 - \int_{H^2} f(\lambda, \omega_i, \omega_o)\, |\cos\theta_i|\, d\omega_i .
\tag{1}
$$

For a non-opaque interface the integral runs over the full sphere and
includes the transmission lobe; a lossless dielectric interface
(R + T = 1) then has zero emissivity, which is correct: the emission of a
window comes from absorption in its *volume*, not from the interface.

> **A2 — Local thermodynamic equilibrium.** Every surface radiates in LTE
> at a well-defined local temperature T(x), and Kirchhoff's law in its
> directional-spectral form (1) holds.

*Motivation.* LTE holds for condensed matter at terrestrial temperatures
and for the time scales of imaging; it is the standard assumption of
thermal-infrared remote sensing and signature modelling [5, 6, 8, 11, 24].
Violations (luminescence, non-equilibrium plasmas, lasers) are out of scope.

> **A3 — Emissivity from the BSDF.** The emissivity is *not* an independent
> material parameter. It is defined by equation (1) from the BSDF attached
> to the shape, so that a single material description is valid from the
> SWIR (reflective) to the LWIR (emissive).

*Motivation.* (i) Energy conservation: ε + ρ_dh = 1 is enforced by
construction; an independent emissivity could not be guaranteed consistent
with the BSDF. (ii) Directional behaviour: a smooth conductor has
ε(θ) = 1 − F(θ), and a rough surface has the angular emissivity implied by
its microfacet distribution [8, 21]; these fall out of (1) automatically.
(iii) It matches how measured data is used in practice: hemispherical
reflectance spectra of the ASTER/ECOSTRESS libraries [9, 10] are converted
to emissivity through ε = 1 − ρ under a Lambertian assumption — in our
framework one simply sets a `diffuse` reflectance of 1 − ε(λ) (Section 3.4).

> **A4 — Unpolarized transport.** Thermal emission and reflection are
> simulated without polarization state (non-polarized variants).

*Motivation.* Thermal emission is unpolarized at normal incidence and only
partially polarized at grazing angles (the s and p Fresnel reflectances
differ); this affects radiance only for polarization-sensitive sensors.
Mitsuba's polarized variants could be enabled later; the formulation of
Section 3.2 extends to Mueller-matrix transport unchanged because it only
uses BSDF evaluation and sampling. 📝 Phase 5.

> **A5 — BSDF energy loss is interpreted as absorption.** For BSDFs whose
> hemispherical albedo is less than the physical one because of modelling
> approximations (single-scattering microfacet models lose energy at high
> roughness [21, 22]), the missing energy is treated as absorbed and hence
> contributes to emission.

*Motivation.* This is the only choice consistent with (1) and keeps the
renderer energy-conserving. The error is the same one that already affects
visible renderings with these BSDFs; multiple-scattering microfacet models
[22] remove it. The effect is quantified in Section 6 (planned).

### 3.2 The difference formulation

Inserting (1) into the rendering equation [19] for the outgoing radiance of
a thermal surface,

$$
L_o(\omega_o) = \varepsilon(\omega_o)\, B(T) + \int f(\omega_i,\omega_o)\,
L_i(\omega_i)\, |\cos\theta_i|\, d\omega_i
= B(T) + \int f(\omega_i,\omega_o)\, \big(L_i(\omega_i) - B(T)\big)\,
|\cos\theta_i|\, d\omega_i .
\tag{2}
$$

Equation (2) is exact and requires *no* evaluation of the emissivity or of
the BSDF's albedo. It is the surface analogue of the *net-exchange*
formulation of Monte Carlo radiative heat transfer [7, 41], where the
transported quantity is the difference between two blackbody radiances.

**Path-space estimator.** Let a path have vertices x₀ (camera), x₁, x₂, …,
with BSDF sampling weights w_k = f cos θ / p at each scattering vertex and
throughput β_k = ∏_{j<k} w_j. Expanding (2) recursively:

$$
L(x_0 \leftarrow x_1) = B_1 + w_1\,(B_2 - B_1) + w_1 w_2\,(B_3 - B_2) + \dots
= \sum_{k \ge 1} \beta_k\,\big(B_k - B_{k-1}\big), \qquad B_0 := 0,
\tag{3}
$$

where B_k is the blackbody radiance of vertex k, zero for a non-thermal
surface, and where a ray that escapes the scene (or hits a non-thermal
emitter with radiance L_e) contributes β_k (L_e − B_{k−1}). In words: **add
the full blackbody radiance B_k at every thermal vertex, and subtract
B_{k−1} (the previous vertex's blackbody radiance) once for every
continuation ray.** The subtraction is what turns B_k into ε_k B_k in
expectation: E[w_k] = ρ_dh, so E[B_k − w_k B_k] = (1 − ρ_dh) B_k = ε_k B_k.

**Properties.**

* *Unbiased for any BSDF,* as long as the BSDF sampling routine is unbiased
  for its albedo (E[w] = ρ_dh), which is a weaker requirement than being
  able to evaluate ρ_dh.
* *Zero variance in the important special cases.* If w_k equals the albedo
  deterministically — diffuse (w = ρ), smooth conductor (w = F(θ)), smooth
  dielectric (w = 1) — the emission estimate B_k (1 − w_k) is exact per
  sample. In particular, an isothermal enclosure of such materials renders
  with zero variance (`test02_isothermal_enclosure_specular`), and a diffuse
  plate against a cold background renders (1 − ρ)B exactly
  (`test01_diffuse_plate_cold_background`).
* *Variance scales with temperature differences,* not with absolute
  radiance: in a scene where all temperatures lie within ±10 K of 300 K the
  terms B_k − B_{k−1} are at most ±16 % of B in the LWIR, whereas an estimator
  that samples ε_k B_k directly has variance proportional to B². This is
  why the formulation was chosen over the obvious alternative of adding an
  `albedo()` method to every BSDF and multiplying B by a stochastic
  estimate of 1 − albedo.
* *Graceful truncation.* When the path is cut by `max_depth` after vertex k,
  the missing term is β_{k+1}(L(x_{k+1}) − B_k), i.e. the error is
  proportional to how much the unseen radiance differs from B_k. In a
  near-isothermal environment this is nearly zero, whereas the standard
  formulation drops β_{k+1} L(x_{k+1}) entirely. Operationally, the
  subtraction of B_k is *not* performed when no continuation ray is traced.
* *Russian roulette* is unbiased unchanged: the continuation (including its
  −B_k term) is either dropped or reweighted by 1/p_rr as a whole.

**Multiple importance sampling.** The term B_k at a thermal vertex reached
by BSDF sampling is an emitter contribution and is MIS-weighted against
next-event estimation exactly as in Mitsuba's `path` integrator [18];
next-event estimation of a thermal emitter returns B(T) at the sampled
point. The subtracted term −B_{k−1} is *not* an emitter contribution: it is
part of the integrand of the continuation and is therefore always applied
with the plain BSDF sampling weight and no MIS weight.

**Null surfaces.** Shapes with a `null` BSDF are not path vertices. The
subtraction belongs to the first surface encountered by the continuation
ray — null or not — or to the ray escaping; it is applied exactly once per
scattering event (tracked by a `diff_pending` flag).

> **A6 — Surfaces without a thermal emitter are at 0 K.** They reflect and
> transmit but do not emit.

*Motivation.* This coincides with Mitsuba's existing semantics (an
un-tagged surface emits nothing), keeps visible-light scenes unchanged, and
makes the user's intent explicit: an IR scene must assign a temperature to
every surface, for which a Python helper will be provided (Section 6).
The alternative — treating un-tagged surfaces as being at the temperature
of the previous vertex — would silently invent emission.

> **A7 — Thermal emitters report blackbody radiance.** The `thermal`
> emitter's `eval()` returns B(λ, T(x)); the emissivity is applied by
> thermal-aware integrators through (3).

*Motivation.* This keeps the emitter plugin free of integrator logic and
makes next-event estimation trivially correct. The cost is that integrators
that are not thermal-aware (`path`, `volpath`, `direct`, …) render thermal
surfaces as blackbodies; the `Scene` constructor therefore logs a warning
whenever thermal emitters are used with such an integrator
(`Integrator::thermal_aware()`), and `test01` documents the behaviour.

### 3.3 Temperature fields

The surface temperature is a texture (`temperature` parameter, kelvin), so
it can be constant, painted (bitmap), procedural, or a mesh attribute
(`mesh_attribute` texture) produced by an external thermal solver — the
standard coupling used by signature-prediction tools [25, 42]. Textures are
evaluated with `eval_1()`, i.e. as a scalar field independent of
wavelength.

`twosided` defaults to **true** for thermal emitters (unlike `area`):
physically both faces of a thin surface at temperature T radiate, and for
closed meshes the inside is never seen, so the default is safe and avoids
black back-faces when a camera is placed inside a cavity.

### 3.4 Using measured emissivity spectra

Spectral libraries [9, 10] publish directional-hemispherical reflectance
ρ(λ) (ECOSTRESS) or emissivity ε(λ) = 1 − ρ(λ) measured with an integrating
sphere, i.e. under the assumption that the surface is Lambertian or that
only the hemispherical quantity is needed [8]. In this framework such data
is used by attaching

```python
'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'irregular',
          'wavelengths': lam_nm, 'values': 1 - eps}},
'emitter': {'type': 'thermal', 'temperature': T}
```

The emissivity then follows exactly: ε(λ) = 1 − (1 − ε(λ)). A loader for
the ECOSTRESS ASCII format is planned (Section 6). For surfaces that are
not Lambertian in the IR (metals, glass, water, smooth paints), the correct
approach is to use the IR optical constants with a (rough) conductor or
dielectric BSDF; Phase 2 adds those data sets.

---

## 4. Implementation reference

### 4.1 Files

| File | Change | Status |
|---|---|---|
| `include/mitsuba/render/emitter.h` | New flag `EmitterFlags::Thermal = 0x100` | ✅ |
| `include/mitsuba/render/integrator.h` | New virtual `Integrator::thermal_aware()` (default `false`) | ✅ |
| `src/render/scene.cpp` | Warning when thermal emitters meet a non-thermal-aware integrator | ✅ |
| `src/render/python/emitter.cpp` | Python binding of `EmitterFlags.Thermal` | ✅ |
| `src/emitters/thermal.cpp` | New `thermal` emitter plugin | ✅ |
| `src/integrators/path_ir.cpp` | New `path_ir` integrator (difference formulation) | ✅ |
| `src/emitters/CMakeLists.txt`, `src/integrators/CMakeLists.txt` | Plugin registration | ✅ |
| `src/emitters/tests/test_thermal.py` | Unit tests of the emitter | ✅ |
| `src/integrators/tests/test_path_ir.py` | Analytic validation of the integrator | ✅ |
| `docs/ir/IR_TECHNICAL_REPORT.md` | This document | ✅ |
| `src/python/python/ir/{__init__,radiometry,materials,scene}.py` | `mitsuba.ir` helper package (Section 7.1) | ✅ |
| `resources/ir_data/ior/*_ir.{eta,k}.spd`, `make_ior_ir.py`, `README_ir.md` | IR optical constants and generator (Section 6) | ✅ |
| `src/python_tests/test_ir.py` | Tests of the helper package and data | ✅ |
| `examples/ir/` | Nine example scripts and README (Section 7.2) | ✅ |
| `include/mitsuba/render/thermal.h` | Shared Planck function and constants | ✅ |
| `include/mitsuba/render/medium.h`, `src/render/medium.cpp`, `src/render/python/medium_v.cpp` | `temperature` volume and `is_thermal()` on all media | ✅ |
| `src/integrators/volpath_ir.cpp` | Volumetric IR path tracer (Section 6.5) | ✅ |
| `src/emitters/thermalsky.cpp` | Clear-sky thermal emitter (Section 6.6) | ✅ |
| `src/python/python/ir/atmosphere.py` | Sky emissivity, air media, HITRAN helper (Sections 6.6, 6.7) | ✅ |
| `src/integrators/tests/test_volpath_ir.py`, `src/emitters/tests/test_thermalsky.py` | Tests | ✅ |

### 4.2 `thermal` emitter

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `temperature` | float or texture | 300 | Surface temperature [K]; spatially varying through any texture |
| `twosided` | bool | true | Emit from both faces |
| `wavelength_min`, `wavelength_max` | float | 360, 14000 | Range for *emitter-side* wavelength sampling (only used by light tracing, e.g. `ptracer`); camera-based integrators sample wavelengths from the film's response functions |
| `sampling_weight` | float | 1 | Inherited: relative probability of next-event estimation picking this emitter |

Requires a spectral variant. `eval`, `sample_direction`, `eval_direction`
return B(λ, T(x)) (W m⁻² sr⁻¹ nm⁻¹); `sample_ray` and `sample_wavelengths`
sample uniformly in the wavelength range (the Planck spectrum of a textured
temperature field has no cheap inverse CDF; uniform sampling is unbiased
and the range is explicit).

### 4.3 `path_ir` integrator

Parameters are those of `path` (`max_depth`, `rr_depth`, `clamp_direct`,
`clamp_indirect`). The loop state of the path tracer is extended by
`B_prev` (the blackbody radiance of the last scattering vertex, per
wavelength) and `diff_pending` (whether its subtraction is still owed by
the current continuation ray). Clamping applies only to emitted (positive)
contributions, never to the difference terms, so that clamping cannot bias
the mean of a near-isothermal scene upward.

Non-thermal emitters (sun, sky, lamps, `area`, `envmap`, …) are handled
exactly as in `path`, so mixed scenes work (`test07_thermal_plus_solar_mixed`).
Participating media are not handled (see `volpath_ir`, Section 6.5).

### 4.4 Interaction with the rest of Mitsuba

* **Next-event estimation** uses `Scene::sample_emitter_direction`, which
  picks an emitter proportionally to `sampling_weight` (uniform by
  default). In an LWIR scene *every* shape is an emitter, so a small hot
  object or the sun should be given a larger `sampling_weight`; an
  automatic power-based weighting is a candidate improvement (📝).
* **Differentiable rendering**: the integrator keeps the detached-sampling
  logic of `path`; the temperature texture is exposed as a
  non-differentiable parameter for now (📝 differentiable temperature for
  inverse thermography).
* **Hidden emitters / visibility** are honoured unchanged: a thermal emitter
  hidden from a ray class has B = 0 for that class.

---

## 5. Spectral sampling, bands and films

Mitsuba's spectral variants trace *four* wavelengths per path (hero
wavelength sampling [17]). The wavelengths are drawn by the sensor: if the
film defines sensor response functions (`specfilm`), they are importance
sampled from the sum of the responses; otherwise from a CIE-based
distribution over 360–830 nm. **An IR scene must therefore use `specfilm`
with response functions that cover the band of interest**; a `hdrfilm`
would (a) sample visible wavelengths and (b) convert to XYZ, giving zero
for IR radiance.

A single-band film with an ideal response over 8–14 µm:

```python
'film': {
    'type': 'specfilm', 'width': 640, 'height': 512,
    'lwir': {'type': 'regular', 'wavelength_min': 8000, 'wavelength_max': 14000,
             'values': '1, 1'},
}
```

The channel value is ∫ L_λ R(λ) dλ in W m⁻² sr⁻¹ (Section 2.3). Several
bands can be rendered at once by adding several response spectra; the
sampler then covers their union. Detector response curves for typical
InGaAs, InSb/HgCdTe and microbolometer cameras, and a helper that builds
these film dictionaries, are planned (📝 Phase 4).

> **A8 — Hero-wavelength sampling across the band.** The four path
> wavelengths are drawn from the film response distribution; all spectral
> quantities (Planck, optical constants) are evaluated per wavelength.

*Motivation.* This is Mitsuba's native mechanism and is unbiased for any
response function. Spectral variance is low for smooth integrands (Planck
over an 8–14 µm band varies by ≈ 1.3×); it becomes relevant for
line-structured atmospheric transmission (📝 Phase 3), where a spectral
resolution choice will be documented.

Things that silently do **not** work in the IR and are flagged here:

| Component | Problem | Remedy |
|---|---|---|
| `rgb`/`srgb` textures and reflectances | Spectral upsampling is defined on 360–830 nm only | Use `uniform`, `regular`, `irregular` spectra or `.spd` files (`mitsuba.ir.validate` warns) |
| `hdrfilm` | Converts to XYZ; zero outside the visible | Use `specfilm` |
| `blackbody` spectrum plugin | Masks wavelengths outside its `wavelength_min/max` (default 360–830 nm) | Set the range, or use `thermal` |
| Conductor material presets (`material="Al"`, …) | Data cover 360–830 nm only | IR optical constants (Section 6) |
| `sunsky`/`envmap` | Visible-light models | `mitsuba.ir.sun` and the `thermalsky` emitter (Sections 6.4, 6.6) |

---

## 6. Infrared material data

Visible-light material descriptions do not extrapolate into the infrared
(Section 1.3). Phase 2 adds optical constants for the materials that
dominate IR scenes and the means to use measured emissivity spectra.
All data files live in `resources/ir_data/ior/` with the suffix `_ir`
(provenance table: `resources/ir_data/ior/README_ir.md`, generator:
`make_ior_ir.py`) and plug into every conductor BSDF through the existing
preset mechanism, e.g. `{'type': 'roughconductor', 'material': 'Al_ir', 'alpha': 0.2}`.

### 6.1 Metals: Lorentz–Drude model

> **A9 — Metals follow the Lorentz–Drude model of Rakić et al. (1998).**
> The complex permittivity is ε(ω) = 1 − f₀ω_p²/(ω² + iΓ₀ω) + Σ_j f_j ω_p² /
> ((ω_j² − ω²) − iΓ_jω), with the published parameters for Ag, Al, Au, Be,
> Cr, Cu, Ni, Pd, Pt, Ti and W [28]. The model is evaluated analytically
> (`mitsuba.ir.drude_lorentz_nk`) and tabulated from 0.3 to 20 µm.

*Motivation.* (i) The LD model is a physically constrained fit to measured
optical constants over 0.1–6 eV (0.2–12.4 µm) and is the standard
representation of metals in optics codes; (ii) in the far infrared a metal is
a Drude conductor, so extrapolating the fit beyond 12.4 µm to 14–20 µm is
the physically expected behaviour (the Hagen–Rubens limit, ε_n ∝ √(ν/σ)
[37]); (iii) the parameters were **verified** against the independent
tabulation of the same model in the refractiveindex.info database [43]:
maximum relative deviation of n and k over 0.3–12.4 µm is 1.3·10⁻⁴ for all
eleven metals. The implied normal emissivities at 10 µm (Al 1.2 %, Au 1.3 %,
Ag 1.1 %, Cu 0.8 %, Fe 2.4 %, Ti 7.3 %) agree with handbook values for
polished metals [5, 11].

*Caveat (stated to users in the data files).* These are **bulk, clean,
polished** metals. Real metal surfaces carry oxide layers, contamination and
roughness that raise the emissivity by up to an order of magnitude (oxidised
steel 0.6–0.9 [11]). Roughness is modelled with `roughconductor`; oxide
layers and paints must be described by their own (dielectric or measured)
material. Iron is not in [28]; `Fe_ir` uses the tabulation of Querry [44].

### 6.2 Dielectrics, windows, water

> **A10 — Smooth absorbing dielectrics are opaque Fresnel interfaces.**
> Water, glass in the LWIR, sapphire beyond 6 µm, etc. are rendered with the
> conductor BSDFs, which implement the exact Fresnel equations for a complex
> index n + ik, i.e. as opaque bodies whose emissivity is 1 − F(θ).

*Motivation.* The penetration depth λ/(4πk) is 16 µm for water at 10 µm
(k = 0.051 [30]) and 1.4 µm for fused silica at 10 µm (k = 0.55 [47]):
anything thicker than a few tens of micrometres is opaque and the interface
alone determines emission and reflection. **Validity limit:** in the SWIR
water has k ≈ 10⁻⁴ (penetration ≈ 1 mm) and glass k ≈ 10⁻⁶; there, thin
layers transmit and the opaque model is wrong — a window/medium model must
be used instead (`window_ior`, and Phase 3 media).

> **A11 — IR windows use a scalar, band-averaged refractive index.** For
> transparent materials (Ge, Si, ZnSe, ZnS, CaF₂, BaF₂, NaCl, sapphire
> < 6 µm) Mitsuba's `dielectric`/`roughdielectric` BSDFs take a scalar IOR;
> `mitsuba.ir.window_ior(material, band)` returns the Planck-weighted band
> average of n (and of k, so that the caller can check transparency).

*Motivation.* Dispersion within one IR band is small for these materials
(Ge: 4.03 → 4.00 over 8–14 µm; ZnSe: 2.41 → 2.38), so the error of a scalar
IOR on Fresnel reflectance is < 0.3 %. Chromatic refraction effects are
lost, which is irrelevant for radiometric imaging.

Sources (all converted from the public-domain refractiveindex.info database
[43]; the `.spd` headers carry the full citation): Si [46], Ge [48, 49],
ZnSe/ZnS/BaF₂/NaCl [45], sapphire and Fe [44], CaF₂ [50], fused silica
[47], water [30]. Sources that provide only n are transparent in their
stated range; their k is stored as 10⁻⁸ (Mitsuba rejects all-zero spectra),
which is radiometrically indistinguishable from zero.

### 6.3 Measured emissivity spectra (ECOSTRESS / ASTER)

> **A12 — Spectral-library materials are Lambertian.** A hemispherical
> reflectance spectrum ρ(λ) from the ECOSTRESS [9] or ASTER [10] libraries
> is used as the reflectance of a `diffuse` BSDF; Kirchhoff's law then gives
> exactly the measured emissivity ε(λ) = 1 − ρ(λ) (`load_ecostress`,
> `diffuse_from_emissivity`, `diffuse_from_reflectance`).

*Motivation.* The libraries measure directional-hemispherical reflectance
(10° incidence) with an integrating sphere; the Lambertian interpretation is
the one used to derive emissivity in the libraries themselves and in land
surface temperature retrieval [8, 9]. The neglected angular variation of
emissivity is a few percent up to 60° for most natural surfaces and larger
for water and smooth man-made surfaces [8], for which the Fresnel models of
Section 6.2 are preferable. Library files are not redistributed (download
from https://speclib.jpl.nasa.gov); the loader handles the ASCII format.

### 6.4 Solar source

> **A13 — The sun is a 5772 K blackbody of 1361 W m⁻² total irradiance.**
> `mitsuba.ir.sun` builds a `directional` emitter with
> E(λ) = τ(λ)·π B(λ, 5772 K)(R_sun/AU)², where τ(λ) is an optional
> user-supplied atmospheric transmittance table.

*Motivation.* The IAU nominal solar values [27] give the correct total
irradiance by construction; the blackbody deviates from the measured
extraterrestrial spectrum [26] by up to ≈ 10 % in parts of the SWIR
(Fraunhofer and molecular bands), which can be corrected by passing a
measured spectrum as τ(λ)·E. Atmospheric transmission (H₂O, CO₂ bands) is
essential in the MWIR/SWIR; Section 6.7 provides a HITRAN line-by-line
transmittance for τ(λ). Without it the helper produces top-of-atmosphere
sunlight.

### 6.5 Emissive participating media (`volpath_ir`)

> **A14 — Media are in local thermodynamic equilibrium and emit
> σ_a B(λ, T_m).** A medium with a ``temperature`` field radiates
> isotropically with its absorption coefficient; Kirchhoff's law for volumes.

*Motivation.* This is the standard source term of the radiative transfer
equation for gases and aerosols at terrestrial temperatures [5 §11, 6 §10];
it is what makes the atmosphere both attenuate and *add* radiance ("path
radiance") in the infrared. Mitsuba's ``volpath`` ignores emission; the new
``volpath_ir`` integrator adds it. The `temperature` is a `Volume`, so
heterogeneous fields (exhaust plumes from CFD) are supported through the
`heterogeneous` medium.

**Difference formulation for media.** Along a ray through a medium,

$$
L = \int_0^d T(s)\,\sigma_a B(T_m)\,ds + \int_0^d T(s)\,\sigma_s L_s\,ds + T(d)\,L(y).
$$

Because the phase function is normalised, the in-scattered radiance can be
written as L_s = B(T_m) + ∫ p (L_i − B(T_m)) dω — exactly the surface
identity (2) with the BSDF replaced by a lossless phase function. The
absorption and scattering branches of a real collision therefore combine into
a single rule: **at a real collision add β (B(T_m) − B_prev), multiply β by
the single-scattering albedo, and set B_prev := B(T_m)**; the pending
subtraction of B_prev is settled at the first event of every continuation
(real collision, surface hit, or escape), never at null collisions. The
estimator is unbiased (Section 8, `test02`–`test08`), has zero variance in an
isothermal medium (every collision contributes zero; `test03`), and needs
neither σ_a at collisions nor a separate emission-sampling strategy. With
spectrally varying extinction, the thermal term uses the real-collision
weight σ_t / p_real, i.e. volpath's hero-wavelength scattering weight with
the albedo removed, which keeps the spectral MIS of `volpath` intact.

Media without a temperature are at 0 K (consistent with A6): they attenuate
and scatter but do not emit. Since the atmosphere always emits in the
MWIR/LWIR, an air medium must be given its air temperature
(`mitsuba.ir.air_medium`).

### 6.6 Clear-sky thermal radiance (`thermalsky`)

> **A15 — The clear sky is a plane-parallel isothermal slab at the
> near-surface air temperature with zenith emissivity ε_z(λ).** Its
> downwelling radiance is L(θ, λ) = ε(θ, λ) B(λ, T_air) with
> ε(θ) = 1 − (1 − ε_z)^{sec θ}, i.e. optical depth τ_z sec θ; below the
> horizon the emitter returns B(T_air).

*Motivation.* The sec θ air-mass scaling is the plane-parallel
approximation that underlies both the measured angular distribution of
clear-sky longwave radiance (Unsworth & Monteith 1975 [33]: radiance rises
towards the horizon and approaches blackbody emission at the air
temperature) and operational models [11 §1, 12]. Using the screen-level air
temperature as the slab temperature slightly overestimates the window
radiance for the upper atmosphere, which is colder; this is partly absorbed
into the empirical ε_z. For the zenith emissivity, `mitsuba.ir.sky_emissivity_idso`
implements Idso's (1981) [32] clear-sky formulas (full spectrum, 8–14 µm,
10.5–12.5 µm) in terms of air temperature and vapour pressure (Magnus
formula, WMO No. 8). Typical result: T_air = 293 K, RH 60 % → ε_z(8–14 µm)
= 0.40, apparent zenith temperature 245 K, horizon 293 K
(`examples/ir/08_sky.py`). A spectral ε_z(λ) table (e.g. 1 − τ from a
MODTRAN run) can be supplied instead of the band constant; clouds and
aerosol are not modelled.

### 6.7 Molecular transmittance from HITRAN

> **A16 — Line-by-line molecular absorption (H₂O, CO₂, …) from HITRAN plus
> the empirical Roberts–Selby–Biberman water-vapour continuum; no aerosols,
> no O₃.** `mitsuba.ir.hitran_absorption` computes σ_a(λ) of humid air with
> the HITRAN Application Programming Interface [51] (Voigt profiles, ±25
> cm⁻¹ line wings, TIPS partition sums, 0.05 cm⁻¹ grid) and adds, for
> 8–14 µm, the continuum of Roberts et al. (1976) [55]: k = C_s(ν,T) ρ_w
> (e + 0.002 (P − e)), C_s = [4.18 + 5578 e^{−0.00787 ν}] e^{1800(1/T − 1/296)}
> cm² g⁻¹ atm⁻¹. Both feed `air_medium` and the `sun` transmittance.

*Motivation.* HITRAN [35] is the reference spectroscopic database and
`hapi` its official line-by-line tool. Lines alone reproduce the band
structure that bounds the atmospheric windows — for a 1 km sea-level path
at 288 K, RH 50 % (6.4 g m⁻³): τ = 0.02 at 6.5–7.0 µm and 0.53 at 7.5–8.0 µm
(the H₂O ν₂ wing that forms the short-wave edge of the LWIR window), 0.94
across 9–13 µm, 0.32 at 13.5–14 µm and 0 beyond 14.5 µm (CO₂ ν₂ band); in
the MWIR τ = 0.04 in the CO₂ ν₃ band at 4.2–4.4 µm and 0.998 at 3.5–4.1 µm.
What lines do *not* give is the smooth absorption of the window itself: the
water-vapour continuum, which scales with e_a² and rises towards 13 µm. The
RSB fit [55] (the LOWTRAN continuum) supplies it with ≈ 20 % accuracy —
0.035, 0.049 and 0.095 km⁻¹ at 8, 10 and 12 µm for the conditions above,
lowering the 9–13 µm transmittance from 0.94 to 0.88 — and was chosen over
MT_CKD [52] because its closed form needs no external tables; MT_CKD is the
modern reference and a user-supplied σ_a(λ) table replaces the fit
transparently. Grid-step check: 0.25 → 0.02 cm⁻¹ changes 1 km band averages
by < 0.2 %, so 0.05 cm⁻¹ is adequate. **Not included:** aerosol extinction,
ozone (9.6 µm, minor for horizontal sea-level paths), CH₄/N₂O beyond their
fixed mixing ratios, and any vertical structure; slant paths through the
whole atmosphere require a layered model (MODTRAN-type), whose output can be
passed to `air_medium`/`sun`.

### 6.8 Scene temperatures from a surface energy balance (`mitsuba.ir.thermal`)

An infrared image is only as good as the temperatures it is rendered from.
`mitsuba.ir.thermal` computes them for outdoor scenes from a single
parametrised weather description, including the shadows cast by the scene
geometry. The approach is the one used by signature-prediction codes
(PRISM, MuSES [42], DIRSIG's thermal model [25]) and in the thermal
remote-sensing literature [24, 56]: every surface sample point owns a
one-dimensional transient conduction problem whose surface boundary
condition is the net radiative-convective flux.

> **A17 — One-dimensional conduction per surface element.** Each sampled
> point of a shape (a UV-grid texel, or a vertex for meshes without UVs)
> is a column of finite volumes in depth: semi-infinite (ground, modelled
> to 1 m with geometrically growing cells and a fixed deep temperature
> equal to the daily mean air temperature) or a slab of given thickness
> (walls, sheet metal, leaves) whose back is adiabatic, exposed to air, or
> coupled to an interior temperature through a conductance. Lateral
> conduction between elements is neglected.

*Motivation.* For the diurnal cycle the thermal penetration depth
δ = √(2a/ω) is ≈ 0.1 m for asphalt and concrete and ≈ 0.05 m for dry soil
[57], much smaller than the lateral extent of the features that produce
IR contrast (shadows, objects), so heat flows essentially normal to the
surface; this is the standard assumption of terrain temperature models
[56, 58]. Thin metal skins are nearly isothermal through their thickness
and are treated as lumped slabs (two nodes). The solver is backward Euler
with the T⁴ term linearised about the previous step (radiative coefficient
4εσT³) and a vectorised Thomas algorithm; it is unconditionally stable, so
5-minute steps are used. Validation (`test_ir_thermal.py`): steady-state
energy balance reproduced to 10⁻³ K; the Carslaw–Jaeger amplitude decay
e^{−z/δ} of a sinusoidally forced semi-infinite solid reproduced to 1 % at
three depths.

> **A18 — Surface energy balance.** The surface node receives
> q = α_s [E_dn cos θ_i V + F_sky E_dh + (1 − F_sky) ρ_g E_gh] + ε [F_sky L_sky
> + (1 − F_sky) σT_sur⁴ − σT_s⁴] + h (T_air − T_s) + q_int, where V ∈ {0, 1} is
> the solar visibility of the element (shadow ray through the scene),
> F_sky the cosine-weighted sky-view factor (stratified hemisphere rays
> through the scene), α_s the broadband solar absorptance, ε the broadband
> longwave emissivity, h = 5.7 + 3.8 u (McAdams/Jürges [59]), T_sur the
> temperature of the unresolved surroundings (default: air temperature)
> and q_int an optional internal flux.

*Motivation.* The radiative terms are the standard decomposition of
incident shortwave into direct, sky-diffuse and ground-reflected parts and
of longwave into sky and terrain parts weighted by the sky-view factor
[24, 56, 60]; obtaining V and F_sky by ray casting through the actual
scene (Section 6.8.1) is what produces geometrically correct *thermal
shadows*. Multiple shortwave reflections and the longwave exchange between
resolved scene surfaces are linearised to an environment at T_air; this is
the usual first-order treatment [56] and errs mostly for deep courtyards and
under dense canopies. Evapotranspiration (vegetation, wet soil) and
precipitation are not modelled; `MATERIALS['grass']` and
`['vegetation_canopy']` therefore run warmer than real transpiring
vegetation in the afternoon. Broadband α_s and ε are material parameters
(table in `heat.MATERIALS`, values after Oke [57], Incropera [61] and
Jacobs [56]); the *spectral* emissivity used for rendering still comes from
the BSDF (A3) — the user is responsible for choosing consistent pairs
(e.g. a paint BSDF with ε ≈ 0.9 for `steel_sheet_painted`).

> **A19 — One parametrised weather model with a periodic diurnal cycle.**
> `Weather(date, latitude, longitude, utc_offset, T_min, T_max,
> relative_humidity, wind_speed, cloud_cover, aod500, pressure, ground_albedo)`
> provides: solar position by Michalsky's algorithm [62]; clear-sky direct
> and diffuse irradiance by Bird & Hulstrom [63] with precipitable water
> from the surface humidity (Leckner [64]); cloud correction by Kasten &
> Czeplak [65]; air temperature by Parton & Logan [66] (minimum at sunrise,
> maximum 1.86 h after the sinusoidal peak, exponential night cooling);
> sky longwave by Idso [32] with Crawford & Duchon's cloud correction [67].
> The same day is repeated for the spin-up (3 days by default).

*Motivation.* These are the simplest published models that capture the
first-order behaviour of each quantity with a handful of physically named
parameters, each validated against measurements in its source; they are
the models behind many engineering tools (Bird: NREL; Parton–Logan: crop
and soil models; Kasten–Czeplak: solar-energy practice). Validation in
`test_ir_thermal.py`: solar elevation at the solstices to 0.15°, azimuth
to 1°; Bird clear-sky noon values (DNI 924, DHI 115, GHI 1039 W m⁻² for
AOD 0.1, W = 1.5 cm); sunrise/sunset of Stockholm in July to 0.5 h.
Periodic forcing is a modelling choice: the solver needs a spun-up state,
and a realistic multi-day sequence can be obtained by chaining `Weather`
objects (planned).

> **A20 — Internal heat sources.** A per-material internal flux
> `q_internal` [W m⁻²] at the back node and/or an interior temperature
> `T_internal` coupled through `U_internal` [W m⁻² K⁻¹] represent engines,
> heated buildings, electronics.

*Motivation.* Both forms occur in practice (a known dissipated power per
area; a known interior temperature behind a skin of known thermal
resistance); they are validated against the closed-form equilibrium in
`test04` (q = 200 W m⁻² and U → ∞).

#### 6.8.1 Shadows and sky view from the scene

`ThermalScene` loads the user's scene (sensors, integrators and emitters
stripped) and samples each simulated shape: analytic shapes (rectangle,
sphere, disk, cylinder) on a UV grid through `Shape::eval_parameterization`,
meshes on a UV grid when a rasterised overlap test shows their UV map to
be unique (e.g. `rectangle`, which is a 4-vertex mesh in Mitsuba), and at
their vertices otherwise (Mitsuba's `cube` and many imported models map
several faces onto the same UV region, which a texture cannot represent;
`mesh_mode` overrides the choice). Note that Mitsuba assigns a *black* BSDF
to a shape that has an emitter but no BSDF, so a `thermal` shape without a
BSDF renders as a blackbody (ε = 1). The vertex
density of a mesh therefore sets its thermal resolution. For every 5-minute step with the sun above the horizon a shadow
ray is cast from each element towards the sun (`Scene::ray_test`,
vectorised); elements facing away or occluded receive no direct
irradiance. The sky-view factor F_sky is the cosine-weighted fraction of
128 stratified (Hammersley) hemisphere directions, randomly rotated per
element, that reach the sky. Objects not simulated (no material given)
still cast shadows. Cost: 33 000 elements × 3 days at 5-minute steps run
in ≈ 5 s on 12 cores (`examples/ir/10_thermal_solver.py`).

#### 6.8.2 Output (texture mode)

`ThermalResult.temperature(key, t)` returns the surface temperature field
of a shape at an output time; `texture(key, t)` wraps a UV-sampled field as
a raw float bitmap texture; `write_mesh(key, t, file)` exports a
vertex-sampled mesh to PLY (world coordinates) with a `vertex_temperature`
attribute; `apply(scene_dict, t)` returns a copy of the scene in which each
simulated shape carries a `thermal` emitter with its texture or mesh
attribute, ready for `path_ir`/`volpath_ir` (Section 7.2, example 10). Example 10 shows the
expected behaviour for Stockholm on 15 July: open asphalt 16 °C at 05:00 →
49 °C at 14:00 → 22 °C at 23:00 (air 14–26 °C), asphalt 0.8 m west of a
wall shaded in the morning lagging the open ground by 16 K at 11:00 and
still by 4 K at 14:00 after the shadow has moved (the thermal shadow), a 2
mm steel box over a 60 °C compartment staying at 30–40 °C through the
night, and a leaf canopy following the air temperature closely.

#### 6.8.3 Shading-time evaluation: resolution-free thermal shadows

The texture mode above resolves thermal shadows only to its texel size and
its cost grows with texels × time steps regardless of what the camera sees.
The default for rendering is therefore a second formulation that evaluates
the surface temperature *at each shading point inside the integrator*.

> **A21 — Linear response of the column.** With the radiative term
> linearised, the column is a linear time-invariant system, so the surface
> temperature of a point x at the render time t is
> T(x, t) = T_ref(t) + c · Σ_k G_per(t − t_k) [q(x, t_k) − q_ref(t_k)] Δt,
> where G is the surface-temperature response to a unit flux impulse (one
> small 1-D solve per material with the column's own solver, held at its
> mean reference temperature), periodised over the day because the forcing
> is periodic, q_ref the forcing of a fully exposed reference element of the
> same material and orientation whose temperature T_ref(t) is obtained with
> the full nonlinear column, and c the quasi-steady correction
> h_lin / (h_conv + 4εσ T_mid³) for the temperature dependence of the
> radiative conductance (T_mid halfway between T_ref and the point).

*Motivation.* This is the response-factor / conduction-transfer-function
method of building thermal simulation [68, 69], applied per shading point.
The only position-dependent terms of the forcing are the direct solar term
α_s DNI_k cos θ_k V_k(x) and the sky-view terms (F_sky(x) − 1)(…), so the
whole spatial dependence reduces to ray-casting queries at the exact hit
point: Σ_k w_k cos θ_k V_k(x) with precomputed weights
w_k = G_per(lag_k) DNI_k (one shadow ray per daylight step with
cos θ_k > 0, ≈ 50–100 rays), and F_sky from a few cosine-weighted sky rays
drawn from the integrator's sampler (an unbiased estimate; T is linear in
F_sky). Thermal shadows thereby acquire the geometric sharpness of the ray
tracer at any image resolution, with a cost proportional to the number of
shading points — the same scaling as next-event estimation — instead of to
a texture resolution.

*Bias.* Spatially none. Temporally the forcing discretisation Δt of the
solver (10 min) is kept for the weights, while the sun *direction* is
jittered within each step per shading sample (interpolating consecutive
table entries), so the visibility is integrated over each step rather than
sampled at its centre: moving shadow edges show no time-step banding, at
the price of a little noise in the penumbra of the thermal shadow. The
linearisation about the reference trajectory is the one approximation of
the formulation; the correction c removes most of it: against the per-texel
nonlinear solver, the roof-over-asphalt scene is reproduced to a mean of
< 0.3 K and a maximum of < 1 K at all times of day, including texels under
the roof whose deviation from the exposed reference reaches −17 K
(`test_ir_thermal_dynamic.py::test02`). Diagnosed and fixed on the way:
the response function must be corrected with the radiative conductance *at
the time of day* of the render (the reference element is 25 K warmer at
14:00 than its daily mean), not merely with the deviation; and the two
formulations must agree on the time-step convention (a 10-minute offset is
0.6 K during the morning rise).

*Implementation.* `thermal_dynamic` emitter (per-shape tables: T_ref,
α_s, ε, h_conv, h_lin, C_ref, C_sky, weights w_k and sun directions, built
by `ThermalScene.dynamic(scene_dict, t)` from `DynamicModel`), new
`EmitterFlags::ThermalDynamic`, `Emitter::thermal_*()` accessors, and the
evaluation routine `thermal_dynamic_temperature()` in
`include/mitsuba/render/thermal.h`, called by `path_ir` and `volpath_ir`
for BSDF-sampled hits *and* for next-event estimation (the emitter's own
`eval()` returns the reference temperature, so no ray tracing happens
inside emitter calls — which keeps the CUDA variant safe). Integrator
parameters: `thermal_up` (zenith direction) and `thermal_sky_rays`
(default 8). With all tables zero the emitter reproduces `thermal`
bit-for-bit (`test03`); a 2 cm cover over asphalt renders a shadow edge of
≤ 2 pixels at 25 cm and at 6 cm per pixel (`test04`), whereas a 20 m roof
correctly produces a *soft* thermal shadow because its optical shadow
sweeps ≈ 17 m across the day. Validity: the quasi-steady correction is
exact in steady state and approximate during fast transients; large
deviations (> 30 K, e.g. sunlit metal next to deep shade) should prefer a
reference of the matching orientation (`ThermalScene` uses the shape's
mean normal) or the texture mode for verification.

---

## 7. Helper library and examples

### 7.1 `mitsuba.ir` (Python)

| Function | Purpose |
|---|---|
| `planck`, `planck_dT` | Planck's law and its temperature derivative (W m⁻² sr⁻¹ nm⁻¹) |
| `band_radiance(lmin, lmax, T, srf)` | ∫ B R dλ, optionally with a detector response |
| `apparent_temperature(L, lmin, lmax, srf)` | Inverse of the above (brightness temperature), vectorised over images |
| `BANDS`, `band_range`, `band_film`, `boxcar_srf` | Band definitions and `specfilm` construction (channel = band radiance) |
| `solar_irradiance_toa`, `sun`, `sky_blackbody` | Sources (Section 6.4; sky as an effective-temperature blackbody until Phase 3) |
| `thermalize(scene, T, overrides)` | Attach `thermal` emitters to every shape without an emitter (assumption A6) |
| `validate(scene, band)` | Warn about visible-light constructs: `path`, `hdrfilm`, RGB values, `d65`, non-IR conductor presets, spectra not covering the band, shapes at 0 K |
| `conductor_ir`, `opaque_ir`, `window_ior`, `load_ior`, `IR_MATERIALS` | Material builders and data access (Section 6) |
| `drude_lorentz_nk`, `normal_emissivity`, `hemispherical_emissivity`, `fresnel_conductor` | Analytic material physics |
| `load_ecostress`, `diffuse_from_emissivity`, `diffuse_from_reflectance` | Spectral-library workflow (Section 6.3) |
| `sky_emissivity_idso`, `vapour_pressure`, `thermalsky` | Clear-sky emissivity and `thermalsky` emitter dictionaries (Section 6.6) |
| `air_medium`, `medium_box`, `hitran_absorption`, `transmittance`, `atmosphere.band_average` | Emissive air media and HITRAN line-by-line absorption (Sections 6.5, 6.7) |
| `thermal.Weather`, `thermal.solar_position`, `thermal.bird_clear_sky` | Parametrised weather and solar forcing (Section 6.8, A19) |
| `thermal.Material`, `thermal.MATERIALS`, `thermal.Column` | Thermal material properties and the 1-D solver (A17, A18, A20) |
| `thermal.ThermalScene`, `thermal.ThermalResult` | Scene sampling, shadows, sky view, time integration, temperature textures (Section 6.8.1–2) |
| `thermal.ThermalScene.dynamic`, `thermal.DynamicModel` | Shading-time (linear-response) model: `thermal_dynamic` emitters evaluated by the integrators (Section 6.8.3) |
| `materials.rii_to_nk`, `write_spd` | Conversion of refractiveindex.info pages (formulas 1–7 and tabulated data) |

### 7.2 Examples (`examples/ir/`)

| Script | Demonstrates | Key result |
|---|---|---|
| `01_lwir_basic.py` | Thermal emitters, `thermalize`, Kirchhoff coupling, band film, apparent temperature | Rough-Al box at 300 K reads 275 K on top (mirrors the 250 K sky) and 289 K on the side |
| `02_bands.py` | Same scene in SWIR/MWIR/LWIR with sunlight; separation of reflected and emitted components | Thermal share of the image: SWIR 0.0 %, MWIR 45.7 %, LWIR 99.8 % — the regimes of Table 1 |
| `03_materials.py` | `*_ir` presets; six spheres at 320 K | Apparent temperatures: matte paint 317.9 K, polished Al 288.0 K, polished Fe 288.3 K, water 319.6 K, glass 314.6 K |
| `04_kirchhoff_checks.py` | Rendered vs. analytic radiance | All cases within 2.6·10⁻⁴ (Section 8) |
| `05_temperature_texture.py` | Temperature field as a raw float bitmap | Hot spot 340 K with ε = 0.9 under a 260 K sky: rendered 334.5 K vs. expected 334.0 K |
| `06_ecostress.py` | Measured reflectance spectra as Lambertian IR materials | Rendered radiance equals ε_band·B to 0.07 % |
| `07_atmosphere.py` | `volpath_ir` with an emissive air layer of 10 m–10 km in front of a 350 K plate (gray or `--hitran` absorption) | Matches τ L_up + (1−τ) B_air (hemispherical slab term for L_down) to < 0.3 % at short range, ≤ 2 % at 10 km (Monte Carlo noise, 1024 spp) |
| `08_sky.py` | `thermalsky` angular radiance; polished Al plate mirroring the sky | Sky 245 K (zenith) → 293 K (horizon); 300 K Al plate reads 245–293 K depending on tilt |
| `09_hitran_transmittance.py` | HITRAN H₂O + CO₂ transmittance over 1 km, 2.5–5.5 and 6.5–15.5 µm, with/without RSB continuum; attenuated sun and air medium | Window edges at 7.5–8 µm (H₂O) and 13.5–14 µm (CO₂), CO₂ 4.3 µm band opaque, continuum slope in the window |
| `10_thermal_solver.py` | `mitsuba.ir.thermal`: asphalt, concrete wall, heated steel box, tree in Stockholm on 2026-07-15; probes through the day, LWIR renders at 08/14/20 h, diurnal curves | Moving optical shadows with lagging thermal shadows; asphalt 16–49 °C, wall 18–27 °C, box 30–40 °C |
| `11_thermal_dynamic.py` | Same scene at 14:00 with `thermal_dynamic` emitters vs the 25 cm texture solver, close-up of the wall/tree shadows, timings | Resolution-free shadow edges; cost scales with shading points |

---

## 8. Validation

Each test in `src/integrators/tests/test_path_ir.py` compares a
band-integrated radiance to a closed-form reference derived from Planck's
law and Kirchhoff's law. Tests run on every spectral variant that was
built (`scalar_spectral`, `llvm_ad_spectral`, `cuda_ad_spectral`).

| Test | Configuration | Reference | Tolerance | Status |
|---|---|---|---|---|
| `test01` | Diffuse plate (ρ=0.3, 300 K), black background | (1−ρ) B | 0.2 % (zero-variance case) | ✅ |
| `test01` (b) | Same with `path` integrator | B (documents the non-thermal-aware behaviour) | 0.2 % | ✅ |
| `test02` | Camera inside isothermal specular-conductor sphere | B | 0.2 % (zero variance) | ✅ |
| `test03` | Camera inside isothermal diffuse sphere (ρ=0.7) | B | 1.5 % (MIS noise) | ✅ |
| `test04` | Diffuse plate (320 K) inside blackbody cavity (280 K) | (1−ρ) B(T₁) + ρ B(T₂) | 1.5 % | ✅ |
| `test05` | Conductor plate (n=3, k=4) at 0°, 45°, 75° | (1 − F(θ)) B — directional Kirchhoff | 0.3 % | ✅ |
| `test06` | Blackbody plate, 8–14 µm band | 54.93 W m⁻² sr⁻¹ | 1 % | ✅ |
| `test07` | Thermal diffuse plate + directional source (MWIR) | (1−ρ) B + ρ E/π | 0.5 % (sun `sampling_weight` 10⁴) | ✅ |

Phase 2 adds `src/python_tests/test_ir.py` (13 tests on all spectral variants):

| Test | Check | Reference |
|---|---|---|
| `test01` | Stefan–Boltzmann integral, Wien's law, LWIR band value 54.93, dB/dT | σT⁴/π, b/T [5] |
| `test02` | Apparent-temperature inversion round-trips in SWIR/MWIR/LWIR, with a Gaussian response | — |
| `test03` | ∫E_sun dλ = 1361 W m⁻² | IAU 2015 [27] |
| `test04` | LD normal emissivities of Al/Au/Ag/Cu/W at 10 µm in 0.3–3 %; LD vs. shipped Johnson–Christy Au data in the visible (< 15 %) | [28, 37] |
| `test05` | All 22 `*_ir` presets load in `conductor`; the sampled Fresnel weight of `Al_ir` equals the Fresnel formula from the data | — |
| `test06` | Band-averaged n of Ge (≈ 4.0), ZnSe (≈ 2.4); water opaque in the LWIR | [48, 45, 30] |
| `test07` | `validate` flags `path`, `hdrfilm`, RGB, visible presets, untagged shapes; `thermalize` semantics | — |
| `test08` | Sun on a diffuse plate gives ρE/π with the physically scaled spectrum | — |
| `test09` | ECOSTRESS parser (units, ordering) and material construction | [9] |

Phase 3 adds `src/integrators/tests/test_volpath_ir.py` (9 tests) and
`src/emitters/tests/test_thermalsky.py` (5 tests), all spectral variants:

| Test | Configuration | Reference | Tolerance |
|---|---|---|---|
| `volpath_ir/test02` | Cold absorbing slab (σ = 0.7 m⁻¹, 1 m, 280 K) before a 320 K blackbody plate; spectral and gray extinction | τ B(T₁) + (1−τ) B(T_m) | 1.5 % |
| `volpath_ir/test03` | Isothermal slab + blackbody plate | B(T), zero variance | 0.2 % |
| `volpath_ir/test03b` | Isothermal slab + plate of albedo 0.3 under a black sky | τ L_up + (1−τ) B | 1.5 % |
| `volpath_ir/test04` | Hot absorbing slab, no background | (1−τ) B(T_m) | 1.5 % |
| `volpath_ir/test05` | Purely scattering slab at 350 K | 0 (no absorption → no emission) | 2 % of B |
| `volpath_ir/test06` | Camera inside a diffuse isothermal cavity filled with a scattering/absorbing medium | B(T) | 2 % |
| `volpath_ir/test07` | No media: plate in cavity | same as `path_ir` | 1.5 % |
| `volpath_ir/test08` | Wavelength-dependent σ_a (irregular spectrum) | ∫ e^{−σ(λ)d} B dλ | 2 % |
| `thermalsky/test01–05` | Zenith/horizon/below-horizon values, isotropic model, `up` vector, spectral ε_z, sampling consistency, conductor plate reflecting the zenith (F(0) ε_z B) | closed forms | ≤ 0.3 % |

Phase 3b adds `src/python_tests/test_ir_thermal.py` (7 tests):

| Test | Check | Reference |
|---|---|---|
| `test01` | Solar elevation/azimuth at both solstices (Stockholm, Sydney), vectorised call, `sun_direction` frame conventions | 90° − φ ± δ; [62] |
| `test02` | Bird clear-sky noon values, air-mass dependence, aerosol and cloud sensitivities | [63, 65] |
| `test03` | Sunrise/sunset, T_min/T_max reproduction and afternoon maximum, irradiance and sky longwave ranges, h, precipitable water | [66, 32] |
| `test04` | Steady-state slab with 0/300/800 W m⁻² absorbed; internal flux; interior boundary | energy balance (Newton) |
| `test05` | Periodic penetration into a semi-infinite solid at three depths | e^{−z/δ}, Carslaw & Jaeger |
| `test06` | Roof over asphalt: F_sky under the roof < 0.3, thermal shadow > 10 K at noon, lag (15:00 warmer than 12:00, evening warmer than air), texture round trip and `path_ir` render | — |

`src/python_tests/test_ir_thermal_dynamic.py` (5 tests, Section 6.8.3):
impulse-response properties (positivity, decay, steady-state gain 1/h for a
thin slab, periodisation); agreement with the per-texel solver (< 1 K max,
< 0.3 K mean); bit-for-bit equivalence with `thermal` for zero tables in
an isothermal cavity (BSDF hits and next-event estimation); shadow-edge
width ≤ 2 px at two image resolutions and agreement of the unshaded
temperature with T_ref; `path_ir` and `volpath_ir` agreement.

**Convergence study** (isothermal cavity, L/B, 4 seeds): diffuse wall is
exactly 1.0000 at any sample count (zero variance, as derived in Section
3.2); rough Al (α = 0.3): 0.998 ± 0.011 at 256 spp, 1.003 ± 0.001 at 4096,
1.002 ± 0.002 at 32 768 spp; rough Al (α = 0.05): 1.007 ± 0.001 at 32 768 spp.
The residual of ≈ 0.5 % for the near-specular rough conductor over 64-bounce
paths is above the statistical error and is noted as an open item (suspected
single-precision effects in the microfacet eval/pdf ratio at grazing
incidence; it does not affect diffuse or smooth materials).

`src/emitters/tests/test_thermal.py` checks the emitter in isolation:
Planck values at 250–1500 K, two-sidedness, agreement with the existing
`blackbody` spectrum, textured temperature, and the consistency of
`sample_direction`/`pdf_direction`/`eval_direction`.

**Results (2026-11-12, all three spectral variants):** 53/53 tests pass
(26 emitter tests, 27 integrator tests). Observed accuracy in the
zero-variance configurations is limited only by single-precision Planck
evaluation and the 20 nm test band (< 0.1 %).

A practical lesson from `test07`: with uniform emitter selection and two
emitters, the solar term of each sample is either 0 or twice its mean,
giving a relative standard error of 1/√spp (6 % at 256 spp). In scenes where
every surface emits, strong non-thermal sources (sun, hot exhausts) must be
given a correspondingly large `sampling_weight`; see Section 4.4.

---

## 9. Limitations and roadmap

| Phase | Content | Status |
|---|---|---|
| 1 | Thermal emission of surfaces, Kirchhoff coupling, `path_ir`, validation | ✅ |
| 2 | IR material data (Section 6), `mitsuba.ir` helpers, examples | ✅ |
| 3 | Emissive media (`volpath_ir`, medium `temperature`), `thermalsky`, Idso clear-sky emissivity, HITRAN transmittance | ✅ |
| 3b | Scene temperatures: `mitsuba.ir.thermal` energy-balance solver with scene shadows, parametrised weather, internal heat; shading-time evaluation (`thermal_dynamic`) | ✅ |
| 4 | Sensor model: detector response curves [40], apparent-temperature output, NETD noise model | 📝 |
| 5 | Polarized thermal emission; multiple-scattering microfacet BSDFs [22]; differentiable temperature | 📝 |

Known limitations of the present state:

* Emissivity of rough surfaces follows the single-scattering microfacet
  models (A5).
* Uniform emitter selection for next-event estimation (Section 4.4).
* Material data cover clean bulk materials; paints, oxides, vegetation,
  soils and fabrics must come from spectral libraries (Section 6.3) or user
  measurements. Only hemispherical (Lambertian) use of library data is
  supported; measured BRDFs are not.
* Atmosphere: legacy RSB continuum instead of MT_CKD, no aerosols, no O₃,
  no clouds, horizontal homogeneous paths only (A15, A16); the sky is an
  isothermal slab at the screen-level air temperature.
* Residual ≈ 0.5 % in long near-specular rough-conductor paths (Section 8).
* Thermal solver: no lateral conduction, no evapotranspiration or
  precipitation, surroundings linearised to T_air, periodic single-day
  forcing, broadband α_s/ε separate from the rendering BSDF (A17–A20).

---

## 10. References

[1] M. Planck, "Ueber das Gesetz der Energieverteilung im Normalspectrum," *Annalen der Physik* 309(3), 553–563, 1901.

[2] G. Kirchhoff, "Ueber das Verhältniss zwischen dem Emissionsvermögen und dem Absorptionsvermögen der Körper für Wärme und Licht," *Annalen der Physik* 185(2), 275–301, 1860.

[3] F. E. Nicodemus, "Directional reflectance and emissivity of an opaque surface," *Applied Optics* 4(7), 767–775, 1965.

[4] F. E. Nicodemus, J. C. Richmond, J. J. Hsia, I. W. Ginsberg, T. Limperis, *Geometrical Considerations and Nomenclature for Reflectance*, NBS Monograph 160, National Bureau of Standards, 1977.

[5] J. R. Howell, M. P. Mengüç, R. Siegel, *Thermal Radiation Heat Transfer*, 6th ed., CRC Press, 2016.

[6] M. F. Modest, *Radiative Heat Transfer*, 3rd ed., Academic Press, 2013.

[7] J. R. Howell, "The Monte Carlo method in radiative heat transfer," *Journal of Heat Transfer* 120(3), 547–560, 1998.

[8] W. C. Snyder, Z. Wan, "BRDF models to predict spectral reflectance and emissivity in the thermal infrared," *IEEE Transactions on Geoscience and Remote Sensing* 36(1), 214–225, 1998.

[9] S. K. Meerdink, S. J. Hook, D. A. Roberts, E. A. Abbott, "The ECOSTRESS spectral library version 1.0," *Remote Sensing of Environment* 230, 111196, 2019.

[10] A. M. Baldridge, S. J. Hook, C. I. Grove, G. Rivera, "The ASTER spectral library version 2.0," *Remote Sensing of Environment* 113(4), 711–715, 2009.

[11] M. Vollmer, K.-P. Möllmann, *Infrared Thermal Imaging: Fundamentals, Research and Applications*, 2nd ed., Wiley-VCH, 2018.

[12] C. J. Willers, *Electro-Optical System Analysis and Design: A Radiometry Perspective*, SPIE Press, 2013.

[13] R. D. Hudson, *Infrared System Engineering*, Wiley, 1969.

[14] ISO 20473:2007, *Optics and photonics — Spectral bands*, International Organization for Standardization, 2007.

[15] W. Jakob, S. Speierer, N. Roussel, M. Nimier-David, D. Vicini, T. Zeltner, B. Nicolet, M. Crespo, V. Leroy, Z. Zhang, *Mitsuba 3 renderer*, version 3.x, https://mitsuba-renderer.org, 2022.

[16] W. Jakob, S. Speierer, N. Roussel, D. Vicini, "Dr.Jit: A just-in-time compiler for differentiable rendering," *ACM Transactions on Graphics* 41(4), 2022.

[17] A. Wilkie, S. Nawaz, M. Droske, A. Weidlich, J. Hanika, "Hero wavelength spectral sampling," *Computer Graphics Forum* 33(4), 123–131, 2014.

[18] E. Veach, L. J. Guibas, "Optimally combining sampling techniques for Monte Carlo rendering," *Proc. SIGGRAPH '95*, 419–428, 1995.

[19] J. T. Kajiya, "The rendering equation," *Proc. SIGGRAPH '86*, 143–150, 1986.

[20] Bureau International des Poids et Mesures, *The International System of Units (SI)*, 9th ed., 2019 (exact values of h, c, k_B).

[21] E. Heitz, "Understanding the masking-shadowing function in microfacet-based BRDFs," *Journal of Computer Graphics Techniques* 3(2), 48–107, 2014.

[22] E. Heitz, J. Hanika, E. d'Eon, C. Dachsbacher, "Multiple-scattering microfacet BSDFs with the Smith model," *ACM Transactions on Graphics* 35(4), 2016.

[23] B. Walter, S. R. Marschner, H. Li, K. E. Torrance, "Microfacet models for refraction through rough surfaces," *Proc. Eurographics Symposium on Rendering*, 195–206, 2007.

[24] J. R. Schott, *Remote Sensing: The Image Chain Approach*, 2nd ed., Oxford University Press, 2007.

[25] A. A. Goodenough, S. D. Brown, "DIRSIG5: Next-generation remote sensing data and image simulation framework," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* 10(11), 4818–4833, 2017.

[26] C. A. Gueymard, "The sun's total and spectral irradiance for solar energy applications and solar radiation models," *Solar Energy* 76(4), 423–453, 2004.

[27] A. Prša et al., "Nominal values for selected solar and planetary quantities: IAU 2015 Resolution B3," *The Astronomical Journal* 152(2), 41, 2016 (nominal solar effective temperature 5772 K).

[28] A. D. Rakić, A. B. Djurišić, J. M. Elazar, M. L. Majewski, "Optical properties of metallic films for vertical-cavity optoelectronic devices," *Applied Optics* 37(22), 5271–5283, 1998.

[29] M. A. Ordal, L. L. Long, R. J. Bell, S. E. Bell, R. R. Bell, R. W. Alexander, C. A. Ward, "Optical properties of the metals Al, Co, Cu, Au, Fe, Pb, Ni, Pd, Pt, Ag, Ti, and W in the infrared and far infrared," *Applied Optics* 22(7), 1099–1119, 1983.

[30] G. M. Hale, M. R. Querry, "Optical constants of water in the 200-nm to 200-µm wavelength region," *Applied Optics* 12(3), 555–563, 1973.

[31] R. Kitamura, L. Pilon, M. Jonasz, "Optical constants of silica glass from extreme ultraviolet to far infrared at near room temperature," *Applied Optics* 46(33), 8118–8133, 2007.

[32] S. B. Idso, "A set of equations for full spectrum and 8- to 14-µm and 10.5- to 12.5-µm thermal radiation from cloudless skies," *Water Resources Research* 17(2), 295–304, 1981.

[33] M. H. Unsworth, J. L. Monteith, "Long-wave radiation at the ground: I. Angular distribution of incoming radiation," *Quarterly Journal of the Royal Meteorological Society* 101(427), 13–24, 1975.

[34] A. Berk, P. Conforti, R. Kennett, T. Perkins, F. Hawes, J. van den Bosch, "MODTRAN6: a major upgrade of the MODTRAN radiative transfer code," *Proc. SPIE* 9088, 90880H, 2014.

[35] I. E. Gordon et al., "The HITRAN2020 molecular spectroscopic database," *Journal of Quantitative Spectroscopy and Radiative Transfer* 277, 107949, 2022.

[36] J. Novák, I. Georgiev, J. Hanika, W. Jarosz, "Monte Carlo methods for volumetric light transport simulation," *Computer Graphics Forum* 37(2), 551–576, 2018.

[37] E. Hagen, H. Rubens, "Über Beziehungen des Reflexions- und Emissionsvermögens der Metalle zu ihrem elektrischen Leitvermögen," *Annalen der Physik* 316(8), 873–901, 1903.

[38] G. F. Evans, M. D. McCool, "Stratified wavelength clusters for efficient spectral Monte Carlo rendering," *Proc. Graphics Interface*, 42–49, 1999.

[39] M. N. Polyanskiy, "Refractiveindex.info database of optical constants," *Scientific Data* 11, 94, 2024. *(also cited as [43])*

[40] A. Rogalski, *Infrared Detectors*, 2nd ed., CRC Press, 2011.

[41] M. Cherkaoui, J.-L. Dufresne, R. Fournier, J.-Y. Grandpeix, A. Lahellec, "Monte Carlo simulation of radiation in gases with a narrow-band model and a net-exchange formulation," *Journal of Heat Transfer* 118(2), 401–407, 1996.

[42] K. Johnson, A. Curran, D. Less, D. Levanen, E. Marttila, T. Gonda, J. Jones, "MuSES: A new heat and signature management design tool for virtual prototyping," *Proc. 9th Annual Ground Target Modeling & Simulation Conference*, 1998.

[43] M. N. Polyanskiy, "Refractiveindex.info database of optical constants," *Scientific Data* 11, 94, 2024. Database: https://refractiveindex.info (CC0 1.0).

[44] M. R. Querry, *Optical Constants*, Contractor Report CRDC-CR-85034, U.S. Army Chemical Research and Development Center, 1985.

[45] M. R. Querry, *Optical Constants of Minerals and Other Materials from the Millimeter to the Ultraviolet*, Contractor Report CRDEC-CR-88009, 1987.

[46] D. Franta, A. Dubroka, C. Wang, A. Giglia, J. Vohánka, P. Franta, I. Ohlídal, "Temperature-dependent dispersion model of float zone crystalline silicon," *Applied Surface Science* 421, 405–419, 2017.

[47] D. Franta, D. Nečas, I. Ohlídal, A. Giglia, "Optical characterization of SiO₂ thin films using universal dispersion model over wide spectral range," *Proc. SPIE* 9890, 989014, 2016.

[48] J. H. Burnett, S. G. Kaplan, E. Stover, A. Phenis, "Refractive index measurements of Ge," *Proc. SPIE* 9974, 99740X, 2016.

[49] T. Amotchkina, M. Trubetskov, D. Hahner, V. Pervak, "Characterization of e-beam evaporated Ge, YbF₃, ZnS, and LaF₃ thin films for laser-oriented coatings," *Applied Optics* 59(5), A40–A47, 2020.

[50] H. H. Li, "Refractive index of alkaline earth halides and its wavelength and temperature derivatives," *Journal of Physical and Chemical Reference Data* 9(1), 161–289, 1980.

[51] R. V. Kochanov, I. E. Gordon, L. S. Rothman, P. Wcisło, C. Hill, J. S. Wilzewski, "HITRAN Application Programming Interface (HAPI): A comprehensive approach to working with spectroscopic data," *Journal of Quantitative Spectroscopy and Radiative Transfer* 177, 15–30, 2016.

[52] E. J. Mlawer, V. H. Payne, J.-L. Moncet, J. S. Delamere, M. J. Alvarado, D. C. Tobin, "Development and recent evaluation of the MT_CKD model of continuum absorption," *Philosophical Transactions of the Royal Society A* 370, 2520–2556, 2012.

[53] J. L. Monteith, M. H. Unsworth, *Principles of Environmental Physics*, 4th ed., Academic Press, 2013 (ch. 5, long-wave radiation of clear skies).

[54] World Meteorological Organization, *Guide to Meteorological Instruments and Methods of Observation*, WMO-No. 8, 2008 (Magnus-type saturation vapour pressure formula).

[55] R. E. Roberts, J. E. A. Selby, L. M. Biberman, "Infrared continuum absorption by atmospheric water vapor in the 8–12-µm window," *Applied Optics* 15(9), 2085–2090, 1976.

[56] P. A. Jacobs, *Thermal Infrared Characterization of Ground Targets and Backgrounds*, 2nd ed., SPIE Press, 2006.

[57] T. R. Oke, *Boundary Layer Climates*, 2nd ed., Routledge, 1987 (Table 2.1: thermal properties of natural and urban materials).

[58] L. K. Balick, L. E. Link, R. K. Scoggins, J. L. Solomon, *Thermal Modeling of Terrain Surface Elements*, Technical Report EL-81-2, U.S. Army Engineer Waterways Experiment Station, 1981.

[59] W. H. McAdams, *Heat Transmission*, 3rd ed., McGraw-Hill, 1954; as used in J. A. Duffie, W. A. Beckman, *Solar Engineering of Thermal Processes*, 4th ed., Wiley, 2013 (h = 5.7 + 3.8 V).

[60] F. Lindberg, B. Holmer, S. Thorsson, "SOLWEIG 1.0 – Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings," *International Journal of Biometeorology* 52, 697–713, 2008.

[61] T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. DeWitt, *Fundamentals of Heat and Mass Transfer*, 7th ed., Wiley, 2011.

[62] J. J. Michalsky, "The Astronomical Almanac's algorithm for approximate solar position (1950–2050)," *Solar Energy* 40(3), 227–235, 1988.

[63] R. E. Bird, R. L. Hulstrom, *A Simplified Clear Sky Model for Direct and Diffuse Insolation on Horizontal Surfaces*, SERI/TR-642-761, Solar Energy Research Institute, 1981.

[64] B. Leckner, "The spectral distribution of solar radiation at the earth's surface—elements of a model," *Solar Energy* 20(2), 143–150, 1978.

[65] F. Kasten, G. Czeplak, "Solar and terrestrial radiation dependent on the amount and type of cloud," *Solar Energy* 24(2), 177–189, 1980.

[66] W. J. Parton, J. A. Logan, "A model for diurnal variation in soil and air temperature," *Agricultural Meteorology* 23, 205–216, 1981.

[68] D. G. Stephenson, G. P. Mitalas, "Cooling load calculations by thermal response factor method," *ASHRAE Transactions* 73(1), III.1.1–III.1.7, 1967.

[69] G. P. Mitalas, D. G. Stephenson, "Room thermal response factors," *ASHRAE Transactions* 73(1), III.2.1–III.2.10, 1967.

[67] T. M. Crawford, C. E. Duchon, "An improved parameterization for estimating effective atmospheric emissivity for use in calculating daytime downwelling longwave radiation," *Journal of Applied Meteorology* 38(4), 474–480, 1999.

---

## Appendix A — Changelog

| Date | Version | Change |
|---|---|---|
| 2026-11-13 | 0.5 | Shading-time thermal model: `thermal_dynamic` emitter, `EmitterFlags::ThermalDynamic`, `Emitter::thermal_*()` accessors, `thermal_dynamic_temperature()` in `path_ir`/`volpath_ir` (hits and NEE), `DynamicModel` / `ThermalScene.dynamic()`, 5 tests, example 11, assumption A21, references [68, 69]. |
| 2026-11-12 | 0.4 | Phase 3b: `mitsuba.ir.thermal` (solar geometry, Bird clear sky, parametrised weather, 1-D energy-balance solver with scene shadows and sky-view factors, materials table, temperature textures), 7 tests, example 10, assumptions A17–A20, references [56]–[67]. |
| 2026-11-12 | 0.3.1 | Atmosphere: RSB water-vapour continuum added to `hitran_absorption` (default on), grid step 0.05 cm⁻¹, example 09 extended beyond the band edges; A16 and §6.7 revised after review of the transmittance figure. |
| 2026-11-12 | 0.3 | Phase 3: medium `temperature`, `volpath_ir` (difference formulation for emissive media), `thermalsky` emitter, `mitsuba.ir.atmosphere` (Idso emissivity, air media, HITRAN line-by-line via hapi), 14 new tests, examples 07–09, assumptions A14–A16, references [51]–[54]. |
| 2026-11-12 | 0.2 | Phase 2: `*_ir` optical constants (22 presets, Lorentz–Drude metals verified to 1.3·10⁻⁴ against [43]; dielectrics/water from [43]), `mitsuba.ir` package (radiometry, materials, scene helpers), 13 new tests, `examples/ir/` (6 scripts), assumptions A9–A13, convergence study. |
| 2026-11-12 | 0.1 | Phase 1: `EmitterFlags::Thermal`, `Integrator::thermal_aware()`, `thermal` emitter, `path_ir` integrator, scene-level compatibility warning, unit and validation tests, this report. |

## Appendix B — Build notes

* Built from a fresh recursive clone with Clang 23 / Ninja, `CMAKE_BUILD_TYPE=Release`,
  variants `scalar_rgb`, `scalar_spectral`, `llvm_ad_spectral`, `cuda_ad_spectral`
  (`build/mitsuba.conf`), `MI_SPLIT_MODE=OFF`.
* `cmake` is taken from a project-local virtual environment (`.venv-build`,
  created with `uv`), which also provides `pytest` and `numpy`.
* The configure step is run with `-DMI_NANOBIND_FOUND=0` because a system
  `nanobind` package is installed; Mitsuba's bundled copy is first on the
  include path and is the one actually compiled against.
* Run tests with `source build/setpath.sh && pytest src/emitters/tests/test_thermal.py src/integrators/tests/test_path_ir.py`.
