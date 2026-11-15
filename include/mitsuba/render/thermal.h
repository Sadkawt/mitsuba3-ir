#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/bsdf.h>
#include <drjit/array.h>
#include <drjit/math.h>
#include <drjit/while_loop.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * Physical constants and Planck's law shared by the infrared extension
 * (``thermal`` emitter, ``thermalsky`` emitter, ``volpath_ir`` integrator).
 * See docs/ir/IR_TECHNICAL_REPORT.md, section 2.2.
 */
namespace thermal {
    // Exact SI (2019) values
    constexpr double c_light  = 299792458.0;     /// Speed of light [m/s]
    constexpr double h_planck = 6.62607015e-34;  /// Planck constant [J s]
    constexpr double k_boltz  = 1.380649e-23;    /// Boltzmann constant [J/K]

    /// First radiation constant for spectral radiance, 2 h c^2 [W m^2 sr^-1]
    constexpr double c1L = 2.0 * h_planck * c_light * c_light;
    /// Second radiation constant, h c / k_B [m K]
    constexpr double c2 = h_planck * c_light / k_boltz;
}

/**
 * \brief Planck's law: blackbody spectral radiance B(lambda, T)
 *
 * \param wavelengths  Wavelengths in nanometers (one value per spectral sample)
 * \param T            Absolute temperature in kelvin (one value per lane)
 * \return             Spectral radiance in W m^-2 sr^-1 nm^-1. Zero for
 *                     T <= 0 and whenever exp() overflows (very cold
 *                     surfaces at short wavelengths).
 */
template <typename Wavelength, typename Float>
Wavelength planck_radiance(const Wavelength &wavelengths, const Float &T) {
    using ScalarFloat = dr::scalar_t<Float>;

    Wavelength lambda  = wavelengths * ScalarFloat(1e-9),
               lambda2 = dr::square(lambda),
               lambda5 = dr::square(lambda2) * lambda;

    Wavelength x = ScalarFloat(thermal::c2) / (lambda * T);
    Wavelength B = ScalarFloat(thermal::c1L) / (lambda5 * (dr::exp(x) - ScalarFloat(1)));

    // Density per nanometer rather than per meter
    B *= ScalarFloat(1e-9);

    return dr::select(T > ScalarFloat(0), B, Wavelength(ScalarFloat(0)));
}

/**
 * \brief Surface temperature of a thermal-dynamic emitter at a surface point,
 * evaluated at shading time (linear-response model; report section 6.8.3).
 *
 * T = T_ref + c( alpha * [sum_k w_k cos(theta_k) V_k - C_ref] + (F_sky - 1) C_sky )
 *
 * where V_k is the visibility of the sun at forcing step k (shadow ray),
 * F_sky the cosine-weighted sky-view factor (``sky_rays`` rays) and c() the
 * quasi-steady correction for the temperature dependence of the radiative
 * conductance. All ray casting happens in the caller's kernel.
 *
 * \param scene    Scene used for the shadow and sky rays
 * \param emitter  Per-lane emitter pointer (flags must contain ThermalDynamic)
 * \param p, n     Surface point and outward geometric normal (world space)
 * \param up       Zenith direction (world space)
 * \param sampler  Sampler for the sky-view rays
 * \param sky_rays Number of cosine-weighted hemisphere rays
 * \param time     Ray time
 */
/**
 * \brief First intersection along ``ray`` that is not a null-BSDF interface
 *
 * Shapes with a ``null`` BSDF (medium boundaries, cut-outs) are transparent
 * to the IR integrators; the shading-time thermal model must see through
 * them in the same way. Crosses at most 16 null interfaces per ray.
 */
template <typename Float, typename Spectrum, typename Scene, typename Ray3f>
SurfaceInteraction<Float, Spectrum>
thermal_trace_opaque(const Scene *scene, const Ray3f &ray_, dr::mask_t<Float> active) {
    using Mask   = dr::mask_t<Float>;
    using UInt32 = dr::uint32_array_t<Float>;
    using SurfaceInteraction3f = SurfaceInteraction<Float, Spectrum>;

    SurfaceInteraction3f si = scene->ray_intersect(
        ray_, +RayFlags::Default, Mask(false), UInt32((uint32_t) RayMask::Secondary), active);
    if (!scene->has_null_shapes())
        return si;

    struct Loop {
        Ray3f ray;
        SurfaceInteraction3f si;
        Mask active;
        UInt32 n;
        DRJIT_STRUCT(Loop, ray, si, active, n)
    } l = { ray_, si, active, UInt32(0) };

    dr::tie(l) = dr::while_loop(dr::make_tuple(l),
        [](const Loop &l) { return l.active; },
        [scene](Loop &l) {
            Mask valid = l.si.is_valid();
            Mask is_null = false;
            if constexpr (dr::is_jit_v<Float>) {
                is_null = valid && has_flag(l.si.bsdf()->flags(), BSDFFlags::Null);
            } else {
                if (valid)
                    is_null = has_flag(l.si.bsdf()->flags(), BSDFFlags::Null);
            }
            l.active &= is_null && (l.n < 16u);
            dr::masked(l.ray, l.active) = l.si.spawn_ray(l.ray.d);
            SurfaceInteraction3f si2 = scene->ray_intersect(
                l.ray, +RayFlags::Default, Mask(false),
                UInt32((uint32_t) RayMask::Secondary), l.active);
            dr::masked(l.si, l.active) = si2;
            l.n += 1;
        });
    return l.si;
}

template <typename Float, typename Spectrum, typename Scene, typename EmitterPtr,
          typename Sampler, typename Wavelength>
Float thermal_dynamic_temperature(const Scene *scene, const EmitterPtr &emitter,
                                  const Point<Float, 3> &p, const Normal<Float, 3> &n,
                                  const Vector<dr::scalar_t<Float>, 3> &up, Sampler *sampler,
                                  uint32_t sky_rays, const Float &time,
                                  const Wavelength &wavelengths, dr::mask_t<Float> active) {
    DRJIT_MARK_USED(wavelengths);
    using Mask     = dr::mask_t<Float>;
    using UInt32   = dr::uint32_array_t<Float>;
    using Vector3f = Vector<Float, 3>;
    using Point3f  = Point<Float, 3>;
    using Ray3f    = Ray<Point3f, Spectrum>;
    using ScalarFloat = dr::scalar_t<Float>;

    // Shadow rays leave the surface slightly along the normal
    Float offset = dr::maximum(dr::max(dr::abs(p)) * ScalarFloat(1e-4), ScalarFloat(1e-4));
    Point3f o = p + n * offset;

    // --- direct solar history: sum_k w_k cos(theta_k) V_k ---------------------
    // Table entry k is the centre of forcing step k, and consecutive entries
    // are consecutive daylight steps (DynamicModel.tables). The sun direction
    // is jittered symmetrically within the step, t_k + u dt with u in
    // [-1/2, 1/2), using the central-difference tangent of the sun path, so
    // that visibility is averaged over the step instead of sampled at its
    // centre (no time-step banding of moving shadow edges).
    UInt32 n_steps = emitter->thermal_steps();
    Float u_time = sampler->next_1d(active) - ScalarFloat(0.5);
    bool has_null = scene->has_null_shapes();
    struct Loop {
        UInt32 k;
        Float acc;
        DRJIT_STRUCT(Loop, k, acc)
    } ls = { UInt32(0), Float(0.f) };

    dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
        [n_steps](const Loop &l) { return l.k < n_steps; },
        [&](Loop &l) {
            auto [s0, w] = emitter->thermal_forcing(l.k, active);
            UInt32 kp = dr::select(l.k > 0u, l.k - 1u, l.k),
                   kn = dr::minimum(l.k + 1u, n_steps - 1u);
            Vector3f sp = emitter->thermal_forcing(kp, active).first,
                     sn = emitter->thermal_forcing(kn, active).first;
            Float span = Float(kn - kp);
            Vector3f tangent = dr::select(span > ScalarFloat(0), (sn - sp) / span, Vector3f(0.f));
            Vector3f s = dr::normalize(s0 + u_time * tangent);
            Float cos_i = dr::dot(n, s);
            Mask lit = active && (cos_i > ScalarFloat(1e-4)) && (w != ScalarFloat(0));
            Ray3f ray(o, s, time, wavelengths);
            Mask occluded;
            if (has_null)
                occluded = thermal_trace_opaque<Float, Spectrum>(scene, ray, lit).is_valid();
            else
                occluded = scene->ray_test(ray, Mask(false), UInt32((uint32_t) RayMask::Secondary), lit);
            dr::masked(l.acc, lit && !occluded) += w * cos_i;
            l.k += 1;
        });

    // --- sky-view factor and radiative exchange: cosine-weighted hemisphere rays
    // A direction either reaches the sky (unoccluded and above the horizon),
    // hits another thermal-dynamic surface whose exchange fields give the
    // deviation of its radiosity history from the generic surroundings, or
    // hits other geometry / the lower hemisphere (generic surroundings: no
    // correction).
    Float alpha = emitter->thermal_alpha(),
          eps   = emitter->thermal_emissivity();
    Float F_sky = 1.f;
    Float dT_exchange = 0.f;
    if (sky_rays > 0) {
        Frame<Float> frame(n);
        Float visible = 0.f;
        UInt32 m_recv = emitter->thermal_material_index();
        Float h_air = emitter->thermal_h_air(),
              s_ref = emitter->thermal_s_ref();
        for (uint32_t i = 0; i < sky_rays; ++i) {
            Point<Float, 2> u = sampler->next_2d(active);
            Vector3f d_local = warp::square_to_cosine_hemisphere(u);
            Vector3f d = frame.to_world(d_local);
            Mask above = dr::dot(d, Vector3f(up)) > ScalarFloat(0);
            Ray3f ray(o, d, time, wavelengths);
            auto si_hit = thermal_trace_opaque<Float, Spectrum>(scene, ray, active);
            Mask hit = si_hit.is_valid();
            dr::masked(visible, above && !hit) += ScalarFloat(1);
            auto hit_emitter = si_hit.emitter(scene, UInt32((uint32_t) RayMask::Secondary), hit);
            Mask has_em = hit && (hit_emitter != nullptr);
            if constexpr (dr::is_jit_v<Float>) {
                auto [d_lw, d_sw] = hit_emitter->thermal_exchange(si_hit, m_recv, h_air, s_ref, has_em);
                dr::masked(dT_exchange, has_em) += eps * d_lw + alpha * d_sw;
            } else {
                // Scalar variants: EmitterPtr is a raw pointer (null for
                // geometry without an emitter)
                if (has_em) {
                    auto [d_lw, d_sw] = hit_emitter->thermal_exchange(si_hit, m_recv, h_air, s_ref, has_em);
                    dT_exchange += eps * d_lw + alpha * d_sw;
                }
            }
        }
        F_sky = visible / ScalarFloat(sky_rays);
        dT_exchange /= ScalarFloat(sky_rays);
    }

    // --- linear response and quasi-steady correction ---------------------------
    Float dT_lin = alpha * (ls.acc - emitter->thermal_c_ref())
                   + (F_sky - ScalarFloat(1)) * emitter->thermal_c_sky()
                   + dT_exchange;
    Float h_lin  = emitter->thermal_h_total(),
          h_conv = emitter->thermal_h_conv(),
          t_ref  = emitter->thermal_t_ref();
    // Quasi-steady correction for the temperature dependence of the
    // radiative conductance: the response function was linearised with the
    // conductance h_lin; the actual conductance at this point and time is
    // h_conv + 4 eps sigma T_mid^3 with T_mid halfway between the reference
    // and the point temperature (report 6.8.3).
    Float T_mid = t_ref + ScalarFloat(0.5) * dT_lin;
    Float h_act = h_conv + ScalarFloat(4.0 * 5.670374419e-8) * eps * T_mid * T_mid * T_mid;
    Float corr = dr::select(h_lin > ScalarFloat(0),
                            h_lin / dr::maximum(h_act, ScalarFloat(0.25) * h_lin),
                            Float(1.f));
    return t_ref + dT_lin * corr;
}

/**
 * \brief Radiance scale factor B(lambda, T) / B(lambda, T_ref) of a
 * thermal-dynamic emitter at surface point (p, n), where T is the
 * shading-time temperature. Integrators multiply the emitter's own result
 * (``eval()`` or the next-event-estimation weight, both evaluated at T_ref)
 * by it, which preserves everything the emitter applied to that result
 * (one-sided culling, masks). Returns 1 outside the ``active`` lanes and in
 * non-spectral variants.
 */
template <typename Float, typename Spectrum, typename Scene, typename EmitterPtr,
          typename Sampler, typename Wavelength>
unpolarized_spectrum_t<Spectrum>
thermal_dynamic_scale(const Scene *scene, const EmitterPtr &emitter,
                      const Point<Float, 3> &p, const Normal<Float, 3> &n,
                      const Vector<dr::scalar_t<Float>, 3> &up, Sampler *sampler,
                      uint32_t sky_rays, const Float &time,
                      const Wavelength &wavelengths, dr::mask_t<Float> active) {
    using UnpolarizedSpectrum = unpolarized_spectrum_t<Spectrum>;
    using ScalarFloat = dr::scalar_t<Float>;
    if constexpr (is_spectral_v<Spectrum>) {
        Float T = thermal_dynamic_temperature<Float, Spectrum>(
            scene, emitter, p, n, up, sampler, sky_rays, time, wavelengths, active);
        UnpolarizedSpectrum B     = planck_radiance(wavelengths, T),
                            B_ref = planck_radiance(wavelengths, Float(emitter->thermal_t_ref()));
        return dr::select(active, dr::select(B_ref > ScalarFloat(0), B / B_ref, ScalarFloat(0)),
                          UnpolarizedSpectrum(ScalarFloat(1)));
    } else {
        DRJIT_MARK_USED(scene); DRJIT_MARK_USED(emitter); DRJIT_MARK_USED(p);
        DRJIT_MARK_USED(n); DRJIT_MARK_USED(up); DRJIT_MARK_USED(sampler);
        DRJIT_MARK_USED(sky_rays); DRJIT_MARK_USED(time);
        DRJIT_MARK_USED(wavelengths); DRJIT_MARK_USED(active);
        return UnpolarizedSpectrum(ScalarFloat(1));
    }
}

NAMESPACE_END(mitsuba)
