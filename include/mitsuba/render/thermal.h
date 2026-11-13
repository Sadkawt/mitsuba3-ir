#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
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
template <typename Float, typename Spectrum, typename Scene, typename EmitterPtr,
          typename Sampler>
Float thermal_dynamic_temperature(const Scene *scene, const EmitterPtr &emitter,
                                  const Point<Float, 3> &p, const Normal<Float, 3> &n,
                                  const Vector<dr::scalar_t<Float>, 3> &up, Sampler *sampler,
                                  uint32_t sky_rays, const Float &time,
                                  dr::mask_t<Float> active) {
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
    // The sun direction is jittered within each forcing step (one stratum
    // sample per shading point, interpolating consecutive table entries), so
    // that the visibility is integrated over the step instead of being
    // sampled at its centre: this removes the time-step banding of moving
    // shadow edges without bias.
    UInt32 n_steps = emitter->thermal_steps();
    Float u_time = sampler->next_1d(active);
    struct Loop {
        UInt32 k;
        Float acc;
        DRJIT_STRUCT(Loop, k, acc)
    } ls = { UInt32(0), Float(0.f) };

    dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
        [n_steps](const Loop &l) { return l.k < n_steps; },
        [&](Loop &l) {
            auto [s0, w] = emitter->thermal_forcing(l.k, active);
            UInt32 k1 = dr::minimum(l.k + 1, n_steps - 1);
            auto [s1, w1] = emitter->thermal_forcing(k1, active);
            // Interpolate towards the next entry only when it is a
            // consecutive daylight step (non-zero weight)
            Vector3f s = dr::select(w1 != ScalarFloat(0), dr::normalize(s0 + u_time * (s1 - s0)), s0);
            Float cos_i = dr::dot(n, s);
            Mask lit = active && (cos_i > ScalarFloat(1e-4)) && (w != ScalarFloat(0));
            Ray3f ray(o, s, time);
            Mask occluded = scene->ray_test(ray, Mask(false), UInt32((uint32_t) RayMask::Secondary), lit);
            dr::masked(l.acc, lit && !occluded) += w * cos_i;
            l.k += 1;
        });

    // --- sky-view factor: cosine-weighted hemisphere rays ---------------------
    Float F_sky = 1.f;
    if (sky_rays > 0) {
        Frame<Float> frame(n);
        Float visible = 0.f;
        for (uint32_t i = 0; i < sky_rays; ++i) {
            Point<Float, 2> u = sampler->next_2d(active);
            Vector3f d_local = warp::square_to_cosine_hemisphere(u);
            Vector3f d = frame.to_world(d_local);
            Mask above = dr::dot(d, Vector3f(up)) > ScalarFloat(0);
            Mask occluded = scene->ray_test(Ray3f(o, d, time), Mask(false),
                                            UInt32((uint32_t) RayMask::Secondary), active && above);
            dr::masked(visible, above && !occluded) += ScalarFloat(1);
        }
        F_sky = visible / ScalarFloat(sky_rays);
    }

    // --- linear response and quasi-steady correction ---------------------------
    Float alpha  = emitter->thermal_alpha(),
          dT_lin = alpha * (ls.acc - emitter->thermal_c_ref())
                   + (F_sky - ScalarFloat(1)) * emitter->thermal_c_sky();
    Float h_lin  = emitter->thermal_h_total(),
          h_conv = emitter->thermal_h_conv(),
          eps    = emitter->thermal_emissivity(),
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

NAMESPACE_END(mitsuba)
