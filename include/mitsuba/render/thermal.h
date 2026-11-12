#pragma once

#include <mitsuba/core/fwd.h>
#include <drjit/array.h>
#include <drjit/math.h>

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

NAMESPACE_END(mitsuba)
