#include <mitsuba/render/texture.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _spectrum-thermal:

Thermal grey-body spectrum (:monosp:`thermal`)
----------------------------------------------

.. pluginparameters::

 * - temperature
   - |float| or |texture|
   - Absolute temperature in Kelvin. May be spatially varying (e.g. a heat map
     baked from a thermal solver such as TAITherm and imported as a floating
     point bitmap). (Default: 5000 K)
   - |exposed|, |differentiable|

 * - emissivity
   - |float| or |texture|
   - Grey-body emissivity :math:`\varepsilon \in [0, 1]`, assumed constant across
     wavelength. May be spatially varying. (Default: 1, i.e. an ideal black body)
   - |exposed|, |differentiable|

 * - wavelength_min / wavelength_max
   - |float|
   - Spectral range (nm) outside of which the emission is clamped to zero.
     (Defaults: 360 nm / 830 nm -- widen these for infrared work.)

This spectrum evaluates spatially varying grey-body thermal emission following
Planck's law:

.. math::

    L_e(\lambda, \mathbf{x}) = \varepsilon(\mathbf{x}) \,
        \frac{2 h c^2}{\lambda^5}
        \frac{1}{\exp\!\left(\frac{h c}{\lambda\, k\, T(\mathbf{x})}\right) - 1}

Both the temperature :math:`T(\mathbf{x})` and the emissivity
:math:`\varepsilon(\mathbf{x})` can be provided as textures, which makes it the
natural input for per-texel heat maps. It is intended to be attached to a shape's
:monosp:`radiance` field (the lightweight emissive field), but also works as the
:monosp:`radiance` of an :monosp:`area` emitter.

Like the :ref:`blackbody <spectrum-blackbody>` spectrum, it carries physical units
(:math:`W\,m^{-2}\,sr^{-1}\,nm^{-1}`), so the scene should be modeled in meters,
and it is only available in **spectral** rendering modes.

.. tabs::
    .. code-tab:: python

        'type': 'sphere',
        'radiance': {
            'type': 'thermal',
            'temperature': {'type': 'bitmap', 'filename': 'heat_K.exr', 'raw': True},
            'emissivity':  {'type': 'bitmap', 'filename': 'emissivity.png', 'raw': True}
        }

 */

template <typename Float, typename Spectrum>
class ThermalSpectrum final : public Texture<Float, Spectrum> {
public:
    MI_IMPORT_TYPES(Texture)

    // Natural constants (SI)
    constexpr static ScalarFloat c = ScalarFloat(2.99792458e+8);   /// Speed of light
    constexpr static ScalarFloat h = ScalarFloat(6.62607004e-34);  /// Planck constant
    constexpr static ScalarFloat k = ScalarFloat(1.38064852e-23);  /// Boltzmann constant

    /// First and second radiation constants
    constexpr static ScalarFloat c0 = 2 * h * c * c;
    constexpr static ScalarFloat c1 = h * c / k;

    ThermalSpectrum(const Properties &props) : Texture(props) {
        if constexpr (!is_spectral_v<Spectrum>)
            Throw("The 'thermal' spectrum is only available in spectral "
                  "rendering modes (it evaluates Planck's law per wavelength).");

        // Temperature (Kelvin) can be far outside [0, 1], so use the unbounded path
        m_temperature = props.get_unbounded_texture<Texture>("temperature", 5000.f);
        // Grey-body emissivity in [0, 1]
        m_emissivity  = props.get_texture<Texture>("emissivity", 1.f);

        m_wavelength_range = ScalarVector2f(
            props.get<ScalarFloat>("wavelength_min", MI_CIE_MIN),
            props.get<ScalarFloat>("wavelength_max", MI_CIE_MAX));
    }

    void traverse(TraversalCallback *cb) override {
        cb->put("temperature", m_temperature, ParamFlags::Differentiable);
        cb->put("emissivity",  m_emissivity,  ParamFlags::Differentiable);
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        if constexpr (is_spectral_v<Spectrum>) {
            // Per-texel temperature (K) and grey-body emissivity
            Float temperature = m_temperature->eval_1(si, active);
            Float emissivity  = m_emissivity->eval_1(si, active);

            // Planck's law. The 1e-9 factors convert between densities per unit
            // nanometer and per unit meter (wavelengths are stored in nm).
            Wavelength lambda  = si.wavelengths * 1e-9f,
                       lambda2 = dr::square(lambda),
                       lambda5 = dr::square(lambda2) * lambda;

            dr::mask_t<Wavelength> in_range = active &&
                (si.wavelengths >= m_wavelength_range.x()) &&
                (si.wavelengths <= m_wavelength_range.y());

            UnpolarizedSpectrum P =
                1e-9f * c0 / (lambda5 * (dr::exp(c1 / (lambda * temperature)) - 1.f));

            return (emissivity * P) & in_range;
        } else {
            DRJIT_MARK_USED(si);
            DRJIT_MARK_USED(active);
            Throw("ThermalSpectrum::eval(): only available in spectral modes!");
        }
    }

    Float eval_1(const SurfaceInteraction3f &si, Mask active) const override {
        // Mean spectral radiance over the sampled wavelengths (scalar proxy)
        return dr::mean(eval(si, active));
    }

    bool is_spatially_varying() const override {
        return m_temperature->is_spatially_varying() ||
               m_emissivity->is_spatially_varying();
    }

    ScalarVector2f wavelength_range() const override { return m_wavelength_range; }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "ThermalSpectrum[" << std::endl
            << "  temperature = " << string::indent(m_temperature) << "," << std::endl
            << "  emissivity = " << string::indent(m_emissivity) << "," << std::endl
            << "  wavelength_range = " << m_wavelength_range << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(ThermalSpectrum)

private:
    ref<Texture> m_temperature;
    ref<Texture> m_emissivity;
    ScalarVector2f m_wavelength_range;

    MI_TRAVERSE_CB(Texture, m_temperature, m_emissivity)
};

MI_EXPORT_PLUGIN(ThermalSpectrum)
NAMESPACE_END(mitsuba)
