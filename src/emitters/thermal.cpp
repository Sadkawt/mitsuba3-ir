#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>
#include <drjit/traversable_base.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-thermal:

Thermal emitter (:monosp:`thermal`)
-----------------------------------

.. pluginparameters::

 * - temperature
   - |float| or |texture|
   - Absolute surface temperature in Kelvin. A texture can be used to describe
     a spatially varying temperature field. (Default: 300 K)
   - |exposed|

 * - twosided
   - |bool|
   - Emit from both sides of the surface. (Default: |true|)

 * - wavelength_min, wavelength_max
   - |float|
   - Wavelength range (in nanometers) used when *this emitter* has to sample
     wavelengths, which only happens in light-tracing integrators
     (``ptracer``). Camera-based integrators sample wavelengths from the
     sensor response functions of the film instead. (Default: 360, 14000)

This plugin turns the shape it is attached to into a thermal (Planckian)
radiator. It is the basis of infrared rendering in Mitsuba: every surface with
a temperature above absolute zero emits radiation according to Planck's law,

.. math::

    B_\lambda(\lambda, T) = \frac{2 h c^2}{\lambda^5}
        \frac{1}{\exp\left(\frac{h c}{\lambda k_B T}\right) - 1},

expressed here in :math:`\mathrm{W\,m^{-2}\,sr^{-1}\,nm^{-1}}`. The scene
must therefore be modeled in meters. The physical constants are the exact
2019 SI values.

**Emissivity is not a parameter of this plugin.** For a surface in local
thermodynamic equilibrium, Kirchhoff's law states that the directional
spectral emissivity equals one minus the directional-hemispherical
reflectance (plus transmittance for non-opaque surfaces),
:math:`\varepsilon(\lambda, \omega) = 1 - \rho_{dh}(\lambda, \omega)`.
The emissivity therefore follows from the BSDF attached to the shape, which
guarantees energy conservation and makes the same material description valid
in the reflective (SWIR), mixed (MWIR) and emissive (LWIR) regimes.

The plugin reports the *blackbody* radiance :math:`B_\lambda(T)` of the
surface; thermal-aware integrators such as :ref:`path_ir <integrator-path_ir>`
apply Kirchhoff's law through the identity

.. math::

    L_o = B(T) + \int_{S^2} f(\omega_i, \omega_o)\,
        \big(L_i(\omega_i) - B(T)\big)\, |\cos\theta_i|\, d\omega_i,

which needs no explicit evaluation of the emissivity. Integrators that are
not thermal-aware render the surface as a blackbody and a warning is issued
when the scene is loaded.

To use measured emissivity spectra (e.g. from the ECOSTRESS spectral
library), set the reflectance of a :ref:`diffuse <bsdf-diffuse>` BSDF to
:math:`1 - \varepsilon(\lambda)`; see the infrared technical report in
``docs/ir/``.

.. tabs::
    .. code-tab:: xml
        :name: thermal-emitter

        <shape type="obj">
            <string name="filename" value="tank.obj"/>
            <bsdf type="roughconductor">
                <string name="material" value="Fe_ir"/>
            </bsdf>
            <emitter type="thermal">
                <float name="temperature" value="310"/>
            </emitter>
        </shape>

    .. code-tab:: python

        'type': 'obj',
        'filename': 'tank.obj',
        'bsdf': {
            'type': 'roughconductor',
            'material': 'Fe_ir'
        },
        'emitter': {
            'type': 'thermal',
            'temperature': 310.0
        }

 */

template <typename Float, typename Spectrum>
class ThermalEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_shape, m_medium)
    MI_IMPORT_TYPES(Scene, Shape, Texture)

    // Exact SI (2019) values of the physical constants
    constexpr static double c_light = 299792458.0;     /// Speed of light [m/s]
    constexpr static double h_planck = 6.62607015e-34; /// Planck constant [J s]
    constexpr static double k_boltz = 1.380649e-23;    /// Boltzmann constant [J/K]

    /// First radiation constant for spectral radiance, 2 h c^2 [W m^2 sr^-1]
    constexpr static double c1L = 2.0 * h_planck * c_light * c_light;
    /// Second radiation constant, h c / k_B [m K]
    constexpr static double c2 = h_planck * c_light / k_boltz;

    ThermalEmitter(const Properties &props) : Base(props) {
        if constexpr (!is_spectral_v<Spectrum>)
            Throw("The 'thermal' emitter requires a spectral variant of "
                  "Mitsuba (e.g. scalar_spectral, llvm_ad_spectral, "
                  "cuda_ad_spectral).");

        if (props.has_property("to_world"))
            Throw("Found a 'to_world' transformation -- this is not allowed. "
                  "The thermal emitter inherits this transformation from its "
                  "parent shape.");

        m_temperature = props.get_texture<Texture>("temperature", 300.f);
        m_twosided = props.get<bool>("twosided", true);
        m_wavelength_range = ScalarVector2f(
            props.get<ScalarFloat>("wavelength_min", 360.f),
            props.get<ScalarFloat>("wavelength_max", 14000.f));

        if (m_wavelength_range.x() <= 0.f ||
            m_wavelength_range.y() <= m_wavelength_range.x())
            Throw("Invalid wavelength range [%f, %f] nm.",
                  m_wavelength_range.x(), m_wavelength_range.y());

        m_flags = +EmitterFlags::Surface | +EmitterFlags::Thermal;
        if (m_temperature->is_spatially_varying())
            m_flags |= +EmitterFlags::SpatiallyVarying;
    }

    void traverse(TraversalCallback *cb) override {
        Base::traverse(cb);
        cb->put("temperature", m_temperature, ParamFlags::NonDifferentiable);
    }

    /**
     * Planck's law, evaluated at the wavelengths stored in ``si`` and the
     * temperature of the surface at ``si``. Units: W m^-2 sr^-1 nm^-1.
     */
    UnpolarizedSpectrum planck(const SurfaceInteraction3f &si, Mask active) const {
        if constexpr (is_spectral_v<Spectrum>) {
            Float T = m_temperature->eval_1(si, active);

            // Wavelengths are stored in nanometers; convert to meters.
            UnpolarizedSpectrum lambda  = si.wavelengths * 1e-9f,
                                lambda2 = dr::square(lambda),
                                lambda5 = dr::square(lambda2) * lambda;

            // exp() overflows to +inf for very cold surfaces / short
            // wavelengths; the radiance then correctly evaluates to zero.
            UnpolarizedSpectrum x = ScalarFloat(c2) / (lambda * T);
            UnpolarizedSpectrum B = ScalarFloat(c1L) / (lambda5 * (dr::exp(x) - 1.f));

            // Density per nanometer rather than per meter
            B *= 1e-9f;

            return dr::select(active && T > 0.f, B, 0.f);
        } else {
            DRJIT_MARK_USED(si);
            DRJIT_MARK_USED(active);
            return 0.f;
        }
    }

    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        Spectrum result = depolarizer<Spectrum>(planck(si, active));
        if (!m_twosided)
            result = dr::select(Frame3f::cos_theta(si.wi) > 0.f, result, 0.f);
        return result;
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &sample2, const Point2f &sample3,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // 1. Sample spatial component
        auto [ps, pos_weight] = sample_position(time, sample2, active);

        // 2. Sample directional component. Two-sided emitters reuse the first
        // sample dimension to pick a side.
        Vector3f local;
        if (m_twosided) {
            Point2f sample_dir = sample3;
            Mask flip = sample_dir.x() >= .5f;
            sample_dir.x() = dr::select(flip, dr::fmadd(sample_dir.x(), 2.f, -1.f),
                                        sample_dir.x() * 2.f);
            local = warp::square_to_cosine_hemisphere(sample_dir);
            dr::masked(local.z(), flip) = -local.z();
        } else {
            local = warp::square_to_cosine_hemisphere(sample3);
        }

        // 3. Sample spectral component
        SurfaceInteraction3f si(ps, dr::zeros<Wavelength>());
        auto [wavelength, wav_weight] =
            sample_wavelengths(si, wavelength_sample, active);
        si.time = time;
        si.wavelengths = wavelength;

        // Note: some terms cancelled out with `warp::square_to_cosine_hemisphere_pdf`.
        Spectrum weight = pos_weight * wav_weight *
                          (m_twosided ? 2.f : 1.f) * dr::Pi<ScalarFloat>;

        return { si.spawn_ray(si.to_world(local)),
                 depolarizer<Spectrum>(weight) };
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        if constexpr (drjit::is_jit_v<Float>) {
            if (!m_shape)
                return { dr::zeros<DirectionSample3f>(), 0.f };
        } else {
            Assert(m_shape, "Can't sample from a thermal emitter without an "
                            "associated Shape.");
        }

        // Importance sample the shape wrt. solid angle at 'it'
        DirectionSample3f ds = m_shape->sample_direction(it, sample, active);
        active &= ds.pdf != 0.f;
        if (!m_twosided)
            active &= dr::dot(ds.d, ds.n) < 0.f;

        SurfaceInteraction3f si(ds, it.wavelengths);
        UnpolarizedSpectrum spec = planck(si, active) / ds.pdf;
        ds.emitter = this;
        return { ds, depolarizer<Spectrum>(spec) & active };
    }

    Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        Float dp = dr::dot(ds.d, ds.n);
        if (m_twosided)
            dp = -dr::abs(dp);
        active &= dp < 0.f;

        if constexpr (drjit::is_jit_v<Float>) {
            if (!m_shape)
                return 0.f;
        } else {
            Assert(m_shape, "The thermal emitter has no associated Shape!");
        }

        Float value = m_shape->pdf_direction(it, ds, active);
        return dr::select(active, value, 0.f);
    }

    Spectrum eval_direction(const Interaction3f &it,
                            const DirectionSample3f &ds,
                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        Float dp = dr::dot(ds.d, ds.n);
        if (m_twosided)
            dp = -dr::abs(dp);
        active &= dp < 0.f;

        SurfaceInteraction3f si(ds, it.wavelengths);
        UnpolarizedSpectrum spec = planck(si, active);
        return dr::select(active, depolarizer<Spectrum>(spec), 0.f);
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float time, const Point2f &sample,
                    Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSamplePosition, active);

        if constexpr (drjit::is_jit_v<Float>) {
            if (!m_shape)
                return { dr::zeros<PositionSample3f>(), 0.f };
        } else {
            Assert(m_shape, "Cannot sample from a thermal emitter without an "
                            "associated Shape.");
        }

        PositionSample3f ps = m_shape->sample_position(time, sample, active);
        Float weight = dr::select(active && ps.pdf > 0.f, dr::rcp(ps.pdf), Float(0.f));
        return { ps, weight };
    }

    /**
     * Uniform wavelength sampling within ``[wavelength_min, wavelength_max]``.
     * Only used by light tracing integrators; camera-based integrators obtain
     * wavelengths from the sensor/film response functions.
     */
    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f &si, Float sample,
                       Mask active) const override {
        if constexpr (is_spectral_v<Spectrum>) {
            Wavelength u = math::sample_shifted<Wavelength>(sample);
            ScalarFloat extent = m_wavelength_range.y() - m_wavelength_range.x();
            Wavelength wavelengths = dr::fmadd(u, extent, m_wavelength_range.x());

            SurfaceInteraction3f si2 = si;
            si2.wavelengths = wavelengths;
            return { wavelengths, depolarizer<Spectrum>(planck(si2, active) * extent) };
        } else {
            DRJIT_MARK_USED(si);
            DRJIT_MARK_USED(sample);
            DRJIT_MARK_USED(active);
            return { dr::zeros<Wavelength>(), 0.f };
        }
    }

    ScalarBoundingBox3f bbox() const override { return m_shape->bbox(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "ThermalEmitter[" << std::endl
            << "  temperature = " << string::indent(m_temperature) << "," << std::endl
            << "  twosided = " << m_twosided << "," << std::endl
            << "  wavelength_range = " << m_wavelength_range << "," << std::endl
            << "  surface_area = ";
        if (m_shape) oss << m_shape->surface_area();
        else         oss << "  <no shape attached!>";
        oss << "," << std::endl;
        if (m_medium) oss << string::indent(m_medium);
        else         oss << "  <no medium attached!>";
        oss << std::endl << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(ThermalEmitter)
private:
    ref<Texture> m_temperature;
    bool m_twosided;
    ScalarVector2f m_wavelength_range;

    MI_TRAVERSE_CB(Base, m_temperature)
};

MI_EXPORT_PLUGIN(ThermalEmitter)
NAMESPACE_END(mitsuba)
