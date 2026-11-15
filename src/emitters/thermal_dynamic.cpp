#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/thermal.h>
#include <drjit/traversable_base.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-thermal_dynamic:

Shading-time thermal emitter (:monosp:`thermal_dynamic`)
--------------------------------------------------------

.. pluginparameters::

 * - t_ref
   - |float|
   - Surface temperature [K] of the reference element of this material at
     the render time (fully exposed, no shadows, sky-view factor 1).

 * - alpha_solar
   - |float|
   - Broadband solar absorptance of the thermal model.

 * - weights
   - |string|
   - Comma-separated list of ``K`` weights :math:`w_k` [K]: the surface
     temperature response at the render time to the direct normal
     irradiance of forcing step ``k`` (response function × DNI × Δt).

 * - sun_directions
   - |string|
   - Comma-separated list of ``3K`` components of the unit vectors towards
     the sun at the forcing steps (world space).

 * - c_ref
   - |float|
   - :math:`\sum_k w_k \cos\theta_{ref,k}` of the reference element [K].

 * - c_sky
   - |float|
   - Temperature change per unit decrease of the sky-view factor [K].

 * - h_total, h_conv, emissivity
   - |float|
   - Surface conductance :math:`h_{conv} + 4\varepsilon\sigma T_{lin}^3` used to
     linearise the response function, the convective coefficient and the
     broadband emissivity, used for the quasi-steady nonlinearity correction.

 * - material_index, h_air, s_ref
   - |int|, |float|, |float|
   - Receiver index of this shape's thermal material into the exchange fields
     and the baselines of the longwave / shortwave exchange fields for that
     material (:math:`\sum_k G_k \sigma T_{air,k}^4`, :math:`\sum_k G_k \rho_g E_{gh,k}`) [K].

 * - exchange_lw_<m>, exchange_sw_<m>
   - |texture|
   - Radiative-exchange fields of *this* surface for receivers of thermal
     material ``m``: the convolution of its longwave radiosity / reflected
     solar irradiance history with the receiver's response function [K per
     unit view factor]. Produced by ``ThermalScene.dynamic()`` from the
     per-element solver; optional.

 * - twosided
   - |bool|
   - Emit from both faces. (Default: |true|)

Thermal emitter whose surface temperature is not stored but *evaluated at
shading time* by the thermal-aware integrators (``path_ir``, ``volpath_ir``)
with the linear-response model of the surface energy balance:

.. math::

    T(x) = T_{ref} + c\Big(\alpha_s\big[\textstyle\sum_k w_k \cos\theta_k\, V_k(x) - C_{ref}\big]
           + (F_{sky}(x) - 1)\, C_{sky}\Big),

where :math:`V_k(x)` is the visibility of the sun at forcing step ``k``
(a shadow ray cast through the scene from ``x``), :math:`F_{sky}(x)` the
cosine-weighted sky-view factor (hemisphere rays) and :math:`c(\cdot)` the
correction for the temperature dependence of the radiative conductance.
Thermal shadows therefore have the geometric sharpness of the ray tracer
at any image resolution, and the cost scales with the number of shading
points rather than with a texture resolution. The tables are produced by
``mitsuba.ir.thermal.ThermalScene.dynamic()`` from a ``Weather`` and a
``Material``; see docs/ir/IR_TECHNICAL_REPORT.md, section 6.8.3.

Integrators that are not thermal-aware see the reference temperature.

 */

template <typename Float, typename Spectrum>
class ThermalDynamicEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_shape, m_medium)
    MI_IMPORT_TYPES(Scene, Shape, Texture)
    using FloatStorage = DynamicBuffer<Float>;

    ThermalDynamicEmitter(const Properties &props) : Base(props) {
        if constexpr (!is_spectral_v<Spectrum>)
            Throw("The 'thermal_dynamic' emitter requires a spectral variant of Mitsuba.");
        if (props.has_property("to_world"))
            Throw("Found a 'to_world' transformation -- this is not allowed. "
                  "The emitter inherits this transformation from its parent shape.");

        m_t_ref   = props.get<ScalarFloat>("t_ref");
        m_alpha   = props.get<ScalarFloat>("alpha_solar", 0.7f);
        m_c_ref   = props.get<ScalarFloat>("c_ref", 0.f);
        m_c_sky   = props.get<ScalarFloat>("c_sky", 0.f);
        m_h_total = props.get<ScalarFloat>("h_total", 0.f);
        m_h_conv  = props.get<ScalarFloat>("h_conv", 0.f);
        m_eps     = props.get<ScalarFloat>("emissivity", 0.9f);
        m_twosided = props.get<bool>("twosided", true);
        m_material_index = (uint32_t) props.get<int>("material_index", 0);
        m_h_air = props.get<ScalarFloat>("h_air", 0.f);
        m_s_ref = props.get<ScalarFloat>("s_ref", 0.f);
        for (uint32_t m = 0; ; ++m) {
            std::string lw = "exchange_lw_" + std::to_string(m),
                        sw = "exchange_sw_" + std::to_string(m);
            if (!props.has_property(lw))
                break;
            m_exchange_lw.push_back(props.get_texture<Texture>(lw));
            m_exchange_sw.push_back(props.has_property(sw) ? props.get_texture<Texture>(sw)
                                                           : props.get_texture<Texture>(sw, 0.f));
        }

        std::vector<ScalarFloat> w = parse_list(props.get<std::string_view>("weights", ""));
        std::vector<ScalarFloat> d = parse_list(props.get<std::string_view>("sun_directions", ""));
        if (d.size() != 3 * w.size())
            Throw("'sun_directions' must hold 3 values per entry of 'weights' (%zu vs %zu).",
                  d.size(), w.size());
        m_steps = (uint32_t) w.size();
        if (m_steps == 0) {
            // Keep the tables non-empty: the (masked) gather in the
            // integrator's loop is recorded symbolically in JIT variants
            w.push_back(0.f);
            d = { 0.f, 0.f, 1.f };
        }
        uint32_t n = (uint32_t) w.size();
        std::vector<ScalarFloat> dx(n), dy(n), dz(n);
        for (uint32_t k = 0; k < n; ++k) {
            dx[k] = d[3 * k]; dy[k] = d[3 * k + 1]; dz[k] = d[3 * k + 2];
        }
        m_weights = dr::load<FloatStorage>(w.data(), w.size());
        m_sun_x   = dr::load<FloatStorage>(dx.data(), dx.size());
        m_sun_y   = dr::load<FloatStorage>(dy.data(), dy.size());
        m_sun_z   = dr::load<FloatStorage>(dz.data(), dz.size());

        m_flags = +EmitterFlags::Surface | +EmitterFlags::Thermal |
                  +EmitterFlags::ThermalDynamic | +EmitterFlags::SpatiallyVarying;
    }

    static std::vector<ScalarFloat> parse_list(std::string_view text) {
        std::vector<ScalarFloat> out;
        for (auto &tok : string::tokenize(text, " ,\t\n")) {
            if (!tok.empty())
                out.push_back((ScalarFloat) std::stod(std::string(tok)));
        }
        return out;
    }

    void traverse(TraversalCallback *cb) override {
        Base::traverse(cb);
        cb->put("t_ref", m_t_ref, ParamFlags::NonDifferentiable);
    }

    // --- thermal-dynamic accessors -------------------------------------------
    std::pair<Vector3f, Float> thermal_forcing(UInt32 k, Mask active) const override {
        Float w = dr::gather<Float>(m_weights, k, active);
        Vector3f s(dr::gather<Float>(m_sun_x, k, active),
                   dr::gather<Float>(m_sun_y, k, active),
                   dr::gather<Float>(m_sun_z, k, active));
        return { s, w };
    }
    uint32_t thermal_steps() const override { return m_steps; }
    ScalarFloat thermal_t_ref() const override { return m_t_ref; }
    ScalarFloat thermal_alpha() const override { return m_alpha; }
    ScalarFloat thermal_c_ref() const override { return m_c_ref; }
    ScalarFloat thermal_c_sky() const override { return m_c_sky; }
    ScalarFloat thermal_h_total() const override { return m_h_total; }
    ScalarFloat thermal_h_conv() const override { return m_h_conv; }
    ScalarFloat thermal_emissivity() const override { return m_eps; }
    uint32_t thermal_material_index() const override { return m_material_index; }
    ScalarFloat thermal_h_air() const override { return m_h_air; }
    ScalarFloat thermal_s_ref() const override { return m_s_ref; }

    std::pair<Float, Float> thermal_exchange(const SurfaceInteraction3f &si, UInt32 m,
                                             Float h_air, Float s_ref, Mask active) const override {
        if (m_exchange_lw.empty())
            return { 0.f, 0.f };
        Float lw = 0.f, sw = 0.f;
        Mask any = false;
        for (uint32_t i = 0; i < (uint32_t) m_exchange_lw.size(); ++i) {
            Mask sel = active && (m == i);
            dr::masked(lw, sel) = m_exchange_lw[i]->eval_1(si, sel) - h_air;
            dr::masked(sw, sel) = m_exchange_sw[i]->eval_1(si, sel) - s_ref;
            any |= sel;
        }
        return { dr::select(any, lw, 0.f), dr::select(any, sw, 0.f) };
    }

    // --- emitter interface: the reference temperature is used where the
    //     integrator does not evaluate the dynamic model -------------------------
    UnpolarizedSpectrum planck_ref(const Wavelength &wavelengths, Mask active) const {
        if constexpr (is_spectral_v<Spectrum>)
            return dr::select(active, planck_radiance(wavelengths, Float(m_t_ref)), 0.f);
        else {
            DRJIT_MARK_USED(wavelengths); DRJIT_MARK_USED(active);
            return 0.f;
        }
    }

    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        Spectrum result = depolarizer<Spectrum>(planck_ref(si.wavelengths, active));
        if (!m_twosided)
            result = dr::select(Frame3f::cos_theta(si.wi) > 0.f, result, 0.f);
        return result;
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &sample2, const Point2f &sample3,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        auto [ps, pos_weight] = sample_position(time, sample2, active);
        Vector3f local;
        if (m_twosided) {
            Point2f sample_dir = sample3;
            Mask flip = sample_dir.x() >= .5f;
            sample_dir.x() = dr::select(flip, dr::fmadd(sample_dir.x(), 2.f, -1.f), sample_dir.x() * 2.f);
            local = warp::square_to_cosine_hemisphere(sample_dir);
            dr::masked(local.z(), flip) = -local.z();
        } else {
            local = warp::square_to_cosine_hemisphere(sample3);
        }
        SurfaceInteraction3f si(ps, dr::zeros<Wavelength>());
        auto [wavelength, wav_weight] = sample_wavelengths(si, wavelength_sample, active);
        si.time = time;
        si.wavelengths = wavelength;
        Spectrum weight = pos_weight * wav_weight * (m_twosided ? 2.f : 1.f) * dr::Pi<ScalarFloat>;
        return { si.spawn_ray(si.to_world(local)), depolarizer<Spectrum>(weight) };
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);
        if constexpr (drjit::is_jit_v<Float>) {
            if (!m_shape)
                return { dr::zeros<DirectionSample3f>(), 0.f };
        } else {
            Assert(m_shape, "Can't sample from a thermal emitter without an associated Shape.");
        }
        DirectionSample3f ds = m_shape->sample_direction(it, sample, active);
        active &= ds.pdf != 0.f;
        if (!m_twosided)
            active &= dr::dot(ds.d, ds.n) < 0.f;
        UnpolarizedSpectrum spec = planck_ref(it.wavelengths, active) / ds.pdf;
        ds.emitter = this;
        return { ds, depolarizer<Spectrum>(spec) & active };
    }

    Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds, Mask active) const override {
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
        return dr::select(active, m_shape->pdf_direction(it, ds, active), 0.f);
    }

    Spectrum eval_direction(const Interaction3f &it, const DirectionSample3f &ds, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        Float dp = dr::dot(ds.d, ds.n);
        if (m_twosided)
            dp = -dr::abs(dp);
        active &= dp < 0.f;
        return dr::select(active, depolarizer<Spectrum>(planck_ref(it.wavelengths, active)), 0.f);
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float time, const Point2f &sample, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSamplePosition, active);
        if constexpr (drjit::is_jit_v<Float>) {
            if (!m_shape)
                return { dr::zeros<PositionSample3f>(), 0.f };
        } else {
            Assert(m_shape, "Cannot sample from a thermal emitter without an associated Shape.");
        }
        PositionSample3f ps = m_shape->sample_position(time, sample, active);
        Float weight = dr::select(active && ps.pdf > 0.f, dr::rcp(ps.pdf), Float(0.f));
        return { ps, weight };
    }

    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f & /*si*/, Float sample, Mask active) const override {
        if constexpr (is_spectral_v<Spectrum>) {
            Wavelength u = math::sample_shifted<Wavelength>(sample);
            ScalarFloat lo = 360.f, extent = 20000.f - 360.f;
            Wavelength wavelengths = dr::fmadd(u, extent, lo);
            return { wavelengths, depolarizer<Spectrum>(planck_ref(wavelengths, active) * extent) };
        } else {
            DRJIT_MARK_USED(sample); DRJIT_MARK_USED(active);
            return { dr::zeros<Wavelength>(), 0.f };
        }
    }

    ScalarBoundingBox3f bbox() const override { return m_shape->bbox(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "ThermalDynamicEmitter[" << std::endl
            << "  t_ref = " << m_t_ref << "," << std::endl
            << "  alpha_solar = " << m_alpha << "," << std::endl
            << "  steps = " << m_steps << "," << std::endl
            << "  c_ref = " << m_c_ref << ", c_sky = " << m_c_sky << "," << std::endl
            << "  h_total = " << m_h_total << ", h_conv = " << m_h_conv << ", emissivity = " << m_eps << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(ThermalDynamicEmitter)
private:
    ScalarFloat m_t_ref, m_alpha, m_c_ref, m_c_sky, m_h_total, m_h_conv, m_eps, m_h_air, m_s_ref;
    bool m_twosided;
    uint32_t m_steps, m_material_index;
    FloatStorage m_weights, m_sun_x, m_sun_y, m_sun_z;
    std::vector<ref<Texture>> m_exchange_lw, m_exchange_sw;

    MI_TRAVERSE_CB(Base, m_weights, m_sun_x, m_sun_y, m_sun_z, m_exchange_lw, m_exchange_sw)
};

MI_EXPORT_PLUGIN(ThermalDynamicEmitter)
NAMESPACE_END(mitsuba)
