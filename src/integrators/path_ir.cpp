#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/thermal.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-path_ir:

Infrared path tracer (:monosp:`path_ir`)
----------------------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1
     corresponds to :math:`\infty`). A value of 1 will only render directly
     visible light sources. 2 will lead to single-bounce (direct-only)
     illumination, and so on. (Default: -1)

 * - rr_depth
   - |int|
   - Specifies the path depth, at which the implementation will begin to use
     the *russian roulette* path termination criterion. For example, if set to
     1, then path generation may randomly cease after encountering directly
     visible surfaces. (Default: 5)

 * - clamp_direct, clamp_indirect
   - |float|
   - Clamp positive (emitted) contributions of direct/indirect paths. The
     thermal difference terms are never clamped. (Default: 0, i.e. disabled)

This integrator is the unidirectional path tracer of Mitsuba (:ref:`path
<integrator-path>`) extended with *thermal self-emission* of surfaces for
infrared rendering. It handles emitters flagged as ``EmitterFlags::Thermal``
(see :ref:`thermal <emitter-thermal>`), which report the blackbody radiance
:math:`B(\lambda, T)` of a surface, and applies the directional spectral
emissivity implied by Kirchhoff's law without ever evaluating it explicitly.

**Difference formulation.** For an opaque surface in local thermodynamic
equilibrium with BSDF :math:`f`, Kirchhoff's law gives the emissivity
:math:`\varepsilon(\omega_o) = 1 - \int f(\omega_i,\omega_o)|\cos\theta_i|\,
d\omega_i`. Substituting into the rendering equation,

.. math::

    L_o = \varepsilon B + \int f L_i |\cos\theta_i| d\omega_i
        = B + \int f \,(L_i - B)\, |\cos\theta_i|\, d\omega_i .

The same identity holds for non-opaque surfaces when the integral runs over
the full sphere (reflection and transmission lobes). A path tracer therefore
estimates thermal emission by (i) adding the *full* blackbody radiance
:math:`B(T_k)` of every thermal vertex :math:`k` (with the usual multiple
importance sampling between emitter and BSDF sampling), and (ii) subtracting
:math:`B(T_k)` once, weighted by the BSDF sampling weight, whenever a
continuation ray leaves vertex :math:`k`. The subtraction is applied at the
first surface hit after the vertex (null surfaces included) or when the ray
escapes the scene; it is *not* applied when the path is truncated by
``max_depth``, which amounts to the assumption that the unknown remaining
radiance equals :math:`B(T_k)`---an excellent approximation inside
near-isothermal enclosures and the reason why this formulation truncates
with far less bias than the standard one.

Surfaces without a thermal emitter are treated as being at 0 K: they reflect
and transmit but do not emit, which coincides with the conventional
semantics of Mitsuba.

The formulation has two desirable properties. First, it is exact for any
BSDF for which the Monte Carlo weight :math:`f|\cos\theta_i| / p` equals the
albedo (e.g. diffuse, smooth conductors, smooth dielectrics): an isothermal
enclosure then renders with zero variance. Second, differences of similar
temperatures cancel, so the variance in a typical LWIR scene, where all
temperatures lie within a few tens of Kelvin, is much lower than that of a
formulation that samples the emissivity explicitly.

Non-thermal emitters (sun, sky, lamps) are handled exactly as in the regular
path tracer, so mixed scenes (e.g. MWIR with solar reflections) are supported.

.. note:: This integrator does not handle participating media. See
   ``volpath_ir`` for emissive media (atmosphere).

.. tabs::
    .. code-tab::  xml
        :name: path_ir-integrator

        <integrator type="path_ir">
            <integer name="max_depth" value="16"/>
        </integrator>

    .. code-tab:: python

        'type': 'path_ir',
        'max_depth': 16

 */

template <typename Float, typename Spectrum>
class PathIRIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth,
                   m_clamp_direct, m_clamp_indirect, clamp_contribution)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    PathIRIntegrator(const Properties &props) : Base(props) {
        m_thermal_up = dr::normalize(props.get<ScalarVector3f>("thermal_up", ScalarVector3f(0.f, 0.f, 1.f)));
        m_sky_rays = props.get<uint32_t>("thermal_sky_rays", 8);
    }

    bool thermal_aware() const override { return true; }

    std::pair<Spectrum, Bool> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const Ray3f &ray_,
                                     const Medium * /* medium */,
                                     Bool active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if (unlikely(m_max_depth == 0))
            return { 0.f, false };

        // --------------------- Configure loop state ----------------------

        Ray3f ray                     = ray_;
        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;
        Float eta                     = 1.f;
        PreliminaryIntersection3f pi  = dr::zeros<PreliminaryIntersection3f>();
        Mask valid_ray                = false;
        BSDFContext bsdf_ctx;

        // Null tests fold to literals in scenes without null shapes
        bool has_null = scene->has_null_shapes();

        // State of the last scattering vertex of the path. Null crossings
        // are not path vertices and leave it unchanged.
        struct Vertex {
            Interaction3f si;
            Float bsdf_pdf;
            Bool bsdf_delta;
            UInt32 depth;
            UInt32 ray_mask;
            DRJIT_STRUCT(Vertex, si, bsdf_pdf, bsdf_delta, depth, ray_mask)
        } vertex = { dr::zeros<Interaction3f>(), 1.f, true, 0,
                     +RayMask::Primary };

        struct LoopState {
            Ray3f ray;
            PreliminaryIntersection3f pi;
            Spectrum throughput;
            Spectrum result;
            Float eta;
            Mask valid_ray;
            Vertex vertex;
            // Blackbody radiance of the last scattering vertex (0 when the
            // vertex is not a thermal emitter), and whether its subtraction
            // is still pending for the current continuation ray.
            Spectrum B_prev;
            Mask diff_pending;
            Bool active;
            Sampler* sampler;
            DRJIT_STRUCT(LoopState, ray, pi, throughput, result, eta,
                         valid_ray, vertex, B_prev, diff_pending, active,
                         sampler)
        } ls = {
            ray,
            pi,
            throughput,
            result,
            eta,
            valid_ray,
            vertex,
            dr::zeros<Spectrum>(),
            false,
            active,
            sampler
        };

        // First bounce is usually coherent - don't reorder threads. The
        // camera mask hides emitters marked as invisible.
        ls.pi = scene->ray_intersect_preliminary(ls.ray,
                                                 /* coherent = */ true,
                                                 +RayMask::Primary,
                                                 ls.active);

        dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
            [](const LoopState& ls) { return ls.active; },
            [this, scene, bsdf_ctx, has_null](LoopState& ls) {

            // dr::while_loop implicitly masks all code in the loop using the
            // 'active' flag, so there is no need to pass it to every function

            // Fill out all information of the interaction
            SurfaceInteraction3f si = scene->compute_surface_interaction(
                ls.ray, ls.pi, +RayFlags::Default);

            // ------------------ Thermal difference term ------------------

            // The continuation ray from the last scattering vertex x
            // estimates int f (L_i - B(T_x)) |cos| dw. The '-B(T_x)' part is
            // accounted for once per continuation, at the first surface hit
            // (including null surfaces) or when the ray escapes. The weight
            // is the plain BSDF sampling weight (no MIS: this term is not an
            // emitter contribution).
            dr::masked(ls.result, ls.diff_pending) -= ls.throughput * ls.B_prev;
            ls.diff_pending = false;

            // ---------------------- Direct emission ----------------------

            // The emitter lookup uses the mask of the ray that produced 'si'
            EmitterPtr emitter = si.emitter(scene, ls.vertex.ray_mask);
            ls.valid_ray |= emitter != nullptr;

            // Emitted radiance at this vertex: thermal emitters report the
            // full blackbody radiance B(T), other emitters their radiance.
            Spectrum L_e = dr::zeros<Spectrum>();
            Mask is_thermal = false;
            if (dr::any_or<true>(emitter != nullptr)) {
                L_e = emitter->eval(si);
                is_thermal = has_flag(emitter->flags(), EmitterFlags::Thermal);

                // Shading-time thermal model: evaluate the surface
                // temperature from shadow and sky rays (report 6.8.3)
                Mask is_dyn = has_flag(emitter->flags(), EmitterFlags::ThermalDynamic);
                if (dr::any_or<true>(is_dyn)) {
                    Float T = thermal_dynamic_temperature<Float, Spectrum>(
                        scene, emitter, si.p, si.n, m_thermal_up, ls.sampler,
                        m_sky_rays, si.time, si.wavelengths, is_dyn);
                    if constexpr (is_spectral_v<Spectrum>)
                        dr::masked(L_e, is_dyn) = depolarizer<Spectrum>(
                            planck_radiance(si.wavelengths, T));
                }

                DirectionSample3f ds(scene, si, ls.vertex.si,
                                     ls.vertex.ray_mask);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!ls.vertex.bsdf_delta))
                    em_pdf = scene->pdf_emitter_direction(ls.vertex.si, ds,
                                                          !ls.vertex.bsdf_delta);

                // Compute MIS weight for emitter sample from previous bounce
                Float mis_bsdf = mis_weight(ls.vertex.bsdf_pdf, em_pdf);

                // Accumulate and potentially clamp outliers
                ls.result += clamp_contribution(
                    ls.throughput * (L_e * mis_bsdf),
                    /* direct = */ ls.vertex.depth <= 1);
            }

            if (dr::none_or<false>(si.is_valid())) {
                ls.active = false;
                return; // early exit for scalar mode
            }

            BSDFPtr bsdf = si.bsdf();
            UInt32 bsdf_flags = bsdf->flags();

            // Continue tracing the path at this point? Null crossings don't
            // count when checking for the max_depth cutoff.
            Bool depth_ok  = ls.vertex.depth + 1 < m_max_depth,
                 can_cross = has_null ? has_flag(bsdf_flags, BSDFFlags::Null)
                                      : Mask(false),
                 active_next = si.is_valid() && (depth_ok || can_cross);

            if (dr::none_or<false>(active_next)) {
                ls.active = false;
                return; // early exit for scalar mode
            }

            // ---------------------- Emitter sampling ----------------------

            // Perform emitter sampling?
            Mask active_em = active_next && depth_ok &&
                             has_flag(bsdf_flags, BSDFFlags::Smooth);

            DirectionSample3f ds = dr::zeros<DirectionSample3f>();
            Spectrum em_weight = dr::zeros<Spectrum>();
            Vector3f wo = dr::zeros<Vector3f>();

            if (dr::any_or<true>(active_em)) {
                // Sample the emitter. The sample is detached, and its weight
                // is attached to the emitter and to the motion of 'si'.
                std::tie(ds, em_weight) = scene->sample_emitter_direction(
                    si, ls.sampler->next_2d(), true, active_em);
                active_em &= (ds.pdf != 0.f);
                wo = si.to_local(ds.d);

                // Thermal-dynamic emitters: replace the reference radiance by
                // the shading-time temperature at the sampled point
                Mask dyn_em = active_em &&
                              has_flag(ds.emitter->flags(), EmitterFlags::ThermalDynamic);
                if (dr::any_or<true>(dyn_em)) {
                    Float T = thermal_dynamic_temperature<Float, Spectrum>(
                        scene, ds.emitter, ds.p, ds.n, m_thermal_up, ls.sampler,
                        m_sky_rays, si.time, si.wavelengths, dyn_em);
                    if constexpr (is_spectral_v<Spectrum>) {
                        UnpolarizedSpectrum B     = planck_radiance(si.wavelengths, T),
                                            B_ref = planck_radiance(si.wavelengths,
                                                                    Float(ds.emitter->thermal_t_ref()));
                        dr::masked(em_weight, dyn_em) *= depolarizer<Spectrum>(
                            dr::select(B_ref > 0.f, B / B_ref, 0.f));
                    }
                }
            }

            // ------ Evaluate BSDF * cos(theta) and sample direction -------

            Float sample_1 = ls.sampler->next_1d();
            Point2f sample_2 = ls.sampler->next_2d();

            auto [bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight]
                = bsdf->eval_pdf_sample(bsdf_ctx, si, wo, sample_1, sample_2);

            // --------------- Emitter sampling contribution ----------------

            if (dr::any_or<true>(active_em)) {
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Compute the MIS weight
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                // Accumulate and potentially clamp outliers
                dr::masked(ls.result, active_em) += clamp_contribution(
                    ls.throughput * (bsdf_val * em_weight * mis_em),
                    ls.vertex.depth == 0);
            }

            // ---------------------- BSDF sampling ----------------------

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            ls.ray = si.spawn_ray(si.to_world(bsdf_sample.wo));

            // A null crossing is not a path vertex
            Mask scattered = has_null ? !bsdf_sample.is_null() : Mask(true);

            // When the path tracer is differentiated, we must be careful that
            // the generated Monte Carlo samples are detached (i.e. don't track
            // derivatives) to avoid bias resulting from the combination of moving
            // samples and discontinuous visibility. We need to re-evaluate the
            // BSDF differentiably with the detached sample in that case. A
            // null crossing keeps the attached weight of the null lobe.
            if (dr::grad_enabled(ls.ray)) {
                ls.ray = dr::detach(ls.ray);

                // Recompute 'wo' to propagate derivatives to cosine term
                Vector3f wo_2 = si.to_local(ls.ray.d);
                auto [bsdf_val_2, bsdf_pdf_2] = bsdf->eval_pdf(bsdf_ctx, si, wo_2);
                bsdf_weight[bsdf_pdf_2 > 0.f && scattered] =
                    bsdf_val_2 / dr::detach(bsdf_pdf_2);
            }

            // ------ Update loop variables based on current interaction ------

            ls.throughput *= bsdf_weight;
            ls.eta *= bsdf_sample.eta;
            ls.valid_ray |= scattered && si.is_valid();

            dr::masked(ls.vertex, scattered) = Vertex{
                Interaction3f(si), bsdf_sample.pdf,
                bsdf_sample.is_delta(),
                ls.vertex.depth + 1,
                // A specular event leaves the visibility class unchanged
                dr::select(bsdf_sample.is_delta(), ls.vertex.ray_mask,
                           +RayMask::Secondary) };

            // A scattering event at a thermal vertex starts a new continuation
            // whose integrand is (L_i - B(T)). Null crossings keep the state
            // of the last scattering vertex.
            dr::masked(ls.B_prev, scattered) =
                dr::select(is_thermal, L_e, dr::zeros<Spectrum>());
            dr::masked(ls.diff_pending, scattered) = true;

            // -------------------- Stopping criterion ---------------------

            Float throughput_max = dr::max(unpolarized_spectrum(ls.throughput));

            Float rr_prob = dr::minimum(throughput_max * dr::square(ls.eta), .95f);
            Mask rr_active = ls.vertex.depth >= m_rr_depth,
                 rr_continue = ls.sampler->next_1d() < rr_prob;

            // Differentiable variants of the renderer require the russian
            // roulette sampling weight to be detached to avoid bias. This is a
            // no-op in non-differentiable variants.
            ls.throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

            ls.active = active_next && (depth_ok || !scattered) &&
                        (!rr_active || rr_continue) && (throughput_max != 0.f);

            // Reorder threads based on the shape they hit.
            ls.pi = scene->ray_intersect_preliminary(ls.ray,
                                                     /* coherent = */ false,
                                                     ls.vertex.ray_mask,
                                                     ls.active,
                                                     /* reorder = */ jit_flag(JitFlag::LoopRecord));
        });

        return {
            /* spec  = */ dr::select(ls.valid_ray, ls.result, 0.f),
            /* valid = */ ls.valid_ray
        };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("PathIRIntegrator[\n"
            "  max_depth = %u,\n"
            "  rr_depth = %u,\n"
            "  clamp_direct = %f,\n"
            "  clamp_indirect = %f\n"
            "]", m_max_depth, m_rr_depth, m_clamp_direct, m_clamp_indirect);
    }

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach(dr::select(dr::isfinite(w), w, 0.f));
    }

    MI_DECLARE_CLASS(PathIRIntegrator)
private:
    /// Zenith direction and number of sky-view rays for thermal_dynamic emitters
    ScalarVector3f m_thermal_up;
    uint32_t m_sky_rays;
};

MI_EXPORT_PLUGIN(PathIRIntegrator)
NAMESPACE_END(mitsuba)
