#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/thermal.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-volpath_ir:

Infrared volumetric path tracer (:monosp:`volpath_ir`)
------------------------------------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1 corresponds to
     :math:`\infty`). (Default: -1)

 * - rr_depth
   - |int|
   - Specifies the minimum path depth, after which the implementation will start to use the
     *russian roulette* path termination criterion. (Default: 5)

 * - clamp_direct, clamp_indirect
   - |float|
   - Clamp positive (emitted) contributions of direct/indirect paths. The thermal
     difference terms are never clamped. (Default: 0, i.e. disabled)

 * - ambient_temperature
   - |float|
   - Temperature [K] assigned to surfaces that carry no emitter (for example
     instanced geometry such as grass blades, which cannot hold emitters).
     They then radiate like ``thermal`` surfaces at this temperature, with
     the emissivity given by their BSDF. Null-BSDF interfaces are excluded.
     (Default: 0, i.e. disabled: such surfaces are at 0 K)

This integrator is the volumetric path tracer of Mitsuba (:ref:`volpath
<integrator-volpath>`) extended with thermal self-emission of *surfaces*
(:ref:`thermal <emitter-thermal>` emitters, exactly as in :ref:`path_ir
<integrator-path_ir>`) and of *participating media* with a ``temperature``.

**Emissive media.** A medium at temperature :math:`T_m` with absorption
coefficient :math:`\sigma_a` emits :math:`\sigma_a B(\lambda, T_m)` per unit
length (Kirchhoff's law for volumes, local thermodynamic equilibrium). The
radiative transfer equation along a ray,

.. math::

    L(x) = \int_0^d T(s)\,\sigma_a B(T_m)\,ds + \int_0^d T(s)\,\sigma_s L_s\,ds + T(d)\,L(y),

is estimated in the same *difference formulation* as for surfaces: since the
phase function integrates to one, the in-scattered radiance can be written as
:math:`L_s = B(T_m) + \int p\,(L_i - B(T_m))\,d\omega`, and the absorption and
scattering branches of a real collision combine to the single rule

* at a real collision, add :math:`\beta\,(B(T_m) - B_{prev})`, multiply the
  throughput by the single-scattering albedo, and set :math:`B_{prev} := B(T_m)`;
* at the first event after any scattering vertex (surface or medium),
  subtract :math:`\beta B_{prev}`.

The estimator is unbiased, has zero variance in an isothermal medium (every
collision contributes zero and the path simply loses weight), and needs no
explicit evaluation of the absorption coefficient at collisions. Spectrally
varying extinction is handled with the same hero-wavelength ratio weights as
``volpath``; the thermal emission at a real collision uses the real-collision
weight :math:`\sigma_t / p_{real}` (the scattering weight with the albedo
removed).

Media without a temperature are at 0 K: they attenuate and scatter but do
not emit. Note that the atmosphere also *emits* in the infrared (path
radiance), so an atmospheric medium should always be given its air
temperature.

.. tabs::
    .. code-tab::  xml
        :name: volpath_ir-integrator

        <integrator type="volpath_ir">
            <integer name="max_depth" value="32"/>
        </integrator>

    .. code-tab:: python

        'type': 'volpath_ir',
        'max_depth': 32

 */
template <typename Float, typename Spectrum>
class VolumetricPathIRIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth,
                   m_clamp_direct, m_clamp_indirect, clamp_contribution)
    MI_IMPORT_TYPES(Scene, Sampler, Emitter, EmitterPtr, BSDF, BSDFPtr,
                     Medium, MediumPtr, PhaseFunctionContext)

    VolumetricPathIRIntegrator(const Properties &props) : Base(props) {
        m_thermal_up = dr::normalize(props.get<ScalarVector3f>("thermal_up", ScalarVector3f(0.f, 0.f, 1.f)));
        m_sky_rays = props.get<uint32_t>("thermal_sky_rays", 8);
        m_ambient_temperature = props.get<ScalarFloat>("ambient_temperature", 0.f);
    }

    bool thermal_aware() const override { return true; }

    MI_INLINE
    Float index_spectrum(const UnpolarizedSpectrum &spec, const UInt32 &idx) const {
        Float m = spec[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            dr::masked(m, idx == 1u) = spec[1];
            dr::masked(m, idx == 2u) = spec[2];
        } else {
            DRJIT_MARK_USED(idx);
        }
        return m;
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const Ray3f &ray_,
                                     const Medium *initial_medium,
                                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        Mask valid_ray = false;
        Ray3f ray = ray_;

        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);

        Spectrum throughput(1.f), result(0.f);
        MediumPtr medium = initial_medium;
        MediumInteraction3f mei = dr::zeros<MediumInteraction3f>();
        Mask specular_chain = active;
        UInt32 depth = 0;
        UInt32 ray_mask = +RayMask::Primary;

        // Null tests fold to literals in scenes without null shapes
        bool has_null = scene->has_null_shapes();

        UInt32 channel = 0;
        if (is_rgb_v<Spectrum>) {
            uint32_t n_channels = (uint32_t) dr::size_v<Spectrum>;
            channel = (UInt32) dr::minimum(sampler->next_1d(active) * n_channels, n_channels - 1);
        }

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        Mask needs_intersection = true;
        Interaction3f last_scatter_event = dr::zeros<Interaction3f>();
        Float last_scatter_direction_pdf = 1.f;

        struct LoopState {
            Mask active;
            UInt32 depth;
            Ray3f ray;
            Spectrum throughput;
            Spectrum result;
            SurfaceInteraction3f si;
            MediumInteraction3f mei;
            MediumPtr medium;
            Float eta;
            Interaction3f last_scatter_event;
            Float last_scatter_direction_pdf;
            Mask needs_intersection;
            Mask specular_chain;
            Mask valid_ray;
            UInt32 ray_mask;
            // Blackbody radiance of the last scattering vertex (surface or
            // medium) and whether its subtraction is still pending
            Spectrum B_prev;
            Mask diff_pending;
            Sampler* sampler;

            DRJIT_STRUCT(LoopState, active, depth, ray, throughput, result, \
                si, mei, medium, eta, last_scatter_event, \
                last_scatter_direction_pdf, needs_intersection, \
                specular_chain, valid_ray, ray_mask, B_prev, diff_pending, \
                sampler)
        } ls = {
            active,
            depth,
            ray,
            throughput,
            result,
            si,
            mei,
            medium,
            eta,
            last_scatter_event,
            last_scatter_direction_pdf,
            needs_intersection,
            specular_chain,
            valid_ray,
            ray_mask,
            dr::zeros<Spectrum>(),
            false,
            sampler
        };

        dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
            [](const LoopState& ls) { return ls.active; },
            [this, scene, channel, has_null](LoopState& ls) {

            Mask& active = ls.active;
            UInt32& depth = ls.depth;
            Ray3f& ray = ls.ray;
            Spectrum& throughput = ls.throughput;
            Spectrum& result = ls.result;
            SurfaceInteraction3f& si = ls.si;
            MediumInteraction3f& mei = ls.mei;
            MediumPtr& medium = ls.medium;
            Float& eta = ls.eta;
            Interaction3f& last_scatter_event = ls.last_scatter_event;
            Float& last_scatter_direction_pdf = ls.last_scatter_direction_pdf;
            Mask& needs_intersection = ls.needs_intersection;
            Mask& specular_chain = ls.specular_chain;
            Mask& valid_ray = ls.valid_ray;
            UInt32& ray_mask = ls.ray_mask;
            Spectrum& B_prev = ls.B_prev;
            Mask& diff_pending = ls.diff_pending;
            Sampler* sampler = ls.sampler;

            // ----------------- Handle termination of paths ------------------

            // Russian roulette: try to keep path weights equal to one, while accounting for the
            // solid angle compression at refractive index boundaries. Stop with at least some
            // probability to avoid getting stuck (e.g. due to total internal reflection)
            active &= dr::any(unpolarized_spectrum(throughput) != 0.f);
            Float q = dr::minimum(dr::max(unpolarized_spectrum(throughput)) * dr::square(eta), .95f);
            Mask perform_rr = (depth > (uint32_t) m_rr_depth);
            active &= sampler->next_1d(active) < q || !perform_rr;
            dr::masked(throughput, perform_rr) *= dr::rcp(dr::detach(q));

            active &= depth < (uint32_t) m_max_depth;
            if (dr::none_or<false>(active))
                return;

            // ----------------------- Sampling the RTE -----------------------
            Mask active_medium  = active && (medium != nullptr);
            Mask active_surface = active && !active_medium;
            Mask act_null_scatter = false, act_medium_scatter = false,
                 escaped_medium = false;

            // If the medium does not have a spectrally varying extinction,
            // we can perform a few optimizations to speed up rendering
            Mask is_spectral = active_medium;
            Mask not_spectral = false;
            if (dr::any_or<true>(active_medium)) {
                is_spectral &= medium->has_spectral_extinction();
                not_spectral = !is_spectral && active_medium;
            }

            if (dr::any_or<true>(active_medium)) {
                mei = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                dr::masked(ray.maxt, active_medium && medium->is_homogeneous() && mei.is_valid()) = mei.t;
                Mask intersect = needs_intersection && active_medium;
                if (dr::any_or<true>(intersect))
                    dr::masked(si, intersect) = scene->ray_intersect(
                        ray, +RayFlags::Default, false, ray_mask, intersect);
                needs_intersection &= !active_medium;

                dr::masked(mei.t, active_medium && (si.t < mei.t)) = dr::Infinity<Float>;
                if (dr::any_or<true>(is_spectral)) {
                    auto [tr, free_flight_pdf] = medium->transmittance_eval_pdf(mei, si, is_spectral);
                    Float tr_pdf = index_spectrum(free_flight_pdf, channel);
                    dr::masked(throughput, is_spectral) *= dr::select(tr_pdf > 0.f, tr / tr_pdf, 0.f);
                }

                escaped_medium = active_medium && !mei.is_valid();
                active_medium &= mei.is_valid();

                // Handle null and real scatter events
                Float null_scatter_prob = dr::mean(mei.sigma_n / mei.combined_extinction);
                Mask null_scatter = sampler->next_1d(active_medium) < null_scatter_prob;

                act_null_scatter |= null_scatter && active_medium;
                act_medium_scatter |= !act_null_scatter && active_medium;

                if (dr::any_or<true>(is_spectral && act_null_scatter))
                    dr::masked(throughput, is_spectral && act_null_scatter) *=
                        mei.sigma_n / null_scatter_prob;

                dr::masked(depth, act_medium_scatter) += 1;
                dr::masked(last_scatter_event, act_medium_scatter) = mei;
            }

            // Dont estimate lighting if we exceeded number of bounces
            active &= depth < (uint32_t) m_max_depth;
            act_medium_scatter &= active;

            if (dr::any_or<true>(act_null_scatter)) {
                dr::masked(ray.o, act_null_scatter) = mei.p;
                dr::masked(si.t, act_null_scatter) = si.t - mei.t;
            }

            if (dr::any_or<true>(act_medium_scatter)) {
                // ---------------- Thermal emission of the medium ----------------

                // Weight of the *real collision* event (the scattering weight
                // of volpath with the single-scattering albedo removed)
                Float p_real = 1.f - dr::mean(mei.sigma_n / mei.combined_extinction);
                UnpolarizedSpectrum real_w = 1.f;
                if (dr::any_or<true>(is_spectral))
                    dr::masked(real_w, is_spectral) =
                        dr::select(p_real > 0.f, mei.sigma_t / p_real, 0.f);
                Spectrum beta_coll = throughput * depolarizer<Spectrum>(real_w);

                // Blackbody radiance of the medium at the collision point
                Spectrum B_m = dr::zeros<Spectrum>();
                Mask thermal_m = act_medium_scatter && mei.medium->is_thermal();
                if constexpr (is_spectral_v<Spectrum>) {
                    if (dr::any_or<true>(thermal_m)) {
                        Float T_m = mei.medium->temperature(mei, thermal_m);
                        dr::masked(B_m, thermal_m) = depolarizer<Spectrum>(
                            planck_radiance(mei.wavelengths, T_m));
                    }
                } else {
                    DRJIT_MARK_USED(thermal_m);
                }

                // Add B(T_m) and settle the pending difference term
                Spectrum B_sub = dr::select(diff_pending, B_prev, dr::zeros<Spectrum>());
                dr::masked(result, act_medium_scatter) += beta_coll * (B_m - B_sub);
                dr::masked(diff_pending, act_medium_scatter) = false;

                if (dr::any_or<true>(is_spectral))
                    dr::masked(throughput, is_spectral && act_medium_scatter) *=
                        mei.sigma_s / dr::mean(mei.sigma_t / mei.combined_extinction);
                if (dr::any_or<true>(not_spectral))
                    dr::masked(throughput, not_spectral && act_medium_scatter) *= mei.sigma_s / mei.sigma_t;

                PhaseFunctionContext phase_ctx(sampler);
                auto phase = mei.medium->phase_function();

                // --------------------- Emitter sampling ---------------------
                Mask sample_emitters = mei.medium->use_emitter_sampling();
                valid_ray |= act_medium_scatter;
                specular_chain &= !act_medium_scatter;
                specular_chain |= act_medium_scatter && !sample_emitters;

                Mask active_e = act_medium_scatter && sample_emitters;
                if (dr::any_or<true>(active_e)) {
                    auto [emitted, ds] = sample_emitter(mei, scene, sampler, medium, channel, active_e);
                    auto [phase_val, phase_pdf] = phase->eval_pdf(phase_ctx, mei, ds.d, active_e);
                    // 'depth' already counts this scattering event
                    dr::masked(result, active_e) += clamp_contribution(
                        throughput * phase_val * emitted *
                            mis_weight(ds.pdf, dr::select(ds.delta, 0.f, phase_pdf)),
                        depth == 1u);
                }

                // ------------------ Phase function sampling -----------------
                dr::masked(phase, !act_medium_scatter) = nullptr;
                auto [wo, phase_weight, phase_pdf] = phase->sample(phase_ctx, mei,
                    sampler->next_1d(act_medium_scatter),
                    sampler->next_2d(act_medium_scatter),
                    act_medium_scatter);
                act_medium_scatter &= phase_pdf > 0.f;
                Ray3f new_ray  = mei.spawn_ray(wo);
                dr::masked(ray, act_medium_scatter) = new_ray;
                needs_intersection |= act_medium_scatter;
                dr::masked(last_scatter_direction_pdf, act_medium_scatter) = phase_pdf;
                dr::masked(throughput, act_medium_scatter) *= phase_weight;
                dr::masked(ray_mask, act_medium_scatter) = +RayMask::Secondary;

                // The continuation after a medium scattering event estimates
                // int p (L_i - B(T_m)) dw: the phase function is normalized,
                // so B(T_m) is the reference radiance of the new continuation
                dr::masked(B_prev, act_medium_scatter) = B_m;
                dr::masked(diff_pending, act_medium_scatter) = true;
            }

            // --------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium;
            Mask intersect = active_surface && needs_intersection;
            if (dr::any_or<true>(intersect))
                dr::masked(si, intersect) = scene->ray_intersect(
                    ray, +RayFlags::Default, false, ray_mask, intersect);

            // Blackbody radiance of the surface vertex (0 for non-thermal)
            Spectrum B_cur = dr::zeros<Spectrum>();

            if (dr::any_or<true>(active_surface)) {
                // ------------------ Thermal difference term ------------------
                // First event of the continuation (surface hit, null surface or
                // escape): settle the subtraction of the previous vertex's B
                dr::masked(result, active_surface && diff_pending) -= throughput * B_prev;
                dr::masked(diff_pending, active_surface) = false;

                // ---------------- Intersection with emitters ----------------
                Mask ray_from_camera = active_surface && (depth == 0u);
                Mask count_direct = ray_from_camera || specular_chain;
                // Reusing the trace's ray mask hides an invisible environment
                // from escaped depth-0 rays
                EmitterPtr emitter = si.emitter(scene, ray_mask);
                Mask active_e = active_surface && (emitter != nullptr);
                // Rays that see an emitter through the mask are valid samples.
                // This includes escaped rays reaching a visible environment.
                valid_ray |= active_e;
                if (dr::any_or<true>(active_e)) {
                    Float emitter_pdf = 1.0f;
                    if (dr::any_or<true>(active_e && !count_direct)) {
                        // Get the PDF of sampling this emitter using next event estimation
                        DirectionSample3f ds(scene, si, last_scatter_event);
                        emitter_pdf = scene->pdf_emitter_direction(last_scatter_event, ds, active_e);
                    }
                    Spectrum emitted = emitter->eval(si, active_e);
                    Mask is_thermal = active_e && has_flag(emitter->flags(), EmitterFlags::Thermal);

                    // Shading-time thermal model (report 6.8.3)
                    Mask is_dyn = active_e && has_flag(emitter->flags(), EmitterFlags::ThermalDynamic);
                    if (dr::any_or<true>(is_dyn))
                        emitted *= depolarizer<Spectrum>(thermal_dynamic_scale<Float, Spectrum>(
                            scene, emitter, si.p, si.n, m_thermal_up, sampler,
                            m_sky_rays, si.time, si.wavelengths, is_dyn));
                    dr::masked(B_cur, is_thermal) = emitted;

                    Spectrum contrib = dr::select(count_direct, throughput * emitted,
                                                  throughput * mis_weight(last_scatter_direction_pdf, emitter_pdf) * emitted);
                    dr::masked(result, active_e) += clamp_contribution(contrib, depth <= 1u);
                }
            }
            active_surface &= si.is_valid();
            if (dr::any_or<true>(active_surface)) {
                // --------------------- Emitter sampling ---------------------
                BSDFContext ctx;
                BSDFPtr bsdf  = si.bsdf();

                // Ambient temperature of emitter-less surfaces (see path_ir)
                if constexpr (is_spectral_v<Spectrum>) {
                    if (m_ambient_temperature > 0.f) {
                        EmitterPtr hit_em = si.emitter(scene, ray_mask);
                        Mask ambient = active_surface && (hit_em == nullptr) &&
                                       ((bsdf->flags() & ((uint32_t) BSDFFlags::All & ~(uint32_t) BSDFFlags::Null)) != 0u);
                        Spectrum B_amb = depolarizer<Spectrum>(
                            planck_radiance(si.wavelengths, Float(m_ambient_temperature)));
                        dr::masked(result, ambient) += throughput * B_amb;
                        dr::masked(B_cur, ambient) = B_amb;
                        valid_ray |= ambient;
                    }
                }
                Mask active_e = active_surface && bsdf->has_flag(BSDFFlags::Smooth) && (depth + 1 < (uint32_t) m_max_depth);

                if (likely(dr::any_or<true>(active_e))) {
                    auto [emitted, ds] = sample_emitter(si, scene, sampler, medium, channel, active_e);

                    // Query the BSDF for that emitter-sampled direction
                    Vector3f wo       = si.to_local(ds.d);
                    Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
                    bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                    // Determine probability of having sampled that same
                    // direction using BSDF sampling.
                    Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_e);
                    dr::masked(result, active_e) += clamp_contribution(
                        throughput * bsdf_val * mis_weight(ds.pdf, dr::select(ds.delta, 0.f, bsdf_pdf)) * emitted,
                        depth == 0u);
                }

                // ----------------------- BSDF sampling ----------------------
                auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active_surface),
                                                   sampler->next_2d(active_surface), active_surface);
                bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

                dr::masked(throughput, active_surface) *= bsdf_val;
                dr::masked(eta, active_surface) *= bs.eta;

                Ray3f bsdf_ray                  = si.spawn_ray(si.to_world(bs.wo));
                dr::masked(ray, active_surface) = bsdf_ray;
                needs_intersection |= active_surface;

                Mask null = has_null ? bs.is_null() : Mask(false),
                     non_null_bsdf = active_surface && !null;
                dr::masked(depth, non_null_bsdf) += 1;

                // A specular event leaves the visibility class unchanged
                dr::masked(ray_mask, non_null_bsdf && !bs.is_delta()) =
                    +RayMask::Secondary;

                // update the last scatter PDF event if we encountered a non-null scatter event
                dr::masked(last_scatter_event, non_null_bsdf) = si;
                dr::masked(last_scatter_direction_pdf, non_null_bsdf) = bs.pdf;

                // A scattering event at a (thermal) surface starts a new
                // continuation whose integrand is (L_i - B(T)); null
                // crossings keep the state of the last scattering vertex
                dr::masked(B_prev, non_null_bsdf) = B_cur;
                dr::masked(diff_pending, non_null_bsdf) = true;

                valid_ray |= non_null_bsdf;
                specular_chain |= non_null_bsdf && bs.is_delta();
                specular_chain &= !(active_surface && has_flag(bs.sampled_type, BSDFFlags::Smooth));
                act_null_scatter |= active_surface && null;
                Mask has_medium_trans                = active_surface && si.is_medium_transition();
                dr::masked(medium, has_medium_trans) = si.target_medium(ray.d);
            }
            active &= (active_surface | active_medium);
        },
        "Volpath IR integrator");

        return { ls.result, ls.valid_ray };
    }

    /// Samples an emitter in the scene and evaluates its attenuated contribution
    template <typename Interaction>
    std::tuple<Spectrum, DirectionSample3f>
    sample_emitter(const Interaction &ref_interaction, const Scene *scene,
                   Sampler *sampler, MediumPtr medium,
                   UInt32 channel, Mask active) const {
        Spectrum transmittance(1.0f);

        auto [ds, emitter_val] = scene->sample_emitter_direction(ref_interaction, sampler->next_2d(active), false, active);
        dr::masked(emitter_val, ds.pdf == 0.f) = 0.f;
        active &= (ds.pdf != 0.f);

        // Thermal-dynamic emitters: shading-time temperature at the sampled point
        Mask dyn_em = active && has_flag(ds.emitter->flags(), EmitterFlags::ThermalDynamic);
        if (dr::any_or<true>(dyn_em))
            emitter_val *= depolarizer<Spectrum>(thermal_dynamic_scale<Float, Spectrum>(
                scene, ds.emitter, ds.p, ds.n, m_thermal_up, sampler,
                m_sky_rays, ref_interaction.time, ref_interaction.wavelengths, dyn_em));

        if (dr::none_or<false>(active)) {
            return { emitter_val, ds };
        }

        Ray3f ray = ref_interaction.spawn_ray_to(ds);
        Float max_dist = ray.maxt;

        // Potentially escaping the medium if this is the current medium's boundary
        if constexpr (std::is_convertible_v<Interaction, SurfaceInteraction3f>)
            dr::masked(medium, ref_interaction.is_medium_transition()) =
                ref_interaction.target_medium(ray.d);

        Float total_dist = 0.f;
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        Mask needs_intersection = true;
        DirectionSample3f dir_sample = ds;

        struct LoopState {
            Mask active;
            Ray3f ray;
            Float total_dist;
            Mask needs_intersection;
            MediumPtr medium;
            SurfaceInteraction3f si;
            Spectrum transmittance;
            DirectionSample3f dir_sample;
            Sampler* sampler;

            DRJIT_STRUCT(LoopState, active, ray, total_dist, \
                needs_intersection, medium, si, transmittance, \
                dir_sample, sampler)
        } ls = {
            active,
            ray,
            total_dist,
            needs_intersection,
            medium,
            si,
            transmittance,
            dir_sample,
            sampler
        };

        dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
            [](const LoopState& ls) { return dr::detach(ls.active); },
            [this, scene, channel, max_dist](LoopState& ls) {

            Mask& active = ls.active;
            Ray3f& ray = ls.ray;
            Float& total_dist = ls.total_dist;
            Mask& needs_intersection = ls.needs_intersection;
            MediumPtr& medium = ls.medium;
            SurfaceInteraction3f& si = ls.si;
            Spectrum& transmittance = ls.transmittance;
            DirectionSample3f& dir_sample = ls.dir_sample;
            Sampler* sampler = ls.sampler;

            Float remaining_dist = max_dist - total_dist;
            ray.maxt = remaining_dist;
            active &= remaining_dist > 0.f;
            if (dr::none_or<false>(active))
                return;

            Mask escaped_medium = false;
            Mask active_medium  = active && (medium != nullptr);
            Mask active_surface = active && !active_medium;

            if (dr::any_or<true>(active_medium)) {
                auto mei = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                dr::masked(ray.maxt, active_medium && medium->is_homogeneous() && mei.is_valid()) = dr::minimum(mei.t, remaining_dist);
                Mask intersect = needs_intersection && active_medium;
                if (dr::any_or<true>(intersect))
                    dr::masked(si, intersect) = scene->ray_intersect(ray, intersect);

                dr::masked(mei.t, active_medium && (si.t < mei.t)) = dr::Infinity<Float>;
                needs_intersection &= !active_medium;

                Mask is_spectral = medium->has_spectral_extinction() && active_medium;
                Mask not_spectral = !is_spectral && active_medium;
                if (dr::any_or<true>(is_spectral)) {
                    Float t      = dr::minimum(remaining_dist, dr::minimum(mei.t, si.t)) - mei.mint;
                    UnpolarizedSpectrum tr  = dr::exp(-t * mei.combined_extinction);
                    UnpolarizedSpectrum free_flight_pdf = dr::select(si.t < mei.t || mei.t > remaining_dist, tr, tr * mei.combined_extinction);
                    Float tr_pdf = index_spectrum(free_flight_pdf, channel);
                    dr::masked(transmittance, is_spectral) *= dr::select(tr_pdf > 0.f, tr / tr_pdf, 0.f);
                }

                // Handle exceeding the maximum distance by medium sampling
                dr::masked(total_dist, active_medium && (mei.t > remaining_dist) && mei.is_valid()) = dir_sample.dist;
                dr::masked(mei.t, active_medium && (mei.t > remaining_dist)) = dr::Infinity<Float>;

                escaped_medium = active_medium && !mei.is_valid();
                active_medium &= mei.is_valid();
                is_spectral &= active_medium;
                not_spectral &= active_medium;

                dr::masked(total_dist, active_medium) += mei.t;

                if (dr::any_or<true>(active_medium)) {
                    dr::masked(ray.o, active_medium)    = mei.p;
                    dr::masked(si.t, active_medium) = si.t - mei.t;

                    if (dr::any_or<true>(is_spectral))
                        dr::masked(transmittance, is_spectral) *= mei.sigma_n;
                    if (dr::any_or<true>(not_spectral))
                        dr::masked(transmittance, not_spectral) *= mei.sigma_n / mei.combined_extinction;
                }
            }

            // Handle interactions with surfaces
            Mask intersect = active_surface && needs_intersection;
            if (dr::any_or<true>(intersect))
                dr::masked(si, intersect)    = scene->ray_intersect(ray, intersect);
            needs_intersection &= !intersect;
            active_surface |= escaped_medium;
            dr::masked(total_dist, active_surface) += si.t;

            active_surface &= si.is_valid() && active && !active_medium;
            if (dr::any_or<true>(active_surface)) {
                auto bsdf         = si.bsdf();
                Spectrum bsdf_val = bsdf->eval_null(si, active_surface);
                bsdf_val = si.to_world_mueller(bsdf_val, si.wi, si.wi);

                dr::masked(transmittance, active_surface) *= bsdf_val;
            }

            // Update the ray with new origin & t parameter
            dr::masked(ray, active_surface) = si.spawn_ray(ray.d);

            // Account for the rounding-related ray epsilon that Mitsuba
            // adds when spawning rays
            dr::masked(total_dist, active_surface) +=
                dr::detach(dr::dot(ray.o - si.p, ray.d));

            ray.maxt = remaining_dist;
            needs_intersection |= active_surface;

            // Continue tracing through scene if non-zero weights exist
            active &= (active_medium || active_surface) &&
                      dr::any(unpolarized_spectrum(transmittance) != 0.f);

            // If a medium transition is taking place: Update the medium pointer
            Mask has_medium_trans = active_surface && si.is_medium_transition();
            if (dr::any_or<true>(has_medium_trans)) {
                dr::masked(medium, has_medium_trans) = si.target_medium(ray.d);
            }
        },
        "Volpath IR integrator emitter sampling");

        return { ls.transmittance * emitter_val, dir_sample };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("VolumetricPathIRIntegrator[\n"
            "  max_depth = %u,\n"
            "  rr_depth = %u\n"
            "]", m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach(dr::select(dr::isfinite(w), w, 0.f));
    };

    MI_DECLARE_CLASS(VolumetricPathIRIntegrator)
private:
    /// Zenith direction and number of sky-view rays for thermal_dynamic emitters
    ScalarVector3f m_thermal_up;
    uint32_t m_sky_rays;
    ScalarFloat m_ambient_temperature;
};

MI_EXPORT_PLUGIN(VolumetricPathIRIntegrator)
NAMESPACE_END(mitsuba)
