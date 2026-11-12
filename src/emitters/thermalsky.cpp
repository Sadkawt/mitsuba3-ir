#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/thermal.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-thermalsky:

Thermal sky (:monosp:`thermalsky`)
----------------------------------

.. pluginparameters::

 * - temperature
   - |float|
   - Near-surface air temperature in kelvin. (Default: 288.15 K)
   - |exposed|

 * - emissivity_zenith
   - |spectrum|
   - Effective spectral emissivity of the clear sky at the zenith. (Default: 0.8)
   - |exposed|

 * - model
   - |string|
   - ``plane_parallel`` (default): the emissivity increases towards the
     horizon as :math:`\varepsilon(\theta) = 1 - (1 - \varepsilon_z)^{\sec\theta}`;
     ``isotropic``: :math:`\varepsilon(\theta) = \varepsilon_z` everywhere.

 * - up
   - |vector|
   - Zenith direction in world space. (Default: (0, 0, 1))

 * - cos_min
   - |float|
   - Lower bound of :math:`\cos\theta` used in the plane-parallel model, which
     limits the air mass near the horizon. (Default: 0.05, i.e. 20 air masses)

 * - wavelength_min, wavelength_max
   - |float|
   - Range used for emitter-side wavelength sampling (light tracing only).
     (Default: 360, 20000)

Environment emitter modelling the thermal (longwave) radiance of a cloud-free
sky. The downwelling spectral radiance is

.. math::

    L(\theta, \lambda) = \varepsilon(\theta, \lambda)\, B_\lambda(\lambda, T_{air}),

the emission of a plane-parallel atmospheric slab at the near-surface air
temperature with effective optical depth :math:`\tau_z(\lambda) =
-\ln(1 - \varepsilon_z(\lambda))` at the zenith and :math:`\tau_z \sec\theta`
at zenith angle :math:`\theta` (Unsworth & Monteith 1975; Monteith & Unsworth
2013, ch. 5). Below the horizon the emitter returns :math:`B_\lambda(T_{air})`
(an opaque air mass), which only matters when no ground geometry is present.

The zenith emissivity may be a spectrum (e.g. a transmittance-derived table
from MODTRAN) or a band-effective constant; ``mitsuba.ir.sky_emissivity_idso``
provides the empirical clear-sky values of Idso (1981) as a function of air
temperature and water vapour pressure for the full spectrum, the 8-14 um and
the 10.5-12.5 um bands.

Reflected sunlight / scattered skylight are *not* part of this emitter: add a
:ref:`directional <emitter-directional>` sun (``mitsuba.ir.sun``) for the
SWIR/MWIR.

.. tabs::
    .. code-tab:: xml
        :name: thermalsky

        <emitter type="thermalsky">
            <float name="temperature" value="290"/>
            <spectrum name="emissivity_zenith" value="0.35"/>
        </emitter>

    .. code-tab:: python

        'type': 'thermalsky',
        'temperature': 290.0,
        'emissivity_zenith': 0.35

 */

template <typename Float, typename Spectrum>
class ThermalSkyEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags)
    MI_IMPORT_TYPES(Scene, Shape, Texture)

    ThermalSkyEmitter(const Properties &props) : Base(props) {
        if constexpr (!is_spectral_v<Spectrum>)
            Throw("The 'thermalsky' emitter requires a spectral variant of Mitsuba.");

        m_bsphere = BoundingSphere3f(ScalarPoint3f(0.f), 1.f);
        m_surface_area = 4.f * dr::Pi<ScalarFloat>;

        m_temperature = props.get<ScalarFloat>("temperature", 288.15f);
        m_emissivity_zenith = props.get_texture<Texture>("emissivity_zenith", 0.8f);
        if (m_emissivity_zenith->is_spatially_varying())
            Throw("Expected a non-spatially varying emissivity spectrum!");

        std::string_view model = props.get<std::string_view>("model", "plane_parallel");
        if (model == "plane_parallel")
            m_plane_parallel = true;
        else if (model == "isotropic")
            m_plane_parallel = false;
        else
            Throw("Unknown sky model \"%s\" (expected 'plane_parallel' or 'isotropic')", model);

        m_up = dr::normalize(props.get<ScalarVector3f>("up", ScalarVector3f(0.f, 0.f, 1.f)));
        ScalarFloat cos_min = props.get<ScalarFloat>("cos_min", 0.05f);
        if (cos_min <= 0.f || cos_min > 1.f)
            Throw("'cos_min' must lie in (0, 1].");
        m_cos_min = cos_min;
        m_wavelength_range = ScalarVector2f(
            props.get<ScalarFloat>("wavelength_min", 360.f),
            props.get<ScalarFloat>("wavelength_max", 20000.f));

        m_flags = +EmitterFlags::Infinite;
        dr::make_opaque(m_temperature, m_up, m_cos_min);
    }

    void traverse(TraversalCallback *cb) override {
        Base::traverse(cb);
        cb->put("temperature", m_temperature, ParamFlags::NonDifferentiable);
        cb->put("emissivity_zenith", m_emissivity_zenith, ParamFlags::NonDifferentiable);
    }

    void set_scene(const Scene *scene) override {
        if (scene->bbox().valid()) {
            ScalarBoundingSphere3f scene_sphere = scene->bbox().bounding_sphere();
            m_bsphere = BoundingSphere3f(scene_sphere.center, scene_sphere.radius);
            m_bsphere.radius =
                dr::maximum(math::RayEpsilon<Float>,
                            m_bsphere.radius * (1.f + math::RayEpsilon<Float>));
        } else {
            m_bsphere.center = 0.f;
            m_bsphere.radius = math::RayEpsilon<Float>;
        }
        m_surface_area = 4.f * dr::Pi<ScalarFloat> * dr::square(m_bsphere.radius);
        dr::make_opaque(m_bsphere.center, m_bsphere.radius, m_surface_area);
    }

    /// Sky radiance for a world-space direction ``d`` pointing *towards* the sky
    UnpolarizedSpectrum radiance(const Vector3f &d, const Wavelength &wavelengths,
                                 Mask active) const {
        if constexpr (is_spectral_v<Spectrum>) {
            SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
            si.wavelengths = wavelengths;
            UnpolarizedSpectrum eps_z = dr::clip(m_emissivity_zenith->eval(si, active), 0.f, 1.f);

            Float cos_theta = dr::dot(d, m_up);
            UnpolarizedSpectrum eps;
            if (m_plane_parallel) {
                // tau(theta) = tau_z * sec(theta), eps = 1 - exp(-tau)
                Float sec_theta = dr::rcp(dr::maximum(cos_theta, m_cos_min));
                eps = 1.f - dr::pow(1.f - eps_z, sec_theta);
            } else {
                eps = eps_z;
            }
            // Below the horizon: opaque air at T_air
            eps = dr::select(cos_theta > 0.f, eps, UnpolarizedSpectrum(1.f));

            UnpolarizedSpectrum B = planck_radiance(wavelengths, Float(m_temperature));
            return dr::select(active, eps * B, 0.f);
        } else {
            DRJIT_MARK_USED(d); DRJIT_MARK_USED(wavelengths); DRJIT_MARK_USED(active);
            return 0.f;
        }
    }

    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        // For an escaped ray, si.wi = -ray.d, so the direction towards the
        // sky is -si.wi (world space, see the envmap emitter)
        return depolarizer<Spectrum>(radiance(-si.wi, si.wavelengths, active));
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &sample2, const Point2f &sample3,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        Vector3f v0 = warp::square_to_uniform_sphere(sample2);
        Point3f orig = dr::fmadd(v0, m_bsphere.radius, m_bsphere.center);
        Vector3f v1 = warp::square_to_cosine_hemisphere(sample3),
                 dir = Frame3f(-v0).to_world(v1);

        auto [wavelengths, weight] = sample_wavelengths(
            dr::zeros<SurfaceInteraction3f>(), wavelength_sample, active);
        // sample_wavelengths() returns B * extent; apply the angular emissivity
        // of the direction the ray comes from (-dir points towards the sky)
        if constexpr (is_spectral_v<Spectrum>) {
            UnpolarizedSpectrum B = planck_radiance(wavelengths, Float(m_temperature));
            UnpolarizedSpectrum L = radiance(-dir, wavelengths, active);
            weight *= dr::select(B > 0.f, L / B, 0.f) * m_surface_area * dr::Pi<ScalarFloat>;
        }

        return { Ray3f(orig, dir, time, wavelengths), depolarizer<Spectrum>(weight) };
    }

    std::pair<DirectionSample3f, Spectrum> sample_direction(const Interaction3f &it,
                                                            const Point2f &sample,
                                                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        Vector3f d = warp::square_to_uniform_sphere(sample);
        Float pdf = warp::square_to_uniform_sphere_pdf(d);

        Float radius = dr::maximum(m_bsphere.radius, dr::norm(it.p - m_bsphere.center)),
              dist   = 2.f * radius;

        DirectionSample3f ds;
        ds.p       = dr::fmadd(d, dist, it.p);
        ds.n       = -d;
        ds.uv      = sample;
        ds.time    = it.time;
        ds.pdf     = pdf;
        ds.delta   = false;
        ds.emitter = this;
        ds.d       = d;
        ds.dist    = dist;

        return { ds, depolarizer<Spectrum>(radiance(d, it.wavelengths, active)) / ds.pdf };
    }

    Float pdf_direction(const Interaction3f & /*it*/, const DirectionSample3f &ds,
                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        return warp::square_to_uniform_sphere_pdf(ds.d);
    }

    Spectrum eval_direction(const Interaction3f &it, const DirectionSample3f &ds,
                            Mask active) const override {
        return depolarizer<Spectrum>(radiance(ds.d, it.wavelengths, active));
    }

    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f & /*si*/, Float sample,
                       Mask active) const override {
        if constexpr (is_spectral_v<Spectrum>) {
            Wavelength u = math::sample_shifted<Wavelength>(sample);
            ScalarFloat extent = m_wavelength_range.y() - m_wavelength_range.x();
            Wavelength wavelengths = dr::fmadd(u, extent, m_wavelength_range.x());
            UnpolarizedSpectrum B = planck_radiance(wavelengths, Float(m_temperature));
            return { wavelengths, depolarizer<Spectrum>(dr::select(active, B * extent, 0.f)) };
        } else {
            DRJIT_MARK_USED(sample); DRJIT_MARK_USED(active);
            return { dr::zeros<Wavelength>(), 0.f };
        }
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float /*time*/, const Point2f & /*sample*/,
                    Mask /*active*/) const override {
        if constexpr (dr::is_jit_v<Float>) {
            return { dr::zeros<PositionSample3f>(),
                     dr::full<Float>(dr::NaN<ScalarFloat>) };
        } else {
            NotImplementedError("sample_position");
        }
    }

    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "ThermalSkyEmitter[" << std::endl
            << "  temperature = " << m_temperature << "," << std::endl
            << "  emissivity_zenith = " << string::indent(m_emissivity_zenith) << "," << std::endl
            << "  model = " << (m_plane_parallel ? "plane_parallel" : "isotropic") << "," << std::endl
            << "  up = " << m_up << "," << std::endl
            << "  bsphere = " << string::indent(m_bsphere) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(ThermalSkyEmitter)
protected:
    Float m_temperature;
    ref<Texture> m_emissivity_zenith;
    bool m_plane_parallel;
    Vector3f m_up;
    Float m_cos_min;
    ScalarVector2f m_wavelength_range;
    BoundingSphere3f m_bsphere;
    Float m_surface_area;

    MI_TRAVERSE_CB(Base, m_temperature, m_emissivity_zenith, m_up, m_cos_min,
                   m_bsphere, m_surface_area)
};

MI_EXPORT_PLUGIN(ThermalSkyEmitter)
NAMESPACE_END(mitsuba)
