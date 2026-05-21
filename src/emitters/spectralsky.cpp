#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/distr_2d.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/texture.h>
#include <drjit/tensor.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-spectralsky:

Hyperspectral sky emitter (:monosp:`spectralsky`)
-------------------------------------------------

.. pluginparameters::

 * - filename
   - |string|
   - Path to a ``.skyrad`` file produced by ``tools/generate_ir_sky.py``
     (libRadtran/uvspec downwelling-sky radiance table).

 * - data
   - |tensor|
   - Radiance tensor of shape ``(n_theta, n_phi, n_lambda)`` in
     :math:`W\,m^{-2}\,sr^{-1}\,nm^{-1}`. Use together with ``theta``, ``phi``
     and ``wavelengths`` instead of ``filename`` (e.g. from Python).
   - |exposed|, |differentiable|

 * - theta, phi, wavelengths
   - |tensor|
   - 1-D axis arrays: zenith angle [deg, 0..90], azimuth [deg, 0..360) and
     wavelength [nm], all strictly ascending.

 * - scale
   - |float|
   - Radiance multiplier. (Default: 1.0)
   - |exposed|, |differentiable|

 * - to_world
   - |transform|
   - Emitter-to-world transform. Local ``+Z`` is the zenith; local ``+X`` is
     azimuth 0 (the solar principal plane for libRadtran output). (Default: none)

This emitter represents a physically based, spectrally resolved sky dome whose
downwelling radiance is precomputed by libRadtran. Unlike :monosp:`envmap`, it
stores *true spectral radiance* :math:`L(\theta, \phi, \lambda)` rather than RGB
upsampling coefficients, so it is suitable for thermal-infrared and other
hyperspectral rendering in the mitsubaIR fork.

The table covers only the sky hemisphere (zenith angle :math:`0..90^\circ`);
directions below the local horizon (rays travelling downward) emit **zero**
radiance. Directional importance sampling uses a hierarchical warp built from
the band-integrated radiance weighted by the solid-angle Jacobian, so the
emitter combines well with BSDF sampling via MIS.

.. tabs::
    .. code-tab:: python

        from tools.skyrad_io import SkyRadiance
        sky = SkyRadiance.read("renders/sky/lwir_us_standard.skyrad")
        scene = mi.load_dict({ "type": "scene", "sky": sky.to_dict(), ... })

 */

template <typename Float, typename Spectrum>
class SpectralSkyEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_to_world)
    MI_IMPORT_TYPES(Scene, Shape, Texture)

    using Warp         = Hierarchical2D<Float, 0>;
    using WavUInt      = dr::uint32_array_t<Wavelength>;
    using FloatStorage = DynamicBuffer<Float>;

    SpectralSkyEmitter(const Properties &props) : Base(props) {
        m_bsphere = BoundingSphere3f(ScalarPoint3f(0.f), 1.f);
        m_scale   = props.get<ScalarFloat>("scale", 1.f);

        std::vector<ScalarFloat> theta, phi, lambda, data;
        if (props.has_property("filename")) {
            FileResolver *fs = file_resolver();
            fs::path path = fs->resolve(props.get<std::string_view>("filename"));
            m_filename = path.filename().string();
            load_skyrad(path, theta, phi, lambda, data);
        } else if (props.has_property("data")) {
            read_axis(props, "theta",       theta);
            read_axis(props, "phi",         phi);
            read_axis(props, "wavelengths", lambda);
            TensorXf t = props.get_any<TensorXf>("data");
            if (t.ndim() != 3)
                Throw("spectralsky: 'data' must be 3-D (theta, phi, lambda), "
                      "got ndim=%zu", t.ndim());
            auto &&host = dr::migrate(t.array(), AllocType::Host);
            if constexpr (dr::is_jit_v<Float>) dr::sync_thread();
            data.assign(host.data(), host.data() + t.size());
            if (t.shape(0) != theta.size() || t.shape(1) != phi.size() ||
                t.shape(2) != lambda.size())
                Throw("spectralsky: data shape (%zu,%zu,%zu) does not match "
                      "axes (%zu,%zu,%zu)", t.shape(0), t.shape(1), t.shape(2),
                      theta.size(), phi.size(), lambda.size());
        } else {
            Throw("spectralsky: provide either 'filename' or "
                  "'data'+'theta'+'phi'+'wavelengths'.");
        }

        init(theta, phi, lambda, data);
        m_flags = EmitterFlags::Infinite | EmitterFlags::SpatiallyVarying;
    }

    /// Validate axes, upload to device buffers, and build the sampling warp.
    void init(const std::vector<ScalarFloat> &theta,
              const std::vector<ScalarFloat> &phi,
              const std::vector<ScalarFloat> &lambda,
              const std::vector<ScalarFloat> &data) {
        m_n_theta  = (uint32_t) theta.size();
        m_n_phi    = (uint32_t) phi.size();
        m_n_lambda = (uint32_t) lambda.size();
        if (m_n_theta < 2 || m_n_phi < 2 || m_n_lambda < 2)
            Throw("spectralsky: each axis needs at least 2 samples "
                  "(got theta=%u, phi=%u, lambda=%u)",
                  m_n_theta, m_n_phi, m_n_lambda);

        // theta and phi are assumed regularly spaced (the generator always
        // produces linspace grids); wavelength may be irregular.
        m_theta_min = theta.front();
        m_dtheta    = (theta.back() - theta.front()) / (m_n_theta - 1);
        m_phi_min   = phi.front();
        m_dphi      = (phi.back() - phi.front()) / (m_n_phi - 1);
        m_lambda_min = lambda.front();
        m_lambda_max = lambda.back();

        // Device-side buffers.
        size_t shape[3] = { m_n_theta, m_n_phi, m_n_lambda };
        m_data = TensorXf(data.data(), 3, shape);
        m_wavelengths = dr::load<FloatStorage>(lambda.data(), lambda.size());

        build_warp(data, theta);
    }

    /// Build the directional importance-sampling warp from band-integrated
    /// radiance weighted by sin(theta) (the solid-angle Jacobian). An extra
    /// wrap column enforces azimuthal periodicity, as in the envmap emitter.
    void build_warp(const std::vector<ScalarFloat> &data,
                    const std::vector<ScalarFloat> &theta) {
        ScalarVector2u res(m_n_phi + 1, m_n_theta);   // (width=phi, height=theta)
        std::unique_ptr<ScalarFloat[]> power(new ScalarFloat[dr::prod(res)]);

        for (uint32_t ti = 0; ti < m_n_theta; ++ti) {
            ScalarFloat sin_theta =
                dr::sin(theta[ti] * (ScalarFloat) (dr::Pi<double> / 180.0));
            for (uint32_t pj = 0; pj < m_n_phi; ++pj) {
                // Trapezoidal band integral of L over the wavelength axis.
                double integ = 0.0;
                const ScalarFloat *row =
                    &data[((size_t) ti * m_n_phi + pj) * m_n_lambda];
                for (uint32_t k = 0; k + 1 < m_n_lambda; ++k) {
                    // dlambda folded in via build of m_wavelengths spacing is
                    // unnecessary for a *relative* importance map, so we use a
                    // uniform-weight mean (robust to irregular spacing).
                    integ += 0.5 * (row[k] + row[k + 1]);
                }
                power[ti * res.x() + pj] =
                    (ScalarFloat) integ * sin_theta;
            }
            // Wrap column mirrors the first azimuth sample.
            power[ti * res.x() + m_n_phi] = power[ti * res.x()];
        }
        m_warp = Warp(power.get(), res);
    }

    void traverse(TraversalCallback *cb) override {
        Base::traverse(cb);
        cb->put("scale",    m_scale,    ParamFlags::Differentiable);
        cb->put("data",     m_data,     ParamFlags::Differentiable | ParamFlags::Discontinuous);
        cb->put("to_world", m_to_world, ParamFlags::NonDifferentiable);
    }

    void set_scene(const Scene *scene) override {
        if (scene->bbox().valid()) {
            ScalarBoundingSphere3f s = scene->bbox().bounding_sphere();
            m_bsphere = BoundingSphere3f(s.center, s.radius);
            m_bsphere.radius = dr::maximum(math::RayEpsilon<Float>,
                                   m_bsphere.radius * (1.f + math::RayEpsilon<Float>));
        } else {
            m_bsphere.center = 0.f;
            m_bsphere.radius = math::RayEpsilon<Float>;
        }
        dr::make_opaque(m_bsphere.center, m_bsphere.radius);
    }

    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        // Direction the escaping ray was travelling = -si.wi (local frame).
        Vector3f d = dr::normalize(m_to_world.value().inverse() * (-si.wi));
        return depolarizer<Spectrum>(eval_direction_spectrum(d, si.wavelengths, active));
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample,
                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        auto [uv, pdf] = m_warp.sample(sample, nullptr, active);
        active &= pdf > 0.f;

        auto [d_local, sin_theta] = uv_to_direction(uv);
        Vector3f d_world = m_to_world.value() * d_local;

        Float radius = dr::maximum(m_bsphere.radius, dr::norm(it.p - m_bsphere.center));
        Float dist   = 2.f * radius;

        Float inv_sin = dr::safe_rsqrt(dr::maximum(dr::square(sin_theta),
                                                   dr::square(dr::Epsilon<Float>)));

        DirectionSample3f ds;
        ds.p       = dr::fmadd(d_world, dist, it.p);
        ds.n       = -d_world;
        ds.uv      = uv;
        ds.time    = it.time;
        // p(omega) = pdf_uv / (pi^2 * sin_theta)   (hemisphere param.)
        ds.pdf     = dr::select(active,
                        pdf * inv_sin * (1.f / dr::square(dr::Pi<Float>)), 0.f);
        ds.delta   = false;
        ds.emitter = this;
        ds.d       = d_world;
        ds.dist    = dist;

        Spectrum weight =
            depolarizer<Spectrum>(eval_uv_spectrum(uv, it.wavelengths, active)) / ds.pdf;
        return { ds, weight & active };
    }

    Float pdf_direction(const Interaction3f & /*it*/, const DirectionSample3f &ds,
                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        auto [uv, sin_theta, valid] =
            direction_to_uv(m_to_world.value().inverse() * ds.d);
        active &= valid;
        Float inv_sin = dr::safe_rsqrt(dr::maximum(dr::square(sin_theta),
                                                   dr::square(dr::Epsilon<Float>)));
        return dr::select(active,
                  m_warp.eval(uv) * inv_sin * (1.f / dr::square(dr::Pi<Float>)), 0.f);
    }

    Spectrum eval_direction(const Interaction3f &it, const DirectionSample3f &ds,
                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        return depolarizer<Spectrum>(eval_uv_spectrum(ds.uv, it.wavelengths, active));
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &sample2,
                                          const Point2f &sample3,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // 1. Sample direction from the sky toward the scene.
        auto [uv, pdf] = m_warp.sample(sample3, nullptr, active);
        active &= pdf > 0.f;
        auto [d_local, sin_theta] = uv_to_direction(uv);
        Vector3f d_world = m_to_world.value() * d_local;

        Float inv_sin = dr::safe_rsqrt(dr::maximum(dr::square(sin_theta),
                                                   dr::square(dr::Epsilon<Float>)));
        Float dir_pdf = pdf * inv_sin * (1.f / dr::square(dr::Pi<Float>));

        // 2. Spatial origin: uniform disk over the bounding-sphere cross section.
        Point2f offset = warp::square_to_uniform_disk_concentric(sample2);
        Vector3f perp = Frame3f(d_world).to_world(Vector3f(offset.x(), offset.y(), 0.f));
        Point3f origin = m_bsphere.center + (perp - d_world) * m_bsphere.radius;

        // 3. Spectrum.
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.uv   = uv;
        si.time = time;
        auto [wavelengths, weight] = sample_wavelengths(si, wavelength_sample, active);

        Float r2 = dr::square(m_bsphere.radius);
        Ray3f ray(origin, -d_world, time, wavelengths);
        weight *= dr::Pi<Float> * r2 / dir_pdf;
        return { ray, weight & active };
    }

    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f &si, Float sample,
                       Mask active) const override {
        // Uniform spectral sampling over the table's wavelength range, matching
        // the fork's uniform hero-wavelength sampler. PDF = 1 / range.
        Float range = m_lambda_max - m_lambda_min;
        Wavelength wavelengths =
            m_lambda_min + range * math::sample_shifted<Wavelength>(sample);
        UnpolarizedSpectrum value =
            eval_uv_spectrum(si.uv, wavelengths, active);
        return { wavelengths, depolarizer<Spectrum>(value * range) };
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float, const Point2f &, Mask) const override {
        if constexpr (dr::is_jit_v<Float>)
            return { dr::zeros<PositionSample3f>(), dr::NaN<Float> };
        else
            NotImplementedError("sample_position");
    }

    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SpectralSkyEmitter[" << std::endl;
        if (!m_filename.empty())
            oss << "  filename = \"" << m_filename << "\"," << std::endl;
        oss << "  grid = [" << m_n_theta << " theta x " << m_n_phi
            << " phi x " << m_n_lambda << " lambda]," << std::endl
            << "  lambda = [" << m_lambda_min << ", " << m_lambda_max << "] nm,"
            << std::endl << "  scale = " << m_scale << std::endl << "]";
        return oss.str();
    }

protected:
    // ----- direction <-> uv mapping (local z-up, hemisphere) ----------------

    /// uv in [0,1]^2 -> (local direction, sin_theta).  u=phi/2pi, v=theta/(pi/2).
    std::pair<Vector3f, Float> uv_to_direction(const Point2f &uv) const {
        Float phi   = uv.x() * dr::TwoPi<Float>;
        Float theta = uv.y() * (dr::Pi<Float> * 0.5f);
        auto [sp, cp] = dr::sincos(phi);
        auto [st, ct] = dr::sincos(theta);
        return { Vector3f(st * cp, st * sp, ct), st };
    }

    /// local direction -> (uv, sin_theta, valid).  valid=false below horizon.
    std::tuple<Point2f, Float, Mask> direction_to_uv(const Vector3f &d_) const {
        Vector3f d = dr::normalize(d_);
        Float cos_theta = d.z();
        Mask valid = cos_theta > 0.f;
        Float theta = dr::safe_acos(cos_theta);
        Float phi   = dr::atan2(d.y(), d.x());
        phi = dr::select(phi < 0.f, phi + dr::TwoPi<Float>, phi);
        Point2f uv(phi * dr::InvTwoPi<Float>, theta * (dr::InvPi<Float> * 2.f));
        return { uv, dr::sin(theta), valid };
    }

    // ----- spectral table lookup --------------------------------------------

    UnpolarizedSpectrum eval_direction_spectrum(const Vector3f &d_local,
                                                const Wavelength &wavelengths,
                                                Mask active) const {
        auto [uv, sin_theta, valid] = direction_to_uv(d_local);
        return eval_uv_spectrum(uv, wavelengths, active && valid);
    }

    /// Trilinear (theta, phi, lambda) interpolation of the radiance table.
    UnpolarizedSpectrum eval_uv_spectrum(Point2f uv, const Wavelength &wavelengths,
                                         Mask active) const {
        if constexpr (is_spectral_v<Spectrum>) {
            // Fractional angular indices (regular grids).
            Float theta_deg = uv.y() * 90.f;
            Float phi_deg   = uv.x() * 360.f;

            Float tf = dr::clip((theta_deg - m_theta_min) / m_dtheta,
                                0.f, (Float) (m_n_theta - 1));
            UInt32 ti0 = dr::minimum(UInt32(tf), m_n_theta - 2);
            Float  tw  = tf - Float(ti0);
            UInt32 ti1 = ti0 + 1;

            Float pf = (phi_deg - m_phi_min) / m_dphi;
            pf = pf - dr::floor(pf / m_n_phi) * (Float) m_n_phi;  // wrap to [0, n_phi)
            UInt32 pj0 = dr::minimum(UInt32(pf), m_n_phi - 1);
            Float  pw  = pf - Float(pj0);
            UInt32 pj1 = dr::select(pj0 + 1 >= m_n_phi, UInt32(0), pj0 + 1);

            // Irregular wavelength interval (per spectral lane).
            WavUInt lk0 = math::find_interval<WavUInt>(
                m_n_lambda, [&](WavUInt idx) {
                    return dr::gather<Wavelength>(m_wavelengths, idx, active) <= wavelengths;
                });
            WavUInt lk1 = lk0 + 1;
            Wavelength l0 = dr::gather<Wavelength>(m_wavelengths, lk0, active);
            Wavelength l1 = dr::gather<Wavelength>(m_wavelengths, lk1, active);
            Wavelength lw = dr::clip((wavelengths - l0) / (l1 - l0), 0.f, 1.f);
            // Outside the tabulated range -> zero (no extrapolation).
            dr::mask_t<Wavelength> in_band =
                (wavelengths >= m_lambda_min) & (wavelengths <= m_lambda_max);

            auto corner = [&](UInt32 ti, UInt32 pj, WavUInt lk) {
                WavUInt index = (WavUInt(ti) * m_n_phi + WavUInt(pj)) * m_n_lambda + lk;
                return dr::gather<Wavelength>(m_data.array(), index, active);
            };
            // 8 corners: theta x phi x lambda.
            Wavelength c000 = corner(ti0, pj0, lk0), c001 = corner(ti0, pj0, lk1),
                       c010 = corner(ti0, pj1, lk0), c011 = corner(ti0, pj1, lk1),
                       c100 = corner(ti1, pj0, lk0), c101 = corner(ti1, pj0, lk1),
                       c110 = corner(ti1, pj1, lk0), c111 = corner(ti1, pj1, lk1);

            // Interpolate lambda, then phi, then theta.
            Wavelength c00 = dr::lerp(c000, c001, lw), c01 = dr::lerp(c010, c011, lw),
                       c10 = dr::lerp(c100, c101, lw), c11 = dr::lerp(c110, c111, lw);
            Wavelength c0 = dr::lerp(c00, c01, pw), c1 = dr::lerp(c10, c11, pw);
            Wavelength c  = dr::lerp(c0, c1, tw);

            UnpolarizedSpectrum result = c * m_scale;
            return result & (in_band && active);
        } else {
            // The spectral table is meaningless in RGB / monochromatic modes.
            DRJIT_MARK_USED(uv);
            DRJIT_MARK_USED(wavelengths);
            DRJIT_MARK_USED(active);
            return dr::zeros<UnpolarizedSpectrum>();
        }
    }

    // ----- property / file loading ------------------------------------------

    /// Read a 1-D axis array passed as a TensorXf property into a host vector.
    static void read_axis(const Properties &props, const char *name,
                          std::vector<ScalarFloat> &out) {
        if (!props.has_property(name))
            Throw("spectralsky: missing axis property '%s'", name);
        TensorXf t = props.get_any<TensorXf>(name);
        auto &&host = dr::migrate(t.array(), AllocType::Host);
        if constexpr (dr::is_jit_v<Float>) dr::sync_thread();
        out.assign(host.data(), host.data() + t.size());
    }

    /// Parse a binary .skyrad file (see tools/skyrad_io.py for the layout).
    void load_skyrad(const fs::path &path, std::vector<ScalarFloat> &theta,
                     std::vector<ScalarFloat> &phi, std::vector<ScalarFloat> &lambda,
                     std::vector<ScalarFloat> &data) {
        ref<FileStream> fs = new FileStream(path, FileStream::ERead);
        char magic[8];
        fs->read(magic, 8);
        if (std::memcmp(magic, "SKYRAD01", 8) != 0)
            Throw("spectralsky: '%s' is not a valid .skyrad file", path.string());
        uint32_t n_t, n_p, n_l;
        fs->read(n_t); fs->read(n_p); fs->read(n_l);

        auto read_floats = [&](size_t n) {
            std::vector<float> tmp(n);
            fs->read(tmp.data(), n * sizeof(float));
            return std::vector<ScalarFloat>(tmp.begin(), tmp.end());
        };
        theta  = read_floats(n_t);
        phi    = read_floats(n_p);
        lambda = read_floats(n_l);
        data   = read_floats((size_t) n_t * n_p * n_l);
    }

    MI_DECLARE_CLASS(SpectralSkyEmitter)

protected:
    std::string m_filename;
    BoundingSphere3f m_bsphere;
    TensorXf m_data;          // (n_theta, n_phi, n_lambda)
    FloatStorage m_wavelengths;  // (n_lambda,)
    Warp m_warp;
    Float m_scale;

    uint32_t m_n_theta = 0, m_n_phi = 0, m_n_lambda = 0;
    ScalarFloat m_theta_min = 0, m_dtheta = 1;
    ScalarFloat m_phi_min = 0, m_dphi = 1;
    ScalarFloat m_lambda_min = 0, m_lambda_max = 1;

    MI_TRAVERSE_CB(Base, m_bsphere, m_data, m_wavelengths, m_warp, m_scale)
};

MI_EXPORT_PLUGIN(SpectralSkyEmitter)
NAMESPACE_END(mitsuba)
