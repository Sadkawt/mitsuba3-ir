#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/endpoint.h>
#include <mitsuba/render/fwd.h>
#include <drjit/call.h>

NAMESPACE_BEGIN(mitsuba)


/**
 * This list of flags is used to classify the different types of emitters.
 */
enum class EmitterFlags : uint32_t {
    // =============================================================
    //                      Emitter types
    // =============================================================

    /// No flags set (default value)
    Empty                = 0x00000,

    /// The emitter lies at a single point in space
    DeltaPosition        = 0x00001,

    /// The emitter emits light in a single direction
    DeltaDirection       = 0x00002,

    /// The emitter is placed at infinity (i.e., it is an environment map)
    Infinite             = 0x00004,

    /// The emitter is attached to a surface (i.e., it is an area emitter)
    Surface              = 0x00008,

    // =============================================================
    //                    Other lobe attributes
    // =============================================================

    /// The emission depends on the UV coordinates
    SpatiallyVarying     = 0x00010,

    /// The emitter is hidden from directly visible (camera) rays.
    Invisible            = 0x00020,

    /// Light portal (see ``Scene::portals()``), does not emit anything itself
    Portal               = 0x00040,

    /**
     * The emitter importance samples a texture and maps the result onto
     * the shape through ``Shape::eval_parameterization()``.
     */
    SamplesTexture       = 0x00080,

    /**
     * Thermal (Planckian) emitter: ``eval()`` returns the *blackbody* spectral
     * radiance B(lambda, T) of the surface. The directional spectral emissivity
     * is not applied by the emitter; it follows from Kirchhoff's law and is
     * accounted for by thermal-aware integrators (e.g. ``path_ir``) through the
     * difference formulation L = B + int f (L_in - B). Integrators that do
     * not implement this treat the surface as a blackbody (emissivity 1).
     */
    Thermal              = 0x00100,

    /**
     * Thermal emitter whose surface temperature is evaluated at shading time
     * by thermal-aware integrators from the solar/sky forcing history and
     * shadow rays (linear-response model, see ``thermal_dynamic``). The
     * emitter exposes the per-shape data through ``thermal_*()`` accessors.
     */
    ThermalDynamic       = 0x00200,

    // =============================================================
    //                  Compound lobe attributes
    // =============================================================

    /// Delta function in either position or direction
    Delta        = DeltaPosition | DeltaDirection,
};

MI_DECLARE_ENUM_OPERATORS(EmitterFlags)

template <typename Float, typename Spectrum>
class MI_EXPORT_LIB Emitter : public Endpoint<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Endpoint, m_shape)
    MI_IMPORT_TYPES()

    /// Is this an environment map light emitter?
    bool is_environment() const {
        return has_flag(m_flags, EmitterFlags::Infinite) &&
               !has_flag(m_flags, EmitterFlags::Delta);
    }

    /// Is this a light portal? (see ``Scene::portals()``)
    bool is_portal() const { return has_flag(m_flags, EmitterFlags::Portal); }

    /// Relative weight of sampling this emitter. Zero when hidden from secondary rays.
    ScalarFloat sampling_weight() const {
        ShapeVisibility v = visibility();
        return (v == ShapeVisibility::Secondary || v == ShapeVisibility::All)
                   ? m_sampling_weight : 0.f;
    }

    /// Ray categories that can see this emitter
    ShapeVisibility visibility() const {
        return m_shape ? m_shape->visibility() : m_visibility;
    }

    /// Return the complete set of emitter flags
    uint32_t flags() const {
        ShapeVisibility v = visibility();
        bool visible = v == ShapeVisibility::Primary || v == ShapeVisibility::All;
        return m_flags | (visible ? 0u : (uint32_t) EmitterFlags::Invisible);
    }

    void traverse(TraversalCallback *callback) override;

    void parameters_changed(const std::vector<std::string> &keys = {}) override;

    // =============================================================
    //   Shading-time thermal model (EmitterFlags::ThermalDynamic)
    // =============================================================

    /**
     * \brief Forcing table entry ``k``: unit vector towards the sun and the
     * weight (K per unit cos(theta)) with which the direct solar irradiance of
     * that time step contributes to the surface temperature at the render
     * time. Zero for emitters that are not thermal-dynamic.
     */
    virtual std::pair<Vector3f, Float> thermal_forcing(UInt32 k, Mask active = true) const;

    /// Number of forcing table entries (0 if not thermal-dynamic)
    virtual uint32_t thermal_steps() const { return 0; }
    /// Reference surface temperature at the render time [K]
    virtual ScalarFloat thermal_t_ref() const { return 0.f; }
    /// Broadband solar absorptance of the thermal model
    virtual ScalarFloat thermal_alpha() const { return 0.f; }
    /// Weighted reference direct-solar sum (K / alpha), subtracted from the shadow-ray sum
    virtual ScalarFloat thermal_c_ref() const { return 0.f; }
    /// Temperature change per unit decrease of the sky-view factor [K]
    virtual ScalarFloat thermal_c_sky() const { return 0.f; }
    /// Surface conductance h_conv + 4 eps sigma T_lin^3 used to linearise the response function [W m^-2 K^-1]
    virtual ScalarFloat thermal_h_total() const { return 0.f; }
    /// Convective heat transfer coefficient [W m^-2 K^-1]
    virtual ScalarFloat thermal_h_conv() const { return 0.f; }
    /// Broadband longwave emissivity of the thermal model
    virtual ScalarFloat thermal_emissivity() const { return 0.f; }

    /// Return whether the emitter parameters have changed
    bool dirty() const { return m_dirty; }

    /// Modify the emitter's ``dirty`` flag
    void set_dirty(bool dirty) { m_dirty = dirty; }

    /// This is both a class and the base of various Mitsuba plugins
    MI_DECLARE_PLUGIN_BASE_CLASS(Emitter)

protected:
    Emitter(const Properties &props);

protected:
    /// Combined flags for all properties of this emitter.
    uint32_t m_flags;

    /// Sampling weight
    ScalarFloat m_sampling_weight;

    /// Ray categories that can see the emitter
    ShapeVisibility m_visibility;

    /// True if the emitter's parameters have changed
    bool m_dirty = false;

private:
    friend class mitsuba::Scene<Float, Spectrum>;

    MI_TRAVERSE_CB(Base)
};

MI_EXTERN_CLASS(Emitter)
NAMESPACE_END(mitsuba)

// -----------------------------------------------------------------------
// Enables vectorized method calls on Dr.Jit arrays of emitters
// -----------------------------------------------------------------------

DRJIT_CALL_TEMPLATE_BEGIN(mitsuba::Emitter)
    DRJIT_CALL_METHOD(sample_ray)
    DRJIT_CALL_METHOD(sample_direction)
    DRJIT_CALL_METHOD(pdf_direction)
    DRJIT_CALL_METHOD(eval_direction)
    DRJIT_CALL_METHOD(sample_position)
    DRJIT_CALL_METHOD(pdf_position)
    DRJIT_CALL_METHOD(eval)
    DRJIT_CALL_METHOD(sample_wavelengths)
    DRJIT_CALL_METHOD(thermal_forcing)
    DRJIT_CALL_GETTER(thermal_steps)
    DRJIT_CALL_GETTER(thermal_t_ref)
    DRJIT_CALL_GETTER(thermal_alpha)
    DRJIT_CALL_GETTER(thermal_c_ref)
    DRJIT_CALL_GETTER(thermal_c_sky)
    DRJIT_CALL_GETTER(thermal_h_total)
    DRJIT_CALL_GETTER(thermal_h_conv)
    DRJIT_CALL_GETTER(thermal_emissivity)
    DRJIT_CALL_GETTER(is_environment)
    DRJIT_CALL_GETTER(flags)
    DRJIT_CALL_GETTER(shape)
    DRJIT_CALL_GETTER(medium)
    DRJIT_CALL_GETTER(sampling_weight)
DRJIT_CALL_END()

// -----------------------------------------------------------------------
