#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/endpoint.h>
#include <mitsuba/render/shape.h>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT Emitter<Float, Spectrum>::Emitter(const Properties &props)
    : Base(props, ObjectType::Emitter) {
    m_sampling_weight = props.get<ScalarFloat>("sampling_weight", 1.f);
    m_visibility = parse_visibility(
        props.get<std::string_view>("visibility", "all"));
}

MI_VARIANT std::pair<typename Emitter<Float, Spectrum>::Vector3f, Float>
Emitter<Float, Spectrum>::thermal_forcing(UInt32 /*k*/, Mask /*active*/) const {
    return { dr::zeros<Vector3f>(), 0.f };
}

MI_VARIANT void Emitter<Float, Spectrum>::traverse(TraversalCallback *cb) {
    cb->put("sampling_weight", m_sampling_weight, ParamFlags::NonDifferentiable);
}

MI_VARIANT void Emitter<Float, Spectrum>::parameters_changed(const std::vector<std::string> &keys) {
    set_dirty(true);
    Base::parameters_changed(keys);
}

MI_INSTANTIATE_CLASS(Emitter)
NAMESPACE_END(mitsuba)
