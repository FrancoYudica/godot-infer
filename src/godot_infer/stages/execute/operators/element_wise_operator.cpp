#include "element_wise_operator.hpp"

namespace gdinfer::execute::operators {

bool ElementwiseOperator::init(godot::RenderingDevice* rd, ShaderCache* shader_cache) {
    const auto* entry = shader_cache->get_or_load(rd, shader_path());
    ERR_FAIL_COND_V_MSG(entry == nullptr, false, String("Unable to load shader: ") + shader_path());
    _shader = entry->shader;
    _pipeline = entry->pipeline;
    return true;
}

std::optional<dispatch::DispatchDescriptor> ElementwiseOperator::build(
    const Physical::Node& node,
    const OperatorContext& ctx) {

    uint32_t total_floats = std::get<execute::shape_inference::ElementwiseDispatchData>(*ctx.dispatch_data).total_elements;

    dispatch::DispatchDescriptor descriptor;
    descriptor.shader = _shader;
    descriptor.pipeline = _pipeline;
    descriptor.input_names = {node.inputs[0]};
    descriptor.output_names = {node.outputs[0]};

    descriptor.push_constants.write(total_floats);
    descriptor.push_constants.ensure_alignment();

    descriptor.workgroup_count_x = (total_floats + 63) / 64;

    return descriptor;
}

} // namespace gdinfer::execute::operators
