#include "gemm_operator.hpp"

namespace gdinfer::execute::operators {

bool GemmOperator::init(godot::RenderingDevice* rd, ShaderCache* shader_cache) {
    const auto* entry = shader_cache->get_or_load(rd, "shaders/gemm.glsl");
    ERR_FAIL_COND_V_MSG(entry == nullptr, false, "Failed to load GEMM shader.");
    _shader = entry->shader;
    _pipeline = entry->pipeline;
    return true;
}

std::optional<dispatch::DispatchDescriptor> GemmOperator::build(
    const Physical::Node& node,
    const OperatorContext& ctx) {

    const auto& d = std::get<execute::shape_inference::GemmDispatchData>(*ctx.dispatch_data);

    dispatch::DispatchDescriptor descriptor;
    descriptor.shader = _shader;
    descriptor.pipeline = _pipeline;
    descriptor.input_names = {node.inputs[0], node.inputs[1], node.inputs[2]};
    descriptor.output_names = {node.outputs[0]};

    descriptor.push_constants.write(d.M);
    descriptor.push_constants.write(d.N);
    descriptor.push_constants.write(d.K);
    descriptor.push_constants.write(d.alpha);
    descriptor.push_constants.write(d.beta);
    descriptor.push_constants.ensure_alignment();

    descriptor.workgroup_count_x = (d.M + 7) / 8;
    descriptor.workgroup_count_y = (d.N + 7) / 8;

    return descriptor;
}

} // namespace gdinfer::execute::operators
