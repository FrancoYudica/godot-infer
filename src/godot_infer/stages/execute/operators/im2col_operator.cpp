#include "im2col_operator.hpp"

namespace gdinfer::execute::operators {

bool Im2ColOperator::init(godot::RenderingDevice* rd, ShaderCache* shader_cache) {
    const auto* entry = shader_cache->get_or_load(rd, "shaders/im2col.glsl");
    ERR_FAIL_COND_V_MSG(entry == nullptr, false, "Failed to load Im2Col shader.");
    _shader = entry->shader;
    _pipeline = entry->pipeline;
    return true;
}

std::optional<dispatch::DispatchDescriptor> Im2ColOperator::build(
    const Physical::Node& node,
    const OperatorContext& ctx) {

    const auto& d = std::get<execute::shape_inference::Im2ColDispatchData>(*ctx.dispatch_data);

    dispatch::DispatchDescriptor descriptor;
    descriptor.shader = _shader;
    descriptor.pipeline = _pipeline;
    descriptor.input_names = {node.inputs[0]};
    descriptor.output_names = {node.outputs[0]};

    descriptor.push_constants.write(d.in_channels);
    descriptor.push_constants.write(d.in_height);
    descriptor.push_constants.write(d.in_width);
    descriptor.push_constants.write(d.out_height);
    descriptor.push_constants.write(d.out_width);
    descriptor.push_constants.write(d.kernel_h);
    descriptor.push_constants.write(d.kernel_w);
    descriptor.push_constants.write(d.padding_top);
    descriptor.push_constants.write(d.padding_left);
    descriptor.push_constants.write(d.stride_y);
    descriptor.push_constants.write(d.stride_x);
    descriptor.push_constants.write(d.dilation_y);
    descriptor.push_constants.write(d.dilation_x);
    descriptor.push_constants.ensure_alignment();

    uint32_t total_workers = d.out_height * d.out_width * d.in_channels * d.kernel_w * d.kernel_h;
    descriptor.workgroup_count_x = (total_workers + 63) / 64;

    return descriptor;
}

} // namespace gdinfer::execute::operators
