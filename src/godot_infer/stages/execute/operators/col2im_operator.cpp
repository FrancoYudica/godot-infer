#include "col2im_operator.hpp"

namespace gdinfer::execute::operators {

bool Col2ImOperator::init(godot::RenderingDevice* rd, ShaderCache* shader_cache) {
    const auto* entry = shader_cache->get_or_load(rd, "shaders/col2im.glsl");
    ERR_FAIL_COND_V_MSG(entry == nullptr, false, "Failed to load Col2Im shader.");
    _shader = entry->shader;
    _pipeline = entry->pipeline;
    return true;
}

std::optional<dispatch::DispatchDescriptor> Col2ImOperator::build(
    const Physical::Node& node,
    const OperatorContext& ctx) {

    ERR_FAIL_COND_V_MSG(node.outputs.size() != 1, std::nullopt, "Col2Im: expected 1 output");
    ERR_FAIL_COND_V_MSG(node.inputs.size() != 2, std::nullopt, "Col2Im: expected 2 inputs");

    const auto& d = std::get<execute::shape_inference::Col2ImDispatchData>(*ctx.dispatch_data);

    dispatch::DispatchDescriptor descriptor;
    descriptor.shader = _shader;
    descriptor.pipeline = _pipeline;
    // col2im.glsl bindings: 0=col data, 1=bias (inputs, in order), 2=output.
    descriptor.input_names = {node.inputs[0], node.inputs[1]};
    descriptor.output_names = {node.outputs[0]};

    descriptor.push_constants.write(d.in_height);
    descriptor.push_constants.write(d.in_width);
    descriptor.push_constants.write(d.out_channels);
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

    descriptor.workgroup_count_x = (d.out_width + 7) / 8;
    descriptor.workgroup_count_y = (d.out_height + 7) / 8;
    descriptor.workgroup_count_z = d.out_channels;

    return descriptor;
}

} // namespace gdinfer::execute::operators
