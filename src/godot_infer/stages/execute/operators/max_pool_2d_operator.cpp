#include "max_pool_2d_operator.hpp"

namespace gdinfer::execute::operators {

bool MaxPool2DOperator::init(godot::RenderingDevice* rd, ShaderCache* shader_cache) {
    const auto* entry = shader_cache->get_or_load(rd, "shaders/max_pool_2d.glsl");
    ERR_FAIL_COND_V_MSG(entry == nullptr, false, "Failed to load MaxPool2D shader.");
    _shader = entry->shader;
    _pipeline = entry->pipeline;
    return true;
}

std::optional<dispatch::DispatchDescriptor> MaxPool2DOperator::build(
    const Physical::Node& node,
    const OperatorContext& ctx) {

    const auto& d = std::get<execute::shape_inference::MaxPool2DDispatchData>(*ctx.dispatch_data);

    dispatch::DispatchDescriptor descriptor;
    descriptor.shader = _shader;
    descriptor.pipeline = _pipeline;
    descriptor.input_names = {node.inputs[0]};
    descriptor.output_names = {node.outputs[0]};

    descriptor.push_constants.write(d.batch_size);
    descriptor.push_constants.write(d.channels);
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

    descriptor.workgroup_count_x = (d.out_width + 7) / 8;
    descriptor.workgroup_count_y = (d.out_height + 7) / 8;
    descriptor.workgroup_count_z = d.batch_size * d.channels;

    return descriptor;
}

} // namespace gdinfer::execute::operators
