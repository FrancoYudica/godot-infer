#include "reshape_operator.hpp"

namespace gdinfer::execute::operators {

bool ReshapeOperator::init(godot::RenderingDevice* rd, ShaderCache* shader_cache) {
    // Reshape is always a zero-copy alias; it never dispatches a compute shader.
    return true;
}

std::optional<dispatch::DispatchDescriptor> ReshapeOperator::build(
    const Physical::Node& node,
    const OperatorContext& ctx) {

    const auto& d = std::get<execute::shape_inference::ReshapeDispatchData>(*ctx.dispatch_data);

    ERR_FAIL_COND_V_MSG(node.inputs.size() != 1, std::nullopt, "Reshape: expected 1 input");
    ERR_FAIL_COND_V_MSG(node.outputs.size() != 1, std::nullopt, "Reshape: expected 1 output");

    auto result = ctx.activations_tm->create_alias(
        node.inputs[0],
        node.outputs[0],
        d.target_shape);

    ERR_FAIL_COND_V_MSG(!result.success, std::nullopt, result.error.c_str());

    ctx.frame_deletion_stack->push([name = node.outputs[0], tm = ctx.activations_tm]() {
        tm->remove_alias(name);
    });

    // No GPU work: the alias above is Reshape's entire effect.
    return std::nullopt;
}

} // namespace gdinfer::execute::operators
