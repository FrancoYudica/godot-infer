#include "reshape_operator.hpp"

namespace gdinfer::execute::operators {

bool ReshapeOperator::init(godot::RenderingDevice* rd) {
    // Reshape is always a zero-copy alias; it never dispatches a compute shader.
    return true;
}

void ReshapeOperator::dispatch(
    const Physical::Node& node,
    const OperatorContext& ctx) {

    const auto& d = std::get<execute::shape_inference::ReshapeDispatchData>(*ctx.dispatch_data);

    ERR_FAIL_COND_MSG(node.inputs.size() != 1, "Reshape: expected 1 input");
    ERR_FAIL_COND_MSG(node.outputs.size() != 1, "Reshape: expected 1 output");

    auto result = ctx.activations_tm->create_alias(
        node.inputs[0],
        node.outputs[0],
        d.target_shape);

    ERR_FAIL_COND_MSG(!result.success, result.error.c_str());

    ctx.frame_deletion_stack->push([name = node.outputs[0], tm = ctx.activations_tm]() {
        tm->remove_alias(name);
    });
}

void ReshapeOperator::destroy(godot::RenderingDevice* rd) {
}

} // namespace gdinfer::execute::operators
