#include "matrix_transpose_operator.hpp"

namespace gdinfer::execute::operators {

bool MatrixTransposeOperator::init(godot::RenderingDevice* rd, ShaderCache* shader_cache) {
    const auto* entry = shader_cache->get_or_load(rd, "shaders/matrix_transpose.glsl");
    ERR_FAIL_COND_V_MSG(entry == nullptr, false, "MatrixTransposeOperator: failed to load matrix_transpose shader.");
    _shader = entry->shader;
    _pipeline = entry->pipeline;
    return true;
}

std::optional<dispatch::DispatchDescriptor> MatrixTransposeOperator::build(
    const Physical::Node& node,
    const OperatorContext& ctx) {

    ERR_FAIL_COND_V_MSG(node.inputs.size() != 1, std::nullopt, "MatrixTranspose: expected 1 input");
    ERR_FAIL_COND_V_MSG(node.outputs.size() != 1, std::nullopt, "MatrixTranspose: expected 1 output");

    const auto& d = std::get<execute::shape_inference::MatrixTransposeDispatchData>(*ctx.dispatch_data);

    dispatch::DispatchDescriptor descriptor;
    descriptor.shader = _shader;
    descriptor.pipeline = _pipeline;
    descriptor.input_names = {node.inputs[0]};
    descriptor.output_names = {node.outputs[0]};

    descriptor.push_constants.write(d.rows);
    descriptor.push_constants.write(d.cols);
    descriptor.push_constants.ensure_alignment();

    descriptor.workgroup_count_x = (d.rows + 7) / 8;
    descriptor.workgroup_count_y = (d.cols + 7) / 8;

    return descriptor;
}

} // namespace gdinfer::execute::operators
