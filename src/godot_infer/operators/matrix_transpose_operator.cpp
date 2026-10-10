#include "matrix_transpose_operator.hpp"

#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

namespace gdinfer {

bool MatrixTransposeOperator::init(godot::RenderingDevice* rd) {
    const String path = "shaders/matrix_transpose.glsl";
    const String& shader_path = gdinfer::Utils::get_project_relative_path(path);
    _shader = gdinfer::Utils::load_shader(rd, shader_path);

    ERR_FAIL_COND_V_MSG(
        !_shader.is_valid(),
        false,
        "MatrixTransposeOperator: failed to load matrix_transpose shader.");

    _pipeline = rd->compute_pipeline_create(_shader);
    return _shader.is_valid() && _pipeline.is_valid();
}

void MatrixTransposeOperator::dispatch(
    const Physical::Node& node,
    const OperatorContext& ctx) {

    ERR_FAIL_COND_MSG(node.inputs.size() != 1, "MatrixTranspose: expected 1 input");
    ERR_FAIL_COND_MSG(node.outputs.size() != 1, "MatrixTranspose: expected 1 output");

    const auto& d = std::get<passes::MatrixTransposeDispatchData>(*ctx.dispatch_data);
    uint32_t M = d.rows;
    uint32_t N = d.cols;

    auto resolve = [&](const std::string& name) -> RID {
        RID rid = ctx.activations_tm->get_buffer_rid(name);
        if (rid.is_valid()) { return rid; }
        return ctx.initializers_tm->get_buffer_rid(name);
    };

    RID src_buf = resolve(node.inputs[0]);
    RID dst_buf = ctx.activations_tm->get_buffer_rid(node.outputs[0]);

    auto make_uniform = [&](const RID& rid, int binding) {
        Ref<RDUniform> u;
        u.instantiate();
        u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
        u->set_binding(binding);
        u->add_id(rid);
        return u;
    };

    TypedArray<RDUniform> uniforms;
    uniforms.push_back(make_uniform(src_buf, 0));
    uniforms.push_back(make_uniform(dst_buf, 1));

    RID uniform_set = ctx.rd->uniform_set_create(uniforms, _shader, 0);
    ctx.frame_deletion_stack->push([uniform_set, rd = ctx.rd]() {
        if (rd->uniform_set_is_valid(uniform_set)) {
            rd->free_rid(uniform_set);
        }
    });

    PushConstants pc{
        .rows = M,
        .cols = N};
    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(pc));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(pc));

    ctx.rd->compute_list_bind_compute_pipeline(ctx.compute_list, _pipeline);
    ctx.rd->compute_list_bind_uniform_set(ctx.compute_list, uniform_set, 0);
    ctx.rd->compute_list_set_push_constant(ctx.compute_list, pc_bytes, pc_bytes.size());

    uint32_t wg_x = (M + 7) / 8;
    uint32_t wg_y = (N + 7) / 8;
    ctx.rd->compute_list_dispatch(ctx.compute_list, wg_x, wg_y, 1);
}

void MatrixTransposeOperator::destroy(godot::RenderingDevice* rd) {
    if (_pipeline.is_valid()) {
        rd->free_rid(_pipeline);
    }
    if (_shader.is_valid()) {
        rd->free_rid(_shader);
    }
}

} // namespace gdinfer
