#include "im2col_operator.hpp"

#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

namespace gdinfer {

bool Im2ColOperator::init(godot::RenderingDevice* rd) {
    const String path = "shaders/im2col.glsl";
    const String& shader_path = gdinfer::Utils::get_project_relative_path(path);
    _shader = gdinfer::Utils::load_shader(rd, shader_path);

    ERR_FAIL_COND_V_MSG(
        !_shader.is_valid(),
        false,
        "Failed to load Im2Col shader.");

    _pipeline = rd->compute_pipeline_create(_shader);
    return _shader.is_valid() && _pipeline.is_valid();
}

void gdinfer::Im2ColOperator::dispatch(
    const gdinfer::Physical::Node& node,
    const OperatorContext& ctx) {

    RID input_sb = ctx.activations_tm->get_buffer_rid(node.inputs[0]);
    RID out_buf = ctx.activations_tm->get_buffer_rid(node.outputs[0]);

    const auto& d = std::get<passes::Im2ColDispatchData>(*ctx.dispatch_data);

    auto make_uniform = [&](const RID& rid, int binding) {
        Ref<RDUniform> u;
        u.instantiate();
        u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
        u->set_binding(binding);
        u->add_id(rid);
        return u;
    };

    TypedArray<RDUniform> uniforms;
    uniforms.push_back(make_uniform(input_sb, 0));
    uniforms.push_back(make_uniform(out_buf, 1));

    RID uniform_set_rid = ctx.rd->uniform_set_create(uniforms, _shader, 0);
    ctx.frame_deletion_stack->push([uniform_set_rid, rd = ctx.rd]() {
        if (rd->uniform_set_is_valid(uniform_set_rid)) {
            rd->free_rid(uniform_set_rid);
        }
    });

    PushConstants pc{
        .in_channels = d.in_channels,
        .in_height = d.in_height,
        .in_width = d.in_width,
        .out_height = d.out_height,
        .out_width = d.out_width,
        .kernel_h = d.kernel_h,
        .kernel_w = d.kernel_w,
        .padding_top = d.padding_top,
        .padding_left = d.padding_left,
        .stride_y = d.stride_y,
        .stride_x = d.stride_x,
        .dilation_y = d.dilation_y,
        .dilation_x = d.dilation_x};

    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(PushConstants));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(PushConstants));

    ctx.rd->compute_list_bind_compute_pipeline(ctx.compute_list, _pipeline);
    ctx.rd->compute_list_bind_uniform_set(ctx.compute_list, uniform_set_rid, 0);
    ctx.rd->compute_list_set_push_constant(ctx.compute_list, pc_bytes, pc_bytes.size());

    uint32_t total_workers = d.out_height * d.out_width * d.in_channels * d.kernel_w * d.kernel_h;
    uint32_t workgroup_count = (total_workers + 63) / 64;
    ctx.rd->compute_list_dispatch(ctx.compute_list, workgroup_count, 1, 1);
}

void gdinfer::Im2ColOperator::destroy(godot::RenderingDevice* rd) {
    if (_pipeline.is_valid()) {
        rd->free_rid(_pipeline);
    }
    if (_shader.is_valid()) {
        rd->free_rid(_shader);
    }
}

} // namespace gdinfer