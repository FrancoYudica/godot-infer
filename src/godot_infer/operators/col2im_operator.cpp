#include "col2im_operator.hpp"

#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

namespace gdinfer {

bool Col2ImOperator::init(godot::RenderingDevice* rd) {
    const String path = "shaders/col2im.glsl";
    const String& shader_path = gdinfer::Utils::get_project_relative_path(path);
    _shader = gdinfer::Utils::load_shader(rd, shader_path);

    ERR_FAIL_COND_V_MSG(
        !_shader.is_valid(),
        false,
        "Failed to load Col2Im shader.");

    _pipeline = rd->compute_pipeline_create(_shader);
    return _shader.is_valid() && _pipeline.is_valid();
}

void gdinfer::Col2ImOperator::dispatch(
    const gdinfer::Physical::Node& node,
    const OperatorContext& ctx) {

    ERR_FAIL_COND_MSG(node.outputs.size() != 1, "Col2Im: expected 1 output");
    ERR_FAIL_COND_MSG(node.inputs.size() != 2, "Col2Im: expected 2 inputs");

    auto resolve = [&](const std::string& name) -> RID {
        RID r = ctx.activations_tm->get_buffer_rid(name);
        if (r.is_valid()) { return r; }
        return ctx.initializers_tm->get_buffer_rid(name);
    };

    RID input_sb = ctx.activations_tm->get_buffer_rid(node.inputs[0]);
    RID bias_sb = resolve(node.inputs[1]);
    RID out_buf = ctx.activations_tm->get_buffer_rid(node.outputs[0]);

    const auto& d = std::get<passes::Col2ImDispatchData>(*ctx.dispatch_data);

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
    uniforms.push_back(make_uniform(bias_sb, 2));

    RID uniform_set_rid = ctx.rd->uniform_set_create(uniforms, _shader, 0);
    ctx.frame_deletion_stack->push([uniform_set_rid, rd = ctx.rd]() {
        if (rd->uniform_set_is_valid(uniform_set_rid)) {
            rd->free_rid(uniform_set_rid);
        }
    });

    PushConstants pc{
        .in_height = d.in_height,
        .in_width = d.in_width,
        .out_channels = d.out_channels,
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

    uint32_t groups_x = (d.out_width + 7) / 8;
    uint32_t groups_y = (d.out_height + 7) / 8;
    uint32_t groups_z = d.out_channels;
    ctx.rd->compute_list_dispatch(ctx.compute_list, groups_x, groups_y, groups_z);
}

void gdinfer::Col2ImOperator::destroy(godot::RenderingDevice* rd) {
    if (_pipeline.is_valid()) {
        rd->free_rid(_pipeline);
    }
    if (_shader.is_valid()) {
        rd->free_rid(_shader);
    }
}

} // namespace gdinfer