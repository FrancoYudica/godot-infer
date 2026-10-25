#include "texture_output_handler_compute.hpp"

#include "core/core.hpp"

#include <godot_cpp/classes/rd_texture_format.hpp>

using namespace godot;

namespace {
struct PushConstants {
    uint32_t width;
    uint32_t height;
    uint32_t channel_count;
    std::array<int, 1> padding; // Alignment to 16 bytes
};

bool is_supported_format(
    const godot::Ref<godot::RDTextureFormat>& format) {
    switch (format->get_format()) {
    case RenderingDevice::DATA_FORMAT_R8G8B8A8_UNORM:
        return true;
    default:
        ERR_PRINT(
            "InferenceEngine: Unsupported texture format. Only supported: "
            "R8G8B8A8_UNORM");
        return false;
    }
}
} // namespace
namespace gdinfer::execute::io {

bool TextureOutputHandlerCompute::init(godot::RenderingDevice* rd, ShaderCache* shader_cache) {
    const auto* entry = shader_cache->get_or_load(rd, "shaders/texture_write.glsl");
    ERR_FAIL_COND_V_MSG(entry == nullptr, false, "Failed to load texture_write shader.");
    _shader_rid = entry->shader;
    _pipeline_rid = entry->pipeline;

    // Uniform for texture sampler and storage buffer
    Ref<RDUniform> texture_uniform;
    texture_uniform.instantiate();
    texture_uniform->set_uniform_type(RenderingDevice::UNIFORM_TYPE_IMAGE);
    texture_uniform->set_binding(0);

    Ref<RDUniform> tensor_uniform;
    tensor_uniform.instantiate();
    tensor_uniform->set_uniform_type(
        RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
    tensor_uniform->set_binding(1);

    _uniforms.push_back(texture_uniform);
    _uniforms.push_back(tensor_uniform);

    return true;
}

void TextureOutputHandlerCompute::dispatch(
    const std::unique_ptr<OutputDesc::BaseData>& desc,
    const OutputHandlerContext& ctx) {
    OutputDesc::Texture* texture_desc =
        dynamic_cast<OutputDesc::Texture*>(desc.get());

    ERR_FAIL_COND_MSG(
        !texture_desc,
        "InferenceEngine: Failed to cast OutputDesc to Texture.");

    auto format = ctx.rd->texture_get_format(texture_desc->target_texture);

    ERR_FAIL_COND_MSG(
        !is_supported_format(format),
        "InferenceEngine: Unsupported texture format.");

    _texture_width = format->get_width();
    _texture_height = format->get_height();
    _texture_channels = 3;

    const std::vector<int64_t>& tensor_shape =
        ctx.activations_tm->get_tensor_shape(texture_desc->tensor_name);

    const std::vector<int64_t> texture_shape = {
        1,
        _texture_channels,
        _texture_height,
        _texture_width};

    ERR_FAIL_COND_MSG(
        tensor_shape.size() != texture_shape.size(),
        "InferenceEngine: Texture shape " +
            Utils::shape_to_str(texture_shape) +
            " does not match tensor shape " +
            Utils::shape_to_str(tensor_shape) + ".");

    RID tensor_rid =
        ctx.activations_tm->get_buffer_rid(texture_desc->tensor_name);

    // Prepare uniforms for uniform set creation
    _uniforms[0]->clear_ids();
    _uniforms[0]->add_id(texture_desc->target_texture);
    _uniforms[1]->clear_ids();
    _uniforms[1]->add_id(tensor_rid);

    RID uniform_set_rid = ctx.rd->uniform_set_create(
        {_uniforms[0], _uniforms[1]},
        _shader_rid,
        0);

    PushConstants pc{
        .width = _texture_width,
        .height = _texture_height,
        .channel_count = _texture_channels};

    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(PushConstants));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(PushConstants));

    ctx.rd->compute_list_bind_compute_pipeline(ctx.compute_list, _pipeline_rid);
    ctx.rd->compute_list_bind_uniform_set(ctx.compute_list, uniform_set_rid, 0);
    ctx.rd->compute_list_set_push_constant(ctx.compute_list, pc_bytes, pc_bytes.size());

    // Makes sure to dispatch in groups of 8x8
    uint32_t x_groups = (_texture_width + 7) / 8;
    uint32_t y_groups = (_texture_height + 7) / 8;
    ctx.rd->compute_list_dispatch(ctx.compute_list, x_groups, y_groups, 1);

    ctx.frame_deletion_stack->push([uniform_set_rid, rd = ctx.rd]() {
        if (rd->uniform_set_is_valid(uniform_set_rid)) {
            rd->free_rid(uniform_set_rid);
        }
    });
}

Variant TextureOutputHandlerCompute::download(
    const std::unique_ptr<OutputDesc::BaseData>& desc,
    RenderingDevice* rd,
    Ref<execute::tensors::TensorResourceManager> activations_tm) {
    return nullptr;
}
} // namespace gdinfer::execute::io