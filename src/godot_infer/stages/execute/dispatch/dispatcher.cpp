#include "dispatcher.hpp"

#include <godot_cpp/classes/rd_uniform.hpp>

using namespace godot;

namespace gdinfer::execute::dispatch {

namespace {

Ref<RDUniform> make_uniform(RID rid, int binding) {
    Ref<RDUniform> u;
    u.instantiate();
    u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
    u->set_binding(binding);
    u->add_id(rid);
    return u;
}

} // namespace

void execute(
    RenderingDevice* rd,
    const DispatchDescriptor& descriptor,
    Ref<tensors::TensorResourceManager> activations_tm,
    Ref<tensors::TensorResourceManager> initializers_tm,
    int64_t compute_list,
    DeletionStack* frame_deletion_stack) {

    auto resolve_input = [&](const std::string& name) -> RID {
        RID rid = activations_tm->get_buffer_rid(name);
        if (rid.is_valid()) {
            return rid;
        }
        return initializers_tm->get_buffer_rid(name);
    };

    TypedArray<RDUniform> uniforms;
    for (const auto& name : descriptor.input_names) {
        uniforms.push_back(make_uniform(resolve_input(name), static_cast<int>(uniforms.size())));
    }
    for (const auto& name : descriptor.output_names) {
        uniforms.push_back(make_uniform(activations_tm->get_buffer_rid(name), static_cast<int>(uniforms.size())));
    }

    RID uniform_set_rid = rd->uniform_set_create(uniforms, descriptor.shader, 0);
    frame_deletion_stack->push([uniform_set_rid, rd]() {
        if (rd->uniform_set_is_valid(uniform_set_rid)) {
            rd->free_rid(uniform_set_rid);
        }
    });

    const auto& pc_data = descriptor.push_constants.data();
    PackedByteArray pc_bytes;
    pc_bytes.resize(static_cast<int>(pc_data.size()));
    memcpy(pc_bytes.ptrw(), pc_data.data(), pc_data.size());

    rd->compute_list_bind_compute_pipeline(compute_list, descriptor.pipeline);
    rd->compute_list_bind_uniform_set(compute_list, uniform_set_rid, 0);
    rd->compute_list_set_push_constant(compute_list, pc_bytes, pc_bytes.size());

    rd->compute_list_dispatch(
        compute_list,
        descriptor.workgroup_count_x,
        descriptor.workgroup_count_y,
        descriptor.workgroup_count_z);
}

} // namespace gdinfer::execute::dispatch
