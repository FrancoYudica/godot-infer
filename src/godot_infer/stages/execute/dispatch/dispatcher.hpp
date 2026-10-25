#pragma once
#include "core/core.hpp"
#include "stages/execute/deletion_stack.hpp"
#include "stages/execute/tensors/tensors.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

namespace gdinfer::execute::dispatch {

// Packs push-constant fields in shader declaration order. Call ensure_alignment() once done.
class PushConstantWriter {
  public:
    PushConstantWriter() = default;

    template <typename T>
    void write(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>, "PushConstantWriter can only write trivially copyable types.");
        const auto* ptr = reinterpret_cast<const uint8_t*>(&value);
        _data.insert(_data.end(), ptr, ptr + sizeof(T));
    }

    const std::vector<uint8_t>& data() const {
        return _data;
    }

    void ensure_alignment() {
        static constexpr size_t alignment = 16; // Align to 16 bytes for GPU push constants
        size_t current_size = _data.size();
        size_t padding_needed = (alignment - (current_size % alignment)) % alignment;
        _data.insert(_data.end(), padding_needed, 0);
    }

    bool is_valid() const {
        static constexpr size_t max_size = 64; // Maximum size for GPU push constants
        return !_data.empty() && _data.size() <= max_size;
    }

  private:
    std::vector<uint8_t> _data;
};

// One kernel dispatch's data, built by an operator's build() and run by execute() below.
// Bindings: input_names first from 0, then output_names, in order -- shaders must match.
struct DispatchDescriptor {
    RID shader;
    RID pipeline;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    PushConstantWriter push_constants;
    uint32_t workgroup_count_x = 1;
    uint32_t workgroup_count_y = 1;
    uint32_t workgroup_count_z = 1;
};

// Resolves names to buffer RIDs (inputs fall back to initializers) and dispatches descriptor.
void execute(
    godot::RenderingDevice* rd,
    const DispatchDescriptor& descriptor,
    Ref<tensors::TensorResourceManager> activations_tm,
    Ref<tensors::TensorResourceManager> initializers_tm,
    int64_t compute_list,
    DeletionStack* frame_deletion_stack);

} // namespace gdinfer::execute::dispatch
