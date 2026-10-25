#pragma once
#include "core/core.hpp"
#include "stages/compile/lowering/lowering.hpp"
#include "stages/execute/deletion_stack.hpp"
#include "stages/execute/dispatch/dispatcher.hpp"
#include "stages/execute/shader_cache.hpp"
#include "stages/execute/shape_inference/shape_inference.hpp"
#include "stages/execute/tensors/tensors.hpp"
#include "stages/execute/utils.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <optional>

namespace gdinfer::execute::operators {

namespace Physical = compile::lowering::Physical;

struct OperatorContext {
    // Only Reshape uses these two (its alias is a build()-time side effect); others just read
    // dispatch_data below.
    Ref<execute::tensors::TensorResourceManager> activations_tm;
    DeletionStack* frame_deletion_stack;
    // This node's precomputed dispatch payload; see stages/execute/shape_inference/dispatch_data.hpp.
    const execute::shape_inference::DispatchData* dispatch_data;
};

class IOperator {
  public:
    virtual ~IOperator() = default;
    // Looks up this operator's shader+pipeline in shader_cache, which owns them from here on.
    virtual bool init(godot::RenderingDevice* rd, ShaderCache* shader_cache) = 0;
    // Builds this node's dispatch descriptor. Nullopt means no GPU work (e.g. Reshape's alias).
    virtual std::optional<dispatch::DispatchDescriptor> build(
        const Physical::Node& node,
        const OperatorContext& ctx) = 0;
};

} // namespace gdinfer::execute::operators