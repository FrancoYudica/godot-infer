#pragma once
#include "core/core.hpp"
#include "passes/dispatch_data.hpp"
#include "stages/compile/lowering/lowering.hpp"
#include "tensors/tensors.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

namespace gdinfer {
struct OperatorContext {
    RenderingDevice* rd;
    Ref<TensorResourceManager> initializers_tm;
    Ref<TensorResourceManager> activations_tm;
    int64_t compute_list;
    DeletionStack* frame_deletion_stack;
    const ShapeTable* shape_table;
    // Lets an operator reach sibling nodes from the same logical op, e.g. Col2Im resolving
    // ConvTranspose's original input via graph->nodes[source_ops->at(node.source_op)...].
    const Physical::Graph* graph;
    const passes::SourceOpTable* source_ops;
    // This node's precomputed dispatch payload; see passes/dispatch_data.hpp.
    const passes::DispatchData* dispatch_data;
};

class IOperator {
  public:
    virtual ~IOperator() = default;
    virtual bool init(godot::RenderingDevice* rd) = 0;
    virtual void dispatch(
        const gdinfer::Physical::Node& node,
        const OperatorContext& ctx) = 0;
    virtual void destroy(godot::RenderingDevice* rd) = 0;
};

} // namespace gdinfer