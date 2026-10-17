#pragma once
#include "core/core.hpp"
#include "stages/compile/lowering/lowering.hpp"
#include "stages/execute/deletion_stack.hpp"
#include "stages/execute/shape_inference/shape_inference.hpp"
#include "stages/execute/tensors/tensors.hpp"
#include "stages/execute/utils.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

namespace gdinfer::execute::operators {

namespace Physical = compile::lowering::Physical;
using execute::shape_inference::ShapeTable;

struct OperatorContext {
    RenderingDevice* rd;
    Ref<execute::tensors::TensorResourceManager> initializers_tm;
    Ref<execute::tensors::TensorResourceManager> activations_tm;
    int64_t compute_list;
    DeletionStack* frame_deletion_stack;
    const ShapeTable* shape_table;
    // Lets an operator reach sibling nodes from the same logical op, e.g. Col2Im resolving
    // ConvTranspose's original input via graph->nodes[source_ops->at(node.source_op)...].
    const Physical::Graph* graph;
    const compile::lowering::SourceOpTable* source_ops;
    // This node's precomputed dispatch payload; see stages/execute/shape_inference/dispatch_data.hpp.
    const execute::shape_inference::DispatchData* dispatch_data;
};

class IOperator {
  public:
    virtual ~IOperator() = default;
    virtual bool init(godot::RenderingDevice* rd) = 0;
    virtual void dispatch(
        const Physical::Node& node,
        const OperatorContext& ctx) = 0;
    virtual void destroy(godot::RenderingDevice* rd) = 0;
};

} // namespace gdinfer::execute::operators