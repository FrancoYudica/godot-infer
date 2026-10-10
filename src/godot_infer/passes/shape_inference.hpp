#pragma once
#include "core/core.hpp"
#include "dispatch_data.hpp"
#include "stages/compile/lowering/lowering.hpp" // for SourceOpTable

namespace gdinfer::passes {

struct ShapeInferenceResult {
    ShapeTable shapes;
    // dispatch_data[i] describes graph.nodes[i]; see DispatchData in dispatch_data.hpp.
    DispatchDataTable dispatch_data;
    OperationResult status;
};

// Forward shape propagation over the physical graph, one configurator call per logical op
// (via source_ops) so a fanned-out op like Conv2d resolves all its physical nodes at once.
ShapeInferenceResult infer_shapes(
    const Physical::Graph& graph,
    const SourceOpTable& source_ops,
    const ShapeTable& input_shapes);

} // namespace gdinfer::passes