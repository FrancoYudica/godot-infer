#pragma once
#include "core/core.hpp" // IWYU pragma: export

namespace onnx {
class GraphProto;
} // namespace onnx

namespace gdinfer::Logical {

// Parses every node in `proto` into `graph.nodes`. `graph.initializers` must
// already be populated. Fails on an unsupported operator.
OperationResult parse_nodes(const onnx::GraphProto& proto, Graph& graph);

} // namespace gdinfer::Logical
