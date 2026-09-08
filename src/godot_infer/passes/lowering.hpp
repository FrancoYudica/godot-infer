#pragma once
#include "core/core.hpp" // IWYU pragma: export

namespace gdinfer::passes {

struct LoweringResult {
    Physical::Graph graph;
    OperationResult status;
};

LoweringResult lower(const Logical::Graph& logical_graph);

} // namespace gdinfer::passes
