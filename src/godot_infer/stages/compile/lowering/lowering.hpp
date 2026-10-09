#pragma once
#include "core/core.hpp"                    // IWYU pragma: export
#include "stages/compile/parser/types.hpp"  // IWYU pragma: export

namespace gdinfer::passes {

struct LoweringResult {
    Physical::Graph graph;
    OperationResult status;
};

LoweringResult lower(const compile::parser::Graph& graph);

} // namespace gdinfer::passes
