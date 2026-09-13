#pragma once
#include "core/core.hpp" // IWYU pragma: export
#include "types.hpp"     // IWYU pragma: export

namespace gdinfer::compile::parser {

/**
 * Checks per-node arity (input/output counts) and attribute values
 * (kernel dimensions, pad/stride counts, Gemm transB requirement).
 */
OperationResult validate_parse(const Graph& graph);

} // namespace gdinfer::compile::parser
