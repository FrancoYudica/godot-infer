#pragma once
#include "core/core.hpp" // IWYU pragma: export
#include "types.hpp"     // IWYU pragma: export

namespace gdinfer::compile::lowering {

/**
 * Validates a Physical::Graph after lowering.
 * Checks edge connectivity (every input is a defined tensor), per-node arity,
 * attribute variant types, and op-specific invariants (e.g. Col2Im's zero-output-padding
 * requirement, Reshape's Standard-mode target_shape).
 * Call between lower() and infer_shapes().
 */
OperationResult lowering_validation(const Physical::Graph& graph);

} // namespace gdinfer::compile::lowering
