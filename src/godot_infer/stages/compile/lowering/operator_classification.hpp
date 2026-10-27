#pragma once
#include "types.hpp"

namespace gdinfer::compile::lowering {

enum class OperatorClassification : uint8_t {
    Elementwise,
    InjectiveViewOnly,
    ReductionSink,
    OpaqueBarrier
};

// Static, compile-independent metadata per Physical::Operator, never computed by walking
// the graph. See "Operator Classification" in the vault for what each value means and why.
OperatorClassification classify_operator(Physical::Operator op);

} // namespace gdinfer::compile::lowering
