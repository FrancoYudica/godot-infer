#pragma once
#include "core/core.hpp"                   // IWYU pragma: export
#include "stages/compile/parser/types.hpp" // IWYU pragma: export

namespace gdinfer::passes {

// Maps one logical op to the physical nodes it lowered into, plus its original attrs for
// shape inference to use.
struct SourceOpRecord {
    compile::parser::Operator op_type;

    // Original pre-lowering attributes
    std::variant<
        std::monostate,
        compile::parser::GemmAttrs,
        compile::parser::ConvAttrs,
        compile::parser::ConvTransposeAttrs,
        compile::parser::MaxPool2DAttrs>
        attrs;

    // the logical op's own original input tensor names
    std::vector<std::string> original_inputs;

    // indices into LoweringResult::graph.nodes, in emission order
    std::vector<size_t> physical_nodes;
};

using SourceOpTable = std::vector<SourceOpRecord>;

struct LoweringResult {
    Physical::Graph graph;
    SourceOpTable source_ops;
    OperationResult status;
};

LoweringResult lower(const compile::parser::Graph& graph);

} // namespace gdinfer::passes
