#pragma once
#include "core/core.hpp" // IWYU pragma: export
#include "types.hpp"

#include <cstdint>

namespace gdinfer::compile::parser {

struct ParseResult {
    Graph graph;
    OperationResult status;
};

ParseResult parse(const uint8_t* data, size_t size);

} // namespace gdinfer::compile::parser
