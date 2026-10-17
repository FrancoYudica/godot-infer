#pragma once
#include "types.hpp"

namespace gdinfer::compile::lowering::utils {

std::string op_name(Physical::Operator op);

void print(const Physical::Graph& graph);

} // namespace gdinfer::compile::lowering::utils
