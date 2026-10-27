#include "utils.hpp"

#include "core/utils.hpp"

#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace gdinfer::compile::lowering::utils {

std::string op_name(Physical::Operator op) {
    switch (op) {
    case Physical::Operator::Gemm:
        return "Gemm";
    case Physical::Operator::ReLU:
        return "ReLU";
    case Physical::Operator::Sigmoid:
        return "Sigmoid";
    case Physical::Operator::Im2Col:
        return "Im2Col";
    case Physical::Operator::Col2Im:
        return "Col2Im";
    case Physical::Operator::Reshape:
        return "Reshape";
    case Physical::Operator::MatrixTranspose:
        return "MatrixTranspose";
    case Physical::Operator::MaxPool2D:
        return "MaxPool2D";
    default:
        return "Unknown";
    }
}

void print(const Physical::Graph& graph) {
    auto to_gstring = [](const std::string& s) {
        return godot::String(s.c_str());
    };

    godot::UtilityFunctions::print("Input names: " + gdinfer::Utils::get_iterator_str(graph.input_names.begin(), graph.input_names.end()));
    godot::UtilityFunctions::print("Initializers:");
    for (const auto& [name, tensor] : graph.initializers) {
        godot::UtilityFunctions::print(" " + to_gstring(name) + ": ", gdinfer::Utils::get_iterator_str(tensor.shape.begin(), tensor.shape.end()));
    }
    for (const auto& node : graph.nodes) {
        godot::UtilityFunctions::print("Node: ", godot::String(op_name(node.op).c_str()));
        godot::UtilityFunctions::print(" inputs: ", gdinfer::Utils::get_iterator_str(node.inputs.begin(), node.inputs.end()));
        godot::UtilityFunctions::print(" outputs: ", gdinfer::Utils::get_iterator_str(node.outputs.begin(), node.outputs.end()));
    }
}

} // namespace gdinfer::compile::lowering::utils
