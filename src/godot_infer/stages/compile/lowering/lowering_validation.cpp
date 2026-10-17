#include "lowering_validation.hpp"

#include "utils.hpp"

#include <unordered_set>

namespace gdinfer::compile::lowering {

namespace {

std::string node_ctx(size_t idx, Physical::Operator op) {
    return "[node " + std::to_string(idx) + " (" + utils::op_name(op) + ")] ";
}

OperationResult check_connectivity(const Physical::Graph& graph) {
    std::unordered_set<std::string> defined;

    for (const auto& name : graph.input_names) {
        defined.insert(name);
    }
    for (const auto& [name, _] : graph.initializers) {
        defined.insert(name);
    }

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        const std::string ctx = node_ctx(i, node.op);

        for (const auto& input : node.inputs) {
            if (!defined.contains(input)) {
                return {.success = false, .error = ctx + "input tensor '" + input + "' is not defined"};
            }
        }

        for (const auto& output : node.outputs) {
            if (node.op != Physical::Operator::Reshape && defined.contains(output)) {
                return {.success = false, .error = ctx + "output tensor '" + output + "' already defined (duplicate output)"};
            }
            defined.insert(output);
        }
    }
    return OPERATION_OK;
}

OperationResult check_node_arity(size_t idx, const Physical::Node& node) {
    const std::string ctx = node_ctx(idx, node.op);

    switch (node.op) {
    case Physical::Operator::Im2Col:
        if (node.inputs.empty()) {
            return {.success = false, .error = ctx + "expected at least 1 input, got 0"};
        }
        if (node.outputs.size() != 1) {
            return {.success = false, .error = ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        }
        break;
    case Physical::Operator::Gemm:
    case Physical::Operator::Conv:
        if (node.inputs.size() != 3) {
            return {.success = false, .error = ctx + "expected 3 inputs, got " + std::to_string(node.inputs.size())};
        }
        if (node.outputs.size() != 1) {
            return {.success = false, .error = ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        }
        break;
    case Physical::Operator::Col2Im:
        if (node.inputs.size() != 2) {
            return {.success = false, .error = ctx + "expected 2 inputs, got " + std::to_string(node.inputs.size())};
        }
        if (node.outputs.size() != 1) {
            return {.success = false, .error = ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        }
        break;
    case Physical::Operator::Reshape:
    case Physical::Operator::MatrixTranspose:
    case Physical::Operator::ReLU:
    case Physical::Operator::Sigmoid:
    case Physical::Operator::MaxPool2D:
        if (node.inputs.size() != 1) {
            return {.success = false, .error = ctx + "expected 1 input, got " + std::to_string(node.inputs.size())};
        }
        if (node.outputs.size() != 1) {
            return {.success = false, .error = ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        }
        break;
    case Physical::Operator::Unknown:
        return {.success = false, .error = ctx + "op is Unknown - node cannot be dispatched"};
    }
    return OPERATION_OK;
}

OperationResult check_node_attributes(size_t idx, const Physical::Node& node) {
    const std::string ctx = node_ctx(idx, node.op);

    switch (node.op) {
    case Physical::Operator::Im2Col:
    case Physical::Operator::Conv:
        if (!std::holds_alternative<Physical::ConvAttrs>(node.attributes)) {
            return {.success = false, .error = ctx + "expected ConvAttributes variant"};
        }
        break;

    case Physical::Operator::Gemm:
        if (!std::holds_alternative<Physical::GemmAttrs>(node.attributes)) {
            return {.success = false, .error = ctx + "expected GemmAttributes variant"};
        }
        break;

    case Physical::Operator::Col2Im:
        if (!std::holds_alternative<Physical::Col2ImAttrs>(node.attributes)) {
            return {.success = false, .error = ctx + "expected Col2ImAttributes variant"};
        }
        {
            const auto& attrs = std::get<Physical::Col2ImAttrs>(node.attributes);
            if (attrs.output_padding_x != 0 || attrs.output_padding_y != 0) {
                return {.success = false, .error = ctx + "only zero output padding supported"};
            }
        }
        break;

    case Physical::Operator::Reshape:
        if (!std::holds_alternative<Physical::ReshapeAttrs>(node.attributes)) {
            return {.success = false, .error = ctx + "expected ReshapeAttributes variant"};
        }
        break;

    case Physical::Operator::MaxPool2D:
        if (!std::holds_alternative<Physical::MaxPool2DAttrs>(node.attributes)) {
            return {.success = false, .error = ctx + "expected MaxPool2DAttributes variant"};
        }
        break;

    case Physical::Operator::ReLU:
    case Physical::Operator::Sigmoid:
    case Physical::Operator::MatrixTranspose:
        break;

    case Physical::Operator::Unknown:
        return {.success = false, .error = ctx + "op is Unknown"};
    }

    return OPERATION_OK;
}
} // namespace
OperationResult lowering_validation(const Physical::Graph& graph) {
    auto conn = check_connectivity(graph);
    if (!conn.success) {
        return {.success = false, .error = conn.error};
    }

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];

        auto arity = check_node_arity(i, node);
        if (!arity.success) {
            return {.success = false, .error = arity.error};
        }

        auto attrs = check_node_attributes(i, node);
        if (!attrs.success) {
            return {.success = false, .error = attrs.error};
        }
    }

    return OPERATION_OK;
}
} // namespace gdinfer::compile::lowering
