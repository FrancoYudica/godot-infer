#include "parse_validation.hpp"

#include "utils.hpp"

namespace gdinfer::compile::parser {
namespace {

OperationResult check_kernel_pads_strides_dilations(
    const std::string& ctx,
    const std::vector<int64_t>& kernel_shape,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& dilations) {
    if (kernel_shape.size() != 2) {
        return {
            .success = false,
            .error = ctx + "only 2D convolutions supported, got kernel_shape dims: " + std::to_string(kernel_shape.size())};
    }
    if (kernel_shape[0] <= 0 || kernel_shape[1] <= 0) {
        return {
            .success = false,
            .error = ctx + "kernel dimensions must be positive"};
    }
    if (pads.size() != 4) {
        return {
            .success = false,
            .error = ctx + "expected 4 pad values, got: " + std::to_string(pads.size())};
    }
    if (strides.size() != 2) {
        return {
            .success = false,
            .error = ctx + "expected 2 stride values, got: " + std::to_string(strides.size())};
    }
    if (dilations.size() != 2) {
        return {
            .success = false,
            .error = ctx + "expected 2 dilation values, got: " + std::to_string(dilations.size())};
    }

    return OPERATION_OK;
}

std::string node_ctx(size_t idx, Operator op) {
    return "[node " + std::to_string(idx) + " (" + utils::op_name(op) + ")] ";
}

OperationResult check_node_arity(size_t idx, const Node& node) {
    const std::string ctx = node_ctx(idx, node.op);

    switch (node.op) {
    case Operator::Conv:
    case Operator::ConvTranspose:
        if (node.inputs.size() < 3) {
            return {.success = false, .error = ctx + "expected at least 3 inputs (activation, weights, bias), got " + std::to_string(node.inputs.size())};
        }
        if (node.outputs.size() != 1) {
            return {.success = false, .error = ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        }
        break;
    case Operator::Gemm:
        if (node.inputs.size() != 3) {
            return {.success = false, .error = ctx + "expected 3 inputs, got " + std::to_string(node.inputs.size())};
        }
        if (node.outputs.size() != 1) {
            return {.success = false, .error = ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        }
        break;
    case Operator::Im2Col:
        if (node.inputs.size() != 1) {
            return {.success = false, .error = ctx + "expected 1 input, got " + std::to_string(node.inputs.size())};
        }
        if (node.outputs.size() != 1) {
            return {.success = false, .error = ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        }
        break;
    case Operator::Reshape:
        if (node.inputs.size() != 2) {
            return {.success = false, .error = ctx + "expected 2 inputs (data, shape), got " + std::to_string(node.inputs.size())};
        }
        if (node.outputs.size() != 1) {
            return {.success = false, .error = ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        }
        break;
    case Operator::ReLU:
    case Operator::Sigmoid:
    case Operator::MaxPool2D:
        if (node.inputs.size() != 1) {
            return {.success = false, .error = ctx + "expected 1 input, got " + std::to_string(node.inputs.size())};
        }
        if (node.outputs.size() != 1) {
            return {.success = false, .error = ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        }
        break;

    case Operator::Unknown:
        return {.success = false, .error = ctx + "op is Unknown"};
    }
    return OPERATION_OK;
}

} // namespace

OperationResult validate_parse(const Graph& graph) {
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        const std::string ctx = node_ctx(i, node.op);

        auto arity = check_node_arity(i, node);
        if (!arity.success) {
            return arity;
        }

        switch (node.op) {
        case Operator::Gemm:
            if (!std::holds_alternative<GemmAttrs>(node.attributes)) {
                return {.success = false, .error = ctx + "expected GemmAttributes variant"};
            }
            if (!std::get<GemmAttrs>(node.attributes).transB) {
                return {.success = false, .error = ctx + "unsupported transB value `false`"};
            }
            break;

        case Operator::Conv:
        case Operator::Im2Col: {
            if (!std::holds_alternative<ConvAttrs>(node.attributes)) {
                return {.success = false, .error = ctx + "expected ConvAttributes variant"};
            }
            const auto& attrs = std::get<ConvAttrs>(node.attributes);
            auto vr = check_kernel_pads_strides_dilations(ctx, attrs.kernel_shape, attrs.pads, attrs.strides, attrs.dilations);
            if (!vr.success) {
                return vr;
            }
            break;
        }

        case Operator::ConvTranspose: {
            if (!std::holds_alternative<ConvTransposeAttrs>(node.attributes)) {
                return {.success = false, .error = ctx + "expected ConvTransposeAttributes variant"};
            }
            const auto& attrs = std::get<ConvTransposeAttrs>(node.attributes);
            auto vr = check_kernel_pads_strides_dilations(ctx, attrs.kernel_shape, attrs.pads, attrs.strides, attrs.dilations);
            if (!vr.success) {
                return vr;
            }
            if (attrs.output_padding.size() != 2) {
                return {.success = false, .error = ctx + "expected 2 output padding values, got: " + std::to_string(attrs.output_padding.size())};
            }

            break;
        }

        case Operator::MaxPool2D:
            if (!std::holds_alternative<MaxPool2DAttrs>(node.attributes)) {
                return {.success = false, .error = ctx + "expected MaxPool2DAttributes variant"};
            }
            {
                const auto& attrs = std::get<MaxPool2DAttrs>(node.attributes);
                auto vr = check_kernel_pads_strides_dilations(ctx, attrs.kernel_shape, attrs.pads, attrs.strides, attrs.dilations);
                if (!vr.success) {
                    return vr;
                }
            }
            break;

        case Operator::Reshape:
        case Operator::ReLU:
        case Operator::Sigmoid:
            break;

        case Operator::Unknown:
            return {.success = false, .error = ctx + "op is Unknown"};
        }
    }
    return OPERATION_OK;
}

} // namespace gdinfer::compile::parser
