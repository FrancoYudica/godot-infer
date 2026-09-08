#include "node_parser.hpp"

#include "onnx/onnx_pb.h"

#include <algorithm>
#include <godot_cpp/classes/project_settings.hpp>
#include <string_view>

namespace gdinfer::Logical {
namespace {

const std::unordered_map<std::string, Operator> operator_names = {
    {"Relu", Operator::ReLU},
    {"Sigmoid", Operator::Sigmoid},
    {"Gemm", Operator::Gemm},
    {"Conv", Operator::Conv},
    {"ConvTranspose", Operator::ConvTranspose},
    {"Im2Col", Operator::Im2Col},
    {"MaxPool", Operator::MaxPool2D},
    {"Reshape", Operator::Reshape},
    {"Unknown", Operator::Unknown}};

void _copy_io(const onnx::NodeProto& node, Node& n) {
    for (const auto& inp : node.input()) {
        n.inputs.push_back(inp);
    }
    for (const auto& out : node.output()) {
        n.outputs.push_back(out);
    }
}

void _try_read_ints_attr(const onnx::NodeProto& node, std::string_view name, std::vector<int64_t>& out) {
    for (const auto& attr : node.attribute()) {
        if (attr.name() == name) {
            out.assign(attr.ints().begin(), attr.ints().end());
            return;
        }
    }
}

void _default_if_empty(std::vector<int64_t>& v, size_t size, int64_t value) {
    if (v.empty()) {
        v.assign(size, value);
    }
}

void _warn_unhandled_attrs(const onnx::NodeProto& node, std::initializer_list<std::string_view> known_names) {
    for (const auto& attr : node.attribute()) {
        bool known = std::ranges::any_of(known_names, [&](std::string_view n) { return attr.name() == n; });
        if (!known) {
            godot::UtilityFunctions::push_warning("Unhandled attribute: " + godot::String(attr.name().c_str()) + " for operator: " + godot::String(node.name().c_str()));
        }
    }
}

Node _parse_relu_node(const onnx::NodeProto& node) {
    Node n;
    n.op = Operator::ReLU;
    _copy_io(node, n);
    return n;
}

Node _parse_sigmoid_node(const onnx::NodeProto& node) {
    Node n;
    n.op = Operator::Sigmoid;
    _copy_io(node, n);
    return n;
}

Node _parse_gemm_node(const onnx::NodeProto& node) {
    Node n;
    n.op = Operator::Gemm;
    _copy_io(node, n);

    n.attributes.emplace<GemmAttrs>();
    auto& gemm = std::get<GemmAttrs>(n.attributes);
    for (const auto& attr : node.attribute()) {
        if (attr.name() == "alpha") { gemm.alpha = attr.f(); }
        if (attr.name() == "beta") { gemm.beta = attr.f(); }
        if (attr.name() == "transB") { gemm.transB = (attr.i() == 1); }
    }
    return n;
}

// Derives kernel_shape from the weight tensor's shape when the attribute is absent.
void _fallback_kernel_shape_from_weights(const onnx::NodeProto& node, const Graph& graph, std::vector<int64_t>& kernel_shape) {
    if (!kernel_shape.empty() || node.input_size() <= 1) {
        return;
    }
    auto it = graph.initializers.find(node.input(1));
    if (it == graph.initializers.end()) {
        return;
    }
    for (size_t i = 2; i < it->second.shape.size(); ++i) {
        kernel_shape.push_back(it->second.shape[i]);
    }
}

Node _parse_conv_node(const onnx::NodeProto& node, const Graph& graph) {
    Node n;
    n.op = Operator::Conv;
    _copy_io(node, n);

    n.attributes.emplace<ConvAttrs>();
    auto& conv = std::get<ConvAttrs>(n.attributes);
    _try_read_ints_attr(node, "kernel_shape", conv.kernel_shape);
    _try_read_ints_attr(node, "pads", conv.pads);
    _try_read_ints_attr(node, "strides", conv.strides);
    _try_read_ints_attr(node, "dilations", conv.dilations);

    _fallback_kernel_shape_from_weights(node, graph, conv.kernel_shape);
    _default_if_empty(conv.strides, conv.kernel_shape.size(), 1);
    _default_if_empty(conv.pads, conv.kernel_shape.size() * 2, 0);
    _default_if_empty(conv.dilations, conv.kernel_shape.size(), 1);
    return n;
}

Node _parse_im2col_node(const onnx::NodeProto& node, const Graph& graph) {
    Node n;
    n.op = Operator::Im2Col;
    _copy_io(node, n);

    n.attributes.emplace<ConvAttrs>();
    auto& conv = std::get<ConvAttrs>(n.attributes);
    _try_read_ints_attr(node, "kernel_shape", conv.kernel_shape);
    _try_read_ints_attr(node, "pads", conv.pads);
    _try_read_ints_attr(node, "strides", conv.strides);
    _try_read_ints_attr(node, "dilations", conv.dilations);

    _fallback_kernel_shape_from_weights(node, graph, conv.kernel_shape);
    _default_if_empty(conv.strides, conv.kernel_shape.size(), 1);
    _default_if_empty(conv.pads, conv.kernel_shape.size() * 2, 0);
    _default_if_empty(conv.dilations, conv.kernel_shape.size(), 1);
    return n;
}

Node _parse_conv_transpose_node(const onnx::NodeProto& node, const Graph& graph) {
    Node n;
    n.op = Operator::ConvTranspose;
    _copy_io(node, n);

    n.attributes.emplace<ConvTransposeAttrs>();
    auto& conv = std::get<ConvTransposeAttrs>(n.attributes);
    _try_read_ints_attr(node, "kernel_shape", conv.kernel_shape);
    _try_read_ints_attr(node, "pads", conv.pads);
    _try_read_ints_attr(node, "strides", conv.strides);
    _try_read_ints_attr(node, "output_padding", conv.output_padding);
    _try_read_ints_attr(node, "dilations", conv.dilations);

    _fallback_kernel_shape_from_weights(node, graph, conv.kernel_shape);
    _default_if_empty(conv.strides, conv.kernel_shape.size(), 1);
    _default_if_empty(conv.pads, conv.kernel_shape.size() * 2, 0);
    _default_if_empty(conv.output_padding, conv.kernel_shape.size(), 0);
    _default_if_empty(conv.dilations, conv.kernel_shape.size(), 1);
    return n;
}

Node _parse_reshape_node(const onnx::NodeProto& node) {
    Node n;
    n.op = Operator::Reshape;
    _copy_io(node, n);
    return n;
}

Node _parse_maxpool_node(const onnx::NodeProto& node) {
    Node n;
    n.op = Operator::MaxPool2D;
    _copy_io(node, n);

    n.attributes.emplace<MaxPool2DAttrs>();
    auto& max_pool = std::get<MaxPool2DAttrs>(n.attributes);
    _try_read_ints_attr(node, "kernel_shape", max_pool.kernel_shape);
    _try_read_ints_attr(node, "pads", max_pool.pads);
    _try_read_ints_attr(node, "strides", max_pool.strides);
    _try_read_ints_attr(node, "dilations", max_pool.dilations);
    _warn_unhandled_attrs(node, {"kernel_shape", "pads", "strides", "dilations"});

    _default_if_empty(max_pool.strides, max_pool.kernel_shape.size(), 1);
    _default_if_empty(max_pool.pads, max_pool.kernel_shape.size() * 2, 0);
    _default_if_empty(max_pool.dilations, max_pool.kernel_shape.size(), 1);
    return n;
}

} // namespace

OperationResult parse_nodes(const onnx::GraphProto& proto, Graph& graph) {
    for (const auto& node : proto.node()) {
        auto it = operator_names.find(node.op_type());
        if (it == operator_names.end()) {
            return {.success = false, .error = "unsupported operator '" + node.op_type() + "'"};
        }

        switch (it->second) {
        case Operator::ReLU:
            graph.nodes.push_back(_parse_relu_node(node));
            break;
        case Operator::Sigmoid:
            graph.nodes.push_back(_parse_sigmoid_node(node));
            break;
        case Operator::Gemm:
            graph.nodes.push_back(_parse_gemm_node(node));
            break;
        case Operator::Conv:
            graph.nodes.push_back(_parse_conv_node(node, graph));
            break;
        case Operator::Im2Col:
            graph.nodes.push_back(_parse_im2col_node(node, graph));
            break;
        case Operator::ConvTranspose:
            graph.nodes.push_back(_parse_conv_transpose_node(node, graph));
            break;
        case Operator::Reshape:
            graph.nodes.push_back(_parse_reshape_node(node));
            break;
        case Operator::MaxPool2D:
            graph.nodes.push_back(_parse_maxpool_node(node));
            break;
        case Operator::Unknown:
            return {.success = false, .error = "operator '" + node.op_type() + "' resolved to the Unknown sentinel"};
        }
    }
    return OPERATION_OK;
}

} // namespace gdinfer::Logical
