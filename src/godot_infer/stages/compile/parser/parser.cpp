#include "parser.hpp"

#include "node_parser.hpp"
#include "onnx/onnx_pb.h"

#include <cstring>

namespace gdinfer {

namespace Logical {
namespace {

void _parse_inputs(const onnx::GraphProto& proto, Graph& graph) {
    const auto& init = proto.initializer();
    std::unordered_set<std::string> initializer_names;
    for (const auto& t : init) {
        initializer_names.insert(t.name());
    }

    for (const auto& input : proto.input()) {
        if (initializer_names.contains(input.name())) {
            continue;
        }
        graph.input_names.push_back(input.name());
        graph.input_shapes[input.name()] = {};
        for (const auto& dim : input.type().tensor_type().shape().dim()) {
            graph.input_shapes[input.name()].push_back(dim.dim_value());
        }
    }
}

// Copies `raw_data`'s bytes into a vector of T via memcpy.
template <typename T>
std::vector<T> _read_raw_data(const std::string& raw_data) {
    const size_t count = raw_data.size() / sizeof(T);
    std::vector<T> values(count);
    std::memcpy(values.data(), raw_data.data(), count * sizeof(T));
    return values;
}

// Reads a tensor's values into `out.data`, converting to float.
OperationResult _parse_tensor_data(const onnx::TensorProto& tensor, Tensor& out) {
    switch (tensor.data_type()) {
    case onnx::TensorProto::FLOAT: {
        if (tensor.float_data_size() > 0) {
            out.data.assign(tensor.float_data().begin(), tensor.float_data().end());
        } else if (!tensor.raw_data().empty()) {
            out.data = _read_raw_data<float>(tensor.raw_data());
        }
        return OPERATION_OK;
    }

    case onnx::TensorProto::INT64: {
        std::vector<int64_t> values;
        if (tensor.int64_data_size() > 0) {
            values.assign(tensor.int64_data().begin(), tensor.int64_data().end());
        } else if (!tensor.raw_data().empty()) {
            values = _read_raw_data<int64_t>(tensor.raw_data());
        }
        out.data.reserve(values.size());
        for (int64_t v : values) {
            out.data.push_back(static_cast<float>(v));
        }
        return OPERATION_OK;
    }

    default:
        return {
            .success = false,
            .error = "unsupported initializer data type '" + onnx::TensorProto::DataType_Name(tensor.data_type()) + "' for tensor '" + tensor.name() + "'"};
    }
}

OperationResult _parse_initializers(const onnx::GraphProto& proto, Graph& graph) {
    // Goes through each initializer tensor
    for (const auto& tensor : proto.initializer()) {
        Tensor t;

        // Gets name and dimension
        t.name = tensor.name();
        for (auto dim : tensor.dims()) {
            t.shape.push_back(dim);
        }

        auto data_result = _parse_tensor_data(tensor, t);
        if (!data_result.success) {
            return data_result;
        }

        graph.initializers[t.name] = std::move(t);
    }
    return OPERATION_OK;
}

} // namespace

} // namespace Logical
namespace passes {
namespace {

ParseResult _parse_model(onnx::ModelProto& model) {
    const onnx::GraphProto& proto = model.graph();
    Logical::Graph graph;
    Logical::_parse_inputs(proto, graph);

    auto initializers_result = Logical::_parse_initializers(proto, graph);
    if (!initializers_result.success) {
        return {.graph = {}, .status = initializers_result};
    }

    auto nodes_result = Logical::parse_nodes(proto, graph);
    if (!nodes_result.success) {
        return {.graph = {}, .status = nodes_result};
    }

    return {.graph = std::move(graph), .status = OPERATION_OK};
}

} // namespace

ParseResult parse(const uint8_t* data, size_t size) {
    onnx::ModelProto model;
    if (!model.ParseFromArray(data, static_cast<int>(size))) {
        return {.graph = {}, .status = {.success = false, .error = "failed to deserialize ONNX protobuf from bytes"}};
    }

    return _parse_model(model);
}

} // namespace passes
} // namespace gdinfer
