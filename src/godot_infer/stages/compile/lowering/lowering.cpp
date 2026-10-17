#include "lowering.hpp"

namespace gdinfer::compile::lowering {

namespace {

OperationResult low_relu(const compile::parser::Node& node, Physical::Graph& graph) {
    Physical::Node n;
    n.inputs = node.inputs;
    n.outputs = node.outputs;
    n.op = Physical::Operator::ReLU;
    graph.nodes.push_back(std::move(n));
    return OPERATION_OK;
}

OperationResult low_sigmoid(const compile::parser::Node& node, Physical::Graph& graph) {
    Physical::Node n;
    n.inputs = node.inputs;
    n.outputs = node.outputs;
    n.op = Physical::Operator::Sigmoid;
    graph.nodes.push_back(std::move(n));
    return OPERATION_OK;
}

OperationResult low_gemm(const compile::parser::Node& node, Physical::Graph& graph) {
    const auto& l_attrs = std::get<compile::parser::GemmAttrs>(node.attributes);

    Physical::Node n;
    n.inputs = node.inputs;
    n.outputs = node.outputs;

    Physical::GemmAttrs attrs;
    attrs.alpha = l_attrs.alpha;
    attrs.beta = l_attrs.beta;
    n.attributes = attrs;

    n.op = Physical::Operator::Gemm;
    graph.nodes.push_back(std::move(n));
    return OPERATION_OK;
}

OperationResult low_conv(const compile::parser::Node& node, Physical::Graph& graph) {
    const auto& l_attrs = std::get<compile::parser::ConvAttrs>(node.attributes);

    const std::string col_name = node.outputs[0] + "__col";
    // GEMM writes [OH*OW, OC]; the matrix transpose below transposes to [OC, OH*OW] (BCHW).
    const std::string flat_name = node.outputs[0] + "__flat";

    Physical::Node im2col;
    im2col.inputs = {node.inputs[0], node.inputs[1]}; // activation, weights
    im2col.outputs = {col_name};

    // ONNX pads: [top, left, bottom, right]
    // ONNX shapes: [H, W]
    im2col.attributes = Physical::ConvAttrs{
        .kernel_h = static_cast<uint32_t>(l_attrs.kernel_shape[0]),
        .kernel_w = static_cast<uint32_t>(l_attrs.kernel_shape[1]),
        .padding_top = static_cast<uint32_t>(l_attrs.pads[0]),
        .padding_left = static_cast<uint32_t>(l_attrs.pads[1]),
        .padding_bottom = static_cast<uint32_t>(l_attrs.pads[2]),
        .padding_right = static_cast<uint32_t>(l_attrs.pads[3]),
        .stride_y = static_cast<uint32_t>(l_attrs.strides[0]),
        .stride_x = static_cast<uint32_t>(l_attrs.strides[1]),
        .dilation_y = static_cast<uint32_t>(l_attrs.dilations[0]),
        .dilation_x = static_cast<uint32_t>(l_attrs.dilations[1])};

    im2col.op = Physical::Operator::Im2Col;

    Physical::Node gemm;
    gemm.inputs = {col_name, node.inputs[1], node.inputs[2]};
    gemm.outputs = {flat_name}; // [OH*OW, OC]
    gemm.attributes = Physical::GemmAttrs{.alpha = 1.0F, .beta = 1.0F};
    gemm.op = Physical::Operator::Gemm;

    // GPU matrix transpose [OH*OW, OC] -> [OC, OH*OW], then relabel as 4D BCHW.
    Physical::Node transpose;
    transpose.inputs = {flat_name};
    transpose.outputs = node.outputs;
    transpose.op = Physical::Operator::MatrixTranspose;

    graph.nodes.push_back(std::move(im2col));
    graph.nodes.push_back(std::move(gemm));
    graph.nodes.push_back(std::move(transpose));
    return OPERATION_OK;
}

OperationResult low_im2col(const compile::parser::Node& node, Physical::Graph& graph) {
    const auto& l_attrs = std::get<compile::parser::ConvAttrs>(node.attributes);

    Physical::Node n;
    n.inputs = {node.inputs[0]};
    n.outputs = {node.outputs[0]};

    n.attributes = Physical::ConvAttrs{
        .kernel_h = static_cast<uint32_t>(l_attrs.kernel_shape[0]),
        .kernel_w = static_cast<uint32_t>(l_attrs.kernel_shape[1]),
        .padding_top = static_cast<uint32_t>(l_attrs.pads[0]),
        .padding_left = static_cast<uint32_t>(l_attrs.pads[1]),
        .padding_bottom = static_cast<uint32_t>(l_attrs.pads[2]),
        .padding_right = static_cast<uint32_t>(l_attrs.pads[3]),
        .stride_y = static_cast<uint32_t>(l_attrs.strides[0]),
        .stride_x = static_cast<uint32_t>(l_attrs.strides[1]),
        .dilation_y = static_cast<uint32_t>(l_attrs.dilations[0]),
        .dilation_x = static_cast<uint32_t>(l_attrs.dilations[1])};

    n.op = Physical::Operator::Im2Col;
    graph.nodes.push_back(std::move(n));
    return OPERATION_OK;
}

OperationResult low_conv_transpose(const compile::parser::Node& node, Physical::Graph& graph) {
    const auto& l_attrs = std::get<compile::parser::ConvTransposeAttrs>(node.attributes);

    const std::string flat_name = node.inputs[0] + "__flat";
    const std::string gemm_name = node.outputs[0] + "__gemm";
    const std::string col2im_name = node.outputs[0] + "__col2im";

    // BCHW [1, IC, IH, IW] -> [IH*IW, IC] so GEMM gets row-major input. Correct for batch=1
    // only; batch>1 needs a 4D permute.
    Physical::Node pre_transpose;
    pre_transpose.op = Physical::Operator::MatrixTranspose;
    pre_transpose.inputs = {node.inputs[0]};
    pre_transpose.outputs = {flat_name};

    // ONNX ConvTranspose weights are [ic, oc, kH, kW]. The GEMM shader always
    // computes A x B^T, so transpose to [oc*kH*kW, ic] at load time.
    const std::string& weight_name = node.inputs[1];
    if (!graph.initializers.contains(weight_name)) {
        return {.success = false, .error = "ConvTranspose: weight tensor '" + weight_name + "' not found in initializers"};
    }

    const std::string transposed_weight_name = weight_name + "__T";
    {
        const Tensor& w = graph.initializers.at(weight_name);
        const int64_t K = w.shape[0]; // in_channels
        int64_t N = 1;
        for (size_t i = 1; i < w.shape.size(); ++i) {
            N *= w.shape[i]; // out_channels * kH * kW
        }

        Tensor w_T;
        w_T.name = transposed_weight_name;
        w_T.shape = {N, K};
        w_T.data.resize(static_cast<size_t>(N * K));
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t n = 0; n < N; ++n) {
                w_T.data[n * K + k] = w.data[k * N + n];
            }
        }
        graph.initializers[transposed_weight_name] = std::move(w_T);
    }

    // Gemm: [b*ih*iw, ic] x [oc*kh*kw, ic]^T -> [b*ih*iw, oc*kh*kw]
    // beta = 0.0 because Col2Im will handle bias to avoid multiple accumulation.
    Physical::GemmAttrs gemm_attrs{.alpha = 1.0F, .beta = 0.0F};

    Physical::Node gemm;
    gemm.op = Physical::Operator::Gemm;
    gemm.inputs = {flat_name, transposed_weight_name, node.inputs[2]};
    gemm.outputs = {gemm_name};
    gemm.attributes = gemm_attrs;

    // Col2Im: [b*ih*iw, oc*kh*kw] -> [out_h*out_w, oc]
    Physical::Col2ImAttrs col2im_attrs = {
        .kernel_h = static_cast<uint32_t>(l_attrs.kernel_shape[0]),
        .kernel_w = static_cast<uint32_t>(l_attrs.kernel_shape[1]),
        .padding_top = static_cast<uint32_t>(l_attrs.pads[0]),
        .padding_left = static_cast<uint32_t>(l_attrs.pads[1]),
        .padding_bottom = static_cast<uint32_t>(l_attrs.pads[2]),
        .padding_right = static_cast<uint32_t>(l_attrs.pads[3]),
        .stride_y = static_cast<uint32_t>(l_attrs.strides[0]),
        .stride_x = static_cast<uint32_t>(l_attrs.strides[1]),
        .dilation_y = static_cast<uint32_t>(l_attrs.dilations[0]),
        .dilation_x = static_cast<uint32_t>(l_attrs.dilations[1]),
        .output_padding_y = static_cast<uint32_t>(l_attrs.output_padding[0]),
        .output_padding_x = static_cast<uint32_t>(l_attrs.output_padding[1])};

    Physical::Node col2im;
    col2im.op = Physical::Operator::Col2Im;
    col2im.inputs = {gemm_name, node.inputs[2]}; // Col data and Bias
    col2im.outputs = {col2im_name};
    col2im.attributes = col2im_attrs;

    // Reshape 2: [b*oh*ow, oc] -> [b, oc, oh, ow]
    Physical::Node post_reshape;
    post_reshape.op = Physical::Operator::Reshape;
    post_reshape.inputs = {col2im_name};
    post_reshape.outputs = {node.outputs[0]};
    // target_shape is left empty here: configure_conv_transpose_shapes resolves it directly
    // into dispatch_data instead (see Physical::ReshapeAttrs's doc comment).
    post_reshape.attributes = Physical::ReshapeAttrs{};

    graph.nodes.push_back(std::move(pre_transpose));
    graph.nodes.push_back(std::move(gemm));
    graph.nodes.push_back(std::move(col2im));
    graph.nodes.push_back(std::move(post_reshape));
    return OPERATION_OK;
}

OperationResult low_reshape(const compile::parser::Node& node, Physical::Graph& graph) {
    if (!graph.initializers.contains(node.inputs[1])) {
        return {.success = false, .error = "Reshape: shape tensor '" + node.inputs[1] + "' not found in initializers"};
    }
    const auto& it = graph.initializers.find(node.inputs[1]);

    std::vector<int64_t> target_shape;
    for (float v : it->second.data) {
        target_shape.push_back(static_cast<int64_t>(v));
    }

    Physical::Node n;
    n.op = Physical::Operator::Reshape;
    n.inputs = {node.inputs[0]};
    n.outputs = node.outputs;
    n.attributes = Physical::ReshapeAttrs{
        .target_shape = std::move(target_shape)};

    graph.nodes.push_back(std::move(n));
    return OPERATION_OK;
}

OperationResult low_max_pool_2d(const compile::parser::Node& node, Physical::Graph& graph) {
    const auto& l_attrs = std::get<compile::parser::MaxPool2DAttrs>(node.attributes);

    Physical::Node max_pool;
    max_pool.op = Physical::Operator::MaxPool2D;
    max_pool.inputs = {node.inputs[0]};
    max_pool.outputs = {node.outputs[0]};

    max_pool.attributes = Physical::MaxPool2DAttrs{
        .kernel_h = static_cast<uint32_t>(l_attrs.kernel_shape[0]),
        .kernel_w = static_cast<uint32_t>(l_attrs.kernel_shape[1]),
        .padding_top = static_cast<uint32_t>(l_attrs.pads[0]),
        .padding_left = static_cast<uint32_t>(l_attrs.pads[1]),
        .padding_bottom = static_cast<uint32_t>(l_attrs.pads[2]),
        .padding_right = static_cast<uint32_t>(l_attrs.pads[3]),
        .stride_y = static_cast<uint32_t>(l_attrs.strides[0]),
        .stride_x = static_cast<uint32_t>(l_attrs.strides[1]),
        .dilation_y = static_cast<uint32_t>(l_attrs.dilations[0]),
        .dilation_x = static_cast<uint32_t>(l_attrs.dilations[1])};

    graph.nodes.push_back(std::move(max_pool));
    return OPERATION_OK;
}

} // namespace

LoweringResult lower(const compile::parser::Graph& logical_graph) {
    LoweringResult result;
    result.status = OPERATION_OK;
    auto& graph = result.graph;

    graph.input_names = logical_graph.input_names;
    graph.input_shapes = logical_graph.input_shapes;
    graph.initializers = logical_graph.initializers;

    for (const auto& node : logical_graph.nodes) {
        OperationResult op_result;
        const size_t physical_before = graph.nodes.size();

        switch (node.op) {
        case compile::parser::Operator::ReLU:
            op_result = low_relu(node, graph);
            break;
        case compile::parser::Operator::Sigmoid:
            op_result = low_sigmoid(node, graph);
            break;
        case compile::parser::Operator::Gemm:
            op_result = low_gemm(node, graph);
            break;
        case compile::parser::Operator::Conv:
            op_result = low_conv(node, graph);
            break;
        case compile::parser::Operator::Im2Col:
            op_result = low_im2col(node, graph);
            break;
        case compile::parser::Operator::ConvTranspose:
            op_result = low_conv_transpose(node, graph);
            break;
        case compile::parser::Operator::MaxPool2D:
            op_result = low_max_pool_2d(node, graph);
            break;
        case compile::parser::Operator::Reshape:
            op_result = low_reshape(node, graph);
            break;
        case compile::parser::Operator::Unknown:
            result.status = {.success = false, .error = "lowering: encountered unknown op"};
            return result;
        }

        if (!op_result.success) {
            result.status = op_result;
            return result;
        }

        const size_t source_op_index = result.source_ops.size();
        SourceOpRecord record;
        record.op_type = node.op;
        record.attrs = node.attributes;
        record.original_inputs = node.inputs;
        for (size_t i = physical_before; i < graph.nodes.size(); ++i) {
            graph.nodes[i].source_op = source_op_index;
            record.physical_nodes.push_back(i);
        }
        result.source_ops.push_back(std::move(record));
    }

    return result;
}

} // namespace gdinfer::compile::lowering
