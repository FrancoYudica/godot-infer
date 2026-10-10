#include "shape_inference.hpp"

using namespace gdinfer::Physical;

namespace gdinfer::passes {

namespace {

// Helper: looks up a shape and sets an error on result if missing.
const std::vector<int64_t>* require(
    const std::string& name,
    const ShapeTable& shapes,
    ShapeInferenceResult& result) {

    auto it = shapes.find(name);
    if (it == shapes.end()) {
        result.status = {.success = false, .error = "shape_inference: tensor '" + name + "' not found"};
        return nullptr;
    }

    return &it->second;
}

// Shared forward-conv spatial formula, parameterized on raw fields rather than a specific
// attrs struct so it serves both ConvAttrs and MaxPool2DAttrs (same shape, different types).
struct ConvOutputSpatial {
    int64_t out_h;
    int64_t out_w;
};

ConvOutputSpatial conv_forward_output_spatial(
    int64_t h,
    int64_t w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t pad_top,
    int64_t pad_left,
    int64_t pad_bottom,
    int64_t pad_right,
    int64_t stride_y,
    int64_t stride_x,
    int64_t dilation_y,
    int64_t dilation_x) {

    int64_t kh_effective = (kernel_h - 1) * dilation_y + 1;
    int64_t kw_effective = (kernel_w - 1) * dilation_x + 1;
    int64_t pad_vertical = pad_top + pad_bottom;
    int64_t pad_horizontal = pad_left + pad_right;

    return {
        .out_h = (h + pad_vertical - kh_effective) / stride_y + 1,
        .out_w = (w + pad_horizontal - kw_effective) / stride_x + 1};
}

// Gemm: [M, K] x [N, K]^T -> [M, N]. Weights are always stored as [N, K] (transB=true).
std::vector<int64_t> gemm_output_shape(
    const std::vector<int64_t>& in,
    const std::vector<int64_t>& weight) {
    return {in[0], weight[0]};
}

// Standalone Im2Col: [b, c, h, w] -> [out_h*out_w, c*kH*kW].
bool configure_im2col_shapes(const SourceOpRecord& record, const Graph& graph, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;
    const auto& node = graph.nodes[record.physical_nodes[0]];

    const auto* in = require(node.inputs[0], shapes, result);
    if (in == nullptr) {
        return false;
    }

    const auto& attrs = std::get<compile::parser::ConvAttrs>(record.attrs);
    int64_t c = (*in)[1];
    int64_t h = (*in)[2];
    int64_t w = (*in)[3];
    auto spatial = conv_forward_output_spatial(
        h,
        w,
        attrs.kernel_shape[0],
        attrs.kernel_shape[1],
        attrs.pads[0],
        attrs.pads[1],
        attrs.pads[2],
        attrs.pads[3],
        attrs.strides[0],
        attrs.strides[1],
        attrs.dilations[0],
        attrs.dilations[1]);

    shapes[node.outputs[0]] = {spatial.out_h * spatial.out_w, c * attrs.kernel_shape[0] * attrs.kernel_shape[1]};
    result.dispatch_data[record.physical_nodes[0]] = Im2ColDispatchData{
        .in_channels = static_cast<uint32_t>(c),
        .in_height = static_cast<uint32_t>(h),
        .in_width = static_cast<uint32_t>(w),
        .out_height = static_cast<uint32_t>(spatial.out_h),
        .out_width = static_cast<uint32_t>(spatial.out_w),
        .kernel_h = static_cast<uint32_t>(attrs.kernel_shape[0]),
        .kernel_w = static_cast<uint32_t>(attrs.kernel_shape[1]),
        .padding_top = static_cast<uint32_t>(attrs.pads[0]),
        .padding_left = static_cast<uint32_t>(attrs.pads[1]),
        .stride_y = static_cast<uint32_t>(attrs.strides[0]),
        .stride_x = static_cast<uint32_t>(attrs.strides[1]),
        .dilation_y = static_cast<uint32_t>(attrs.dilations[0]),
        .dilation_x = static_cast<uint32_t>(attrs.dilations[1])};
    return true;
}

// Standalone Gemm.
bool configure_gemm_shapes(const SourceOpRecord& record, const Graph& graph, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;
    const auto& node = graph.nodes[record.physical_nodes[0]];

    const auto* in = require(node.inputs[0], shapes, result);
    const auto* w = require(node.inputs[1], shapes, result);
    if (in == nullptr || w == nullptr) {
        return false;
    }

    shapes[node.outputs[0]] = gemm_output_shape(*in, *w);
    const auto& gemm_attrs = std::get<GemmAttrs>(node.attributes);
    result.dispatch_data[record.physical_nodes[0]] = GemmDispatchData{
        .M = static_cast<uint32_t>((*in)[0]),
        .N = static_cast<uint32_t>((*w)[0]),
        .K = static_cast<uint32_t>((*in)[1]),
        .alpha = gemm_attrs.alpha,
        .beta = gemm_attrs.beta};
    return true;
}

// Conv2d lowers to [Im2Col, Gemm, MatrixTranspose] (see low_conv); computes all three shapes.
bool configure_conv_shapes(const SourceOpRecord& record, const Graph& graph, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;
    const auto& im2col_node = graph.nodes[record.physical_nodes[0]];
    const auto& gemm_node = graph.nodes[record.physical_nodes[1]];
    const auto& transpose_node = graph.nodes[record.physical_nodes[2]];

    const auto* in = require(im2col_node.inputs[0], shapes, result);
    if (in == nullptr) {
        return false;
    }

    const auto& attrs = std::get<compile::parser::ConvAttrs>(record.attrs);
    int64_t b = (*in)[0];
    int64_t c = (*in)[1];
    int64_t h = (*in)[2];
    int64_t w = (*in)[3];
    auto spatial = conv_forward_output_spatial(
        h,
        w,
        attrs.kernel_shape[0],
        attrs.kernel_shape[1],
        attrs.pads[0],
        attrs.pads[1],
        attrs.pads[2],
        attrs.pads[3],
        attrs.strides[0],
        attrs.strides[1],
        attrs.dilations[0],
        attrs.dilations[1]);

    // Im2Col's own 2D output.
    shapes[im2col_node.outputs[0]] = {spatial.out_h * spatial.out_w, c * attrs.kernel_shape[0] * attrs.kernel_shape[1]};

    // Gemm's own 2D output: [out_h*out_w, out_c].
    const auto* weight_shape = require(im2col_node.inputs[1], shapes, result);
    if (weight_shape == nullptr) {
        return false;
    }
    int64_t out_c = (*weight_shape)[0];
    shapes[gemm_node.outputs[0]] = {spatial.out_h * spatial.out_w, out_c};

    // The real Conv2d-formula 4D output
    shapes[transpose_node.outputs[0]] = {b, out_c, spatial.out_h, spatial.out_w};

    result.dispatch_data[record.physical_nodes[0]] = Im2ColDispatchData{
        .in_channels = static_cast<uint32_t>(c),
        .in_height = static_cast<uint32_t>(h),
        .in_width = static_cast<uint32_t>(w),
        .out_height = static_cast<uint32_t>(spatial.out_h),
        .out_width = static_cast<uint32_t>(spatial.out_w),
        .kernel_h = static_cast<uint32_t>(attrs.kernel_shape[0]),
        .kernel_w = static_cast<uint32_t>(attrs.kernel_shape[1]),
        .padding_top = static_cast<uint32_t>(attrs.pads[0]),
        .padding_left = static_cast<uint32_t>(attrs.pads[1]),
        .stride_y = static_cast<uint32_t>(attrs.strides[0]),
        .stride_x = static_cast<uint32_t>(attrs.strides[1]),
        .dilation_y = static_cast<uint32_t>(attrs.dilations[0]),
        .dilation_x = static_cast<uint32_t>(attrs.dilations[1])};

    const auto& conv_gemm_attrs = std::get<GemmAttrs>(gemm_node.attributes);
    result.dispatch_data[record.physical_nodes[1]] = GemmDispatchData{
        .M = static_cast<uint32_t>(spatial.out_h * spatial.out_w),
        .N = static_cast<uint32_t>(out_c),
        .K = static_cast<uint32_t>(c * attrs.kernel_shape[0] * attrs.kernel_shape[1]),
        .alpha = conv_gemm_attrs.alpha,
        .beta = conv_gemm_attrs.beta};

    // GEMM's [out_h*out_w, out_c] -> [out_c, out_h*out_w] (BCHW).
    result.dispatch_data[record.physical_nodes[2]] = MatrixTransposeDispatchData{
        .rows = static_cast<uint32_t>(spatial.out_h * spatial.out_w),
        .cols = static_cast<uint32_t>(out_c)};

    return true;
}

// ConvTranspose lowers to [MatrixTranspose, Gemm, Col2Im, Reshape] (see low_conv_transpose);
// the final Reshape relabels Col2Im's 2D output as the real 4D BCHW result.
bool configure_conv_transpose_shapes(const SourceOpRecord& record, const Graph& graph, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;
    const auto& pre_transpose_node = graph.nodes[record.physical_nodes[0]];
    const auto& gemm_node = graph.nodes[record.physical_nodes[1]];
    const auto& col2im_node = graph.nodes[record.physical_nodes[2]];
    const auto& post_reshape_node = graph.nodes[record.physical_nodes[3]];

    // pre_transpose: [b, ic, ih, iw] -> [b*ih*iw, ic].
    const auto* in = require(record.original_inputs[0], shapes, result);
    if (in == nullptr) {
        return false;
    }
    int64_t b = (*in)[0];
    int64_t ic = (*in)[1];
    int64_t ih = (*in)[2];
    int64_t iw = (*in)[3];
    shapes[pre_transpose_node.outputs[0]] = {b * ih * iw, ic};

    // BCHW [1, ic, ih, iw] (memory: [ic, ih*iw]) -> [ih*iw, ic].
    result.dispatch_data[record.physical_nodes[0]] = MatrixTransposeDispatchData{
        .rows = static_cast<uint32_t>(ic),
        .cols = static_cast<uint32_t>(ih * iw)};

    // gemm: [b*ih*iw, ic] x [oc*kh*kw, ic]^T -> [b*ih*iw, oc*kh*kw].
    const auto* gemm_weight = require(gemm_node.inputs[1], shapes, result);
    if (gemm_weight == nullptr) {
        return false;
    }
    shapes[gemm_node.outputs[0]] = {b * ih * iw, (*gemm_weight)[0]};
    const auto& ct_gemm_attrs = std::get<GemmAttrs>(gemm_node.attributes);
    result.dispatch_data[record.physical_nodes[1]] = GemmDispatchData{
        .M = static_cast<uint32_t>(b * ih * iw),
        .N = static_cast<uint32_t>((*gemm_weight)[0]),
        .K = static_cast<uint32_t>(ic),
        .alpha = ct_gemm_attrs.alpha,
        .beta = ct_gemm_attrs.beta};

    // col2im: [b*ih*iw, oc*kh*kw] -> [out_h*out_w, oc].
    const auto& attrs = std::get<compile::parser::ConvTransposeAttrs>(record.attrs);
    int64_t kh_effective = (attrs.kernel_shape[0] - 1) * attrs.dilations[0] + 1;
    int64_t kw_effective = (attrs.kernel_shape[1] - 1) * attrs.dilations[1] + 1;
    int64_t pad_vertical = attrs.pads[0] + attrs.pads[2];
    int64_t pad_horizontal = attrs.pads[1] + attrs.pads[3];

    int64_t out_h = attrs.strides[0] * (ih - 1) + attrs.output_padding[0] + kh_effective - pad_vertical;
    int64_t out_w = attrs.strides[1] * (iw - 1) + attrs.output_padding[1] + kw_effective - pad_horizontal;

    const auto* bias = require(col2im_node.inputs[1], shapes, result);
    if (bias == nullptr) {
        return false;
    }
    int64_t out_c = (*bias)[0];

    shapes[col2im_node.outputs[0]] = {out_h * out_w, out_c};
    result.dispatch_data[record.physical_nodes[2]] = Col2ImDispatchData{
        .in_height = static_cast<uint32_t>(ih),
        .in_width = static_cast<uint32_t>(iw),
        .out_channels = static_cast<uint32_t>(out_c),
        .out_height = static_cast<uint32_t>(out_h),
        .out_width = static_cast<uint32_t>(out_w),
        .kernel_h = static_cast<uint32_t>(attrs.kernel_shape[0]),
        .kernel_w = static_cast<uint32_t>(attrs.kernel_shape[1]),
        .padding_top = static_cast<uint32_t>(attrs.pads[0]),
        .padding_left = static_cast<uint32_t>(attrs.pads[1]),
        .stride_y = static_cast<uint32_t>(attrs.strides[0]),
        .stride_x = static_cast<uint32_t>(attrs.strides[1]),
        .dilation_y = static_cast<uint32_t>(attrs.dilations[0]),
        .dilation_x = static_cast<uint32_t>(attrs.dilations[1])};

    shapes[post_reshape_node.outputs[0]] = {b, out_c, out_h, out_w};
    // Open question: unlike Conv2d's post-GEMM bridge, this stays an alias, not a MatrixTranspose.
    result.dispatch_data[record.physical_nodes[3]] = ReshapeDispatchData{
        .target_shape = {b, out_c, out_h, out_w}};

    return true;
}

// Standalone ONNX Reshape: resolves -1/0 placeholders in the target shape from the shape
// initializer tensor against the input's actual shape.
bool configure_reshape_shapes(const SourceOpRecord& record, const Graph& graph, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;
    const auto& node = graph.nodes[record.physical_nodes[0]];
    const auto& attrs = std::get<ReshapeAttrs>(node.attributes);

    const auto* in = require(node.inputs[0], shapes, result);
    if (in == nullptr) {
        return false;
    }

    int64_t total = 1;
    for (int64_t d : *in) {
        total *= d;
    }

    std::vector<int64_t> out_shape = attrs.target_shape;
    int64_t infer_idx = -1;
    int64_t known_product = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(out_shape.size()); ++i) {
        if (out_shape[i] == -1) {
            infer_idx = i;
        } else if (out_shape[i] == 0) {
            out_shape[i] = (*in)[i];
            known_product *= out_shape[i];
        } else {
            known_product *= out_shape[i];
        }
    }
    if (infer_idx >= 0) {
        out_shape[infer_idx] = total / known_product;
    }

    result.dispatch_data[record.physical_nodes[0]] = ReshapeDispatchData{
        .target_shape = out_shape};
    shapes[node.outputs[0]] = std::move(out_shape);
    return true;
}

// ReLU / Sigmoid: output shape = input shape.
bool configure_elementwise_shapes(const SourceOpRecord& record, const Graph& graph, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;
    const auto& node = graph.nodes[record.physical_nodes[0]];
    const auto* in = require(node.inputs[0], shapes, result);
    if (in == nullptr) {
        return false;
    }
    shapes[node.outputs[0]] = *in;

    uint32_t total_elements = 1;
    for (int64_t d : *in) {
        total_elements *= static_cast<uint32_t>(d);
    }
    result.dispatch_data[record.physical_nodes[0]] = ElementwiseDispatchData{.total_elements = total_elements};
    return true;
}

// MaxPool2D: [b, c, h, w] -> [b, c, out_h, out_w].
bool configure_max_pool_shapes(const SourceOpRecord& record, const Graph& graph, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;
    const auto& node = graph.nodes[record.physical_nodes[0]];
    const auto* in = require(node.inputs[0], shapes, result);
    if (in == nullptr) {
        return false;
    }

    const auto& attrs = std::get<compile::parser::MaxPool2DAttrs>(record.attrs);
    int64_t b = (*in)[0];
    int64_t c = (*in)[1];
    int64_t h = (*in)[2];
    int64_t w = (*in)[3];
    auto spatial = conv_forward_output_spatial(
        h,
        w,
        attrs.kernel_shape[0],
        attrs.kernel_shape[1],
        attrs.pads[0],
        attrs.pads[1],
        attrs.pads[2],
        attrs.pads[3],
        attrs.strides[0],
        attrs.strides[1],
        attrs.dilations[0],
        attrs.dilations[1]);

    shapes[node.outputs[0]] = {b, c, spatial.out_h, spatial.out_w};
    result.dispatch_data[record.physical_nodes[0]] = MaxPool2DDispatchData{
        .batch_size = static_cast<uint32_t>(b),
        .channels = static_cast<uint32_t>(c),
        .in_height = static_cast<uint32_t>(h),
        .in_width = static_cast<uint32_t>(w),
        .out_height = static_cast<uint32_t>(spatial.out_h),
        .out_width = static_cast<uint32_t>(spatial.out_w),
        .kernel_h = static_cast<uint32_t>(attrs.kernel_shape[0]),
        .kernel_w = static_cast<uint32_t>(attrs.kernel_shape[1]),
        .padding_top = static_cast<uint32_t>(attrs.pads[0]),
        .padding_left = static_cast<uint32_t>(attrs.pads[1]),
        .stride_y = static_cast<uint32_t>(attrs.strides[0]),
        .stride_x = static_cast<uint32_t>(attrs.strides[1]),
        .dilation_y = static_cast<uint32_t>(attrs.dilations[0]),
        .dilation_x = static_cast<uint32_t>(attrs.dilations[1])};
    return true;
}
} // namespace
ShapeInferenceResult infer_shapes(
    const Graph& graph,
    const SourceOpTable& source_ops,
    const ShapeTable& input_shapes) {

    ShapeInferenceResult result;
    result.status = OPERATION_OK;
    result.dispatch_data.resize(graph.nodes.size());
    auto& shapes = result.shapes;

    // Seed with caller-provided input shapes
    for (const auto& [name, shape] : input_shapes) {
        shapes[name] = shape;
    }

    // Seed with initializer shapes (weights, biases)
    for (const auto& [name, tensor] : graph.initializers) {
        shapes[name] = tensor.shape;
    }

    // Forward pass over logical ops, already in topological order (lower() built source_ops
    // from the logical graph's own topological order).
    for (const auto& record : source_ops) {
        bool ok = true;
        switch (record.op_type) {
        case compile::parser::Operator::Im2Col:
            ok = configure_im2col_shapes(record, graph, result);
            break;
        case compile::parser::Operator::Gemm:
            ok = configure_gemm_shapes(record, graph, result);
            break;
        case compile::parser::Operator::Conv:
            ok = configure_conv_shapes(record, graph, result);
            break;
        case compile::parser::Operator::ConvTranspose:
            ok = configure_conv_transpose_shapes(record, graph, result);
            break;
        case compile::parser::Operator::Reshape:
            ok = configure_reshape_shapes(record, graph, result);
            break;
        case compile::parser::Operator::ReLU:
        case compile::parser::Operator::Sigmoid:
            ok = configure_elementwise_shapes(record, graph, result);
            break;
        case compile::parser::Operator::MaxPool2D:
            ok = configure_max_pool_shapes(record, graph, result);
            break;
        case compile::parser::Operator::Unknown:
            result.status = {.success = false, .error = "shape_inference: unhandled op"};
            return result;
        }

        if (!ok) {
            return result;
        }
    }

    return result;
}

} // namespace gdinfer::passes
