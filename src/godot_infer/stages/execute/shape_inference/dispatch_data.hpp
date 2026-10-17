#pragma once
#include "core/core.hpp" // IWYU pragma: export

namespace gdinfer::execute::shape_inference {

// Per-operator-type payload holding exactly what dispatch() needs for push constants and
// buffer binding, precomputed once during shape inference.

struct Im2ColDispatchData {
    uint32_t in_channels;
    uint32_t in_height;
    uint32_t in_width;
    uint32_t out_height;
    uint32_t out_width;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t padding_top;
    uint32_t padding_left;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
};

struct GemmDispatchData {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    float alpha;
    float beta;
};

struct MaxPool2DDispatchData {
    uint32_t batch_size;
    uint32_t channels;
    uint32_t in_height;
    uint32_t in_width;
    uint32_t out_height;
    uint32_t out_width;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t padding_top;
    uint32_t padding_left;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
};

struct ElementwiseDispatchData {
    uint32_t total_elements;
};

struct Col2ImDispatchData {
    uint32_t in_height;
    uint32_t in_width;
    uint32_t out_channels;
    uint32_t out_height;
    uint32_t out_width;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t padding_top;
    uint32_t padding_left;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
};

// Resolved target shape to alias the input buffer under.
struct ReshapeDispatchData {
    std::vector<int64_t> target_shape;
};

// rows/cols are the input matrix's dimensions; output is cols x rows.
struct MatrixTransposeDispatchData {
    uint32_t rows;
    uint32_t cols;
};

using DispatchData = std::variant<
    std::monostate,
    Im2ColDispatchData,
    GemmDispatchData,
    MaxPool2DDispatchData,
    ElementwiseDispatchData,
    Col2ImDispatchData,
    ReshapeDispatchData,
    MatrixTransposeDispatchData>;

// Parallel to Physical::Graph::nodes: dispatch_data[i] describes graph.nodes[i].
using DispatchDataTable = std::vector<DispatchData>;

} // namespace gdinfer::execute::shape_inference
