#pragma once
#include "core/core_types.hpp"

namespace gdinfer::compile::lowering::Physical {

enum class Operator {
    Gemm,
    ReLU,
    Sigmoid,
    Conv,
    Im2Col,
    Col2Im,
    MaxPool2D,
    Reshape,
    MatrixTranspose,
    Unknown
};

struct GemmAttrs {
    float alpha = 1.0F;
    float beta = 1.0F;
};

struct ConvAttrs {
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t padding_top;
    uint32_t padding_left;
    uint32_t padding_bottom;
    uint32_t padding_right;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
};

struct Col2ImAttrs {
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t padding_top;
    uint32_t padding_left;
    uint32_t padding_bottom;
    uint32_t padding_right;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
    uint32_t output_padding_y;
    uint32_t output_padding_x;
};

struct MaxPool2DAttrs {
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t padding_top;
    uint32_t padding_left;
    uint32_t padding_bottom;
    uint32_t padding_right;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
};

// Reshape is always a zero-copy alias (no data movement); actual data-moving transposes are
// Physical::Operator::MatrixTranspose nodes instead. May be empty if resolved later by shape inference.
struct ReshapeAttrs {
    std::vector<int64_t> target_shape;
};

struct Node {
    Operator op;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    // Index into the LoweringResult::source_ops table (stages/compile/lowering/lowering.hpp)
    // this node was produced from. Meaningful only for graphs produced by lower().
    size_t source_op = 0;

    std::variant<
        std::monostate,
        GemmAttrs,
        ConvAttrs,
        Col2ImAttrs,
        ReshapeAttrs,
        MaxPool2DAttrs>
        attributes;
};

struct Graph {
    std::vector<std::string> input_names;
    std::unordered_map<std::string, std::vector<int64_t>> input_shapes;
    std::vector<Node> nodes;
    std::unordered_map<std::string, Tensor> initializers;
};
} // namespace gdinfer::compile::lowering::Physical
