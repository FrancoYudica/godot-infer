#include "stages/compile/parser/parser.hpp"
#include "test_utils.hpp"
#include <gtest/gtest.h>

using namespace gdinfer::compile::parser;

TEST(ParserTest, Gemm) {
  auto bytes = load_onnx_fixture("GemmTest1");
  ParseResult result = parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;

  const Graph &graph = result.graph;
  ASSERT_EQ(graph.nodes.size(), 1u);
  EXPECT_EQ(graph.initializers.size(), 2u); // W, B

  const Node &node = graph.nodes[0];
  ASSERT_EQ(node.op, Operator::Gemm);

  const auto &attrs = std::get<GemmAttrs>(node.attributes);
  EXPECT_FLOAT_EQ(attrs.alpha, 1.0f);
  EXPECT_FLOAT_EQ(attrs.beta, 1.0f);
  EXPECT_EQ(attrs.transB, true);
}

TEST(ParserTest, Relu) {
  auto bytes = load_onnx_fixture("ReluTest1");
  ParseResult result = parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;
  ASSERT_EQ(result.graph.nodes.size(), 1u);
  EXPECT_EQ(result.graph.nodes[0].op, Operator::ReLU);
}

TEST(ParserTest, Sigmoid) {
  auto bytes = load_onnx_fixture("SigmoidTest1");
  ParseResult result = parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;
  ASSERT_EQ(result.graph.nodes.size(), 1u);
  EXPECT_EQ(result.graph.nodes[0].op, Operator::Sigmoid);
}

TEST(ParserTest, Conv) {
  auto bytes = load_onnx_fixture("ConvTest1");
  ParseResult result = parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;

  const Graph &graph = result.graph;
  ASSERT_EQ(graph.nodes.size(), 1u);
  EXPECT_EQ(graph.initializers.size(), 2u); // W, B

  const Node &node = graph.nodes[0];
  ASSERT_EQ(node.op, Operator::Conv);

  const auto &attrs = std::get<ConvAttrs>(node.attributes);
  EXPECT_EQ(attrs.kernel_shape, (std::vector<int64_t>{3, 3}));
  EXPECT_EQ(attrs.strides, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(attrs.pads, (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_EQ(attrs.dilations, (std::vector<int64_t>{1, 1}));
}

TEST(ParserTest, ConvTranspose) {
  auto bytes = load_onnx_fixture("ConvTransposeTest1");
  ParseResult result = parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;

  const Graph &graph = result.graph;
  ASSERT_EQ(graph.nodes.size(), 1u);
  EXPECT_EQ(graph.initializers.size(), 2u); // W, B

  const Node &node = graph.nodes[0];
  ASSERT_EQ(node.op, Operator::ConvTranspose);

  const auto &attrs = std::get<ConvTransposeAttrs>(node.attributes);
  EXPECT_EQ(attrs.kernel_shape, (std::vector<int64_t>{3, 3}));
  EXPECT_EQ(attrs.strides, (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(attrs.pads, (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_EQ(attrs.dilations, (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(attrs.output_padding, (std::vector<int64_t>{0, 0}));
}

TEST(ParserTest, MaxPool) {
  auto bytes = load_onnx_fixture("MaxPoolTest1");
  ParseResult result = parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;

  const Graph &graph = result.graph;
  ASSERT_EQ(graph.nodes.size(), 1u);
  EXPECT_EQ(graph.initializers.size(), 0u);

  const Node &node = graph.nodes[0];
  ASSERT_EQ(node.op, Operator::MaxPool2D);

  const auto &attrs = std::get<MaxPool2DAttrs>(node.attributes);
  EXPECT_EQ(attrs.kernel_shape, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(attrs.strides, (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(attrs.pads, (std::vector<int64_t>{0, 0, 0, 0}));
  EXPECT_EQ(attrs.dilations, (std::vector<int64_t>{1, 1}));
}

TEST(ParserTest, Im2Col) {
  auto bytes = load_onnx_fixture("Im2ColTest1");
  ParseResult result = parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;

  const Graph &graph = result.graph;
  ASSERT_EQ(graph.nodes.size(), 1u);
  EXPECT_EQ(graph.initializers.size(), 0u);

  const Node &node = graph.nodes[0];
  ASSERT_EQ(node.op, Operator::Im2Col);

  const auto &attrs = std::get<ConvAttrs>(node.attributes);
  EXPECT_EQ(attrs.kernel_shape, (std::vector<int64_t>{3, 3}));
  EXPECT_EQ(attrs.strides, (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(attrs.pads, (std::vector<int64_t>{0, 0, 0, 0}));
  EXPECT_EQ(attrs.dilations, (std::vector<int64_t>{1, 1}));
}

TEST(ParserTest, Reshape) {
  auto bytes = load_onnx_fixture("ReshapeTest1");
  ParseResult result = parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;

  const Graph &graph = result.graph;
  ASSERT_EQ(graph.nodes.size(), 1u);
  EXPECT_EQ(graph.initializers.size(), 1u); // the shape tensor

  const Node &node = graph.nodes[0];
  EXPECT_EQ(node.op, Operator::Reshape);
  ASSERT_EQ(node.inputs.size(), 2u); // data, shape
  EXPECT_EQ(node.inputs[1], "shape");
}
