#include "core/logical_types.hpp"
#include "stages/compile/parser/parser.hpp"
#include "test_utils.hpp"
#include <gtest/gtest.h>

using namespace gdinfer;

TEST(ParserTest, Gemm) {
  auto bytes = load_onnx_fixture("GemmTest1");
  passes::ParseResult result = passes::parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;

  const Logical::Graph &graph = result.graph;
  ASSERT_EQ(graph.nodes.size(), 1u);
  EXPECT_EQ(graph.initializers.size(), 2u); // W, B

  const Logical::Node &node = graph.nodes[0];
  ASSERT_EQ(node.op, Logical::Operator::Gemm);

  const auto &attrs = std::get<Logical::GemmAttrs>(node.attributes);
  EXPECT_FLOAT_EQ(attrs.alpha, 1.0f);
  EXPECT_FLOAT_EQ(attrs.beta, 1.0f);
  EXPECT_EQ(attrs.transB, true);
}

TEST(ParserTest, Relu) {
  auto bytes = load_onnx_fixture("ReluTest1");
  passes::ParseResult result = passes::parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;
  ASSERT_EQ(result.graph.nodes.size(), 1u);
  EXPECT_EQ(result.graph.nodes[0].op, Logical::Operator::ReLU);
}

TEST(ParserTest, Sigmoid) {
  auto bytes = load_onnx_fixture("SigmoidTest1");
  passes::ParseResult result = passes::parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;
  ASSERT_EQ(result.graph.nodes.size(), 1u);
  EXPECT_EQ(result.graph.nodes[0].op, Logical::Operator::Sigmoid);
}

TEST(ParserTest, Conv) {
  auto bytes = load_onnx_fixture("ConvTest1");
  passes::ParseResult result = passes::parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;

  const Logical::Graph &graph = result.graph;
  ASSERT_EQ(graph.nodes.size(), 1u);
  EXPECT_EQ(graph.initializers.size(), 2u); // W, B

  const Logical::Node &node = graph.nodes[0];
  ASSERT_EQ(node.op, Logical::Operator::Conv);

  const auto &attrs = std::get<Logical::ConvAttrs>(node.attributes);
  EXPECT_EQ(attrs.kernel_shape, (std::vector<int64_t>{3, 3}));
  EXPECT_EQ(attrs.strides, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(attrs.pads, (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_EQ(attrs.dilations, (std::vector<int64_t>{1, 1}));
}

TEST(ParserTest, ConvTranspose) {
  auto bytes = load_onnx_fixture("ConvTransposeTest1");
  passes::ParseResult result = passes::parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;

  const Logical::Graph &graph = result.graph;
  ASSERT_EQ(graph.nodes.size(), 1u);
  EXPECT_EQ(graph.initializers.size(), 2u); // W, B

  const Logical::Node &node = graph.nodes[0];
  ASSERT_EQ(node.op, Logical::Operator::ConvTranspose);

  const auto &attrs = std::get<Logical::ConvTransposeAttrs>(node.attributes);
  EXPECT_EQ(attrs.kernel_shape, (std::vector<int64_t>{3, 3}));
  EXPECT_EQ(attrs.strides, (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(attrs.pads, (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_EQ(attrs.dilations, (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(attrs.output_padding, (std::vector<int64_t>{0, 0}));
}

TEST(ParserTest, MaxPool) {
  auto bytes = load_onnx_fixture("MaxPoolTest1");
  passes::ParseResult result = passes::parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;

  const Logical::Graph &graph = result.graph;
  ASSERT_EQ(graph.nodes.size(), 1u);
  EXPECT_EQ(graph.initializers.size(), 0u);

  const Logical::Node &node = graph.nodes[0];
  ASSERT_EQ(node.op, Logical::Operator::MaxPool2D);

  const auto &attrs = std::get<Logical::MaxPool2DAttrs>(node.attributes);
  EXPECT_EQ(attrs.kernel_shape, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(attrs.strides, (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(attrs.pads, (std::vector<int64_t>{0, 0, 0, 0}));
  EXPECT_EQ(attrs.dilations, (std::vector<int64_t>{1, 1}));
}

TEST(ParserTest, Im2Col) {
  auto bytes = load_onnx_fixture("Im2ColTest1");
  passes::ParseResult result = passes::parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;

  const Logical::Graph &graph = result.graph;
  ASSERT_EQ(graph.nodes.size(), 1u);
  EXPECT_EQ(graph.initializers.size(), 0u);

  const Logical::Node &node = graph.nodes[0];
  ASSERT_EQ(node.op, Logical::Operator::Im2Col);

  const auto &attrs = std::get<Logical::ConvAttrs>(node.attributes);
  EXPECT_EQ(attrs.kernel_shape, (std::vector<int64_t>{3, 3}));
  EXPECT_EQ(attrs.strides, (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(attrs.pads, (std::vector<int64_t>{0, 0, 0, 0}));
  EXPECT_EQ(attrs.dilations, (std::vector<int64_t>{1, 1}));
}

TEST(ParserTest, Reshape) {
  auto bytes = load_onnx_fixture("ReshapeTest1");
  passes::ParseResult result = passes::parse(bytes.data(), bytes.size());
  ASSERT_TRUE(result.status.success) << result.status.error;

  const Logical::Graph &graph = result.graph;
  ASSERT_EQ(graph.nodes.size(), 1u);
  EXPECT_EQ(graph.initializers.size(), 1u); // the shape tensor

  const Logical::Node &node = graph.nodes[0];
  EXPECT_EQ(node.op, Logical::Operator::Reshape);
  ASSERT_EQ(node.inputs.size(), 2u); // data, shape
  EXPECT_EQ(node.inputs[1], "shape");
}
