#include "stages/compile/lowering/operator_classification.hpp"
#include <gtest/gtest.h>

using namespace gdinfer::compile::lowering;

TEST(OperatorClassificationTest, Elementwise) {
  EXPECT_EQ(classify_operator(Physical::Operator::ReLU),
            OperatorClassification::Elementwise);
  EXPECT_EQ(classify_operator(Physical::Operator::Sigmoid),
            OperatorClassification::Elementwise);
}

TEST(OperatorClassificationTest, InjectiveViewOnly) {
  EXPECT_EQ(classify_operator(Physical::Operator::Im2Col),
            OperatorClassification::InjectiveViewOnly);
  EXPECT_EQ(classify_operator(Physical::Operator::Reshape),
            OperatorClassification::InjectiveViewOnly);
  EXPECT_EQ(classify_operator(Physical::Operator::MatrixTranspose),
            OperatorClassification::InjectiveViewOnly);
}

TEST(OperatorClassificationTest, ReductionSink) {
  EXPECT_EQ(classify_operator(Physical::Operator::Gemm),
            OperatorClassification::ReductionSink);
  EXPECT_EQ(classify_operator(Physical::Operator::Col2Im),
            OperatorClassification::ReductionSink);
  EXPECT_EQ(classify_operator(Physical::Operator::MaxPool2D),
            OperatorClassification::ReductionSink);
}

TEST(OperatorClassificationTest, OpaqueBarrier) {
  EXPECT_EQ(classify_operator(Physical::Operator::Unknown),
            OperatorClassification::OpaqueBarrier);
}
