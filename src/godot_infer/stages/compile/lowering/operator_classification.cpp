#include "operator_classification.hpp"

namespace gdinfer::compile::lowering {

OperatorClassification classify_operator(Physical::Operator op) {
    switch (op) {
    case Physical::Operator::ReLU:
    case Physical::Operator::Sigmoid:
        return OperatorClassification::Elementwise;
    case Physical::Operator::Im2Col:
    case Physical::Operator::Reshape:
    case Physical::Operator::MatrixTranspose:
        return OperatorClassification::InjectiveViewOnly;
    case Physical::Operator::Gemm:
    case Physical::Operator::Col2Im:
    case Physical::Operator::MaxPool2D:
        return OperatorClassification::ReductionSink;
    case Physical::Operator::Unknown:
        return OperatorClassification::OpaqueBarrier;
    }
    return OperatorClassification::OpaqueBarrier;
}

} // namespace gdinfer::compile::lowering
