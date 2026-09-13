#include "utils.hpp"

namespace gdinfer::compile::parser::utils {

std::string op_name(Operator op) {
    switch (op) {
    case Operator::Gemm:
        return "Gemm";
    case Operator::ReLU:
        return "ReLU";
    case Operator::Sigmoid:
        return "Sigmoid";
    case Operator::Conv:
        return "Conv";
    case Operator::Im2Col:
        return "Im2Col";
    case Operator::ConvTranspose:
        return "ConvTranspose";
    case Operator::MaxPool2D:
        return "MaxPool2D";
    default:
        return "Unknown";
    }
}

} // namespace gdinfer::compile::parser::utils