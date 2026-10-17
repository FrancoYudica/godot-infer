#pragma once
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/string.hpp>
#include <memory> // IWYU pragma: export
#include <string>
#include <unordered_map>
#include <variant> // IWYU pragma: export
#include <vector>

namespace gdinfer {

struct OperationResult {
    bool success = true;
    std::string error;
};

#define OPERATION_OK                 \
    {                                \
        .success = true, .error = {} \
    }

/**
 * Tensor holds its shape and weight data
 */
struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;
};
} // namespace gdinfer
