#include "utils.hpp"

#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

namespace gdinfer::Utils {

bool tensor_shape_matches(
    const std::vector<int64_t>& shape1,
    const std::vector<int64_t>& shape2) {
    if (shape1.size() != shape2.size()) {
        return false;
    }
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] != shape2[i]) {
            return false;
        }
    }
    return true;
}

uint32_t get_tensor_floats(const std::vector<int64_t>& shape) {
    uint32_t floats = 1;
    for (const auto& dim : shape) {
        floats *= static_cast<uint32_t>(dim);
    }
    return floats;
}

String shape_to_str(const std::vector<int64_t>& shape) {
    String str = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        str += String::num_int64(shape[i]);
        if (i < shape.size() - 1) {
            str += ",";
        }
    }
    str += "]";
    return str;
}
} // namespace gdinfer::Utils