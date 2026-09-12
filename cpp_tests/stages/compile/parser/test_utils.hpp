#pragma once
#include "common/onnx_fixture.hpp"
#include <string>
#include <vector>

inline std::vector<uint8_t> load_onnx_fixture(const std::string& name) {
    return gdinfer::test::load_onnx_fixture(PARSER_TEST_FIXTURES_DIR, name);
}
