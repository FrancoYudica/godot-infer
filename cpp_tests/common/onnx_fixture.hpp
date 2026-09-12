#pragma once
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace gdinfer::test {

// Reads a file into a byte buffer
inline std::vector<uint8_t> read_file_bytes(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
}

inline std::vector<uint8_t> load_onnx_fixture(const std::string &fixtures_dir,
                                              const std::string &name) {
  return read_file_bytes(fixtures_dir + "/" + name + ".onnx");
}

} // namespace gdinfer::test
