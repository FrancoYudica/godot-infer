#pragma once
#include "stages/execute/io/input_handler.hpp"
#include "stages/execute/io/io_descriptor.hpp"

namespace gdinfer::execute::io {
class FloatArrayInputHandler : public IInputHandler {
  public:
    bool upload(
        const std::unique_ptr<InputDesc::BaseData>& desc,
        const InputHandlerContext& ctx) override;

    std::vector<int64_t> get_shape(const std::unique_ptr<InputDesc::BaseData>& desc) const override;
};

} // namespace gdinfer::execute::io