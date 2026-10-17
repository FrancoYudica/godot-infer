#pragma once
#include "stages/execute/io/output_handler.hpp"

namespace gdinfer::execute::io {

class FloatArrayOutputHandler : public IOutputHandler {
  public:
    godot::Variant download(
        const std::unique_ptr<OutputDesc::BaseData>& desc,
        godot::RenderingDevice* rd,
        Ref<execute::tensors::TensorResourceManager> activations_tm) override;
};

} // namespace gdinfer::execute::io