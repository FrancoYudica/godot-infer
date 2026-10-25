#include "output_handler_registry.hpp"

#include "output_handlers/float_array_output_handler.hpp"
#include "output_handlers/texture_output_handler_compute.hpp"

namespace gdinfer::execute::io {

bool OutputHandlerRegistry::init(godot::RenderingDevice* rd, ShaderCache* shader_cache) {
    // Register all supported operators
    ERR_FAIL_COND_V_MSG(
        !_register<FloatArrayOutputHandler>(OutputType::FloatArray, rd, shader_cache),
        false,
        "OutputHandlerRegistry: failed to register FloatArray operator.");

    ERR_FAIL_COND_V_MSG(
        !_register<TextureOutputHandlerCompute>(OutputType::Texture2D, rd, shader_cache),
        false,
        "OutputHandlerRegistry: failed to register Texture operator.");

    return true;
}

const std::unique_ptr<IOutputHandler>& OutputHandlerRegistry::get(
    const OutputType& desc) const {
    return _handlers.at(desc);
}

void OutputHandlerRegistry::destroy(godot::RenderingDevice* rd) {
    for (auto& pair : _handlers) {
        pair.second->destroy(rd);
    }
}

} // namespace gdinfer::execute::io