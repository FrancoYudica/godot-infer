#pragma once
#include "output_handler.hpp"

namespace gdinfer::execute::io {

class OutputHandlerRegistry {
  public:
    bool init(godot::RenderingDevice* rd, ShaderCache* shader_cache);

    const std::unique_ptr<IOutputHandler>& get(
        const OutputType& desc) const;
    void destroy(godot::RenderingDevice* rd);

  private:
    template <typename T>
    bool _register(
        OutputType type,
        godot::RenderingDevice* rd,
        ShaderCache* shader_cache) {
        auto impl = std::make_unique<T>();
        if (impl->init(rd, shader_cache)) {
            _handlers[type] = std::move(impl);
            return true;
        }
        return false;
    }
    std::unordered_map<OutputType, std::unique_ptr<IOutputHandler>>
        _handlers;
};

} // namespace gdinfer::execute::io