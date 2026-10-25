#pragma once
#include "input_handler.hpp"

namespace gdinfer::execute::io {

class InputHandlerRegistry {
  public:
    bool init(godot::RenderingDevice* rd, ShaderCache* shader_cache);

    const std::unique_ptr<IInputHandler>& get(
        const InputType& desc) const;
    void destroy(godot::RenderingDevice* rd);

  private:
    template <typename T>
    bool _register(InputType type, godot::RenderingDevice* rd, ShaderCache* shader_cache) {
        auto impl = std::make_unique<T>();
        if (impl->init(rd, shader_cache)) {
            _handlers[type] = std::move(impl);
            return true;
        }
        return false;
    }
    std::unordered_map<InputType, std::unique_ptr<IInputHandler>>
        _handlers;
};

} // namespace gdinfer::execute::io