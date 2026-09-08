#pragma once
#include "input_handler.hpp"

namespace gdinfer {

class InputHandlerRegistry {
  public:
    bool init(godot::RenderingDevice* rd);

    const std::unique_ptr<gdinfer::IInputHandler>& get(
        const gdinfer::InputType& desc) const;
    void destroy(godot::RenderingDevice* rd);

  private:
    template <typename T>
    bool _register(gdinfer::InputType type, godot::RenderingDevice* rd) {
        auto impl = std::make_unique<T>();
        if (impl->init(rd)) {
            _handlers[type] = std::move(impl);
            return true;
        }
        return false;
    }
    std::unordered_map<gdinfer::InputType, std::unique_ptr<gdinfer::IInputHandler>>
        _handlers;
};

} // namespace gdinfer