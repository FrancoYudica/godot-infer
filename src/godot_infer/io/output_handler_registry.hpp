#pragma once
#include "output_handler.hpp"

namespace gdinfer {

class OutputHandlerRegistry {
  public:
    bool init(godot::RenderingDevice* rd);

    const std::unique_ptr<gdinfer::IOutputHandler>& get(
        const gdinfer::OutputType& desc) const;
    void destroy(godot::RenderingDevice* rd);

  private:
    template <typename T>
    bool _register(
        gdinfer::OutputType type,
        godot::RenderingDevice* rd) {
        auto impl = std::make_unique<T>();
        if (impl->init(rd)) {
            _handlers[type] = std::move(impl);
            return true;
        }
        return false;
    }
    std::unordered_map<gdinfer::OutputType, std::unique_ptr<gdinfer::IOutputHandler>>
        _handlers;
};

} // namespace gdinfer