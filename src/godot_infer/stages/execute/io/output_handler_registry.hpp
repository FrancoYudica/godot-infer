#pragma once
#include "output_handler.hpp"

namespace gdinfer::execute::io {

class OutputHandlerRegistry {
  public:
    bool init(godot::RenderingDevice* rd);

    const std::unique_ptr<IOutputHandler>& get(
        const OutputType& desc) const;
    void destroy(godot::RenderingDevice* rd);

  private:
    template <typename T>
    bool _register(
        OutputType type,
        godot::RenderingDevice* rd) {
        auto impl = std::make_unique<T>();
        if (impl->init(rd)) {
            _handlers[type] = std::move(impl);
            return true;
        }
        return false;
    }
    std::unordered_map<OutputType, std::unique_ptr<IOutputHandler>>
        _handlers;
};

} // namespace gdinfer::execute::io