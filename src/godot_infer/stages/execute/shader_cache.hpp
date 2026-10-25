#pragma once
#include "core/core.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

namespace gdinfer::execute {

// Loads and owns compute shaders + pipelines, keyed by path, so each shader loads once.
class ShaderCache {
  public:
    struct Entry {
        godot::RID shader;
        godot::RID pipeline;
    };

    // Loads (once) and caches shader+pipeline for relative_path. Nullptr on failure.
    const Entry* get_or_load(godot::RenderingDevice* rd, const godot::String& relative_path);

    // Frees every cached shader and pipeline.
    void destroy(godot::RenderingDevice* rd);

  private:
    std::unordered_map<std::string, Entry> _entries;
};

} // namespace gdinfer::execute
