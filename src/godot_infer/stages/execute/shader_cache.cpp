#include "shader_cache.hpp"

#include "stages/execute/utils.hpp"

namespace gdinfer::execute {

const ShaderCache::Entry* ShaderCache::get_or_load(
    godot::RenderingDevice* rd,
    const godot::String& relative_path) {

    std::string key = relative_path.utf8().get_data();

    auto it = _entries.find(key);
    if (it != _entries.end()) {
        return &it->second;
    }

    const godot::String& shader_path = utils::get_project_relative_path(relative_path);
    godot::RID shader = utils::load_shader(rd, shader_path);
    ERR_FAIL_COND_V_MSG(!shader.is_valid(), nullptr, "ShaderCache: failed to load shader: " + relative_path);

    godot::RID pipeline = rd->compute_pipeline_create(shader);
    ERR_FAIL_COND_V_MSG(!pipeline.is_valid(), nullptr, "ShaderCache: failed to create pipeline for shader: " + relative_path);

    auto [inserted_it, _] = _entries.insert({key, Entry{.shader = shader, .pipeline = pipeline}});
    return &inserted_it->second;
}

void ShaderCache::destroy(godot::RenderingDevice* rd) {
    for (auto& [_, entry] : _entries) {
        if (entry.pipeline.is_valid()) {
            rd->free_rid(entry.pipeline);
        }
        if (entry.shader.is_valid()) {
            rd->free_rid(entry.shader);
        }
    }
    _entries.clear();
}

} // namespace gdinfer::execute
