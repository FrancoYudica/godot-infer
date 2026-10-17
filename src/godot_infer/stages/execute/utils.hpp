#pragma once
#include "core/core.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

namespace gdinfer::execute::utils {

/**
 * Gets the project-relative path for a given addon-relative path
 */
godot::String get_project_relative_path(const godot::String& path);

godot::RID load_shader(godot::RenderingDevice* rd, const godot::String& path);

} // namespace gdinfer::execute::utils
