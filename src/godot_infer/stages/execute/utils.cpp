#include "utils.hpp"

#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/variant/string.hpp>

using namespace godot;

namespace gdinfer::execute::utils {

godot::String get_project_relative_path(
    const godot::String& addon_relative_path) {
    static const godot::String base_path = "res://addons/godot_infer/";
    return base_path + addon_relative_path;
}

RID load_shader(RenderingDevice* rd, const godot::String& path) {
    Ref<RDShaderFile> shader_file =
        ResourceLoader::get_singleton()->load(path);

    ERR_FAIL_COND_V_MSG(
        shader_file.is_null(),
        RID(),
        "Failed to load shader file: " + path);

    Ref<RDShaderSPIRV> spirv = shader_file->get_spirv();

    ERR_FAIL_COND_V_MSG(
        spirv.is_null(),
        RID(),
        "Failed to load SPIR-V from shader file: " + path);

    return rd->shader_create_from_spirv(spirv);
}

} // namespace gdinfer::execute::utils
