# gdinfer_core: OperationResult, Tensor, ShapeTable, Logical::/Physical:: types.
# Lowest-level target, no godot_infer dependencies of its own.

file(GLOB_RECURSE GD_INFER_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(gdinfer_core STATIC ${GD_INFER_CORE_SOURCES})
add_library(gdinfer::core ALIAS gdinfer_core)

# godot_infer/ itself (not core/) is the include root, matching the
# project-wide "core/core.hpp"-style include convention.
get_filename_component(GD_INFER_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
target_include_directories(gdinfer_core PUBLIC "${GD_INFER_ROOT_DIR}")

# core.hpp re-exports godot_cpp headers, and utils.cpp calls godot_cpp APIs
# directly, so this is PUBLIC.
target_link_libraries(gdinfer_core PUBLIC godot::cpp)
