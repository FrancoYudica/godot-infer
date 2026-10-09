# gdinfer_stage_lowering: compile::parser::Graph -> Physical::Graph lowering, plus
# post-lowering physical-graph validation. lowering.hpp and lowering_validation.hpp
# are the public entry points.

file(GLOB_RECURSE GD_INFER_STAGE_LOWERING_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

get_filename_component(GD_INFER_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

add_library(gdinfer_stage_lowering STATIC ${GD_INFER_STAGE_LOWERING_SOURCES})
add_library(gdinfer::stage_lowering ALIAS gdinfer_stage_lowering)

target_include_directories(gdinfer_stage_lowering PUBLIC "${GD_INFER_ROOT_DIR}")

# lowering.hpp's public signature consumes compile::parser::Graph and returns
# Physical::Graph (core/physical_types.hpp), so both are PUBLIC.
target_link_libraries(gdinfer_stage_lowering PUBLIC gdinfer::core gdinfer::stage_parser)
