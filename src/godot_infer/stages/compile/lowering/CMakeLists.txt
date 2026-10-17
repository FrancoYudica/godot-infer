# gdinfer_stage_lowering: compile::parser::Graph -> Physical::Graph lowering, plus
# post-lowering physical-graph validation. types.hpp (the Physical:: types this
# stage owns and produces), lowering.hpp, lowering_validation.hpp, and utils.hpp
# are the public entry points -- all four are consumed downstream by the execute
# stage and engine/, since Physical::Graph is lowering's own output artifact.

file(GLOB_RECURSE GD_INFER_STAGE_LOWERING_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

get_filename_component(GD_INFER_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

add_library(gdinfer_stage_lowering STATIC ${GD_INFER_STAGE_LOWERING_SOURCES})
add_library(gdinfer::stage_lowering ALIAS gdinfer_stage_lowering)

target_include_directories(gdinfer_stage_lowering PUBLIC "${GD_INFER_ROOT_DIR}")

# lowering.hpp's public signature consumes compile::parser::Graph (compile::parser's
# Node/Operator/Attrs types), so stage_parser is PUBLIC; types.hpp/utils.cpp use
# core's OperationResult/Tensor and Utils::get_iterator_str, so core is PUBLIC.
target_link_libraries(gdinfer_stage_lowering PUBLIC gdinfer::core gdinfer::stage_parser)
