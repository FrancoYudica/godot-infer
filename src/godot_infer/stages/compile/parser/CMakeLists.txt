# gdinfer_stage_parser: parses ONNX protobuf bytes into Logical::Graph.
# parser.hpp is the only public entry point; node_parser.hpp/.cpp are
# private, used only by parser.cpp. Only module that touches onnx:: types,
# so the generated protobuf sources, onnx include dirs, and protobuf link
# all stay private to this target.

file(GLOB_RECURSE GD_INFER_STAGE_PARSER_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

get_filename_component(GD_INFER_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
get_filename_component(GD_INFER_ONNX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../thirdparty/onnx/onnx" ABSOLUTE)

onnx_generate_protobuf_sources(GD_INFER_STAGE_PARSER_ONNX_SOURCES "${GD_INFER_ONNX_DIR}")

add_library(gdinfer_stage_parser STATIC
    ${GD_INFER_STAGE_PARSER_SOURCES}
    ${GD_INFER_STAGE_PARSER_ONNX_SOURCES}
)
add_library(gdinfer::stage_parser ALIAS gdinfer_stage_parser)

target_compile_definitions(gdinfer_stage_parser PRIVATE ONNX_ML=0)

# onnx/onnx_pb.h and the generated headers never appear in parser.hpp's own
# signature, so PRIVATE.
target_include_directories(gdinfer_stage_parser PRIVATE
    "${GD_INFER_ONNX_DIR}/.."
    "${GD_INFER_ONNX_DIR}"
)

target_include_directories(gdinfer_stage_parser PUBLIC "${GD_INFER_ROOT_DIR}")
target_link_libraries(gdinfer_stage_parser PUBLIC gdinfer::core)
target_link_libraries(gdinfer_stage_parser PRIVATE godot::cpp)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # vcpkg's static protobuf/abseil libs have circular deps; LINK_GROUP:RESCAN
    # re-scans the group until everything resolves, regardless of link order.
    target_link_libraries(gdinfer_stage_parser PRIVATE "$<LINK_GROUP:RESCAN,protobuf::libprotobuf>")
else()
    target_link_libraries(gdinfer_stage_parser PRIVATE protobuf::libprotobuf)
endif()
