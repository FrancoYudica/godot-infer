add_library(gdinfer_test_common INTERFACE)
add_library(gdinfer::test_common ALIAS gdinfer_test_common)


target_include_directories(gdinfer_test_common INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/..")
