add_executable(parser_test parser_test.cpp)

target_compile_definitions(parser_test PRIVATE
    PARSER_TEST_FIXTURES_DIR="${GD_INFER_PROJECT_ROOT}/tests/fixtures/parser")

target_link_libraries(parser_test PRIVATE
    gdinfer::stage_parser
    gdinfer::test_common
    GTest::gtest_main
)

add_test(NAME parser_test COMMAND parser_test)
