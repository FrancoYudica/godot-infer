add_executable(parser_test parser_test.cpp)

target_link_libraries(parser_test PRIVATE
    gdinfer::stage_parser
    GTest::gtest_main
)

add_test(NAME parser_test COMMAND parser_test)
