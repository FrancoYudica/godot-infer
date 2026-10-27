add_executable(lowering_test operator_classification_test.cpp)

target_link_libraries(lowering_test PRIVATE
    gdinfer::stage_lowering
    gdinfer::test_common
    GTest::gtest_main
)

add_test(NAME lowering_test COMMAND lowering_test)
