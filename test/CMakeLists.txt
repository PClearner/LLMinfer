aux_source_directory(${CMAKE_SOURCE_DIR}/test DIR_TEST)
aux_source_directory(${CMAKE_SOURCE_DIR}/test/test_tensor DIR_TEST_TENSOR)

add_executable(test_main ${DIR_TEST} ${DIR_TEST_TENSOR} ${DIR_BASE})
include_directories(test)

target_link_libraries(test_main  glog::glog ${GTEST_LIBRARIES} pthread cudart armadillo)

set_target_properties(test_main PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/test/bin
    )

set_target_properties(test_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


target_include_directories(test_main PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(test_main PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_main PUBLIC ${PROJECT_SOURCE_DIR}/llminfer/include)
target_include_directories(test_main PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(test_main PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})