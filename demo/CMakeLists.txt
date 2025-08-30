add_executable(llama2_infer main.cpp)
target_link_directories(llama2_infer PUBLIC ${CMAKE_SOURCE_DIR}/lib)
target_link_libraries(llama2_infer llmframe)
set_target_properties(llama2_infer PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION OFF  # 关闭分离编译，减少链接复杂度
)

set_target_properties(llama2_infer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/llama2_infer
    )

set_target_properties(llama2_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)