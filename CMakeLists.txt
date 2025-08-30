cmake_minimum_required(VERSION 3.16)
project(llminfer VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the CUDA compiler path or name
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.9/bin/nvcc")

project(llminfer CXX CUDA)
include(cmake/cuda.cmake)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)



# cmake code

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)
# enable_testing()

aux_source_directory(llminfer/source/tensor/ DIR_TENSOR)
aux_source_directory(llminfer/source/base/ DIR_BASE)
aux_source_directory(llminfer/source/op/ DIR_OP)
aux_source_directory(llminfer/source/model/ DIR_MODEL)
aux_source_directory(llminfer/source/op/kernels/cpu DIR_KERNEL_CPU)
aux_source_directory(llminfer/source/op/kernels/cuda DIR_KERNEL_CUDA)
aux_source_directory(llminfer/source/op/kernels/ DIR_KERNEL)
aux_source_directory(llminfer/source/sampler DIR_SAMPLE)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

add_library(llmframe SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_SAMPLE})
target_link_libraries(llmframe sentencepiece glog::glog gtest gtest_main pthread cudart armadillo)
target_link_directories(llmframe PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(llmframe PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llmframe PUBLIC ${PROJECT_SOURCE_DIR}/llminfer/include)
target_include_directories(llmframe PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(llmframe PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(llmframe PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# include_directories(${CMAKE_SOURCE_DIR}/llminfer/include)

# aux_source_directory(${CMAKE_SOURCE_DIR}/llminfer/source/base DIR_BASE)

# add_subdirectory(test)

add_subdirectory(demo)


