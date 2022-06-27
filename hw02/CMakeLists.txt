cmake_minimum_required(VERSION 3.11)


# ------------------------------------------------------------------------------------------------ #
# Set compiler path
# ------------------------------------------------------------------------------------------------ #
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)


# ------------------------------------------------------------------------------------------------ #
# General setting
# ------------------------------------------------------------------------------------------------ #
project(CUDAPractice CXX CUDA)

include_directories(include)
include_directories(/usr/local/cuda/include)


# ------------------------------------------------------------------------------------------------ #
# Compiler flags
# ------------------------------------------------------------------------------------------------ #
# GCC
set(CMAKE_CXX_FLAGS "-O3")
# NVCC
set(CMAKE_CUDA_FLAGS "-v")


# ------------------------------------------------------------------------------------------------ #
# Library
# ------------------------------------------------------------------------------------------------ #
set(REFERENCE_SOURCE_FILES
        src/add/add_reference.cpp
        src/mm/mm_reference.cpp
        src/mv/mv_reference.cpp
        src/reduce/reduce_reference.cpp
        src/softmax/softmax_reference.cpp
        )
set(CUDA_SOURCE_FILES
        src/add/add_cuda.cu
        src/add/add_cuda_wrapper.cpp
        src/mm/mm_cuda.cu
        src/mm/mm_cuda_wrapper.cpp
        src/mv/mv_cuda.cu
        src/mv/mv_cuda_wrapper.cpp
        src/reduce/reduce_cuda.cu
        src/reduce/reduce_cuda_wrapper.cpp
        src/softmax/softmax_cuda.cu
        src/softmax/softmax_cuda_wrapper.cpp
        )
add_library(src_lib ${REFERENCE_SOURCE_FILES} ${CUDA_SOURCE_FILES})

# ------------------------------------------------------------------------------------------------ #
# Executable
# ------------------------------------------------------------------------------------------------ #
add_executable(run_add main_add.cpp)
target_link_libraries(run_add src_lib)

add_executable(run_mm main_mm.cpp)
target_link_libraries(run_mm src_lib)

add_executable(run_reduce main_reduce.cpp)
target_link_libraries(run_reduce src_lib)

add_executable(run_softmax main_softmax.cpp)
target_link_libraries(run_softmax src_lib)

add_executable(run_mv main_mv.cpp)
target_link_libraries(run_mv src_lib)
