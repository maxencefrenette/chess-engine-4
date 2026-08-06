cmake_minimum_required(VERSION 3.25)

project(chess_engine_4_kernels LANGUAGES CXX CUDA)

# CMake 3.25 predates CUDA 13's C++20 dialect table.
set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "--std=c++20")
set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "--std=c++20")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

# Torch expands TORCH_CUDA_ARCH_LIST into target-wide flags. These sources use
# architecture-specific ThunderKittens APIs, so each translation unit must only
# be assembled for its own architecture.
string(
  REPLACE "-gencode arch=compute_100a,code=sm_100a" ""
  CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}"
)
string(
  REPLACE "-gencode arch=compute_120a,code=sm_120a" ""
  CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}"
)

set(THUNDERKITTENS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ThunderKittens")
if(NOT EXISTS "${THUNDERKITTENS_ROOT}/include/kittens.cuh")
  message(FATAL_ERROR "ThunderKittens submodule is missing; run git submodule update --init")
endif()

pybind11_add_module(_chess_engine_4_kernels MODULE
  src/bindings.cpp
  src/sm100/dense.cu
  src/sm120/moe_d128.cu
)

target_include_directories(_chess_engine_4_kernels PRIVATE
  "${THUNDERKITTENS_ROOT}/include"
  "${THUNDERKITTENS_ROOT}/prototype"
)
target_link_libraries(_chess_engine_4_kernels PRIVATE
  ${TORCH_LIBRARIES}
  "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so"
  "${CUDA_CUDART_LIBRARY}"
)
target_compile_definitions(_chess_engine_4_kernels PRIVATE
  TORCH_COMPILE
)
target_compile_options(_chess_engine_4_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    --use_fast_math
    --expt-extended-lambda
    --expt-relaxed-constexpr
    -forward-unknown-to-host-compiler
    -Xcompiler=-fno-strict-aliasing
    -diag-suppress=3189
  >
)

set_source_files_properties(src/sm100/dense.cu PROPERTIES
  COMPILE_DEFINITIONS KITTENS_SM100
  COMPILE_OPTIONS "-gencode=arch=compute_100a,code=sm_100a"
)
set_source_files_properties(src/sm120/moe_d128.cu PROPERTIES
  COMPILE_DEFINITIONS KITTENS_SM120
  COMPILE_OPTIONS "-gencode=arch=compute_120a,code=sm_120a"
)

set_target_properties(_chess_engine_4_kernels PROPERTIES
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
