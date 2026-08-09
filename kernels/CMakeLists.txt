cmake_minimum_required(VERSION 3.25)

project(chess_engine_4_kernels LANGUAGES CXX CUDA)

option(CE4_BUILD_PYTHON "Build the PyTorch training extension" ON)
option(CE4_BUILD_INFERENCE "Build the standalone inference runtime" OFF)
set(CE4_CUDA_ARCH "120a" CACHE STRING "CUDA architecture for the inference runtime")

# CMake 3.25 predates CUDA 13's C++20 dialect table.
set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "--std=c++20")
set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "--std=c++20")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CE4_BUILD_INFERENCE)
  find_package(CUDAToolkit REQUIRED)
  find_package(nlohmann_json CONFIG REQUIRED)
  if(CE4_CUDA_ARCH STREQUAL "80")
    set(CE4_INFERENCE_ARCH_SOURCE src/sm80/inference.cu)
    set(CE4_INFERENCE_ARCH_DEFINITION KITTENS_SM80)
  elseif(CE4_CUDA_ARCH STREQUAL "100a")
    set(CE4_INFERENCE_ARCH_SOURCE src/sm100/inference.cu)
    set(CE4_INFERENCE_ARCH_DEFINITION KITTENS_SM100)
  elseif(CE4_CUDA_ARCH STREQUAL "120a")
    set(CE4_INFERENCE_ARCH_SOURCE src/sm120/inference.cu)
    set(CE4_INFERENCE_ARCH_DEFINITION KITTENS_SM120)
  else()
    message(FATAL_ERROR "CE4 inference supports only CUDA architectures 80, 100a, and 120a")
  endif()
  add_library(ce4_inference STATIC
    src/inference/runtime.cu
    ${CE4_INFERENCE_ARCH_SOURCE}
  )
  target_include_directories(ce4_inference
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ThunderKittens/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ThunderKittens/prototype"
  )
  target_compile_definitions(ce4_inference PRIVATE ${CE4_INFERENCE_ARCH_DEFINITION})
  target_link_libraries(
    ce4_inference
    PUBLIC CUDA::cuda_driver CUDA::cudart CUDA::cublas nlohmann_json::nlohmann_json
  )
  target_compile_options(ce4_inference PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      -O3
      --use_fast_math
      --expt-extended-lambda
      --expt-relaxed-constexpr
      -gencode=arch=compute_${CE4_CUDA_ARCH},code=sm_${CE4_CUDA_ARCH}
    >
  )
  set_target_properties(ce4_inference PROPERTIES
    CUDA_ARCHITECTURES OFF
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
endif()

if(NOT CE4_BUILD_PYTHON)
  return()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

# Torch expands TORCH_CUDA_ARCH_LIST into target-wide flags. These sources use
# architecture-specific ThunderKittens APIs, so each translation unit must only
# be assembled for its own architecture.
string(
  REPLACE "-gencode arch=compute_80,code=sm_80" ""
  CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}"
)
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
  message(FATAL_ERROR "Vendored ThunderKittens source is missing from third_party/ThunderKittens")
endif()

pybind11_add_module(_chess_engine_4_kernels MODULE
  src/bindings.cpp
  src/sm80/dense.cu
  src/sm80/moe.cu
  src/sm100/dense.cu
  src/sm100/moe.cu
  src/sm120/dense.cu
  src/sm120/moe.cu
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

set_source_files_properties(src/sm100/dense.cu src/sm100/moe.cu PROPERTIES
  COMPILE_DEFINITIONS KITTENS_SM100
  COMPILE_OPTIONS "-gencode=arch=compute_100a,code=sm_100a"
)
set_source_files_properties(src/sm80/dense.cu src/sm80/moe.cu PROPERTIES
  COMPILE_DEFINITIONS KITTENS_SM80
  COMPILE_OPTIONS "-gencode=arch=compute_80,code=sm_80"
)
set_source_files_properties(src/sm120/dense.cu PROPERTIES
  COMPILE_DEFINITIONS "KITTENS_SM120;DENSE_USE_ATEN_GEMM"
  COMPILE_OPTIONS "-gencode=arch=compute_120a,code=sm_120a"
)
set_source_files_properties(src/sm120/moe.cu PROPERTIES
  COMPILE_DEFINITIONS KITTENS_SM120
  COMPILE_OPTIONS "-gencode=arch=compute_120a,code=sm_120a"
)

set_target_properties(_chess_engine_4_kernels PROPERTIES
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
