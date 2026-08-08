#include "../inference/bf16_gemm.h"
#include "bf16_gemm.cuh"

#include <cuda_runtime.h>

namespace chess_engine_4::inference {

bool CustomBf16GemmSupported(int rows, int columns, int reduction) {
    return rows % 16 == 0 && columns % 16 == 0 && reduction % 16 == 0;
}

void LaunchCustomBf16Gemm(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows,
    int columns,
    int reduction,
    cudaStream_t stream
) {
    int device = 0;
    cudaDeviceProp properties;
    CUDACHECK(cudaGetDevice(&device));
    CUDACHECK(cudaGetDeviceProperties(&properties, device));
    chess_engine_4::sm80::LaunchBf16Gemm(
        input, weight, output, rows, columns, reduction,
        properties.multiProcessorCount, stream
    );
}

}  // namespace chess_engine_4::inference

#include "../inference/moe_bf16_impl.cuh"
