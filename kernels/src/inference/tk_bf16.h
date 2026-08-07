#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace chess_engine_4::inference {

bool TkBf16GemmSupported(int rows, int columns, int reduction);

void LaunchTkBf16Gemm(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows,
    int columns,
    int reduction,
    cudaStream_t stream
);

}  // namespace chess_engine_4::inference
