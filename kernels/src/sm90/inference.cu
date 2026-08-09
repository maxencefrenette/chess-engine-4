#include "../inference/bf16_gemm.h"

#include <stdexcept>

namespace chess_engine_4::inference {

bool CustomBf16GemmSupported(int rows, int columns, int reduction) {
    // Start from the canonical cuBLAS BF16 path on Hopper. A custom inference
    // GEMM can replace it only after it wins end to end on both H100 and H200.
    return false;
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
    throw std::runtime_error("SM90 uses cuBLAS for dense BF16 GEMMs");
}

}  // namespace chess_engine_4::inference

#include "../inference/moe_bf16_impl.cuh"
