#include "../inference/bf16_gemm.h"

#include <stdexcept>

namespace chess_engine_4::inference {

bool CustomBf16GemmSupported(int rows, int columns, int reduction) {
    // cuBLAS is faster for every batch-256 dense shape on SM120.
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
    throw std::runtime_error("SM120 uses cuBLAS for dense BF16 GEMMs");
}

}  // namespace chess_engine_4::inference

#include "../inference/moe_bf16_impl.cuh"
