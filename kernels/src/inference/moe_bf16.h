#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace chess_engine_4::inference {

constexpr int kMoeExpertCount = 64;
constexpr int kMoeActiveExpertCount = 2;

void LaunchMoeDispatch(
    const __nv_bfloat16* input,
    const __nv_bfloat16* router_logits,
    const float* router_qb_beta,
    __nv_bfloat16* expert_input,
    __nv_bfloat16* expert_probabilities,
    int* expert_offsets,
    int* route_positions,
    int* expert_counts,
    int* expert_cursors,
    int batch_size,
    int d_model,
    int maximum_padded_rows,
    cudaStream_t stream
);

void LaunchMoeExperts(
    const __nv_bfloat16* expert_input,
    const __nv_bfloat16* gate_up_weight,
    const __nv_bfloat16* down_weight,
    const __nv_bfloat16* expert_probabilities,
    const int* expert_offsets,
    __nv_bfloat16* hidden,
    __nv_bfloat16* expert_output,
    int d_model,
    int maximum_padded_rows,
    int multiprocessor_count,
    cudaStream_t stream
);

void LaunchMoeCombine(
    const __nv_bfloat16* residual,
    const __nv_bfloat16* expert_output,
    const int* route_positions,
    __nv_bfloat16* output,
    int batch_size,
    int d_model,
    cudaStream_t stream
);

}  // namespace chess_engine_4::inference
