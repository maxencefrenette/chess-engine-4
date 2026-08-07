#include "moe_bf16.h"

#include "kittens.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace chess_engine_4::inference {
namespace {

using namespace kittens;

constexpr int kTile = 16;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = 256;

__global__ void RouteKernel(
    const __nv_bfloat16* logits,
    int* route_experts,
    __nv_bfloat16* route_probabilities,
    int* expert_counts,
    int batch_size
) {
    const int token = blockIdx.x;
    if (token >= batch_size || threadIdx.x != 0) return;

    float maximum = -INFINITY;
    for (int expert = 0; expert < kMoeExpertCount; ++expert) {
        maximum = fmaxf(maximum, __bfloat162float(logits[token * kMoeExpertCount + expert]));
    }
    float denominator = 0.0f;
    float best_value = -INFINITY;
    float second_value = -INFINITY;
    int best_expert = 0;
    int second_expert = 1;
    for (int expert = 0; expert < kMoeExpertCount; ++expert) {
        const float value = __bfloat162float(logits[token * kMoeExpertCount + expert]);
        denominator += expf(value - maximum);
        if (value > best_value) {
            second_value = best_value;
            second_expert = best_expert;
            best_value = value;
            best_expert = expert;
        } else if (value > second_value) {
            second_value = value;
            second_expert = expert;
        }
    }
    route_experts[token * 2] = best_expert;
    route_experts[token * 2 + 1] = second_expert;
    route_probabilities[token * 2] = __float2bfloat16(expf(best_value - maximum) / denominator);
    route_probabilities[token * 2 + 1] = __float2bfloat16(
        expf(second_value - maximum) / denominator
    );
    atomicAdd(expert_counts + best_expert, 1);
    atomicAdd(expert_counts + second_expert, 1);
}

__global__ void PrefixOffsetsKernel(const int* counts, int* offsets) {
    if (threadIdx.x != 0) return;
    int offset = 0;
    offsets[0] = 0;
    for (int expert = 0; expert < kMoeExpertCount; ++expert) {
        offset += (counts[expert] + kTile - 1) / kTile * kTile;
        offsets[expert + 1] = offset;
    }
}

__global__ void DispatchKernel(
    const __nv_bfloat16* input,
    const int* route_experts,
    const __nv_bfloat16* route_probabilities,
    const int* expert_offsets,
    int* expert_cursors,
    int* route_positions,
    __nv_bfloat16* expert_input,
    __nv_bfloat16* expert_probabilities,
    int routes,
    int d_model
) {
    const int route = blockIdx.x;
    if (route >= routes) return;
    __shared__ int position;
    if (threadIdx.x == 0) {
        const int expert = route_experts[route];
        position = expert_offsets[expert] + atomicAdd(expert_cursors + expert, 1);
        route_positions[route] = position;
        expert_probabilities[position] = route_probabilities[route];
    }
    __syncthreads();
    const int token = route / kMoeActiveExpertCount;
    for (int column = threadIdx.x; column < d_model; column += blockDim.x) {
        expert_input[position * d_model + column] = input[token * d_model + column];
    }
}

__global__ void CombineKernel(
    const __nv_bfloat16* residual,
    const __nv_bfloat16* expert_output,
    const int* route_positions,
    __nv_bfloat16* output,
    int batch_size,
    int d_model
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * d_model) return;
    const int token = index / d_model;
    const int column = index % d_model;
    const int first = route_positions[token * 2];
    const int second = route_positions[token * 2 + 1];
    output[index] = __float2bfloat16(
        __bfloat162float(residual[index])
        + __bfloat162float(expert_output[first * d_model + column])
        + __bfloat162float(expert_output[second * d_model + column])
    );
}

template <int DModel>
struct ExpertSpec {
    static constexpr int kHidden = 2 * DModel;
    static constexpr int kGateUp = 2 * kHidden;
    static constexpr int kOutputTile = DModel >= 256 ? 64 : 16;

    using Activation = gl<bf16, 1, 1, -1, DModel>;
    using GateUpWeight = gl<bf16, 1, kMoeExpertCount, kGateUp, DModel>;
    using Hidden = gl<bf16, 1, 1, -1, kHidden>;
    using DownWeight = gl<bf16, 1, kMoeExpertCount, DModel, kHidden>;
    using Output = gl<bf16, 1, 1, -1, DModel>;
    using Scale = gl<bf16, 1, 1, 1, -1>;
    using Offset = gl<int, 1, 1, 1, kMoeExpertCount + 1>;

    struct GateGlobals {
        Activation input;
        GateUpWeight weight;
        Hidden hidden;
        Offset offsets;
    };
    struct DownGlobals {
        Hidden hidden;
        DownWeight weight;
        Output output;
        Scale probabilities;
        Offset offsets;
    };
};

template <typename Offset>
__device__ int ExpertForRow(const Offset& offsets, int row) {
    int lower = 0;
    int upper = kMoeExpertCount;
    #pragma unroll 6
    while (lower + 1 < upper) {
        const int middle = (lower + upper) / 2;
        if (row < offsets[{middle}]) upper = middle;
        else lower = middle;
    }
    return lower;
}

template <int DModel>
__global__ void ExpertGateKernel(typename ExpertSpec<DModel>::GateGlobals globals) {
    using Spec = ExpertSpec<DModel>;
    const int warp = threadIdx.x / 32;
    const int row_tiles = globals.offsets[{kMoeExpertCount}] / kTile;
    const int output_tiles = Spec::kHidden / Spec::kOutputTile;
    const int tasks = row_tiles * output_tiles;
    for (int task = blockIdx.x * kWarpsPerBlock + warp;
         task < tasks;
         task += gridDim.x * kWarpsPerBlock) {
            const int row_tile = task / output_tiles;
            const int output_tile = task % output_tiles;
            const int expert = ExpertForRow(globals.offsets, row_tile * kTile);
            rt_bf<kTile, kTile> input_tile;
            rt_bf<Spec::kOutputTile, kTile> weight_tile;
            rt_fl<kTile, Spec::kOutputTile> gate;
            rt_fl<kTile, Spec::kOutputTile> up;
            warp::zero(gate);
            warp::zero(up);
            #pragma unroll
            for (int reduction = 0; reduction < DModel / kTile; ++reduction) {
                warp::load(input_tile, globals.input, {0, 0, row_tile, reduction});
                warp::load(weight_tile, globals.weight, {0, expert, output_tile, reduction});
                warp::mma_ABt(gate, input_tile, weight_tile, gate);
                warp::load(
                    weight_tile,
                    globals.weight,
                    {0, expert, Spec::kHidden / Spec::kOutputTile + output_tile, reduction}
                );
                warp::mma_ABt(up, input_tile, weight_tile, up);
            }
            warp::apply(gate, gate, [] __device__(int, int, float value) {
                return value / (1.0f + __expf(-value));
            });
            warp::mul(gate, gate, up);
            warp::store(globals.hidden, gate, {0, 0, row_tile, output_tile});
    }
}

template <int DModel>
__global__ void ExpertDownKernel(typename ExpertSpec<DModel>::DownGlobals globals) {
    using Spec = ExpertSpec<DModel>;
    const int warp = threadIdx.x / 32;
    const int row_tiles = globals.offsets[{kMoeExpertCount}] / kTile;
    const int output_tiles = DModel / Spec::kOutputTile;
    const int tasks = row_tiles * output_tiles;
    for (int task = blockIdx.x * kWarpsPerBlock + warp;
         task < tasks;
         task += gridDim.x * kWarpsPerBlock) {
            const int row_tile = task / output_tiles;
            const int output_tile = task % output_tiles;
            const int expert = ExpertForRow(globals.offsets, row_tile * kTile);
            rt_bf<kTile, kTile> hidden_tile;
            rt_bf<Spec::kOutputTile, kTile> weight_tile;
            rt_fl<kTile, Spec::kOutputTile> result;
            warp::zero(result);
            #pragma unroll
            for (int reduction = 0; reduction < Spec::kHidden / kTile; ++reduction) {
                warp::load(hidden_tile, globals.hidden, {0, 0, row_tile, reduction});
                warp::load(weight_tile, globals.weight, {0, expert, output_tile, reduction});
                warp::mma_ABt(result, hidden_tile, weight_tile, result);
            }
            col_vec<rt_fl<kTile, Spec::kOutputTile>> scale;
            warp::load(scale, globals.probabilities, {row_tile});
            warp::mul_row(result, result, scale);
            warp::store(globals.output, result, {0, 0, row_tile, output_tile});
    }
}

template <int DModel>
void LaunchExpertKernels(
        const __nv_bfloat16* input,
        const __nv_bfloat16* gate_up,
        const __nv_bfloat16* down,
        const __nv_bfloat16* probabilities,
        const int* offsets,
        __nv_bfloat16* hidden,
        __nv_bfloat16* output,
        int maximum_rows,
        int multiprocessors,
        cudaStream_t stream
    ) {
        using Spec = ExpertSpec<DModel>;
        const auto rows = static_cast<std::size_t>(maximum_rows);
        typename Spec::GateGlobals gate_globals{
            .input = typename Spec::Activation{reinterpret_cast<bf16*>(const_cast<__nv_bfloat16*>(input)), nullptr, nullptr, rows, nullptr},
            .weight = typename Spec::GateUpWeight{reinterpret_cast<bf16*>(const_cast<__nv_bfloat16*>(gate_up)), nullptr, nullptr, nullptr, nullptr},
            .hidden = typename Spec::Hidden{reinterpret_cast<bf16*>(hidden), nullptr, nullptr, rows, nullptr},
            .offsets = typename Spec::Offset{const_cast<int*>(offsets), nullptr, nullptr, nullptr, nullptr},
        };
        typename Spec::DownGlobals down_globals{
            .hidden = typename Spec::Hidden{reinterpret_cast<bf16*>(hidden), nullptr, nullptr, rows, nullptr},
            .weight = typename Spec::DownWeight{reinterpret_cast<bf16*>(const_cast<__nv_bfloat16*>(down)), nullptr, nullptr, nullptr, nullptr},
            .output = typename Spec::Output{reinterpret_cast<bf16*>(output), nullptr, nullptr, rows, nullptr},
            .probabilities = typename Spec::Scale{reinterpret_cast<bf16*>(const_cast<__nv_bfloat16*>(probabilities)), nullptr, nullptr, nullptr, rows},
            .offsets = typename Spec::Offset{const_cast<int*>(offsets), nullptr, nullptr, nullptr, nullptr},
        };
        const int blocks = multiprocessors * 2;
        ExpertGateKernel<DModel><<<blocks, kThreads, 0, stream>>>(gate_globals);
        ExpertDownKernel<DModel><<<blocks, kThreads, 0, stream>>>(down_globals);
}

}  // namespace

void LaunchMoeDispatch(
    const __nv_bfloat16* input,
    const __nv_bfloat16* router_logits,
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
) {
    int* route_experts = route_positions + 2 * batch_size;
    __nv_bfloat16* staged_probabilities = expert_probabilities + maximum_padded_rows;
    cudaMemsetAsync(expert_counts, 0, kMoeExpertCount * sizeof(int), stream);
    RouteKernel<<<batch_size, 1, 0, stream>>>(
        router_logits, route_experts, staged_probabilities,
        expert_counts, batch_size
    );
    PrefixOffsetsKernel<<<1, 1, 0, stream>>>(expert_counts, expert_offsets);
    cudaMemsetAsync(expert_cursors, 0, kMoeExpertCount * sizeof(int), stream);
    cudaMemsetAsync(
        expert_input, 0,
        static_cast<std::size_t>(maximum_padded_rows) * d_model * sizeof(__nv_bfloat16),
        stream
    );
    cudaMemsetAsync(
        expert_probabilities, 0,
        maximum_padded_rows * sizeof(__nv_bfloat16), stream
    );
    DispatchKernel<<<batch_size * 2, std::min(d_model, 256), 0, stream>>>(
        input, route_experts, staged_probabilities, expert_offsets,
        expert_cursors, route_positions, expert_input,
        expert_probabilities, batch_size * 2, d_model
    );
}

void LaunchMoeExperts(
    const __nv_bfloat16* input,
    const __nv_bfloat16* gate_up,
    const __nv_bfloat16* down,
    const __nv_bfloat16* probabilities,
    const int* offsets,
    __nv_bfloat16* hidden,
    __nv_bfloat16* output,
    int d_model,
    int maximum_rows,
    int multiprocessors,
    cudaStream_t stream
) {
    switch (d_model) {
        case 128: LaunchExpertKernels<128>(input, gate_up, down, probabilities, offsets, hidden, output, maximum_rows, multiprocessors, stream); break;
        case 256: LaunchExpertKernels<256>(input, gate_up, down, probabilities, offsets, hidden, output, maximum_rows, multiprocessors, stream); break;
        case 512: LaunchExpertKernels<512>(input, gate_up, down, probabilities, offsets, hidden, output, maximum_rows, multiprocessors, stream); break;
        case 1024: LaunchExpertKernels<1024>(input, gate_up, down, probabilities, offsets, hidden, output, maximum_rows, multiprocessors, stream); break;
        default: throw std::runtime_error("unsupported MoE inference width");
    }
}

void LaunchMoeCombine(
    const __nv_bfloat16* residual,
    const __nv_bfloat16* expert_output,
    const int* route_positions,
    __nv_bfloat16* output,
    int batch_size,
    int d_model,
    cudaStream_t stream
) {
    const int elements = batch_size * d_model;
    CombineKernel<<<(elements + 255) / 256, 256, 0, stream>>>(
        residual, expert_output, route_positions, output, batch_size, d_model
    );
}

}  // namespace chess_engine_4::inference
