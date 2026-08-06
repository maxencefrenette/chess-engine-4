#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <pybind11/pybind11.h>

namespace chess_engine_4::sm120::moe_d128 {

using namespace kittens;

constexpr int NUM_EXPERTS = 64;
constexpr int D_MODEL = 128;
constexpr int HIDDEN_DIM = 256;
constexpr int GATE_UP_DIM = 2 * HIDDEN_DIM;
constexpr int TILE = 16;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS = WARPS_PER_BLOCK * 32;

using ActivationGlobal = gl<bf16, 1, 1, -1, D_MODEL>;
using GateUpWeightGlobal = gl<bf16, 1, NUM_EXPERTS, GATE_UP_DIM, D_MODEL>;
using HiddenGlobal = gl<bf16, 1, 1, -1, HIDDEN_DIM>;
using DownWeightGlobal = gl<bf16, 1, NUM_EXPERTS, D_MODEL, HIDDEN_DIM>;
using OutputGlobal = gl<bf16, 1, 1, -1, D_MODEL>;
using ScaleGlobal = gl<bf16, 1, 1, 1, -1>;
using OffsetGlobal = gl<int, 1, 1, 1, NUM_EXPERTS + 1>;

struct GateSwiGLUGlobals {
    ActivationGlobal input;
    GateUpWeightGlobal weight;
    HiddenGlobal hidden;
    OffsetGlobal expert_offsets;
};

struct DownGlobals {
    HiddenGlobal hidden;
    DownWeightGlobal weight;
    OutputGlobal output;
    ScaleGlobal route_probs;
    OffsetGlobal expert_offsets;
};

__device__ __forceinline__ int expert_for_row(
    const OffsetGlobal &expert_offsets,
    int row
) {
    int lower = 0;
    int upper = NUM_EXPERTS;
    #pragma unroll 6
    while (lower + 1 < upper) {
        const int middle = (lower + upper) / 2;
        if (row < expert_offsets[{middle}]) {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    return lower;
}

__global__ void gate_swiglu_kernel(const GateSwiGLUGlobals globals) {
    const int warp = threadIdx.x / 32;
    const int total_row_tiles = globals.expert_offsets[{NUM_EXPERTS}] / TILE;
    const int output_tiles = HIDDEN_DIM / TILE;
    const int total_tasks = total_row_tiles * output_tiles;

    for (
        int task = blockIdx.x * WARPS_PER_BLOCK + warp;
        task < total_tasks;
        task += gridDim.x * WARPS_PER_BLOCK
    ) {
        const int row_tile = task / output_tiles;
        const int hidden_tile = task % output_tiles;
        const int expert = expert_for_row(globals.expert_offsets, row_tile * TILE);

        rt_bf<TILE, TILE> input_tile;
        rt_bf<TILE, TILE> weight_tile;
        rt_fl<TILE, TILE> gate;
        rt_fl<TILE, TILE> up;
        warp::zero(gate);
        warp::zero(up);

        #pragma unroll
        for (int reduction_tile = 0; reduction_tile < D_MODEL / TILE; ++reduction_tile) {
            warp::load(input_tile, globals.input, {0, 0, row_tile, reduction_tile});
            warp::load(
                weight_tile,
                globals.weight,
                {0, expert, hidden_tile, reduction_tile}
            );
            warp::mma_ABt(gate, input_tile, weight_tile, gate);
            warp::load(
                weight_tile,
                globals.weight,
                {0, expert, HIDDEN_DIM / TILE + hidden_tile, reduction_tile}
            );
            warp::mma_ABt(up, input_tile, weight_tile, up);
        }

        warp::apply(
            gate,
            gate,
            [] __device__ (int, int, float value) {
                return value / (1.0f + __expf(-value));
            }
        );
        warp::mul(gate, gate, up);
        warp::store(globals.hidden, gate, {0, 0, row_tile, hidden_tile});
    }
}

__global__ void down_kernel(const DownGlobals globals) {
    const int warp = threadIdx.x / 32;
    const int total_row_tiles = globals.expert_offsets[{NUM_EXPERTS}] / TILE;
    const int output_tiles = D_MODEL / TILE;
    const int total_tasks = total_row_tiles * output_tiles;

    for (
        int task = blockIdx.x * WARPS_PER_BLOCK + warp;
        task < total_tasks;
        task += gridDim.x * WARPS_PER_BLOCK
    ) {
        const int row_tile = task / output_tiles;
        const int output_tile = task % output_tiles;
        const int expert = expert_for_row(globals.expert_offsets, row_tile * TILE);

        rt_bf<TILE, TILE> hidden_tile;
        rt_bf<TILE, TILE> weight_tile;
        rt_fl<TILE, TILE> output_tile_values;
        warp::zero(output_tile_values);

        #pragma unroll
        for (
            int reduction_tile = 0;
            reduction_tile < HIDDEN_DIM / TILE;
            ++reduction_tile
        ) {
            warp::load(hidden_tile, globals.hidden, {0, 0, row_tile, reduction_tile});
            warp::load(
                weight_tile,
                globals.weight,
                {0, expert, output_tile, reduction_tile}
            );
            warp::mma_ABt(output_tile_values, hidden_tile, weight_tile, output_tile_values);
        }
        col_vec<rt_fl<TILE, TILE>> route_scale;
        warp::load(route_scale, globals.route_probs, {row_tile});
        warp::mul_row(output_tile_values, output_tile_values, route_scale);
        warp::store(globals.output, output_tile_values, {0, 0, row_tile, output_tile});
    }
}

void check_tensor(
    const at::Tensor &tensor,
    const char *name,
    at::ScalarType dtype,
    at::IntArrayRef shape
) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(
        tensor.sizes() == shape,
        name,
        " has shape ",
        tensor.sizes(),
        ", expected ",
        shape
    );
}

void forward(
    const at::Tensor &input,
    const at::Tensor &gate_up_weight,
    const at::Tensor &down_weight,
    const at::Tensor &route_probs,
    const at::Tensor &expert_offsets,
    at::Tensor &hidden,
    at::Tensor &output
) {
    TORCH_CHECK(input.dim() == 2, "input must be a matrix");
    TORCH_CHECK(input.size(1) == D_MODEL, "input width must be 128");
    TORCH_CHECK(input.size(0) % TILE == 0, "input rows must be divisible by 16");
    const int64_t rows = input.size(0);
    check_tensor(input, "input", at::kBFloat16, {rows, D_MODEL});
    check_tensor(
        gate_up_weight,
        "gate_up_weight",
        at::kBFloat16,
        {NUM_EXPERTS, GATE_UP_DIM, D_MODEL}
    );
    check_tensor(
        down_weight,
        "down_weight",
        at::kBFloat16,
        {NUM_EXPERTS, D_MODEL, HIDDEN_DIM}
    );
    check_tensor(route_probs, "route_probs", at::kBFloat16, {rows});
    check_tensor(expert_offsets, "expert_offsets", at::kInt, {NUM_EXPERTS + 1});
    check_tensor(hidden, "hidden", at::kBFloat16, {rows, HIDDEN_DIM});
    check_tensor(output, "output", at::kBFloat16, {rows, D_MODEL});
    TORCH_CHECK(
        gate_up_weight.device() == input.device()
            && down_weight.device() == input.device()
            && route_probs.device() == input.device()
            && expert_offsets.device() == input.device()
            && hidden.device() == input.device()
            && output.device() == input.device(),
        "all tensors must be on the same device"
    );

    const c10::cuda::CUDAGuard device_guard(input.device());
    cudaDeviceProp properties;
    CUDACHECK(cudaGetDeviceProperties(&properties, input.get_device()));
    TORCH_CHECK(
        properties.major == 12 && properties.minor == 0,
        "moe_d128_forward requires SM120, got compute capability ",
        properties.major,
        ".",
        properties.minor
    );

    const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    const int blocks = properties.multiProcessorCount * 2;
    GateSwiGLUGlobals gate_globals{
        .input = kittens::py::tensor_to_gl<ActivationGlobal>(input),
        .weight = kittens::py::tensor_to_gl<GateUpWeightGlobal>(gate_up_weight),
        .hidden = kittens::py::tensor_to_gl<HiddenGlobal>(hidden),
        .expert_offsets = kittens::py::tensor_to_gl<OffsetGlobal>(expert_offsets),
    };
    DownGlobals down_globals{
        .hidden = kittens::py::tensor_to_gl<HiddenGlobal>(hidden),
        .weight = kittens::py::tensor_to_gl<DownWeightGlobal>(down_weight),
        .output = kittens::py::tensor_to_gl<OutputGlobal>(output),
        .route_probs = kittens::py::tensor_to_gl<ScaleGlobal>(route_probs),
        .expert_offsets = kittens::py::tensor_to_gl<OffsetGlobal>(expert_offsets),
    };
    gate_swiglu_kernel<<<blocks, THREADS, 0, stream>>>(gate_globals);
    down_kernel<<<blocks, THREADS, 0, stream>>>(down_globals);
    CUDACHECK(cudaPeekAtLastError());
}

}  // namespace chess_engine_4::sm120::moe_d128

void bind_moe_d128_kernels(pybind11::module_ &module) {
    module.def("moe_d128_forward", &chess_engine_4::sm120::moe_d128::forward);
}
