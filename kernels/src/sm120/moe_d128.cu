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
using GateUpGradientGlobal = gl<bf16, 1, 1, -1, GATE_UP_DIM>;
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
    OutputGlobal raw_output;
    OutputGlobal output;
    ScaleGlobal route_probs;
    OffsetGlobal expert_offsets;
};

struct GradHiddenGlobals {
    OutputGlobal grad_output;
    DownWeightGlobal weight;
    HiddenGlobal grad_hidden;
    OffsetGlobal expert_offsets;
};

struct GradGateUpGlobals {
    ActivationGlobal input;
    GateUpWeightGlobal weight;
    HiddenGlobal grad_hidden;
    GateUpGradientGlobal grad_gate_up;
    OffsetGlobal expert_offsets;
};

struct GradInputGlobals {
    GateUpGradientGlobal grad_gate_up;
    GateUpWeightGlobal weight;
    ActivationGlobal grad_input;
    OffsetGlobal expert_offsets;
};

struct DownWeightGradientGlobals {
    OutputGlobal grad_output;
    HiddenGlobal hidden;
    DownWeightGlobal grad_weight;
    OffsetGlobal expert_offsets;
};

struct GateUpWeightGradientGlobals {
    GateUpGradientGlobal grad_gate_up;
    ActivationGlobal input;
    GateUpWeightGlobal grad_weight;
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

template <bool SAVE_RAW_OUTPUT>
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
        if constexpr (SAVE_RAW_OUTPUT) {
            warp::store(
                globals.raw_output,
                output_tile_values,
                {0, 0, row_tile, output_tile}
            );
        }
        col_vec<rt_fl<TILE, TILE>> route_scale;
        warp::load(route_scale, globals.route_probs, {row_tile});
        warp::mul_row(output_tile_values, output_tile_values, route_scale);
        warp::store(globals.output, output_tile_values, {0, 0, row_tile, output_tile});
    }
}

__global__ void scale_gradient_kernel(
    const __nv_bfloat16 *grad_output,
    const __nv_bfloat16 *raw_output,
    const __nv_bfloat16 *route_probs,
    __nv_bfloat16 *grad_unscaled_output,
    __nv_bfloat16 *grad_route_probs,
    int rows
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int column = threadIdx.x;
    const int offset = row * D_MODEL + column;
    const float grad = __bfloat162float(grad_output[offset]);
    grad_unscaled_output[offset] = __float2bfloat16(
        grad * __bfloat162float(route_probs[row])
    );

    float route_grad = grad * __bfloat162float(raw_output[offset]);
    #pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
        route_grad += __shfl_down_sync(0xffffffff, route_grad, delta);
    }
    __shared__ float warp_sums[4];
    if (column % 32 == 0) {
        warp_sums[column / 32] = route_grad;
    }
    __syncthreads();
    if (column < 32) {
        route_grad = column < 4 ? warp_sums[column] : 0.0f;
        #pragma unroll
        for (int delta = 16; delta > 0; delta /= 2) {
            route_grad += __shfl_down_sync(0xffffffff, route_grad, delta);
        }
        if (column == 0) {
            grad_route_probs[row] = __float2bfloat16(route_grad);
        }
    }
}

__global__ void grad_hidden_kernel(const GradHiddenGlobals globals) {
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
        rt_bf<TILE, TILE> grad_tile;
        rt_bf<TILE, TILE, col_l> weight_tile;
        rt_fl<TILE, TILE> result;
        warp::zero(result);
        #pragma unroll
        for (int reduction_tile = 0; reduction_tile < D_MODEL / TILE; ++reduction_tile) {
            warp::load(
                grad_tile,
                globals.grad_output,
                {0, 0, row_tile, reduction_tile}
            );
            warp::load(
                weight_tile,
                globals.weight,
                {0, expert, reduction_tile, hidden_tile}
            );
            warp::mma_AB(result, grad_tile, weight_tile, result);
        }
        warp::store(globals.grad_hidden, result, {0, 0, row_tile, hidden_tile});
    }
}

__global__ void grad_gate_up_kernel(const GradGateUpGlobals globals) {
    const int warp = threadIdx.x / 32;
    const int total_row_tiles = globals.expert_offsets[{NUM_EXPERTS}] / TILE;
    const int hidden_tiles = HIDDEN_DIM / TILE;
    const int total_tasks = total_row_tiles * hidden_tiles;

    for (
        int task = blockIdx.x * WARPS_PER_BLOCK + warp;
        task < total_tasks;
        task += gridDim.x * WARPS_PER_BLOCK
    ) {
        const int row_tile = task / hidden_tiles;
        const int hidden_tile = task % hidden_tiles;
        const int expert = expert_for_row(globals.expert_offsets, row_tile * TILE);
        rt_bf<TILE, TILE> input_tile;
        rt_bf<TILE, TILE> weight_tile;
        rt_bf<TILE, TILE> grad_hidden;
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
        warp::load(grad_hidden, globals.grad_hidden, {0, 0, row_tile, hidden_tile});
        rt_fl<TILE, TILE> grad_gate;
        rt_fl<TILE, TILE> grad_up;
        #pragma unroll
        for (int slot = 0; slot < rt_fl<TILE, TILE>::packed_per_tile; ++slot) {
            const float gate_x = gate.tiles[0][0].data[slot].x;
            const float gate_y = gate.tiles[0][0].data[slot].y;
            const float up_x = up.tiles[0][0].data[slot].x;
            const float up_y = up.tiles[0][0].data[slot].y;
            const float grad_x = __bfloat162float(grad_hidden.tiles[0][0].data[slot].x);
            const float grad_y = __bfloat162float(grad_hidden.tiles[0][0].data[slot].y);
            const float sigmoid_x = 1.0f / (1.0f + __expf(-gate_x));
            const float sigmoid_y = 1.0f / (1.0f + __expf(-gate_y));
            grad_gate.tiles[0][0].data[slot] = make_float2(
                grad_x * up_x * sigmoid_x * (1.0f + gate_x * (1.0f - sigmoid_x)),
                grad_y * up_y * sigmoid_y * (1.0f + gate_y * (1.0f - sigmoid_y))
            );
            grad_up.tiles[0][0].data[slot] = make_float2(
                grad_x * gate_x * sigmoid_x,
                grad_y * gate_y * sigmoid_y
            );
        }
        warp::store(
            globals.grad_gate_up,
            grad_gate,
            {0, 0, row_tile, hidden_tile}
        );
        warp::store(
            globals.grad_gate_up,
            grad_up,
            {0, 0, row_tile, HIDDEN_DIM / TILE + hidden_tile}
        );
    }
}

__global__ void grad_input_kernel(const GradInputGlobals globals) {
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
        rt_bf<TILE, TILE> grad_tile;
        rt_bf<TILE, TILE, col_l> weight_tile;
        rt_fl<TILE, TILE> result;
        warp::zero(result);
        #pragma unroll
        for (int reduction_tile = 0; reduction_tile < GATE_UP_DIM / TILE; ++reduction_tile) {
            warp::load(
                grad_tile,
                globals.grad_gate_up,
                {0, 0, row_tile, reduction_tile}
            );
            warp::load(
                weight_tile,
                globals.weight,
                {0, expert, reduction_tile, output_tile}
            );
            warp::mma_AB(result, grad_tile, weight_tile, result);
        }
        warp::store(globals.grad_input, result, {0, 0, row_tile, output_tile});
    }
}

__global__ void down_weight_gradient_kernel(const DownWeightGradientGlobals globals) {
    const int warp = threadIdx.x / 32;
    constexpr int OUTPUT_TILES = D_MODEL / TILE;
    constexpr int HIDDEN_TILES = HIDDEN_DIM / TILE;
    constexpr int TASKS_PER_EXPERT = OUTPUT_TILES * HIDDEN_TILES;
    const int total_tasks = NUM_EXPERTS * TASKS_PER_EXPERT;

    for (
        int task = blockIdx.x * WARPS_PER_BLOCK + warp;
        task < total_tasks;
        task += gridDim.x * WARPS_PER_BLOCK
    ) {
        const int expert = task / TASKS_PER_EXPERT;
        const int expert_task = task % TASKS_PER_EXPERT;
        const int output_tile = expert_task / HIDDEN_TILES;
        const int hidden_tile = expert_task % HIDDEN_TILES;
        const int first_row_tile = globals.expert_offsets[{expert}] / TILE;
        const int end_row_tile = globals.expert_offsets[{expert + 1}] / TILE;
        rt_fl<TILE, TILE> result;
        warp::zero(result);
        for (int row_tile = first_row_tile; row_tile < end_row_tile; ++row_tile) {
            rt_bf<TILE, TILE, col_l> grad_tile;
            rt_bf<TILE, TILE, col_l> hidden_tile_values;
            warp::load(
                grad_tile,
                globals.grad_output,
                {0, 0, row_tile, output_tile}
            );
            warp::load(
                hidden_tile_values,
                globals.hidden,
                {0, 0, row_tile, hidden_tile}
            );
            warp::mma_AtB(result, grad_tile, hidden_tile_values, result);
        }
        warp::store(
            globals.grad_weight,
            result,
            {0, expert, output_tile, hidden_tile}
        );
    }
}

__global__ void gate_up_weight_gradient_kernel(const GateUpWeightGradientGlobals globals) {
    const int warp = threadIdx.x / 32;
    constexpr int OUTPUT_TILES = GATE_UP_DIM / TILE;
    constexpr int INPUT_TILES = D_MODEL / TILE;
    constexpr int TASKS_PER_EXPERT = OUTPUT_TILES * INPUT_TILES;
    const int total_tasks = NUM_EXPERTS * TASKS_PER_EXPERT;

    for (
        int task = blockIdx.x * WARPS_PER_BLOCK + warp;
        task < total_tasks;
        task += gridDim.x * WARPS_PER_BLOCK
    ) {
        const int expert = task / TASKS_PER_EXPERT;
        const int expert_task = task % TASKS_PER_EXPERT;
        const int output_tile = expert_task / INPUT_TILES;
        const int input_tile = expert_task % INPUT_TILES;
        const int first_row_tile = globals.expert_offsets[{expert}] / TILE;
        const int end_row_tile = globals.expert_offsets[{expert + 1}] / TILE;
        rt_fl<TILE, TILE> result;
        warp::zero(result);
        for (int row_tile = first_row_tile; row_tile < end_row_tile; ++row_tile) {
            rt_bf<TILE, TILE, col_l> grad_tile;
            rt_bf<TILE, TILE, col_l> input_tile_values;
            warp::load(
                grad_tile,
                globals.grad_gate_up,
                {0, 0, row_tile, output_tile}
            );
            warp::load(
                input_tile_values,
                globals.input,
                {0, 0, row_tile, input_tile}
            );
            warp::mma_AtB(result, grad_tile, input_tile_values, result);
        }
        warp::store(
            globals.grad_weight,
            result,
            {0, expert, output_tile, input_tile}
        );
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

template <bool SAVE_RAW_OUTPUT>
void launch_forward(
    const at::Tensor &input,
    const at::Tensor &gate_up_weight,
    const at::Tensor &down_weight,
    const at::Tensor &route_probs,
    const at::Tensor &expert_offsets,
    at::Tensor &hidden,
    at::Tensor &raw_output,
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
    check_tensor(raw_output, "raw_output", at::kBFloat16, {rows, D_MODEL});
    check_tensor(output, "output", at::kBFloat16, {rows, D_MODEL});
    TORCH_CHECK(
        gate_up_weight.device() == input.device()
            && down_weight.device() == input.device()
            && route_probs.device() == input.device()
            && expert_offsets.device() == input.device()
            && hidden.device() == input.device()
            && raw_output.device() == input.device()
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
        .raw_output = kittens::py::tensor_to_gl<OutputGlobal>(raw_output),
        .output = kittens::py::tensor_to_gl<OutputGlobal>(output),
        .route_probs = kittens::py::tensor_to_gl<ScaleGlobal>(route_probs),
        .expert_offsets = kittens::py::tensor_to_gl<OffsetGlobal>(expert_offsets),
    };
    gate_swiglu_kernel<<<blocks, THREADS, 0, stream>>>(gate_globals);
    down_kernel<SAVE_RAW_OUTPUT><<<blocks, THREADS, 0, stream>>>(down_globals);
    CUDACHECK(cudaPeekAtLastError());
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
    launch_forward<false>(
        input,
        gate_up_weight,
        down_weight,
        route_probs,
        expert_offsets,
        hidden,
        output,
        output
    );
}

void training_forward(
    const at::Tensor &input,
    const at::Tensor &gate_up_weight,
    const at::Tensor &down_weight,
    const at::Tensor &route_probs,
    const at::Tensor &expert_offsets,
    at::Tensor &hidden,
    at::Tensor &raw_output,
    at::Tensor &output
) {
    launch_forward<true>(
        input,
        gate_up_weight,
        down_weight,
        route_probs,
        expert_offsets,
        hidden,
        raw_output,
        output
    );
}

void backward(
    const at::Tensor &input,
    const at::Tensor &gate_up_weight,
    const at::Tensor &down_weight,
    const at::Tensor &route_probs,
    const at::Tensor &expert_offsets,
    const at::Tensor &hidden,
    const at::Tensor &raw_output,
    const at::Tensor &grad_output,
    at::Tensor &grad_input,
    at::Tensor &grad_gate_up_weight,
    at::Tensor &grad_down_weight,
    at::Tensor &grad_route_probs,
    at::Tensor &grad_unscaled_output,
    at::Tensor &grad_hidden,
    at::Tensor &grad_gate_up
) {
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
    check_tensor(raw_output, "raw_output", at::kBFloat16, {rows, D_MODEL});
    check_tensor(grad_output, "grad_output", at::kBFloat16, {rows, D_MODEL});
    check_tensor(grad_input, "grad_input", at::kBFloat16, {rows, D_MODEL});
    check_tensor(
        grad_gate_up_weight,
        "grad_gate_up_weight",
        at::kBFloat16,
        {NUM_EXPERTS, GATE_UP_DIM, D_MODEL}
    );
    check_tensor(
        grad_down_weight,
        "grad_down_weight",
        at::kBFloat16,
        {NUM_EXPERTS, D_MODEL, HIDDEN_DIM}
    );
    check_tensor(grad_route_probs, "grad_route_probs", at::kBFloat16, {rows});
    check_tensor(
        grad_unscaled_output,
        "grad_unscaled_output",
        at::kBFloat16,
        {rows, D_MODEL}
    );
    check_tensor(grad_hidden, "grad_hidden", at::kBFloat16, {rows, HIDDEN_DIM});
    check_tensor(grad_gate_up, "grad_gate_up", at::kBFloat16, {rows, GATE_UP_DIM});

    const c10::cuda::CUDAGuard device_guard(input.device());
    cudaDeviceProp properties;
    CUDACHECK(cudaGetDeviceProperties(&properties, input.get_device()));
    TORCH_CHECK(
        properties.major == 12 && properties.minor == 0,
        "moe_d128_backward requires SM120"
    );
    const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    const int blocks = properties.multiProcessorCount * 2;

    scale_gradient_kernel<<<rows, D_MODEL, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(grad_output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(raw_output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(route_probs.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(grad_unscaled_output.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(grad_route_probs.data_ptr()),
        rows
    );
    GradHiddenGlobals grad_hidden_globals{
        .grad_output = kittens::py::tensor_to_gl<OutputGlobal>(grad_unscaled_output),
        .weight = kittens::py::tensor_to_gl<DownWeightGlobal>(down_weight),
        .grad_hidden = kittens::py::tensor_to_gl<HiddenGlobal>(grad_hidden),
        .expert_offsets = kittens::py::tensor_to_gl<OffsetGlobal>(expert_offsets),
    };
    GradGateUpGlobals grad_gate_up_globals{
        .input = kittens::py::tensor_to_gl<ActivationGlobal>(input),
        .weight = kittens::py::tensor_to_gl<GateUpWeightGlobal>(gate_up_weight),
        .grad_hidden = kittens::py::tensor_to_gl<HiddenGlobal>(grad_hidden),
        .grad_gate_up = kittens::py::tensor_to_gl<GateUpGradientGlobal>(grad_gate_up),
        .expert_offsets = kittens::py::tensor_to_gl<OffsetGlobal>(expert_offsets),
    };
    GradInputGlobals grad_input_globals{
        .grad_gate_up = kittens::py::tensor_to_gl<GateUpGradientGlobal>(grad_gate_up),
        .weight = kittens::py::tensor_to_gl<GateUpWeightGlobal>(gate_up_weight),
        .grad_input = kittens::py::tensor_to_gl<ActivationGlobal>(grad_input),
        .expert_offsets = kittens::py::tensor_to_gl<OffsetGlobal>(expert_offsets),
    };
    DownWeightGradientGlobals down_weight_globals{
        .grad_output = kittens::py::tensor_to_gl<OutputGlobal>(grad_unscaled_output),
        .hidden = kittens::py::tensor_to_gl<HiddenGlobal>(hidden),
        .grad_weight = kittens::py::tensor_to_gl<DownWeightGlobal>(grad_down_weight),
        .expert_offsets = kittens::py::tensor_to_gl<OffsetGlobal>(expert_offsets),
    };
    GateUpWeightGradientGlobals gate_up_weight_globals{
        .grad_gate_up = kittens::py::tensor_to_gl<GateUpGradientGlobal>(grad_gate_up),
        .input = kittens::py::tensor_to_gl<ActivationGlobal>(input),
        .grad_weight = kittens::py::tensor_to_gl<GateUpWeightGlobal>(grad_gate_up_weight),
        .expert_offsets = kittens::py::tensor_to_gl<OffsetGlobal>(expert_offsets),
    };
    grad_hidden_kernel<<<blocks, THREADS, 0, stream>>>(grad_hidden_globals);
    grad_gate_up_kernel<<<blocks, THREADS, 0, stream>>>(grad_gate_up_globals);
    grad_input_kernel<<<blocks, THREADS, 0, stream>>>(grad_input_globals);
    down_weight_gradient_kernel<<<blocks, THREADS, 0, stream>>>(down_weight_globals);
    gate_up_weight_gradient_kernel<<<blocks, THREADS, 0, stream>>>(gate_up_weight_globals);
    CUDACHECK(cudaPeekAtLastError());
}

}  // namespace chess_engine_4::sm120::moe_d128

void bind_moe_d128_kernels(pybind11::module_ &module) {
    module.def("moe_d128_forward", &chess_engine_4::sm120::moe_d128::forward);
    module.def(
        "moe_d128_training_forward",
        &chess_engine_4::sm120::moe_d128::training_forward
    );
    module.def("moe_d128_backward", &chess_engine_4::sm120::moe_d128::backward);
}
