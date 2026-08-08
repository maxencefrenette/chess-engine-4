namespace MOE_NAMESPACE {

#ifndef MOE_COMPUTE_CAPABILITY_MAJOR
#define MOE_COMPUTE_CAPABILITY_MAJOR 12
#define MOE_COMPUTE_CAPABILITY_MINOR 0
#define MOE_ARCHITECTURE_NAME "SM120"
#endif

using namespace kittens;

constexpr int NUM_EXPERTS = 64;
constexpr int D_MODEL = MOE_D_MODEL;
constexpr int HIDDEN_DIM = 2 * D_MODEL;
constexpr int GATE_UP_DIM = 2 * HIDDEN_DIM;
constexpr int TILE = 16;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS = WARPS_PER_BLOCK * 32;
constexpr bool USE_WIDE_SM120_SCHEDULE =
    MOE_COMPUTE_CAPABILITY_MAJOR >= 12 && D_MODEL >= 256;
constexpr bool USE_MEDIUM_SM80_SCHEDULE =
    MOE_COMPUTE_CAPABILITY_MAJOR == 8 && D_MODEL >= 256;
constexpr int ACTIVATION_OUTPUT_TILE =
    USE_WIDE_SM120_SCHEDULE ? 64 : (USE_MEDIUM_SM80_SCHEDULE ? 32 : TILE);
constexpr int BACKWARD_OUTPUT_TILE = USE_MEDIUM_SM80_SCHEDULE ? 32 : TILE;
constexpr int WGRAD_OUTPUT_TILE = USE_WIDE_SM120_SCHEDULE ? 32 : TILE;
constexpr int WGRAD_INPUT_TILE = USE_WIDE_SM120_SCHEDULE ? 64 : 32;
constexpr bool USE_WIDE_WGRAD = USE_WIDE_SM120_SCHEDULE || USE_MEDIUM_SM80_SCHEDULE;

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

struct FusedGradGateUpGlobals {
    OutputGlobal grad_output;
    DownWeightGlobal down_weight;
    ActivationGlobal input;
    GateUpWeightGlobal gate_up_weight;
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
    const int output_tiles = HIDDEN_DIM / ACTIVATION_OUTPUT_TILE;
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
        rt_bf<ACTIVATION_OUTPUT_TILE, TILE> weight_tile;
        rt_fl<TILE, ACTIVATION_OUTPUT_TILE> gate;
        rt_fl<TILE, ACTIVATION_OUTPUT_TILE> up;
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
                {
                    0,
                    expert,
                    HIDDEN_DIM / ACTIVATION_OUTPUT_TILE + hidden_tile,
                    reduction_tile
                }
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
    const int output_tiles = D_MODEL / ACTIVATION_OUTPUT_TILE;
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
        rt_bf<ACTIVATION_OUTPUT_TILE, TILE> weight_tile;
        rt_fl<TILE, ACTIVATION_OUTPUT_TILE> output_tile_values;
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
        col_vec<rt_fl<TILE, ACTIVATION_OUTPUT_TILE>> route_scale;
        warp::load(route_scale, globals.route_probs, {row_tile});
        warp::mul_row(output_tile_values, output_tile_values, route_scale);
        warp::store(globals.output, output_tile_values, {0, 0, row_tile, output_tile});
    }
}

__global__ void gate_swiglu_sm80_medium_kernel(
    const GateSwiGLUGlobals globals
) {
    constexpr int BLOCK_TILE = 64;
    using Workers = group<WARPS_PER_BLOCK>;
    extern __shared__ alignment_dummy shared[];
    shared_allocator allocator(reinterpret_cast<int *>(shared));
    auto &input_shared = allocator.allocate<st_bf<BLOCK_TILE, BLOCK_TILE>>();
    auto &weight_shared = allocator.allocate<st_bf<BLOCK_TILE, BLOCK_TILE>>();

    const int warp = threadIdx.x / 32;
    const int row_subtile = warp / 2;
    const int column_subtile = warp % 2;
    const int row_blocks = globals.expert_offsets[{NUM_EXPERTS}] / BLOCK_TILE;
    constexpr int HIDDEN_BLOCKS = HIDDEN_DIM / BLOCK_TILE;
    const int total_tasks = row_blocks * HIDDEN_BLOCKS;

    for (int task = blockIdx.x; task < total_tasks; task += gridDim.x) {
        const int row_block = task / HIDDEN_BLOCKS;
        const int hidden_block = task % HIDDEN_BLOCKS;
        const int expert = expert_for_row(
            globals.expert_offsets,
            row_block * BLOCK_TILE
        );
        rt_fl<TILE, 2 * TILE> gate;
        rt_fl<TILE, 2 * TILE> up;
        warp::zero(gate);
        warp::zero(up);

        #pragma unroll
        for (int reduction_block = 0; reduction_block < D_MODEL / BLOCK_TILE; ++reduction_block) {
            Workers::load(
                input_shared,
                globals.input,
                {0, 0, row_block, reduction_block}
            );
            Workers::load(
                weight_shared,
                globals.weight,
                {0, expert, hidden_block, reduction_block}
            );
            __syncthreads();
            #pragma unroll
            for (int reduction_subtile = 0; reduction_subtile < BLOCK_TILE / TILE; ++reduction_subtile) {
                rt_bf<TILE, TILE> input_tile;
                rt_bf<2 * TILE, TILE> weight_tile;
                auto input_subtile = input_shared.template subtile<TILE, TILE>(
                    {row_subtile, reduction_subtile}
                );
                auto weight_subtile = weight_shared.template subtile<2 * TILE, TILE>(
                    {column_subtile, reduction_subtile}
                );
                warp::load(input_tile, input_subtile);
                warp::load(weight_tile, weight_subtile);
                warp::mma_ABt(gate, input_tile, weight_tile, gate);
            }
            __syncthreads();
            Workers::load(
                weight_shared,
                globals.weight,
                {
                    0,
                    expert,
                    HIDDEN_BLOCKS + hidden_block,
                    reduction_block
                }
            );
            __syncthreads();
            #pragma unroll
            for (int reduction_subtile = 0; reduction_subtile < BLOCK_TILE / TILE; ++reduction_subtile) {
                rt_bf<TILE, TILE> input_tile;
                rt_bf<2 * TILE, TILE> weight_tile;
                auto input_subtile = input_shared.template subtile<TILE, TILE>(
                    {row_subtile, reduction_subtile}
                );
                auto weight_subtile = weight_shared.template subtile<2 * TILE, TILE>(
                    {column_subtile, reduction_subtile}
                );
                warp::load(input_tile, input_subtile);
                warp::load(weight_tile, weight_subtile);
                warp::mma_ABt(up, input_tile, weight_tile, up);
            }
            __syncthreads();
        }
        warp::apply(
            gate,
            gate,
            [] __device__ (int, int, float value) {
                return value / (1.0f + __expf(-value));
            }
        );
        warp::mul(gate, gate, up);
        warp::store(
            globals.hidden,
            gate,
            {
                0,
                0,
                row_block * (BLOCK_TILE / TILE) + row_subtile,
                hidden_block * 2 + column_subtile
            }
        );
    }
}

template <bool SAVE_RAW_OUTPUT>
__global__ void down_sm80_medium_kernel(const DownGlobals globals) {
    constexpr int BLOCK_TILE = 64;
    using Workers = group<WARPS_PER_BLOCK>;
    extern __shared__ alignment_dummy shared[];
    shared_allocator allocator(reinterpret_cast<int *>(shared));
    auto &hidden_shared = allocator.allocate<st_bf<BLOCK_TILE, BLOCK_TILE>>();
    auto &weight_shared = allocator.allocate<st_bf<BLOCK_TILE, BLOCK_TILE>>();

    const int warp = threadIdx.x / 32;
    const int row_subtile = warp / 2;
    const int column_subtile = warp % 2;
    const int row_blocks = globals.expert_offsets[{NUM_EXPERTS}] / BLOCK_TILE;
    constexpr int OUTPUT_BLOCKS = D_MODEL / BLOCK_TILE;
    const int total_tasks = row_blocks * OUTPUT_BLOCKS;

    for (int task = blockIdx.x; task < total_tasks; task += gridDim.x) {
        const int row_block = task / OUTPUT_BLOCKS;
        const int output_block = task % OUTPUT_BLOCKS;
        const int expert = expert_for_row(
            globals.expert_offsets,
            row_block * BLOCK_TILE
        );
        rt_fl<TILE, 2 * TILE> result;
        warp::zero(result);
        #pragma unroll
        for (int reduction_block = 0; reduction_block < HIDDEN_DIM / BLOCK_TILE; ++reduction_block) {
            Workers::load(
                hidden_shared,
                globals.hidden,
                {0, 0, row_block, reduction_block}
            );
            Workers::load(
                weight_shared,
                globals.weight,
                {0, expert, output_block, reduction_block}
            );
            __syncthreads();
            #pragma unroll
            for (int reduction_subtile = 0; reduction_subtile < BLOCK_TILE / TILE; ++reduction_subtile) {
                rt_bf<TILE, TILE> hidden_tile;
                rt_bf<2 * TILE, TILE> weight_tile;
                auto hidden_subtile = hidden_shared.template subtile<TILE, TILE>(
                    {row_subtile, reduction_subtile}
                );
                auto weight_subtile = weight_shared.template subtile<2 * TILE, TILE>(
                    {column_subtile, reduction_subtile}
                );
                warp::load(hidden_tile, hidden_subtile);
                warp::load(weight_tile, weight_subtile);
                warp::mma_ABt(result, hidden_tile, weight_tile, result);
            }
            __syncthreads();
        }
        const int row_tile = row_block * (BLOCK_TILE / TILE) + row_subtile;
        const int output_tile = output_block * 2 + column_subtile;
        if constexpr (SAVE_RAW_OUTPUT) {
            warp::store(globals.raw_output, result, {0, 0, row_tile, output_tile});
        }
        col_vec<rt_fl<TILE, 2 * TILE>> route_scale;
        warp::load(route_scale, globals.route_probs, {row_tile});
        warp::mul_row(result, result, route_scale);
        warp::store(globals.output, result, {0, 0, row_tile, output_tile});
    }
}

template <int REDUCTION_DIM, int OUTPUT_DIM, typename InputGlobal, typename WeightGlobal, typename ResultGlobal>
__global__ void grouped_ab_sm80_medium_kernel(
    const InputGlobal input,
    const WeightGlobal weight,
    const ResultGlobal result_global,
    const OffsetGlobal expert_offsets
) {
    constexpr int BLOCK_TILE = 64;
    using Workers = group<WARPS_PER_BLOCK>;
    extern __shared__ alignment_dummy shared[];
    shared_allocator allocator(reinterpret_cast<int *>(shared));
    auto &input_shared = allocator.allocate<st_bf<BLOCK_TILE, BLOCK_TILE>>();
    auto &weight_shared = allocator.allocate<st_bf<BLOCK_TILE, BLOCK_TILE>>();

    const int warp = threadIdx.x / 32;
    const int row_subtile = warp / 2;
    const int column_subtile = warp % 2;
    const int row_blocks = expert_offsets[{NUM_EXPERTS}] / BLOCK_TILE;
    constexpr int OUTPUT_BLOCKS = OUTPUT_DIM / BLOCK_TILE;
    const int total_tasks = row_blocks * OUTPUT_BLOCKS;

    for (int task = blockIdx.x; task < total_tasks; task += gridDim.x) {
        const int row_block = task / OUTPUT_BLOCKS;
        const int output_block = task % OUTPUT_BLOCKS;
        const int expert = expert_for_row(expert_offsets, row_block * BLOCK_TILE);
        rt_fl<TILE, 2 * TILE> result;
        warp::zero(result);
        #pragma unroll
        for (int reduction_block = 0; reduction_block < REDUCTION_DIM / BLOCK_TILE; ++reduction_block) {
            Workers::load(
                input_shared,
                input,
                {0, 0, row_block, reduction_block}
            );
            Workers::load(
                weight_shared,
                weight,
                {0, expert, reduction_block, output_block}
            );
            __syncthreads();
            #pragma unroll
            for (int reduction_subtile = 0; reduction_subtile < BLOCK_TILE / TILE; ++reduction_subtile) {
                rt_bf<TILE, TILE> input_tile;
                rt_bf<TILE, 2 * TILE, col_l> weight_tile;
                auto input_subtile = input_shared.template subtile<TILE, TILE>(
                    {row_subtile, reduction_subtile}
                );
                auto weight_subtile = weight_shared.template subtile<TILE, 2 * TILE>(
                    {reduction_subtile, column_subtile}
                );
                warp::load(input_tile, input_subtile);
                warp::load(weight_tile, weight_subtile);
                warp::mma_AB(result, input_tile, weight_tile, result);
            }
            __syncthreads();
        }
        warp::store(
            result_global,
            result,
            {
                0,
                0,
                row_block * (BLOCK_TILE / TILE) + row_subtile,
                output_block * 2 + column_subtile
            }
        );
    }
}

template <int OUTPUT_DIM, int INPUT_DIM, typename GradientGlobal, typename InputGlobal, typename ResultGlobal>
__global__ void grouped_atb_sm80_medium_kernel(
    const GradientGlobal gradient,
    const InputGlobal input,
    const ResultGlobal result_global,
    const OffsetGlobal expert_offsets
) {
    constexpr int BLOCK_TILE = 64;
    using Workers = group<WARPS_PER_BLOCK>;
    extern __shared__ alignment_dummy shared[];
    shared_allocator allocator(reinterpret_cast<int *>(shared));
    auto &gradient_shared = allocator.allocate<st_bf<BLOCK_TILE, BLOCK_TILE>>();
    auto &input_shared = allocator.allocate<st_bf<BLOCK_TILE, BLOCK_TILE>>();

    const int warp = threadIdx.x / 32;
    const int output_subtile = warp / 2;
    const int input_subtile_index = warp % 2;
    constexpr int OUTPUT_BLOCKS = OUTPUT_DIM / BLOCK_TILE;
    constexpr int INPUT_BLOCKS = INPUT_DIM / BLOCK_TILE;
    constexpr int TASKS_PER_EXPERT = OUTPUT_BLOCKS * INPUT_BLOCKS;
    constexpr int TOTAL_TASKS = NUM_EXPERTS * TASKS_PER_EXPERT;

    for (int task = blockIdx.x; task < TOTAL_TASKS; task += gridDim.x) {
        const int expert = task / TASKS_PER_EXPERT;
        const int expert_task = task % TASKS_PER_EXPERT;
        const int output_block = expert_task / INPUT_BLOCKS;
        const int input_block = expert_task % INPUT_BLOCKS;
        const int first_row_block = expert_offsets[{expert}] / BLOCK_TILE;
        const int end_row_block = expert_offsets[{expert + 1}] / BLOCK_TILE;
        rt_fl<TILE, 2 * TILE> result;
        warp::zero(result);
        for (int row_block = first_row_block; row_block < end_row_block; ++row_block) {
            Workers::load(
                gradient_shared,
                gradient,
                {0, 0, row_block, output_block}
            );
            Workers::load(
                input_shared,
                input,
                {0, 0, row_block, input_block}
            );
            __syncthreads();
            #pragma unroll
            for (int reduction_subtile = 0; reduction_subtile < BLOCK_TILE / TILE; ++reduction_subtile) {
                rt_bf<TILE, TILE, col_l> gradient_tile;
                rt_bf<TILE, 2 * TILE, col_l> input_tile;
                auto gradient_subtile = gradient_shared.template subtile<TILE, TILE>(
                    {reduction_subtile, output_subtile}
                );
                auto input_subtile = input_shared.template subtile<TILE, 2 * TILE>(
                    {reduction_subtile, input_subtile_index}
                );
                warp::load(gradient_tile, gradient_subtile);
                warp::load(input_tile, input_subtile);
                warp::mma_AtB(result, gradient_tile, input_tile, result);
            }
            __syncthreads();
        }
        warp::store(
            result_global,
            result,
            {
                0,
                expert,
                output_block * 4 + output_subtile,
                input_block * 2 + input_subtile_index
            }
        );
    }
}

__global__ void scale_gradient_kernel(
    const __nv_bfloat16 *grad_output,
    const __nv_bfloat16 *raw_output,
    const __nv_bfloat16 *route_probs,
    __nv_bfloat16 *grad_unscaled_output,
    __nv_bfloat16 *grad_route_probs,
    const int *expert_offsets,
    int rows
) {
    const int row = blockIdx.x;
    if (row >= rows || row >= expert_offsets[NUM_EXPERTS]) {
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
    constexpr int WARPS_PER_ROW = D_MODEL / 32;
    __shared__ float warp_sums[WARPS_PER_ROW];
    if (column % 32 == 0) {
        warp_sums[column / 32] = route_grad;
    }
    __syncthreads();
    if (column < 32) {
        route_grad = column < WARPS_PER_ROW ? warp_sums[column] : 0.0f;
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
    const int output_tiles = HIDDEN_DIM / BACKWARD_OUTPUT_TILE;
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
        rt_bf<TILE, BACKWARD_OUTPUT_TILE, col_l> weight_tile;
        rt_fl<TILE, BACKWARD_OUTPUT_TILE> result;
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
    const int hidden_tiles = HIDDEN_DIM / BACKWARD_OUTPUT_TILE;
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
        rt_bf<BACKWARD_OUTPUT_TILE, TILE> weight_tile;
        rt_bf<TILE, BACKWARD_OUTPUT_TILE> grad_hidden;
        rt_fl<TILE, BACKWARD_OUTPUT_TILE> gate;
        rt_fl<TILE, BACKWARD_OUTPUT_TILE> up;
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
                {
                    0,
                    expert,
                    HIDDEN_DIM / BACKWARD_OUTPUT_TILE + hidden_tile,
                    reduction_tile
                }
            );
            warp::mma_ABt(up, input_tile, weight_tile, up);
        }
        warp::load(grad_hidden, globals.grad_hidden, {0, 0, row_tile, hidden_tile});
        rt_fl<TILE, BACKWARD_OUTPUT_TILE> grad_gate;
        rt_fl<TILE, BACKWARD_OUTPUT_TILE> grad_up;
        #pragma unroll
        for (int column_tile = 0; column_tile < BACKWARD_OUTPUT_TILE / TILE; ++column_tile) {
            #pragma unroll
            for (int slot = 0; slot < rt_fl<TILE, TILE>::packed_per_tile; ++slot) {
                const float gate_x = gate.tiles[0][column_tile].data[slot].x;
                const float gate_y = gate.tiles[0][column_tile].data[slot].y;
                const float up_x = up.tiles[0][column_tile].data[slot].x;
                const float up_y = up.tiles[0][column_tile].data[slot].y;
                const float grad_x = __bfloat162float(
                    grad_hidden.tiles[0][column_tile].data[slot].x
                );
                const float grad_y = __bfloat162float(
                    grad_hidden.tiles[0][column_tile].data[slot].y
                );
                const float sigmoid_x = 1.0f / (1.0f + __expf(-gate_x));
                const float sigmoid_y = 1.0f / (1.0f + __expf(-gate_y));
                grad_gate.tiles[0][column_tile].data[slot] = make_float2(
                    grad_x * up_x * sigmoid_x * (1.0f + gate_x * (1.0f - sigmoid_x)),
                    grad_y * up_y * sigmoid_y * (1.0f + gate_y * (1.0f - sigmoid_y))
                );
                grad_up.tiles[0][column_tile].data[slot] = make_float2(
                    grad_x * gate_x * sigmoid_x,
                    grad_y * gate_y * sigmoid_y
                );
            }
        }
        warp::store(
            globals.grad_gate_up,
            grad_gate,
            {0, 0, row_tile, hidden_tile}
        );
        warp::store(
            globals.grad_gate_up,
            grad_up,
            {0, 0, row_tile, HIDDEN_DIM / BACKWARD_OUTPUT_TILE + hidden_tile}
        );
    }
}

__global__ void grad_gate_up_sm80_medium_kernel(
    const GradGateUpGlobals globals
) {
    constexpr int BLOCK_TILE = 64;
    using Workers = group<WARPS_PER_BLOCK>;
    extern __shared__ alignment_dummy shared[];
    shared_allocator allocator(reinterpret_cast<int *>(shared));
    auto &input_shared = allocator.allocate<st_bf<BLOCK_TILE, BLOCK_TILE>>();
    auto &weight_shared = allocator.allocate<st_bf<BLOCK_TILE, BLOCK_TILE>>();

    const int warp = threadIdx.x / 32;
    const int row_subtile = warp / 2;
    const int column_subtile = warp % 2;
    const int row_blocks = globals.expert_offsets[{NUM_EXPERTS}] / BLOCK_TILE;
    constexpr int HIDDEN_BLOCKS = HIDDEN_DIM / BLOCK_TILE;
    const int total_tasks = row_blocks * HIDDEN_BLOCKS;

    for (int task = blockIdx.x; task < total_tasks; task += gridDim.x) {
        const int row_block = task / HIDDEN_BLOCKS;
        const int hidden_block = task % HIDDEN_BLOCKS;
        const int expert = expert_for_row(
            globals.expert_offsets,
            row_block * BLOCK_TILE
        );
        rt_fl<TILE, 2 * TILE> gate;
        rt_fl<TILE, 2 * TILE> up;
        warp::zero(gate);
        warp::zero(up);

        #pragma unroll
        for (int reduction_block = 0; reduction_block < D_MODEL / BLOCK_TILE; ++reduction_block) {
            Workers::load(
                input_shared,
                globals.input,
                {0, 0, row_block, reduction_block}
            );
            Workers::load(
                weight_shared,
                globals.weight,
                {0, expert, hidden_block, reduction_block}
            );
            __syncthreads();
            #pragma unroll
            for (int reduction_subtile = 0; reduction_subtile < BLOCK_TILE / TILE; ++reduction_subtile) {
                rt_bf<TILE, TILE> input_tile;
                rt_bf<2 * TILE, TILE> weight_tile;
                warp::load(
                    input_tile,
                    input_shared.template subtile<TILE, TILE>(
                        {row_subtile, reduction_subtile}
                    )
                );
                warp::load(
                    weight_tile,
                    weight_shared.template subtile<2 * TILE, TILE>(
                        {column_subtile, reduction_subtile}
                    )
                );
                warp::mma_ABt(gate, input_tile, weight_tile, gate);
            }
            __syncthreads();
            Workers::load(
                weight_shared,
                globals.weight,
                {
                    0,
                    expert,
                    HIDDEN_BLOCKS + hidden_block,
                    reduction_block
                }
            );
            __syncthreads();
            #pragma unroll
            for (int reduction_subtile = 0; reduction_subtile < BLOCK_TILE / TILE; ++reduction_subtile) {
                rt_bf<TILE, TILE> input_tile;
                rt_bf<2 * TILE, TILE> weight_tile;
                warp::load(
                    input_tile,
                    input_shared.template subtile<TILE, TILE>(
                        {row_subtile, reduction_subtile}
                    )
                );
                warp::load(
                    weight_tile,
                    weight_shared.template subtile<2 * TILE, TILE>(
                        {column_subtile, reduction_subtile}
                    )
                );
                warp::mma_ABt(up, input_tile, weight_tile, up);
            }
            __syncthreads();
        }

        rt_bf<TILE, 2 * TILE> grad_hidden;
        warp::load(
            grad_hidden,
            globals.grad_hidden,
            {
                0,
                0,
                row_block * (BLOCK_TILE / TILE) + row_subtile,
                hidden_block * 2 + column_subtile
            }
        );
        rt_fl<TILE, 2 * TILE> grad_gate;
        rt_fl<TILE, 2 * TILE> grad_up;
        #pragma unroll
        for (int column_tile = 0; column_tile < 2; ++column_tile) {
            #pragma unroll
            for (int slot = 0; slot < rt_fl<TILE, TILE>::packed_per_tile; ++slot) {
                const float gate_x = gate.tiles[0][column_tile].data[slot].x;
                const float gate_y = gate.tiles[0][column_tile].data[slot].y;
                const float up_x = up.tiles[0][column_tile].data[slot].x;
                const float up_y = up.tiles[0][column_tile].data[slot].y;
                const float grad_x = __bfloat162float(
                    grad_hidden.tiles[0][column_tile].data[slot].x
                );
                const float grad_y = __bfloat162float(
                    grad_hidden.tiles[0][column_tile].data[slot].y
                );
                const float sigmoid_x = 1.0f / (1.0f + __expf(-gate_x));
                const float sigmoid_y = 1.0f / (1.0f + __expf(-gate_y));
                grad_gate.tiles[0][column_tile].data[slot] = make_float2(
                    grad_x * up_x * sigmoid_x * (1.0f + gate_x * (1.0f - sigmoid_x)),
                    grad_y * up_y * sigmoid_y * (1.0f + gate_y * (1.0f - sigmoid_y))
                );
                grad_up.tiles[0][column_tile].data[slot] = make_float2(
                    grad_x * gate_x * sigmoid_x,
                    grad_y * gate_y * sigmoid_y
                );
            }
        }
        const int row_tile = row_block * (BLOCK_TILE / TILE) + row_subtile;
        const int hidden_tile = hidden_block * 2 + column_subtile;
        warp::store(
            globals.grad_gate_up,
            grad_gate,
            {0, 0, row_tile, hidden_tile}
        );
        warp::store(
            globals.grad_gate_up,
            grad_up,
            {0, 0, row_tile, HIDDEN_DIM / (2 * TILE) + hidden_tile}
        );
    }
}

__global__ void fused_grad_gate_up_kernel(const FusedGradGateUpGlobals globals) {
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
        rt_bf<TILE, TILE> gate_up_weight_tile;
        rt_fl<TILE, TILE> gate;
        rt_fl<TILE, TILE> up;
        warp::zero(gate);
        warp::zero(up);
        #pragma unroll
        for (int reduction_tile = 0; reduction_tile < D_MODEL / TILE; ++reduction_tile) {
            warp::load(input_tile, globals.input, {0, 0, row_tile, reduction_tile});
            warp::load(
                gate_up_weight_tile,
                globals.gate_up_weight,
                {0, expert, hidden_tile, reduction_tile}
            );
            warp::mma_ABt(gate, input_tile, gate_up_weight_tile, gate);
            warp::load(
                gate_up_weight_tile,
                globals.gate_up_weight,
                {0, expert, HIDDEN_DIM / TILE + hidden_tile, reduction_tile}
            );
            warp::mma_ABt(up, input_tile, gate_up_weight_tile, up);
        }

        rt_bf<TILE, TILE> grad_output_tile;
        rt_bf<TILE, TILE, col_l> down_weight_tile;
        rt_fl<TILE, TILE> grad_hidden;
        warp::zero(grad_hidden);
        #pragma unroll
        for (int reduction_tile = 0; reduction_tile < D_MODEL / TILE; ++reduction_tile) {
            warp::load(
                grad_output_tile,
                globals.grad_output,
                {0, 0, row_tile, reduction_tile}
            );
            warp::load(
                down_weight_tile,
                globals.down_weight,
                {0, expert, reduction_tile, hidden_tile}
            );
            warp::mma_AB(grad_hidden, grad_output_tile, down_weight_tile, grad_hidden);
        }

        rt_fl<TILE, TILE> grad_gate;
        rt_fl<TILE, TILE> grad_up;
        #pragma unroll
        for (int slot = 0; slot < rt_fl<TILE, TILE>::packed_per_tile; ++slot) {
            const float gate_x = gate.tiles[0][0].data[slot].x;
            const float gate_y = gate.tiles[0][0].data[slot].y;
            const float up_x = up.tiles[0][0].data[slot].x;
            const float up_y = up.tiles[0][0].data[slot].y;
            const float grad_x = grad_hidden.tiles[0][0].data[slot].x;
            const float grad_y = grad_hidden.tiles[0][0].data[slot].y;
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
    const int output_tiles = D_MODEL / BACKWARD_OUTPUT_TILE;
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
        rt_bf<TILE, BACKWARD_OUTPUT_TILE, col_l> weight_tile;
        rt_fl<TILE, BACKWARD_OUTPUT_TILE> result;
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

template <typename GradientGlobal, typename ActivationGlobal, typename WeightGlobal>
__device__ void wide_weight_gradient(
    const GradientGlobal &gradient,
    const ActivationGlobal &activation,
    const WeightGlobal &grad_weight,
    const OffsetGlobal &expert_offsets,
    int output_blocks,
    int input_blocks
) {
    const int warp = threadIdx.x / 32;
    const int tasks_per_expert = output_blocks * input_blocks;
    const int total_tasks = NUM_EXPERTS * tasks_per_expert;
    for (
        int task = blockIdx.x * WARPS_PER_BLOCK + warp;
        task < total_tasks;
        task += gridDim.x * WARPS_PER_BLOCK
    ) {
        const int expert = task / tasks_per_expert;
        const int expert_task = task % tasks_per_expert;
        const int output_block = expert_task / input_blocks;
        const int input_block = expert_task % input_blocks;
        const int first_row_tile = expert_offsets[{expert}] / TILE;
        const int end_row_tile = expert_offsets[{expert + 1}] / TILE;
        rt_bf<TILE, WGRAD_OUTPUT_TILE, col_l> gradient_tile;
        rt_bf<TILE, WGRAD_INPUT_TILE, col_l> activation_tile;
        rt_fl<WGRAD_OUTPUT_TILE, WGRAD_INPUT_TILE> result;
        warp::zero(result);
        for (int row_tile = first_row_tile; row_tile < end_row_tile; ++row_tile) {
            warp::load(
                gradient_tile,
                gradient,
                {0, 0, row_tile, output_block}
            );
            warp::load(
                activation_tile,
                activation,
                {0, 0, row_tile, input_block}
            );
            warp::mma_AtB(result, gradient_tile, activation_tile, result);
        }
        warp::store(
            grad_weight,
            result,
            {0, expert, output_block, input_block}
        );
    }
}

__global__ void down_weight_gradient_wide_kernel(
    const DownWeightGradientGlobals globals
) {
    wide_weight_gradient(
        globals.grad_output,
        globals.hidden,
        globals.grad_weight,
        globals.expert_offsets,
        D_MODEL / WGRAD_OUTPUT_TILE,
        HIDDEN_DIM / WGRAD_INPUT_TILE
    );
}

__global__ void gate_up_weight_gradient_wide_kernel(
    const GateUpWeightGradientGlobals globals
) {
    wide_weight_gradient(
        globals.grad_gate_up,
        globals.input,
        globals.grad_weight,
        globals.expert_offsets,
        GATE_UP_DIM / WGRAD_OUTPUT_TILE,
        D_MODEL / WGRAD_INPUT_TILE
    );
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
    TORCH_CHECK(input.size(1) == D_MODEL, "input width must be ", D_MODEL);
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
        properties.major == MOE_COMPUTE_CAPABILITY_MAJOR
            && properties.minor == MOE_COMPUTE_CAPABILITY_MINOR,
        MOE_ARCHITECTURE_NAME " MoE forward requires compute capability ",
        MOE_COMPUTE_CAPABILITY_MAJOR,
        ".",
        MOE_COMPUTE_CAPABILITY_MINOR,
        ", got ",
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
    if constexpr (USE_MEDIUM_SM80_SCHEDULE) {
        constexpr int shared_bytes = 2 * sizeof(st_bf<64, 64>);
        gate_swiglu_sm80_medium_kernel<<<blocks, THREADS, shared_bytes, stream>>>(
            gate_globals
        );
        down_sm80_medium_kernel<SAVE_RAW_OUTPUT><<<
            blocks,
            THREADS,
            shared_bytes,
            stream
        >>>(down_globals);
    } else {
        gate_swiglu_kernel<<<blocks, THREADS, 0, stream>>>(gate_globals);
        down_kernel<SAVE_RAW_OUTPUT><<<blocks, THREADS, 0, stream>>>(down_globals);
    }
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
        properties.major == MOE_COMPUTE_CAPABILITY_MAJOR
            && properties.minor == MOE_COMPUTE_CAPABILITY_MINOR,
        MOE_ARCHITECTURE_NAME " MoE backward requires compute capability ",
        MOE_COMPUTE_CAPABILITY_MAJOR,
        ".",
        MOE_COMPUTE_CAPABILITY_MINOR
    );
    const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    const int blocks = properties.multiProcessorCount * 2;

    scale_gradient_kernel<<<rows, D_MODEL, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(grad_output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(raw_output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(route_probs.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(grad_unscaled_output.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(grad_route_probs.data_ptr()),
        reinterpret_cast<const int *>(expert_offsets.data_ptr()),
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
    FusedGradGateUpGlobals fused_grad_gate_up_globals{
        .grad_output = kittens::py::tensor_to_gl<OutputGlobal>(grad_unscaled_output),
        .down_weight = kittens::py::tensor_to_gl<DownWeightGlobal>(down_weight),
        .input = kittens::py::tensor_to_gl<ActivationGlobal>(input),
        .gate_up_weight = kittens::py::tensor_to_gl<GateUpWeightGlobal>(gate_up_weight),
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
    constexpr int shared_bytes = 2 * sizeof(st_bf<64, 64>);
    if constexpr (USE_WIDE_SM120_SCHEDULE) {
        fused_grad_gate_up_kernel<<<blocks, THREADS, 0, stream>>>(
            fused_grad_gate_up_globals
        );
    } else if constexpr (USE_MEDIUM_SM80_SCHEDULE) {
        grouped_ab_sm80_medium_kernel<D_MODEL, HIDDEN_DIM><<<
            blocks,
            THREADS,
            shared_bytes,
            stream
        >>>(
            grad_hidden_globals.grad_output,
            grad_hidden_globals.weight,
            grad_hidden_globals.grad_hidden,
            grad_hidden_globals.expert_offsets
        );
        grad_gate_up_sm80_medium_kernel<<<
            blocks,
            THREADS,
            shared_bytes,
            stream
        >>>(grad_gate_up_globals);
    } else {
        grad_hidden_kernel<<<blocks, THREADS, 0, stream>>>(grad_hidden_globals);
        grad_gate_up_kernel<<<blocks, THREADS, 0, stream>>>(grad_gate_up_globals);
    }
    if constexpr (USE_MEDIUM_SM80_SCHEDULE) {
        grouped_ab_sm80_medium_kernel<GATE_UP_DIM, D_MODEL><<<
            blocks,
            THREADS,
            shared_bytes,
            stream
        >>>(
            grad_input_globals.grad_gate_up,
            grad_input_globals.weight,
            grad_input_globals.grad_input,
            grad_input_globals.expert_offsets
        );
    } else {
        grad_input_kernel<<<blocks, THREADS, 0, stream>>>(grad_input_globals);
    }
    if constexpr (USE_MEDIUM_SM80_SCHEDULE) {
        grouped_atb_sm80_medium_kernel<D_MODEL, HIDDEN_DIM><<<
            blocks,
            THREADS,
            shared_bytes,
            stream
        >>>(
            down_weight_globals.grad_output,
            down_weight_globals.hidden,
            down_weight_globals.grad_weight,
            down_weight_globals.expert_offsets
        );
        grouped_atb_sm80_medium_kernel<GATE_UP_DIM, D_MODEL><<<
            blocks,
            THREADS,
            shared_bytes,
            stream
        >>>(
            gate_up_weight_globals.grad_gate_up,
            gate_up_weight_globals.input,
            gate_up_weight_globals.grad_weight,
            gate_up_weight_globals.expert_offsets
        );
    } else if constexpr (USE_WIDE_WGRAD) {
        constexpr int down_tasks = NUM_EXPERTS
            * (D_MODEL / WGRAD_OUTPUT_TILE)
            * (HIDDEN_DIM / WGRAD_INPUT_TILE);
        constexpr int gate_up_tasks = NUM_EXPERTS
            * (GATE_UP_DIM / WGRAD_OUTPUT_TILE)
            * (D_MODEL / WGRAD_INPUT_TILE);
        constexpr int down_blocks =
            (down_tasks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        constexpr int gate_up_blocks =
            (gate_up_tasks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        down_weight_gradient_wide_kernel<<<down_blocks, THREADS, 0, stream>>>(
            down_weight_globals
        );
        gate_up_weight_gradient_wide_kernel<<<gate_up_blocks, THREADS, 0, stream>>>(
            gate_up_weight_globals
        );
    } else {
        down_weight_gradient_kernel<<<blocks, THREADS, 0, stream>>>(down_weight_globals);
        gate_up_weight_gradient_kernel<<<blocks, THREADS, 0, stream>>>(
            gate_up_weight_globals
        );
    }
    CUDACHECK(cudaPeekAtLastError());
}

}  // namespace MOE_NAMESPACE
