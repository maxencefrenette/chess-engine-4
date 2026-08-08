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

template <bool SAVE_RAW_OUTPUT>
void launch_sm80_medium_forward(
    const GateSwiGLUGlobals &gate_globals,
    const DownGlobals &down_globals,
    int blocks,
    cudaStream_t stream
) {
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
}

void launch_sm80_medium_backward(
    const GradHiddenGlobals &grad_hidden_globals,
    const GradGateUpGlobals &grad_gate_up_globals,
    const GradInputGlobals &grad_input_globals,
    const DownWeightGradientGlobals &down_weight_globals,
    const GateUpWeightGradientGlobals &gate_up_weight_globals,
    int blocks,
    cudaStream_t stream
) {
    constexpr int shared_bytes = 2 * sizeof(st_bf<64, 64>);
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
}
