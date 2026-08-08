constexpr int WGRAD_OUTPUT_TILE = 32;
constexpr int WGRAD_INPUT_TILE = 64;

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

void launch_sm120_wide_backward(
    const FusedGradGateUpGlobals &fused_grad_gate_up_globals,
    const GradInputGlobals &grad_input_globals,
    const DownWeightGradientGlobals &down_weight_globals,
    const GateUpWeightGradientGlobals &gate_up_weight_globals,
    int blocks,
    cudaStream_t stream
) {
    fused_grad_gate_up_kernel<<<blocks, THREADS, 0, stream>>>(
        fused_grad_gate_up_globals
    );
    grad_input_kernel<<<blocks, THREADS, 0, stream>>>(grad_input_globals);
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
}
