#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include "../../../third_party/ThunderKittens/kernels/gemm/common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

// ThunderKittens' reference file contains the Blackwell MXFP8 device kernel
// and quantizer. Capture its definitions in this extension module.
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, module) \
    static void bind_thunderkittens_reference(pybind11::module_ &module)
#define _C _chess_engine_4_kernels_reference
#include "../../../third_party/ThunderKittens/kernels/gemm/mxfp8_b200/mxfp8_b200_gemm.cu"
#undef _C
#undef PYBIND11_MODULE

// Small and medium models do not amortize MXFP8 quantization, so use TK's
// native Blackwell BF16 GEMM for those shapes.
namespace tk_bf16_gemm {
#define main thunderkittens_bf16_benchmark_main
#include "../../../third_party/ThunderKittens/kernels/gemm/bf16_b200/bf16_b200_gemm.cu"
#undef main
}  // namespace tk_bf16_gemm

namespace chess_engine_4::dense {

class PrimaryContextGuard {
public:
    explicit PrimaryContextGuard(int device_index) : device_index_(device_index) {
        CUcontext current = nullptr;
        CUCHECK(cuCtxGetCurrent(&current));
        if (current != nullptr) {
            return;
        }
        CUdevice device;
        CUCHECK(cuDeviceGet(&device, device_index));
        CUCHECK(cuDevicePrimaryCtxRetain(&context_, device));
        CUCHECK(cuCtxPushCurrent(context_));
        pushed_ = true;
    }

    ~PrimaryContextGuard() {
        if (!pushed_) {
            return;
        }
        CUcontext popped;
        CUCHECK(cuCtxPopCurrent(&popped));
        CUCHECK(cuDevicePrimaryCtxRelease(static_cast<CUdevice>(device_index_)));
    }

    PrimaryContextGuard(const PrimaryContextGuard &) = delete;
    PrimaryContextGuard &operator=(const PrimaryContextGuard &) = delete;

private:
    int device_index_ = 0;
    CUcontext context_ = nullptr;
    bool pushed_ = false;
};

using Mxfp8GemmConfig = mxfp8_gemm::config<256, 6, 16, 12, 4, false>;
using Mxfp8GemmGlobals = mxfp8_gemm::globals<Mxfp8GemmConfig>;

constexpr int THREADS = 256;
constexpr int RMS_ROWS_PER_BLOCK = 8;

template <int OUTPUT_TILE, int REDUCTION_TILE>
using Bf16GemmConfig = tk_bf16_gemm::config<
    256,
    OUTPUT_TILE,
    REDUCTION_TILE,
    4,
    true,
    2,
    OUTPUT_TILE / 32
>;

template <int OUTPUT_TILE, int REDUCTION_TILE>
void launch_bf16_gemm(
    const at::Tensor &left,
    const at::Tensor &right,
    at::Tensor &output,
    cudaStream_t stream
) {
    using Config = Bf16GemmConfig<OUTPUT_TILE, REDUCTION_TILE>;
    using Globals = tk_bf16_gemm::globals<Config>;
    Globals globals{
        .a = kittens::py::tensor_to_gl<typename Globals::a_gl>(left),
        .b = kittens::py::tensor_to_gl<typename Globals::b_gl>(right),
        .d = kittens::py::tensor_to_gl<typename Globals::d_gl>(output),
    };
    CUDACHECK(cudaFuncSetAttribute(
        tk_bf16_gemm::kernel<Config>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        globals.dynamic_shared_memory()
    ));
    LaunchConfig<true, true> launch_config(
        globals.grid(),
        globals.block(),
        globals.dynamic_shared_memory(),
        stream,
        Config::CLUSTER_SIZE
    );
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        tk_bf16_gemm::kernel<Config>,
        globals
    ));
}

template <int OUTPUT_TILE>
void dispatch_bf16_reduction(
    const at::Tensor &left,
    const at::Tensor &right,
    at::Tensor &output,
    cudaStream_t stream
) {
    const int reduction = left.size(1);
    if (reduction == 64) {
        launch_bf16_gemm<OUTPUT_TILE, 64>(left, right, output, stream);
    } else {
        TORCH_CHECK(
            reduction % 128 == 0,
            "small dense BF16 GEMM reduction must be 64 or divisible by 128"
        );
        launch_bf16_gemm<OUTPUT_TILE, 128>(left, right, output, stream);
    }
}

namespace transpose_quantize {

using Config = mxfp8_quantize::config;

struct Globals {
    static constexpr int TILE_SIZE = 128;
    static constexpr int K_BLOCK_SIZE = 32;

    using InputTile = st_bf<TILE_SIZE, TILE_SIZE, false>;
    using OutputTile = st_fp8e4m3<TILE_SIZE, TILE_SIZE, false>;
    using ScaleTile = st_fp8e8m0<32, 16, false>;
    using InputGlobal = gl<bf16, 1, 1, -1, -1, InputTile>;
    using OutputGlobal = gl<fp8e4m3, 1, 1, -1, -1, OutputTile>;
    using ScaleGlobal = gl<fp8e8m0, -1, -1, 32, 16, ScaleTile>;

    InputGlobal input;
    OutputGlobal output;
    ScaleGlobal scales;

    __host__ inline dim3 grid() const {
        return dim3(input.cols() / TILE_SIZE, input.rows() / TILE_SIZE);
    }

    __host__ inline int dynamic_shared_memory() const {
        return TILE_SIZE * TILE_SIZE * sizeof(bf16) + 1024;
    }
};

__device__ inline void kernel(const Globals &globals) {
    extern __shared__ int shared[];
    tma_swizzle_allocator allocator(reinterpret_cast<int *>(&shared[0]));
    Globals::InputTile &input_tile = allocator.allocate<Globals::InputTile>();
    Globals::OutputTile &output_tile =
        *reinterpret_cast<Globals::OutputTile *>(&input_tile);
    Globals::ScaleTile &scale_tile = *reinterpret_cast<Globals::ScaleTile *>(
        reinterpret_cast<uint64_t>(&output_tile) + sizeof(output_tile)
    );

    const int thread = threadIdx.x;
    const int input_tile_row = blockIdx.y;
    const int input_tile_column = blockIdx.x;
    __shared__ semaphore input_arrived;
    if (thread == 0) {
        init_semaphore(input_arrived, 0, 1);
        tma::expect(input_arrived, input_tile);
        tma::load_async(
            input_tile,
            globals.input,
            {input_tile_row, input_tile_column},
            input_arrived
        );
    }
    __syncthreads();
    wait(input_arrived, 0);

    constexpr int ROWS_PER_THREAD = 2;
    constexpr int NUM_K_BLOCKS = Globals::TILE_SIZE / Globals::K_BLOCK_SIZE;
    constexpr int VALUES_PER_K_BLOCK = Globals::TILE_SIZE / 2 / NUM_K_BLOCKS;
    bf16_2 values[ROWS_PER_THREAD][NUM_K_BLOCKS][VALUES_PER_K_BLOCK];
    fp8e8m0 scales[ROWS_PER_THREAD][NUM_K_BLOCKS];
    const uint32_t input_address =
        static_cast<uint32_t>(__cvta_generic_to_shared(&input_tile));

    #pragma unroll
    for (int row_slot = 0; row_slot < ROWS_PER_THREAD; ++row_slot) {
        const int output_row = thread + row_slot * 64;
        #pragma unroll
        for (int block_slot = 0; block_slot < NUM_K_BLOCKS; ++block_slot) {
            const int k_block = (block_slot + thread / 8) % NUM_K_BLOCKS;
            #pragma unroll
            for (int value_slot = 0; value_slot < VALUES_PER_K_BLOCK; ++value_slot) {
                const int output_column = k_block * Globals::K_BLOCK_SIZE
                    + ((thread + value_slot) * 2) % Globals::K_BLOCK_SIZE;
                const int first_offset =
                    (output_column * Globals::TILE_SIZE + output_row) * sizeof(bf16);
                const int second_offset =
                    ((output_column + 1) * Globals::TILE_SIZE + output_row) * sizeof(bf16);
                move<bf16>::lds(values[row_slot][block_slot][value_slot].x,
                                input_address + first_offset);
                move<bf16>::lds(values[row_slot][block_slot][value_slot].y,
                                input_address + second_offset);
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int row_slot = 0; row_slot < ROWS_PER_THREAD; ++row_slot) {
        const int output_row = thread + row_slot * 64;
        #pragma unroll
        for (int block_slot = 0; block_slot < NUM_K_BLOCKS; ++block_slot) {
            const int k_block = (block_slot + thread / 8) % NUM_K_BLOCKS;
            bf16_2 maximum = __habs2(values[row_slot][block_slot][0]);
            #pragma unroll
            for (int value_slot = 1; value_slot < VALUES_PER_K_BLOCK; ++value_slot) {
                maximum = __hmax2(
                    maximum,
                    __habs2(values[row_slot][block_slot][value_slot])
                );
            }
            const float scale = max(
                __bfloat162float(__hmax(maximum.x, maximum.y)) * 0.002232142857f,
                0.000000000001f
            );
            scales[row_slot][k_block].__x = __nv_cvt_float_to_e8m0(
                scale,
                __NV_SATFINITE,
                cudaRoundPosInf
            );
            const float inverse_scale = 1.0f / static_cast<float>(scales[row_slot][k_block]);
            #pragma unroll
            for (int value_slot = 0; value_slot < VALUES_PER_K_BLOCK; ++value_slot) {
                const int output_column = k_block * Globals::K_BLOCK_SIZE
                    + ((thread + value_slot) * 2) % Globals::K_BLOCK_SIZE;
                const int output_offset =
                    (output_row * Globals::TILE_SIZE + output_column) * sizeof(fp8e4m3);
                fp8e4m3 output_values[2] = {
                    __nv_fp8_e4m3(
                        __bfloat162float(values[row_slot][block_slot][value_slot].x)
                        * inverse_scale
                    ),
                    __nv_fp8_e4m3(
                        __bfloat162float(values[row_slot][block_slot][value_slot].y)
                        * inverse_scale
                    ),
                };
                asm volatile("{st.shared.b16 [%0], %1;}"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&output_tile))
                           + output_offset)
                       "h"(*reinterpret_cast<uint16_t *>(&output_values[0])));
            }
        }
        const int scale_offset =
            (output_row % 32) * 16 + (output_row / 32) * 4;
        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&scale_tile))
                   + scale_offset)
               "r"(*reinterpret_cast<uint32_t *>(&scales[row_slot][0])));
    }

    __syncthreads();
    if (thread == 0) {
        tma::store_async(
            globals.output,
            output_tile,
            {input_tile_column, input_tile_row}
        );
        tma::store_async(
            globals.scales,
            scale_tile,
            {input_tile_column, input_tile_row, 0, 0}
        );
    }
}

}  // namespace transpose_quantize

template <int THREAD_COUNT>
__device__ float block_sum(float value, float *warp_sums, float *total) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < THREAD_COUNT / 32 ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        if (lane == 0) {
            *total = value;
        }
    }
    __syncthreads();
    return *total;
}

template <int WIDTH, int THREAD_COUNT>
__global__ void rmsnorm_forward_kernel(
    const __nv_bfloat16 *input,
    const __nv_bfloat16 *weight,
    __nv_bfloat16 *output,
    int rows,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    __shared__ float warp_sums[THREAD_COUNT / 32];
    __shared__ float total;
    float local_sum_squares = 0.0f;
    #pragma unroll
    for (int column = threadIdx.x; column < WIDTH; column += THREAD_COUNT) {
        const float value = __bfloat162float(input[row * WIDTH + column]);
        local_sum_squares += value * value;
    }
    const float sum_squares = block_sum<THREAD_COUNT>(
        local_sum_squares,
        warp_sums,
        &total
    );
    const float inverse_rms = rsqrtf(sum_squares / WIDTH + eps);
    #pragma unroll
    for (int column = threadIdx.x; column < WIDTH; column += THREAD_COUNT) {
        const int offset = row * WIDTH + column;
        output[offset] = __float2bfloat16(
            __bfloat162float(input[offset])
            * inverse_rms
            * __bfloat162float(weight[column])
        );
    }
}

template <int HIDDEN_DIM>
__global__ void swiglu_forward_kernel(
    const __nv_bfloat16 *gate_up,
    __nv_bfloat16 *hidden,
    int rows
) {
    constexpr int PAIRS_PER_ROW = HIDDEN_DIM / 2;
    constexpr int BLOCKS_PER_ROW = (PAIRS_PER_ROW + THREADS - 1) / THREADS;
    const int row = blockIdx.x / BLOCKS_PER_ROW;
    if (row >= rows) {
        return;
    }
    const int pair_column = (blockIdx.x % BLOCKS_PER_ROW) * THREADS + threadIdx.x;
    if (pair_column >= PAIRS_PER_ROW) {
        return;
    }
    const auto *gate_up_pairs = reinterpret_cast<const __nv_bfloat162 *>(gate_up);
    auto *hidden_pairs = reinterpret_cast<__nv_bfloat162 *>(hidden);
    const int gate_offset = row * HIDDEN_DIM + pair_column;
    const __nv_bfloat162 gate = gate_up_pairs[gate_offset];
    const __nv_bfloat162 up = gate_up_pairs[gate_offset + PAIRS_PER_ROW];
    const float gate_x = __bfloat162float(gate.x);
    const float gate_y = __bfloat162float(gate.y);
    hidden_pairs[row * PAIRS_PER_ROW + pair_column] = __floats2bfloat162_rn(
        gate_x / (1.0f + expf(-gate_x)) * __bfloat162float(up.x),
        gate_y / (1.0f + expf(-gate_y)) * __bfloat162float(up.y)
    );
}

__global__ void residual_add_kernel(
    const __nv_bfloat16 *residual,
    __nv_bfloat16 *output,
    int elements
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __float2bfloat16(
            __bfloat162float(output[index]) + __bfloat162float(residual[index])
        );
    }
}

template <int HIDDEN_DIM>
__global__ void swiglu_backward_kernel(
    const __nv_bfloat16 *grad_hidden,
    const __nv_bfloat16 *gate_up,
    __nv_bfloat16 *grad_gate_up,
    int rows
) {
    constexpr int PAIRS_PER_ROW = HIDDEN_DIM / 2;
    constexpr int BLOCKS_PER_ROW = (PAIRS_PER_ROW + THREADS - 1) / THREADS;
    const int row = blockIdx.x / BLOCKS_PER_ROW;
    if (row >= rows) {
        return;
    }
    const int pair_column = (blockIdx.x % BLOCKS_PER_ROW) * THREADS + threadIdx.x;
    if (pair_column >= PAIRS_PER_ROW) {
        return;
    }
    const auto *grad_hidden_pairs = reinterpret_cast<const __nv_bfloat162 *>(grad_hidden);
    const auto *gate_up_pairs = reinterpret_cast<const __nv_bfloat162 *>(gate_up);
    auto *grad_gate_up_pairs = reinterpret_cast<__nv_bfloat162 *>(grad_gate_up);
    const int gate_offset = row * HIDDEN_DIM + pair_column;
    const __nv_bfloat162 grad = grad_hidden_pairs[row * PAIRS_PER_ROW + pair_column];
    const __nv_bfloat162 gate = gate_up_pairs[gate_offset];
    const __nv_bfloat162 up = gate_up_pairs[gate_offset + PAIRS_PER_ROW];
    const float gate_x = __bfloat162float(gate.x);
    const float gate_y = __bfloat162float(gate.y);
    const float sigmoid_x = 1.0f / (1.0f + expf(-gate_x));
    const float sigmoid_y = 1.0f / (1.0f + expf(-gate_y));
    const float silu_x = gate_x * sigmoid_x;
    const float silu_y = gate_y * sigmoid_y;
    grad_gate_up_pairs[gate_offset] = __floats2bfloat162_rn(
        __bfloat162float(grad.x) * __bfloat162float(up.x)
            * sigmoid_x * (1.0f + gate_x * (1.0f - sigmoid_x)),
        __bfloat162float(grad.y) * __bfloat162float(up.y)
            * sigmoid_y * (1.0f + gate_y * (1.0f - sigmoid_y))
    );
    grad_gate_up_pairs[gate_offset + PAIRS_PER_ROW] = __floats2bfloat162_rn(
        __bfloat162float(grad.x) * silu_x,
        __bfloat162float(grad.y) * silu_y
    );
}

template <int WIDTH, int THREAD_COUNT>
__global__ void rmsnorm_backward_kernel(
    const __nv_bfloat16 *input,
    const __nv_bfloat16 *weight,
    const __nv_bfloat16 *grad_normalized,
    const __nv_bfloat16 *grad_residual,
    __nv_bfloat16 *grad_input,
    float *grad_weight,
    int rows,
    float eps
) {
    const int first_row = blockIdx.x * RMS_ROWS_PER_BLOCK;
    constexpr int COLUMNS_PER_THREAD = WIDTH / THREAD_COUNT;
    __shared__ float warp_sums[THREAD_COUNT / 32];
    __shared__ float total;
    float local_grad_weight[COLUMNS_PER_THREAD] = {};

    #pragma unroll
    for (int row_in_block = 0; row_in_block < RMS_ROWS_PER_BLOCK; ++row_in_block) {
        const int row = first_row + row_in_block;
        if (row >= rows) {
            break;
        }
        float local_sum_squares = 0.0f;
        #pragma unroll
        for (int slot = 0; slot < COLUMNS_PER_THREAD; ++slot) {
            const int column = threadIdx.x + slot * THREAD_COUNT;
            const float x = __bfloat162float(input[row * WIDTH + column]);
            local_sum_squares += x * x;
        }
        const float sum_squares = block_sum<THREAD_COUNT>(
            local_sum_squares,
            warp_sums,
            &total
        );
        const float inverse_rms = rsqrtf(sum_squares / WIDTH + eps);
        float local_correction = 0.0f;
        #pragma unroll
        for (int slot = 0; slot < COLUMNS_PER_THREAD; ++slot) {
            const int column = threadIdx.x + slot * THREAD_COUNT;
            const int offset = row * WIDTH + column;
            const float x = __bfloat162float(input[offset]);
            const float grad = __bfloat162float(grad_normalized[offset]);
            local_correction += grad * __bfloat162float(weight[column]) * x;
        }
        const float correction = block_sum<THREAD_COUNT>(
            local_correction,
            warp_sums,
            &total
        ) / WIDTH;
        #pragma unroll
        for (int slot = 0; slot < COLUMNS_PER_THREAD; ++slot) {
            const int column = threadIdx.x + slot * THREAD_COUNT;
            const int offset = row * WIDTH + column;
            const float x = __bfloat162float(input[offset]);
            const float grad = __bfloat162float(grad_normalized[offset]);
            const float weighted_grad = grad * __bfloat162float(weight[column]);
            const float dx = weighted_grad * inverse_rms
                - x * inverse_rms * inverse_rms * inverse_rms * correction
                + __bfloat162float(grad_residual[offset]);
            grad_input[offset] = __float2bfloat16(dx);
            local_grad_weight[slot] += grad * x * inverse_rms;
        }
    }
    #pragma unroll
    for (int slot = 0; slot < COLUMNS_PER_THREAD; ++slot) {
        const int column = threadIdx.x + slot * THREAD_COUNT;
        atomicAdd(&grad_weight[column], local_grad_weight[slot]);
    }
}

template <int WIDTH, int THREAD_COUNT>
__global__ void cast_grad_weight_kernel(
    const float *input,
    __nv_bfloat16 *output
) {
    #pragma unroll
    for (int column = threadIdx.x; column < WIDTH; column += THREAD_COUNT) {
        output[column] = __float2bfloat16(input[column]);
    }
}

void mxfp8_gemm(
    const at::Tensor &left,
    const at::Tensor &left_scales,
    const at::Tensor &right,
    const at::Tensor &right_scales,
    at::Tensor &output
) {
    const c10::cuda::CUDAGuard device_guard(left.device());
    const PrimaryContextGuard context_guard(left.get_device());
    Mxfp8GemmGlobals globals{
        .A = kittens::py::tensor_to_gl<typename Mxfp8GemmGlobals::A_gl>(left),
        .A_sc = kittens::py::tensor_to_gl<typename Mxfp8GemmGlobals::A_sc_gl>(left_scales),
        .B = kittens::py::tensor_to_gl<typename Mxfp8GemmGlobals::B_gl>(right),
        .B_sc = kittens::py::tensor_to_gl<typename Mxfp8GemmGlobals::B_sc_gl>(right_scales),
        .D = kittens::py::tensor_to_gl<typename Mxfp8GemmGlobals::D_gl>(output),
    };
    kittens::py::launch_kernel<
        Mxfp8GemmConfig,
        Mxfp8GemmGlobals,
        mxfp8_gemm::kernel<Mxfp8GemmConfig>
    >(globals);
}

void bf16_gemm(
    const at::Tensor &left,
    const at::Tensor &right,
    at::Tensor &output
) {
    const c10::cuda::CUDAGuard device_guard(left.device());
    const PrimaryContextGuard context_guard(left.get_device());
    const auto stream = at::cuda::getCurrentCUDAStream(left.get_device());
    TORCH_CHECK(left.size(1) == right.size(1), "BF16 GEMM reduction dimensions differ");
    TORCH_CHECK(left.size(0) % 128 == 0, "BF16 GEMM rows must be divisible by 128");
    TORCH_CHECK(output.size(1) == right.size(0), "BF16 GEMM output shape is invalid");

    const int output_columns = output.size(1);
    if (output_columns == 32) {
        dispatch_bf16_reduction<32>(left, right, output, stream);
    } else if (output_columns == 64) {
        dispatch_bf16_reduction<64>(left, right, output, stream);
    } else if (output_columns == 128) {
        dispatch_bf16_reduction<128>(left, right, output, stream);
    } else if (output_columns % 256 == 0) {
        dispatch_bf16_reduction<256>(left, right, output, stream);
    } else {
        TORCH_CHECK(false, "unsupported dense BF16 GEMM output width: ", output_columns);
    }
}

void quantize_mxfp8(
    const at::Tensor &input,
    at::Tensor &output,
    at::Tensor &scales
) {
    const c10::cuda::CUDAGuard device_guard(input.device());
    const PrimaryContextGuard context_guard(input.get_device());
    using Config = mxfp8_quantize::config;
    using Globals = mxfp8_quantize::globals;
    Globals globals{
        .A_bf16 = kittens::py::tensor_to_gl<Globals::A_bf16_gl>(input),
        .A_fp8 = kittens::py::tensor_to_gl<Globals::A_fp8_gl>(output),
        .A_sc = kittens::py::tensor_to_gl<Globals::A_sc_gl>(scales),
    };
    kittens::py::launch_kernel<Config, Globals, mxfp8_quantize::kernel>(globals);
}

void quantize_mxfp8_transpose(
    const at::Tensor &input,
    at::Tensor &output,
    at::Tensor &scales
) {
    const c10::cuda::CUDAGuard device_guard(input.device());
    const PrimaryContextGuard context_guard(input.get_device());
    using Config = transpose_quantize::Config;
    using Globals = transpose_quantize::Globals;
    Globals globals{
        .input = kittens::py::tensor_to_gl<Globals::InputGlobal>(input),
        .output = kittens::py::tensor_to_gl<Globals::OutputGlobal>(output),
        .scales = kittens::py::tensor_to_gl<Globals::ScaleGlobal>(scales),
    };
    kittens::py::launch_kernel<Config, Globals, transpose_quantize::kernel>(globals);
}

template <int WIDTH, int THREAD_COUNT>
void launch_rmsnorm_forward(
    const at::Tensor &input,
    const at::Tensor &weight,
    at::Tensor &output,
    float eps,
    cudaStream_t stream
) {
    rmsnorm_forward_kernel<WIDTH, THREAD_COUNT>
        <<<input.size(0), THREAD_COUNT, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16 *>(weight.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()),
            input.size(0),
            eps
        );
}

template <int WIDTH, int THREAD_COUNT>
void launch_rmsnorm_backward(
    const at::Tensor &input,
    const at::Tensor &weight,
    const at::Tensor &grad_normalized,
    const at::Tensor &grad_residual,
    at::Tensor &grad_input,
    at::Tensor &grad_weight_workspace,
    at::Tensor &grad_weight,
    float eps,
    cudaStream_t stream
) {
    const int blocks = (input.size(0) + RMS_ROWS_PER_BLOCK - 1) / RMS_ROWS_PER_BLOCK;
    rmsnorm_backward_kernel<WIDTH, THREAD_COUNT><<<blocks, THREAD_COUNT, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(grad_normalized.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(grad_residual.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(grad_input.data_ptr()),
        grad_weight_workspace.data_ptr<float>(),
        input.size(0),
        eps
    );
    cast_grad_weight_kernel<WIDTH, THREAD_COUNT><<<1, THREAD_COUNT, 0, stream>>>(
        grad_weight_workspace.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 *>(grad_weight.data_ptr())
    );
}

void rmsnorm_forward(
    const at::Tensor &input,
    const at::Tensor &weight,
    at::Tensor &output,
    double eps
) {
    const c10::cuda::CUDAGuard device_guard(input.device());
    const PrimaryContextGuard context_guard(input.get_device());
    const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    const float eps_float = static_cast<float>(eps);
    switch (input.size(1)) {
        case 64: launch_rmsnorm_forward<64, 64>(input, weight, output, eps_float, stream); break;
        case 128: launch_rmsnorm_forward<128, 128>(input, weight, output, eps_float, stream); break;
        case 256: launch_rmsnorm_forward<256, 256>(input, weight, output, eps_float, stream); break;
        case 512: launch_rmsnorm_forward<512, 256>(input, weight, output, eps_float, stream); break;
        case 768: launch_rmsnorm_forward<768, 256>(input, weight, output, eps_float, stream); break;
        case 1024: launch_rmsnorm_forward<1024, 256>(input, weight, output, eps_float, stream); break;
        case 1280: launch_rmsnorm_forward<1280, 256>(input, weight, output, eps_float, stream); break;
        default: TORCH_CHECK(false, "unsupported dense width: ", input.size(1));
    }
}

void swiglu_forward(const at::Tensor &gate_up, at::Tensor &hidden) {
    const c10::cuda::CUDAGuard device_guard(gate_up.device());
    const PrimaryContextGuard context_guard(gate_up.get_device());
    const auto stream = at::cuda::getCurrentCUDAStream(gate_up.get_device());
    const int hidden_dim = hidden.size(1);
    const int rows = hidden.size(0);
    const int blocks_per_row = (hidden_dim / 2 + THREADS - 1) / THREADS;
    const int blocks = rows * blocks_per_row;
    const auto *gate_up_data = reinterpret_cast<const __nv_bfloat16 *>(gate_up.data_ptr());
    auto *hidden_data = reinterpret_cast<__nv_bfloat16 *>(hidden.data_ptr());
    switch (hidden_dim) {
        case 128: swiglu_forward_kernel<128><<<blocks, THREADS, 0, stream>>>(gate_up_data, hidden_data, rows); break;
        case 256: swiglu_forward_kernel<256><<<blocks, THREADS, 0, stream>>>(gate_up_data, hidden_data, rows); break;
        case 512: swiglu_forward_kernel<512><<<blocks, THREADS, 0, stream>>>(gate_up_data, hidden_data, rows); break;
        case 1024: swiglu_forward_kernel<1024><<<blocks, THREADS, 0, stream>>>(gate_up_data, hidden_data, rows); break;
        case 2048: swiglu_forward_kernel<2048><<<blocks, THREADS, 0, stream>>>(gate_up_data, hidden_data, rows); break;
        case 3072: swiglu_forward_kernel<3072><<<blocks, THREADS, 0, stream>>>(gate_up_data, hidden_data, rows); break;
        case 4096: swiglu_forward_kernel<4096><<<blocks, THREADS, 0, stream>>>(gate_up_data, hidden_data, rows); break;
        case 5120: swiglu_forward_kernel<5120><<<blocks, THREADS, 0, stream>>>(gate_up_data, hidden_data, rows); break;
        default: TORCH_CHECK(false, "unsupported SwiGLU hidden width: ", hidden_dim);
    }
}

void residual_add(const at::Tensor &residual, at::Tensor &output) {
    const c10::cuda::CUDAGuard device_guard(residual.device());
    const PrimaryContextGuard context_guard(residual.get_device());
    const auto stream = at::cuda::getCurrentCUDAStream(residual.get_device());
    const int elements = output.numel();
    residual_add_kernel<<<(elements + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(residual.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()),
        elements
    );
}

void swiglu_backward(
    const at::Tensor &grad_hidden,
    const at::Tensor &gate_up,
    at::Tensor &grad_gate_up
) {
    const c10::cuda::CUDAGuard device_guard(grad_hidden.device());
    const PrimaryContextGuard context_guard(grad_hidden.get_device());
    const auto stream = at::cuda::getCurrentCUDAStream(grad_hidden.get_device());
    const int hidden_dim = grad_hidden.size(1);
    const int rows = grad_hidden.size(0);
    const int blocks_per_row = (hidden_dim / 2 + THREADS - 1) / THREADS;
    const int blocks = rows * blocks_per_row;
    const auto *grad_hidden_data =
        reinterpret_cast<const __nv_bfloat16 *>(grad_hidden.data_ptr());
    const auto *gate_up_data = reinterpret_cast<const __nv_bfloat16 *>(gate_up.data_ptr());
    auto *grad_gate_up_data = reinterpret_cast<__nv_bfloat16 *>(grad_gate_up.data_ptr());
    switch (hidden_dim) {
        case 128: swiglu_backward_kernel<128><<<blocks, THREADS, 0, stream>>>(grad_hidden_data, gate_up_data, grad_gate_up_data, rows); break;
        case 256: swiglu_backward_kernel<256><<<blocks, THREADS, 0, stream>>>(grad_hidden_data, gate_up_data, grad_gate_up_data, rows); break;
        case 512: swiglu_backward_kernel<512><<<blocks, THREADS, 0, stream>>>(grad_hidden_data, gate_up_data, grad_gate_up_data, rows); break;
        case 1024: swiglu_backward_kernel<1024><<<blocks, THREADS, 0, stream>>>(grad_hidden_data, gate_up_data, grad_gate_up_data, rows); break;
        case 2048: swiglu_backward_kernel<2048><<<blocks, THREADS, 0, stream>>>(grad_hidden_data, gate_up_data, grad_gate_up_data, rows); break;
        case 3072: swiglu_backward_kernel<3072><<<blocks, THREADS, 0, stream>>>(grad_hidden_data, gate_up_data, grad_gate_up_data, rows); break;
        case 4096: swiglu_backward_kernel<4096><<<blocks, THREADS, 0, stream>>>(grad_hidden_data, gate_up_data, grad_gate_up_data, rows); break;
        case 5120: swiglu_backward_kernel<5120><<<blocks, THREADS, 0, stream>>>(grad_hidden_data, gate_up_data, grad_gate_up_data, rows); break;
        default: TORCH_CHECK(false, "unsupported SwiGLU hidden width: ", hidden_dim);
    }
}

void rmsnorm_backward(
    const at::Tensor &input,
    const at::Tensor &weight,
    const at::Tensor &grad_normalized,
    const at::Tensor &grad_residual,
    at::Tensor &grad_input,
    at::Tensor &grad_weight_workspace,
    at::Tensor &grad_weight,
    double eps
) {
    const c10::cuda::CUDAGuard device_guard(input.device());
    const PrimaryContextGuard context_guard(input.get_device());
    const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    const float eps_float = static_cast<float>(eps);
    switch (input.size(1)) {
        case 64: launch_rmsnorm_backward<64, 64>(input, weight, grad_normalized, grad_residual, grad_input, grad_weight_workspace, grad_weight, eps_float, stream); break;
        case 128: launch_rmsnorm_backward<128, 128>(input, weight, grad_normalized, grad_residual, grad_input, grad_weight_workspace, grad_weight, eps_float, stream); break;
        case 256: launch_rmsnorm_backward<256, 256>(input, weight, grad_normalized, grad_residual, grad_input, grad_weight_workspace, grad_weight, eps_float, stream); break;
        case 512: launch_rmsnorm_backward<512, 256>(input, weight, grad_normalized, grad_residual, grad_input, grad_weight_workspace, grad_weight, eps_float, stream); break;
        case 768: launch_rmsnorm_backward<768, 256>(input, weight, grad_normalized, grad_residual, grad_input, grad_weight_workspace, grad_weight, eps_float, stream); break;
        case 1024: launch_rmsnorm_backward<1024, 256>(input, weight, grad_normalized, grad_residual, grad_input, grad_weight_workspace, grad_weight, eps_float, stream); break;
        case 1280: launch_rmsnorm_backward<1280, 256>(input, weight, grad_normalized, grad_residual, grad_input, grad_weight_workspace, grad_weight, eps_float, stream); break;
        default: TORCH_CHECK(false, "unsupported dense width: ", input.size(1));
    }
}

}  // namespace chess_engine_4::dense

void bind_sm100_dense_kernels(pybind11::module_ &module) {
    bind_thunderkittens_reference(module);
    module.def("dense_mxfp8_gemm", &chess_engine_4::dense::mxfp8_gemm);
    module.def("dense_bf16_gemm", &chess_engine_4::dense::bf16_gemm);
    module.def("dense_quantize_mxfp8", &chess_engine_4::dense::quantize_mxfp8);
    module.def(
        "dense_quantize_mxfp8_transpose",
        &chess_engine_4::dense::quantize_mxfp8_transpose
    );
    module.def("dense_rmsnorm_forward", &chess_engine_4::dense::rmsnorm_forward);
    module.def("dense_swiglu_forward", &chess_engine_4::dense::swiglu_forward);
    module.def("dense_residual_add", &chess_engine_4::dense::residual_add);
    module.def("dense_swiglu_backward", &chess_engine_4::dense::swiglu_backward);
    module.def("dense_rmsnorm_backward", &chess_engine_4::dense::rmsnorm_backward);
}
