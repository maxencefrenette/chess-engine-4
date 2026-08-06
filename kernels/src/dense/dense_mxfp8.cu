#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

// ThunderKittens' reference file contains the Blackwell MXFP8 device kernel,
// quantizer, and its stock Nb=256 binding. Capture that binding so this
// project can expose an additional d128-specialized CUDA entry point from one
// extension module.
static void bind_thunderkittens_reference(pybind11::module_ &module);
static void bind_chess_engine_kernels(pybind11::module_ &module);

PYBIND11_MODULE(_chess_engine_4_kernels, module) {
    bind_chess_engine_kernels(module);
}

#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, module) \
    static void bind_thunderkittens_reference(pybind11::module_ &module)
#define _C _chess_engine_4_kernels_reference
#include "../../../third_party/ThunderKittens/kernels/gemm/mxfp8_b200/mxfp8_b200_gemm.cu"
#undef _C
#undef PYBIND11_MODULE

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

// The stock TK PyTorch binding instantiates Nb=256 even though the underlying
// kernel supports Nb=128. Dense d128 has several projections whose output is
// exactly 128 columns. This specialization avoids padding those projections to
// 256 columns and discarding half of the result.
using NarrowGemmConfig = mxfp8_gemm::config<128, 5, 4, 12, 2, true>;
using NarrowGemmGlobals = mxfp8_gemm::globals<NarrowGemmConfig>;
using WideGemmConfig = mxfp8_gemm::config<256, 6, 16, 12, 4, false>;
using WideGemmGlobals = mxfp8_gemm::globals<WideGemmConfig>;

constexpr int THREADS = 256;
constexpr int RMS_ROWS_PER_BLOCK = 8;
constexpr int MIN_WIDTH = 128;
constexpr int MAX_WIDTH = 2048;

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

__global__ void swiglu_forward_kernel(
    const __nv_bfloat16 *gate_up,
    __nv_bfloat16 *hidden,
    int elements,
    int hidden_dim
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = index / hidden_dim;
    const int column = index % hidden_dim;
    const int row_offset = row * (2 * hidden_dim);
    const float gate = __bfloat162float(gate_up[row_offset + column]);
    const float up = __bfloat162float(gate_up[row_offset + hidden_dim + column]);
    const float silu = gate / (1.0f + expf(-gate));
    hidden[index] = __float2bfloat16(silu * up);
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

__global__ void swiglu_backward_kernel(
    const __nv_bfloat16 *grad_hidden,
    const __nv_bfloat16 *gate_up,
    __nv_bfloat16 *grad_gate_up,
    int elements,
    int hidden_dim
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = index / hidden_dim;
    const int column = index % hidden_dim;
    const int output_offset = row * (2 * hidden_dim);
    const float grad = __bfloat162float(grad_hidden[index]);
    const float gate_value = __bfloat162float(gate_up[output_offset + column]);
    const float up_value = __bfloat162float(gate_up[output_offset + hidden_dim + column]);
    const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
    const float silu = gate_value * sigmoid;
    const float silu_gradient = sigmoid * (1.0f + gate_value * (1.0f - sigmoid));
    grad_gate_up[output_offset + column] = __float2bfloat16(grad * up_value * silu_gradient);
    grad_gate_up[output_offset + hidden_dim + column] = __float2bfloat16(grad * silu);
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

void mxfp8_gemm_narrow(
    const at::Tensor &left,
    const at::Tensor &left_scales,
    const at::Tensor &right,
    const at::Tensor &right_scales,
    at::Tensor &output
) {
    const c10::cuda::CUDAGuard device_guard(left.device());
    const PrimaryContextGuard context_guard(left.get_device());
    NarrowGemmGlobals globals{
        .A = kittens::py::tensor_to_gl<typename NarrowGemmGlobals::A_gl>(left),
        .A_sc = kittens::py::tensor_to_gl<typename NarrowGemmGlobals::A_sc_gl>(left_scales),
        .B = kittens::py::tensor_to_gl<typename NarrowGemmGlobals::B_gl>(right),
        .B_sc = kittens::py::tensor_to_gl<typename NarrowGemmGlobals::B_sc_gl>(right_scales),
        .D = kittens::py::tensor_to_gl<typename NarrowGemmGlobals::D_gl>(output),
    };
    kittens::py::launch_kernel<
        NarrowGemmConfig,
        NarrowGemmGlobals,
        mxfp8_gemm::kernel<NarrowGemmConfig>
    >(globals);
}

void mxfp8_gemm_wide(
    const at::Tensor &left,
    const at::Tensor &left_scales,
    const at::Tensor &right,
    const at::Tensor &right_scales,
    at::Tensor &output
) {
    const c10::cuda::CUDAGuard device_guard(left.device());
    const PrimaryContextGuard context_guard(left.get_device());
    WideGemmGlobals globals{
        .A = kittens::py::tensor_to_gl<typename WideGemmGlobals::A_gl>(left),
        .A_sc = kittens::py::tensor_to_gl<typename WideGemmGlobals::A_sc_gl>(left_scales),
        .B = kittens::py::tensor_to_gl<typename WideGemmGlobals::B_gl>(right),
        .B_sc = kittens::py::tensor_to_gl<typename WideGemmGlobals::B_sc_gl>(right_scales),
        .D = kittens::py::tensor_to_gl<typename WideGemmGlobals::D_gl>(output),
    };
    kittens::py::launch_kernel<
        WideGemmConfig,
        WideGemmGlobals,
        mxfp8_gemm::kernel<WideGemmConfig>
    >(globals);
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
        case 128: launch_rmsnorm_forward<128, 128>(input, weight, output, eps_float, stream); break;
        case 256: launch_rmsnorm_forward<256, 256>(input, weight, output, eps_float, stream); break;
        case 512: launch_rmsnorm_forward<512, 256>(input, weight, output, eps_float, stream); break;
        case 1024: launch_rmsnorm_forward<1024, 256>(input, weight, output, eps_float, stream); break;
        case 2048: launch_rmsnorm_forward<2048, 256>(input, weight, output, eps_float, stream); break;
        default: TORCH_CHECK(false, "unsupported dense width: ", input.size(1));
    }
}

void swiglu_forward(const at::Tensor &gate_up, at::Tensor &hidden) {
    const c10::cuda::CUDAGuard device_guard(gate_up.device());
    const PrimaryContextGuard context_guard(gate_up.get_device());
    const auto stream = at::cuda::getCurrentCUDAStream(gate_up.get_device());
    const int elements = hidden.numel();
    const int hidden_dim = hidden.size(1);
    swiglu_forward_kernel<<<(elements + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(gate_up.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(hidden.data_ptr()),
        elements,
        hidden_dim
    );
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
    const int elements = grad_hidden.numel();
    const int hidden_dim = grad_hidden.size(1);
    swiglu_backward_kernel<<<(elements + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(grad_hidden.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(gate_up.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(grad_gate_up.data_ptr()),
        elements,
        hidden_dim
    );
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
        case 128: launch_rmsnorm_backward<128, 128>(input, weight, grad_normalized, grad_residual, grad_input, grad_weight_workspace, grad_weight, eps_float, stream); break;
        case 256: launch_rmsnorm_backward<256, 256>(input, weight, grad_normalized, grad_residual, grad_input, grad_weight_workspace, grad_weight, eps_float, stream); break;
        case 512: launch_rmsnorm_backward<512, 256>(input, weight, grad_normalized, grad_residual, grad_input, grad_weight_workspace, grad_weight, eps_float, stream); break;
        case 1024: launch_rmsnorm_backward<1024, 256>(input, weight, grad_normalized, grad_residual, grad_input, grad_weight_workspace, grad_weight, eps_float, stream); break;
        case 2048: launch_rmsnorm_backward<2048, 256>(input, weight, grad_normalized, grad_residual, grad_input, grad_weight_workspace, grad_weight, eps_float, stream); break;
        default: TORCH_CHECK(false, "unsupported dense width: ", input.size(1));
    }
}

}  // namespace chess_engine_4::dense

static void bind_chess_engine_kernels(pybind11::module_ &module) {
    bind_thunderkittens_reference(module);
    module.def("dense_mxfp8_gemm_narrow", &chess_engine_4::dense::mxfp8_gemm_narrow);
    module.def("dense_mxfp8_gemm_wide", &chess_engine_4::dense::mxfp8_gemm_wide);
    module.def("dense_quantize_mxfp8", &chess_engine_4::dense::quantize_mxfp8);
    module.def("dense_rmsnorm_forward", &chess_engine_4::dense::rmsnorm_forward);
    module.def("dense_swiglu_forward", &chess_engine_4::dense::swiglu_forward);
    module.def("dense_residual_add", &chess_engine_4::dense::residual_add);
    module.def("dense_swiglu_backward", &chess_engine_4::dense::swiglu_backward);
    module.def("dense_rmsnorm_backward", &chess_engine_4::dense::rmsnorm_backward);
}
