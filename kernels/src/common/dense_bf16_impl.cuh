#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <algorithm>

#ifndef DENSE_ARCH_NAMESPACE
#error "DENSE_ARCH_NAMESPACE must be defined by the architecture translation unit"
#endif
#ifndef DENSE_COMPUTE_CAPABILITY_MAJOR
#error "DENSE_COMPUTE_CAPABILITY_MAJOR must be defined by the architecture translation unit"
#endif
#ifndef DENSE_COMPUTE_CAPABILITY_MINOR
#error "DENSE_COMPUTE_CAPABILITY_MINOR must be defined by the architecture translation unit"
#endif
#ifndef DENSE_ARCHITECTURE_NAME
#error "DENSE_ARCHITECTURE_NAME must be defined by the architecture translation unit"
#endif

namespace DENSE_ARCH_NAMESPACE::dense {
namespace {

constexpr int kThreads = 256;

__device__ float BlockSum(float value, float* warp_sums, float* total) {
    for (int delta = 16; delta > 0; delta /= 2) {
        value += __shfl_down_sync(0xffffffff, value, delta);
    }
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = threadIdx.x < blockDim.x / 32 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        for (int delta = 16; delta > 0; delta /= 2) {
            value += __shfl_down_sync(0xffffffff, value, delta);
        }
        if (lane == 0) *total = value;
    }
    __syncthreads();
    return *total;
}

__global__ void RmsNormForwardKernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows,
    int width,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float warp_sums[kThreads / 32];
    __shared__ float total;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = __bfloat162float(input[row * width + column]);
        sum += value * value;
    }
    const float inverse_rms = rsqrtf(BlockSum(sum, warp_sums, &total) / width + eps);
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const int offset = row * width + column;
        output[offset] = __float2bfloat16(
            __bfloat162float(input[offset]) * inverse_rms * __bfloat162float(weight[column])
        );
    }
}

__global__ void SwiGluForwardKernel(
    const __nv_bfloat16* gate_up,
    __nv_bfloat16* hidden,
    int elements,
    int hidden_dim
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const int row = index / hidden_dim;
    const int column = index % hidden_dim;
    const int gate_offset = row * 2 * hidden_dim + column;
    const float gate = __bfloat162float(gate_up[gate_offset]);
    const float up = __bfloat162float(gate_up[gate_offset + hidden_dim]);
    hidden[index] = __float2bfloat16(gate / (1.0f + __expf(-gate)) * up);
}

__global__ void SwiGluBackwardKernel(
    const __nv_bfloat16* grad_hidden,
    const __nv_bfloat16* gate_up,
    __nv_bfloat16* grad_gate_up,
    int elements,
    int hidden_dim
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const int row = index / hidden_dim;
    const int column = index % hidden_dim;
    const int gate_offset = row * 2 * hidden_dim + column;
    const float grad = __bfloat162float(grad_hidden[index]);
    const float gate = __bfloat162float(gate_up[gate_offset]);
    const float up = __bfloat162float(gate_up[gate_offset + hidden_dim]);
    const float sigmoid = 1.0f / (1.0f + __expf(-gate));
    grad_gate_up[gate_offset] = __float2bfloat16(
        grad * up * sigmoid * (1.0f + gate * (1.0f - sigmoid))
    );
    grad_gate_up[gate_offset + hidden_dim] = __float2bfloat16(grad * gate * sigmoid);
}

__global__ void ResidualAddKernel(
    const __nv_bfloat16* residual,
    __nv_bfloat16* output,
    int elements
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __float2bfloat16(
            __bfloat162float(output[index]) + __bfloat162float(residual[index])
        );
    }
}

__global__ void RmsNormBackwardKernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* grad_normalized,
    const __nv_bfloat16* grad_residual,
    __nv_bfloat16* grad_input,
    float* grad_weight,
    int rows,
    int width,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float warp_sums[kThreads / 32];
    __shared__ float total;
    float square_sum = 0.0f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = __bfloat162float(input[row * width + column]);
        square_sum += value * value;
    }
    const float inverse_rms = rsqrtf(
        BlockSum(square_sum, warp_sums, &total) / width + eps
    );
    float dot = 0.0f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const int offset = row * width + column;
        dot += __bfloat162float(grad_normalized[offset])
            * __bfloat162float(weight[column])
            * __bfloat162float(input[offset]);
    }
    const float correction = BlockSum(dot, warp_sums, &total) / width;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const int offset = row * width + column;
        const float value = __bfloat162float(input[offset]);
        const float grad = __bfloat162float(grad_normalized[offset]);
        grad_input[offset] = __float2bfloat16(
            grad * __bfloat162float(weight[column]) * inverse_rms
            - value * inverse_rms * inverse_rms * inverse_rms * correction
            + __bfloat162float(grad_residual[offset])
        );
        atomicAdd(&grad_weight[column], grad * value * inverse_rms);
    }
}

__global__ void CastKernel(const float* input, __nv_bfloat16* output, int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = __float2bfloat16(input[index]);
}

void CheckBf16Cuda(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be BF16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckDeviceCapability(const at::Tensor& tensor) {
    cudaDeviceProp properties;
    CUDACHECK(cudaGetDeviceProperties(&properties, tensor.get_device()));
    TORCH_CHECK(
        properties.major == DENSE_COMPUTE_CAPABILITY_MAJOR
            && properties.minor == DENSE_COMPUTE_CAPABILITY_MINOR,
        DENSE_ARCHITECTURE_NAME " dense kernels require compute capability ",
        DENSE_COMPUTE_CAPABILITY_MAJOR,
        ".",
        DENSE_COMPUTE_CAPABILITY_MINOR
    );
}

cudaStream_t Stream(const at::Tensor& tensor) {
    return at::cuda::getCurrentCUDAStream(tensor.get_device());
}

}  // namespace

void Bf16Gemm(const at::Tensor& left, const at::Tensor& right, at::Tensor& output) {
    CheckBf16Cuda(left, "left");
    CheckBf16Cuda(right, "right");
    CheckBf16Cuda(output, "output");
    TORCH_CHECK(left.dim() == 2 && right.dim() == 2 && output.dim() == 2, "GEMM tensors must be matrices");
    TORCH_CHECK(left.size(1) == right.size(1), "GEMM reduction dimensions differ");
    TORCH_CHECK(left.size(1) >= 64, DENSE_ARCHITECTURE_NAME " dense kernels require reduction width of at least 64");
    TORCH_CHECK(output.size(0) == left.size(0) && output.size(1) == right.size(0), "invalid GEMM output shape");
    TORCH_CHECK(left.size(0) % kGemmTile == 0 && right.size(0) % kGemmTile == 0 && left.size(1) % kGemmTile == 0, DENSE_ARCHITECTURE_NAME " TK GEMM dimensions must be divisible by 16");
    const c10::cuda::CUDAGuard guard(left.device());
    CheckDeviceCapability(left);
#ifdef DENSE_USE_ATEN_GEMM
    at::mm_out(output, left, right.transpose(0, 1));
#else
    cudaDeviceProp properties;
    CUDACHECK(cudaGetDeviceProperties(&properties, left.get_device()));
    LaunchBf16Gemm(
        reinterpret_cast<const __nv_bfloat16*>(left.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(right.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        left.size(0),
        right.size(0),
        left.size(1),
        properties.multiProcessorCount,
        Stream(left)
    );
    CUDACHECK(cudaPeekAtLastError());
#endif
}

void RmsNormForward(const at::Tensor& input, const at::Tensor& weight, at::Tensor& output, double eps) {
    const c10::cuda::CUDAGuard guard(input.device());
    CheckDeviceCapability(input);
    TORCH_CHECK(input.size(1) >= 64, DENSE_ARCHITECTURE_NAME " dense kernels require width of at least 64");
    const int threads = std::min<int>(input.size(1), kThreads);
    RmsNormForwardKernel<<<input.size(0), threads, 0, Stream(input)>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        input.size(0), input.size(1), static_cast<float>(eps)
    );
}

void SwiGluForward(const at::Tensor& gate_up, at::Tensor& hidden) {
    const c10::cuda::CUDAGuard guard(gate_up.device());
    CheckDeviceCapability(gate_up);
    TORCH_CHECK(hidden.size(1) >= 256, DENSE_ARCHITECTURE_NAME " dense kernels require hidden width of at least 256");
    const int elements = hidden.numel();
    SwiGluForwardKernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, Stream(gate_up)>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(hidden.data_ptr()), elements, hidden.size(1)
    );
}

void ResidualAdd(const at::Tensor& residual, at::Tensor& output) {
    const c10::cuda::CUDAGuard guard(residual.device());
    CheckDeviceCapability(residual);
    TORCH_CHECK(output.size(1) >= 64, DENSE_ARCHITECTURE_NAME " dense kernels require width of at least 64");
    const int elements = output.numel();
    ResidualAddKernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, Stream(residual)>>>(
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), elements
    );
}

void SwiGluBackward(const at::Tensor& grad_hidden, const at::Tensor& gate_up, at::Tensor& grad_gate_up) {
    const c10::cuda::CUDAGuard guard(grad_hidden.device());
    CheckDeviceCapability(grad_hidden);
    TORCH_CHECK(grad_hidden.size(1) >= 256, DENSE_ARCHITECTURE_NAME " dense kernels require hidden width of at least 256");
    const int elements = grad_hidden.numel();
    SwiGluBackwardKernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, Stream(grad_hidden)>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_hidden.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_gate_up.data_ptr()), elements, grad_hidden.size(1)
    );
}

void RmsNormBackward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& grad_normalized,
    const at::Tensor& grad_residual,
    at::Tensor& grad_input,
    at::Tensor& grad_weight_workspace,
    at::Tensor& grad_weight,
    double eps
) {
    const c10::cuda::CUDAGuard guard(input.device());
    CheckDeviceCapability(input);
    TORCH_CHECK(input.size(1) >= 64, DENSE_ARCHITECTURE_NAME " dense kernels require width of at least 64");
    const int threads = std::min<int>(input.size(1), kThreads);
    RmsNormBackwardKernel<<<input.size(0), threads, 0, Stream(input)>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(grad_normalized.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(grad_residual.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr()),
        grad_weight_workspace.data_ptr<float>(), input.size(0), input.size(1), static_cast<float>(eps)
    );
    CastKernel<<<(input.size(1) + kThreads - 1) / kThreads, kThreads, 0, Stream(input)>>>(
        grad_weight_workspace.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(grad_weight.data_ptr()), input.size(1)
    );
    CUDACHECK(cudaPeekAtLastError());
}

}  // namespace DENSE_ARCH_NAMESPACE::dense
