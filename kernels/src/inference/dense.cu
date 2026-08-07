#include "chess_engine_4/inference.h"
#include "tk_bf16.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace chess_engine_4::inference {
namespace {

constexpr int kInputPlanes = 112;
constexpr int kHistoryPlanes = 104;
constexpr int kPlanesPerHistory = 13;

void CheckCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void CheckCublas(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with status " + std::to_string(status)
        );
    }
}

struct HostTensor {
    std::vector<std::int64_t> shape;
    const std::byte *data;
    std::size_t bytes;
};

struct SafetensorsFile {
    nlohmann::json metadata;
    std::unordered_map<std::string, HostTensor> tensors;
    std::vector<std::byte> storage;
};

SafetensorsFile LoadSafetensors(const std::string &path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("Cannot open model: " + path);
    }
    const auto size = static_cast<std::size_t>(stream.tellg());
    stream.seekg(0);
    SafetensorsFile file;
    file.storage.resize(size);
    stream.read(reinterpret_cast<char *>(file.storage.data()), size);
    if (!stream || size < sizeof(std::uint64_t)) {
        throw std::runtime_error("Invalid Safetensors file: " + path);
    }

    std::uint64_t header_size = 0;
    std::memcpy(&header_size, file.storage.data(), sizeof(header_size));
    if (header_size > size - sizeof(header_size)) {
        throw std::runtime_error("Invalid Safetensors header size");
    }
    const auto header = nlohmann::json::parse(
        reinterpret_cast<const char *>(file.storage.data() + sizeof(header_size)),
        reinterpret_cast<const char *>(
            file.storage.data() + sizeof(header_size) + header_size
        )
    );
    file.metadata = header.value("__metadata__", nlohmann::json::object());
    const std::size_t data_start = sizeof(header_size) + header_size;
    for (const auto &[name, value] : header.items()) {
        if (name == "__metadata__") {
            continue;
        }
        if (value.at("dtype").get<std::string>() != "BF16") {
            throw std::runtime_error("Tensor " + name + " is not BF16");
        }
        const auto offsets = value.at("data_offsets").get<std::vector<std::size_t>>();
        if (offsets.size() != 2 || offsets[0] > offsets[1] || data_start + offsets[1] > size) {
            throw std::runtime_error("Tensor " + name + " has invalid data offsets");
        }
        file.tensors.emplace(
            name,
            HostTensor{
                .shape = value.at("shape").get<std::vector<std::int64_t>>(),
                .data = file.storage.data() + data_start + offsets[0],
                .bytes = offsets[1] - offsets[0],
            }
        );
    }
    return file;
}

int MetadataInt(const nlohmann::json &metadata, const char *name) {
    return std::stoi(metadata.at(name).get<std::string>());
}

float MetadataFloat(const nlohmann::json &metadata, const char *name) {
    return std::stof(metadata.at(name).get<std::string>());
}

class DeviceTensor {
public:
    DeviceTensor() = default;

    explicit DeviceTensor(const HostTensor &host) : bytes_(host.bytes), shape_(host.shape) {
        CheckCuda(cudaMalloc(&data_, bytes_), "cudaMalloc weight");
        CheckCuda(cudaMemcpy(data_, host.data, bytes_, cudaMemcpyHostToDevice), "copy weight");
    }

    ~DeviceTensor() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceTensor(DeviceTensor &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          bytes_(other.bytes_),
          shape_(std::move(other.shape_)) {}

    DeviceTensor &operator=(DeviceTensor &&other) noexcept {
        std::swap(data_, other.data_);
        std::swap(bytes_, other.bytes_);
        std::swap(shape_, other.shape_);
        return *this;
    }

    DeviceTensor(const DeviceTensor &) = delete;
    DeviceTensor &operator=(const DeviceTensor &) = delete;

    __nv_bfloat16 *data() { return static_cast<__nv_bfloat16 *>(data_); }
    const __nv_bfloat16 *data() const {
        return static_cast<const __nv_bfloat16 *>(data_);
    }

private:
    void *data_ = nullptr;
    std::size_t bytes_ = 0;
    std::vector<std::int64_t> shape_;
};

const HostTensor &RequireTensor(
    const SafetensorsFile &file,
    const std::string &name,
    std::initializer_list<std::int64_t> shape
) {
    const auto iter = file.tensors.find(name);
    if (iter == file.tensors.end()) {
        throw std::runtime_error("Model is missing tensor " + name);
    }
    if (iter->second.shape != std::vector<std::int64_t>(shape)) {
        throw std::runtime_error("Tensor " + name + " has an unexpected shape");
    }
    return iter->second;
}

template <typename T>
T *Allocate(std::size_t count, const char *name) {
    T *result = nullptr;
    CheckCuda(cudaMalloc(&result, count * sizeof(T)), name);
    return result;
}

__global__ void ExpandPlanes(
    const InputPlane *planes,
    __nv_bfloat16 *output,
    int batch_size,
    int history_length,
    int selected_planes
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = batch_size * selected_planes * 64;
    if (index >= elements) {
        return;
    }
    const int square = index % 64;
    const int selected_plane = (index / 64) % selected_planes;
    const int sample = index / (selected_planes * 64);
    const int history_planes = history_length * kPlanesPerHistory;
    const int source_plane = selected_plane < history_planes
        ? selected_plane
        : kHistoryPlanes + selected_plane - history_planes;
    const InputPlane plane = planes[sample * kInputPlanes + source_plane];
    float value = ((plane.mask >> square) & 1ULL) != 0 ? plane.value : 0.0f;
    if (selected_plane == history_planes + 5) {
        value /= 99.0f;
    }
    output[index] = __float2bfloat16(value);
}

__global__ void AddBias(
    __nv_bfloat16 *values,
    const __nv_bfloat16 *bias,
    int rows,
    int columns
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * columns) {
        values[index] = __float2bfloat16(
            __bfloat162float(values[index]) + __bfloat162float(bias[index % columns])
        );
    }
}

__global__ void RmsNorm(
    const __nv_bfloat16 *input,
    const __nv_bfloat16 *weight,
    __nv_bfloat16 *output,
    int rows,
    int width,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    float sum = 0.0f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = __bfloat162float(input[row * width + column]);
        sum += value * value;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float warp_sums[8];
    __shared__ float inverse_rms;
    if (threadIdx.x % 32 == 0) {
        warp_sums[threadIdx.x / 32] = sum;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < blockDim.x / 32 ? warp_sums[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) {
            inverse_rms = rsqrtf(sum / width + eps);
        }
    }
    __syncthreads();
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const int offset = row * width + column;
        output[offset] = __float2bfloat16(
            __bfloat162float(input[offset])
            * inverse_rms
            * __bfloat162float(weight[column])
        );
    }
}

__global__ void SwiGlu(
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
    const float gate = __bfloat162float(gate_up[row * 2 * hidden_dim + column]);
    const float up = __bfloat162float(gate_up[row * 2 * hidden_dim + hidden_dim + column]);
    hidden[index] = __float2bfloat16((gate / (1.0f + expf(-gate))) * up);
}

__global__ void AddResidual(
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

__global__ void ConvertOutputs(
    const __nv_bfloat16 *policy,
    const __nv_bfloat16 *wdl,
    const __nv_bfloat16 *moves_left,
    float *policy_output,
    float *wdl_output,
    float *moves_left_output,
    int batch_size,
    int policy_size,
    int policy_storage_size
) {
    const int sample = blockIdx.x;
    for (int column = threadIdx.x; column < policy_size; column += blockDim.x) {
        policy_output[sample * policy_size + column] =
            __bfloat162float(policy[sample * policy_storage_size + column]);
    }
    if (threadIdx.x == 0) {
        float logits[3];
        float maximum = -INFINITY;
        for (int i = 0; i < 3; ++i) {
            logits[i] = __bfloat162float(wdl[sample * 32 + i]);
            maximum = fmaxf(maximum, logits[i]);
        }
        float total = 0.0f;
        for (float &logit : logits) {
            logit = expf(logit - maximum);
            total += logit;
        }
        for (int i = 0; i < 3; ++i) {
            wdl_output[sample * 3 + i] = logits[i] / total;
        }
        moves_left_output[sample] = __bfloat162float(moves_left[sample * 32]);
    }
}

}  // namespace

class DenseModel::Impl {
public:
    Impl(const std::string &path, int gpu, int maximum_batch_size) {
        if (maximum_batch_size <= 0 || maximum_batch_size > 1024) {
            throw std::runtime_error("maximum batch size must be in [1, 1024]");
        }
        CheckCuda(cudaSetDevice(gpu), "cudaSetDevice");
        const SafetensorsFile file = LoadSafetensors(path);
        if (file.metadata.at("format") != "chess-engine-4"
            || file.metadata.at("format_version") != "1"
            || file.metadata.at("architecture") != "dense") {
            throw std::runtime_error("Unsupported model format");
        }
        if (file.metadata.at("activation") != "swiglu"
            || file.metadata.at("input_format") != "lc0-classical-112"
            || file.metadata.at("input_normalization")
                != "history-select-rule50-div99-v1"
            || MetadataInt(file.metadata, "input_planes") != kInputPlanes
            || MetadataInt(file.metadata, "board_size") != 8) {
            throw std::runtime_error("Unsupported dense model contract");
        }
        info_ = ModelInfo{
            .d_model = MetadataInt(file.metadata, "d_model"),
            .depth = MetadataInt(file.metadata, "depth"),
            .history_length = MetadataInt(file.metadata, "history_length"),
            .policy_size = MetadataInt(file.metadata, "policy_size"),
            .maximum_batch_size = maximum_batch_size,
        };
        hidden_dim_ = static_cast<int>(
            info_.d_model * MetadataFloat(file.metadata, "expansion_ratio")
        );
        if (info_.d_model <= 0 || info_.depth <= 0
            || info_.history_length <= 0 || info_.history_length > 8
            || info_.policy_size <= 0 || hidden_dim_ <= 0) {
            throw std::runtime_error("Invalid dense model dimensions");
        }
        selected_planes_ = info_.history_length * kPlanesPerHistory + (kInputPlanes - kHistoryPlanes);
        input_dim_ = selected_planes_ * 64;
        policy_storage_size_ = MetadataInt(file.metadata, "policy_storage_size");
        if (policy_storage_size_ < info_.policy_size) {
            throw std::runtime_error("Policy storage is smaller than the lc0 policy output");
        }
        rms_norm_eps_ = MetadataFloat(file.metadata, "rms_norm_eps");

        input_weight_ = DeviceTensor(RequireTensor(
            file, "input.weight", {info_.d_model, input_dim_}
        ));
        input_bias_ = DeviceTensor(RequireTensor(file, "input.bias", {info_.d_model}));
        final_norm_ = DeviceTensor(RequireTensor(
            file, "final_norm.weight", {info_.d_model}
        ));
        policy_weight_ = DeviceTensor(RequireTensor(
            file, "policy.weight", {policy_storage_size_, info_.d_model}
        ));
        policy_bias_ = DeviceTensor(RequireTensor(
            file, "policy.bias", {policy_storage_size_}
        ));
        wdl_weight_ = DeviceTensor(RequireTensor(file, "wdl.weight", {32, info_.d_model}));
        wdl_bias_ = DeviceTensor(RequireTensor(file, "wdl.bias", {32}));
        moves_left_weight_ = DeviceTensor(RequireTensor(
            file, "moves_left.weight", {32, info_.d_model}
        ));
        moves_left_bias_ = DeviceTensor(RequireTensor(file, "moves_left.bias", {32}));
        blocks_.reserve(info_.depth);
        for (int layer = 0; layer < info_.depth; ++layer) {
            const std::string prefix = "blocks." + std::to_string(layer);
            blocks_.push_back(Block{
                .norm = DeviceTensor(RequireTensor(
                    file, prefix + ".norm.weight", {info_.d_model}
                )),
                .gate_up = DeviceTensor(RequireTensor(
                    file,
                    prefix + ".gate_up.weight",
                    {2 * hidden_dim_, info_.d_model}
                )),
                .down = DeviceTensor(RequireTensor(
                    file, prefix + ".down.weight", {info_.d_model, hidden_dim_}
                )),
            });
        }

        CheckCublas(cublasCreate(&cublas_), "cublasCreate");
        CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create stream");
        CheckCublas(cublasSetStream(cublas_, stream_), "cublasSetStream");
        CheckCublas(cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");

        const std::size_t batch = maximum_batch_size;
        device_planes_ = Allocate<InputPlane>(batch * kInputPlanes, "allocate planes");
        expanded_ = Allocate<__nv_bfloat16>(batch * input_dim_, "allocate expanded input");
        x_[0] = Allocate<__nv_bfloat16>(batch * info_.d_model, "allocate x0");
        x_[1] = Allocate<__nv_bfloat16>(batch * info_.d_model, "allocate x1");
        normalized_ = Allocate<__nv_bfloat16>(batch * info_.d_model, "allocate normalized");
        gate_up_ = Allocate<__nv_bfloat16>(batch * 2 * hidden_dim_, "allocate gate_up");
        hidden_ = Allocate<__nv_bfloat16>(batch * hidden_dim_, "allocate hidden");
        policy_ = Allocate<__nv_bfloat16>(batch * policy_storage_size_, "allocate policy");
        wdl_ = Allocate<__nv_bfloat16>(batch * 32, "allocate wdl");
        moves_left_ = Allocate<__nv_bfloat16>(batch * 32, "allocate moves left");
        output_policy_ = Allocate<float>(batch * info_.policy_size, "allocate policy output");
        output_wdl_ = Allocate<float>(batch * 3, "allocate wdl output");
        output_moves_left_ = Allocate<float>(batch, "allocate moves-left output");
    }

    ~Impl() {
        for (const auto &[batch_size, graph] : graphs_) {
            static_cast<void>(batch_size);
            cudaGraphExecDestroy(graph);
        }
        if (cublas_ != nullptr) cublasDestroy(cublas_);
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
        cudaFree(device_planes_);
        cudaFree(expanded_);
        cudaFree(x_[0]);
        cudaFree(x_[1]);
        cudaFree(normalized_);
        cudaFree(gate_up_);
        cudaFree(hidden_);
        cudaFree(policy_);
        cudaFree(wdl_);
        cudaFree(moves_left_);
        cudaFree(output_policy_);
        cudaFree(output_wdl_);
        cudaFree(output_moves_left_);
    }

    void Evaluate(std::span<const InputPlane> planes, int batch_size, Outputs outputs) {
        if (batch_size <= 0 || batch_size > info_.maximum_batch_size) {
            throw std::runtime_error("batch size exceeds model capacity");
        }
        if (planes.size() != static_cast<std::size_t>(batch_size * kInputPlanes)) {
            throw std::runtime_error("input plane count does not match batch size");
        }
        if (outputs.policy.size() != static_cast<std::size_t>(batch_size * info_.policy_size)
            || outputs.wdl.size() != static_cast<std::size_t>(batch_size * 3)
            || outputs.moves_left.size() != static_cast<std::size_t>(batch_size)) {
            throw std::runtime_error("output buffer size does not match batch size");
        }

        CheckCuda(
            cudaMemcpyAsync(
                device_planes_,
                planes.data(),
                planes.size_bytes(),
                cudaMemcpyHostToDevice,
                stream_
            ),
            "copy input planes"
        );
        RunGraph(batch_size);
        CheckCuda(cudaMemcpyAsync(outputs.policy.data(), output_policy_, outputs.policy.size_bytes(),
                                  cudaMemcpyDeviceToHost, stream_), "copy policy output");
        CheckCuda(cudaMemcpyAsync(outputs.wdl.data(), output_wdl_, outputs.wdl.size_bytes(),
                                  cudaMemcpyDeviceToHost, stream_), "copy wdl output");
        CheckCuda(cudaMemcpyAsync(outputs.moves_left.data(), output_moves_left_,
                                  outputs.moves_left.size_bytes(), cudaMemcpyDeviceToHost,
                                  stream_), "copy moves-left output");
        CheckCuda(cudaStreamSynchronize(stream_), "synchronize inference");
    }

    ModelInfo info_{};

private:
    struct Block {
        DeviceTensor norm;
        DeviceTensor gate_up;
        DeviceTensor down;
    };

    void Forward(int batch_size) {
        const int expanded_elements = batch_size * input_dim_;
        ExpandPlanes<<<(expanded_elements + 255) / 256, 256, 0, stream_>>>(
            device_planes_, expanded_, batch_size, info_.history_length, selected_planes_
        );
        Gemm(expanded_, input_weight_.data(), x_[0], batch_size, info_.d_model, input_dim_);
        Bias(x_[0], input_bias_.data(), batch_size, info_.d_model);

        int current = 0;
        for (const Block &block : blocks_) {
            RmsNorm<<<batch_size, 256, 0, stream_>>>(
                x_[current], block.norm.data(), normalized_, batch_size,
                info_.d_model, rms_norm_eps_
            );
            Gemm(
                normalized_, block.gate_up.data(), gate_up_, batch_size,
                2 * hidden_dim_, info_.d_model
            );
            const int hidden_elements = batch_size * hidden_dim_;
            SwiGlu<<<(hidden_elements + 255) / 256, 256, 0, stream_>>>(
                gate_up_, hidden_, hidden_elements, hidden_dim_
            );
            Gemm(hidden_, block.down.data(), x_[1 - current], batch_size,
                 info_.d_model, hidden_dim_);
            const int residual_elements = batch_size * info_.d_model;
            AddResidual<<<(residual_elements + 255) / 256, 256, 0, stream_>>>(
                x_[current], x_[1 - current], residual_elements
            );
            current = 1 - current;
        }
        RmsNorm<<<batch_size, 256, 0, stream_>>>(
            x_[current], final_norm_.data(), normalized_, batch_size,
            info_.d_model, rms_norm_eps_
        );
        Gemm(normalized_, policy_weight_.data(), policy_, batch_size,
             policy_storage_size_, info_.d_model);
        Bias(policy_, policy_bias_.data(), batch_size, policy_storage_size_);
        Gemm(normalized_, wdl_weight_.data(), wdl_, batch_size, 32, info_.d_model);
        Bias(wdl_, wdl_bias_.data(), batch_size, 32);
        Gemm(normalized_, moves_left_weight_.data(), moves_left_, batch_size, 32,
             info_.d_model);
        Bias(moves_left_, moves_left_bias_.data(), batch_size, 32);
        ConvertOutputs<<<batch_size, 256, 0, stream_>>>(
            policy_, wdl_, moves_left_, output_policy_, output_wdl_, output_moves_left_,
            batch_size, info_.policy_size, policy_storage_size_
        );
        CheckCuda(cudaGetLastError(), "dense inference kernels");
    }

    void RunGraph(int batch_size) {
        const auto existing = graphs_.find(batch_size);
        if (existing != graphs_.end()) {
            CheckCuda(cudaGraphLaunch(existing->second, stream_), "launch inference graph");
            return;
        }

        // Warm cuBLAS and initialize TK launch metadata before stream capture.
        Forward(batch_size);
        CheckCuda(cudaStreamSynchronize(stream_), "warm inference graph");
        CheckCuda(
            cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
            "begin inference graph capture"
        );
        Forward(batch_size);
        cudaGraph_t graph = nullptr;
        CheckCuda(cudaStreamEndCapture(stream_, &graph), "end inference graph capture");
        cudaGraphExec_t executable = nullptr;
        const cudaError_t instantiate_status = cudaGraphInstantiate(&executable, graph, 0);
        cudaGraphDestroy(graph);
        CheckCuda(instantiate_status, "instantiate inference graph");
        graphs_.emplace(batch_size, executable);
        CheckCuda(cudaGraphLaunch(executable, stream_), "launch inference graph");
    }

    void Gemm(
        const __nv_bfloat16 *input,
        const __nv_bfloat16 *weight,
        __nv_bfloat16 *output,
        int rows,
        int columns,
        int reduction
    ) {
        if (TkBf16GemmSupported(rows, columns, reduction)) {
            LaunchTkBf16Gemm(input, weight, output, rows, columns, reduction, stream_);
            return;
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CheckCublas(
            cublasGemmEx(
                cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                columns, rows, reduction,
                &alpha,
                weight, CUDA_R_16BF, reduction,
                input, CUDA_R_16BF, reduction,
                &beta,
                output, CUDA_R_16BF, columns,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ),
            "cublasGemmEx"
        );
    }

    void Bias(__nv_bfloat16 *values, const __nv_bfloat16 *bias, int rows, int columns) {
        const int elements = rows * columns;
        AddBias<<<(elements + 255) / 256, 256, 0, stream_>>>(
            values, bias, rows, columns
        );
    }

    int hidden_dim_ = 0;
    int selected_planes_ = 0;
    int input_dim_ = 0;
    int policy_storage_size_ = 0;
    float rms_norm_eps_ = 1e-6f;
    DeviceTensor input_weight_;
    DeviceTensor input_bias_;
    std::vector<Block> blocks_;
    DeviceTensor final_norm_;
    DeviceTensor policy_weight_;
    DeviceTensor policy_bias_;
    DeviceTensor wdl_weight_;
    DeviceTensor wdl_bias_;
    DeviceTensor moves_left_weight_;
    DeviceTensor moves_left_bias_;
    cublasHandle_t cublas_ = nullptr;
    cudaStream_t stream_ = nullptr;
    InputPlane *device_planes_ = nullptr;
    __nv_bfloat16 *expanded_ = nullptr;
    __nv_bfloat16 *x_[2] = {};
    __nv_bfloat16 *normalized_ = nullptr;
    __nv_bfloat16 *gate_up_ = nullptr;
    __nv_bfloat16 *hidden_ = nullptr;
    __nv_bfloat16 *policy_ = nullptr;
    __nv_bfloat16 *wdl_ = nullptr;
    __nv_bfloat16 *moves_left_ = nullptr;
    float *output_policy_ = nullptr;
    float *output_wdl_ = nullptr;
    float *output_moves_left_ = nullptr;
    std::unordered_map<int, cudaGraphExec_t> graphs_;
};

std::unique_ptr<DenseModel> DenseModel::Load(
    const std::string &path,
    int gpu,
    int maximum_batch_size
) {
    return std::unique_ptr<DenseModel>(
        new DenseModel(std::make_unique<Impl>(path, gpu, maximum_batch_size))
    );
}

DenseModel::DenseModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DenseModel::~DenseModel() = default;
DenseModel::DenseModel(DenseModel &&) noexcept = default;
DenseModel &DenseModel::operator=(DenseModel &&) noexcept = default;
const ModelInfo &DenseModel::info() const { return impl_->info_; }

void DenseModel::Evaluate(
    std::span<const InputPlane> planes,
    int batch_size,
    Outputs outputs
) {
    impl_->Evaluate(planes, batch_size, outputs);
}

}  // namespace chess_engine_4::inference
