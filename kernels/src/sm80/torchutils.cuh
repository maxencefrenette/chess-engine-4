#pragma once

#include "kittens.cuh"

#include <ATen/core/Tensor.h>
#include <torch/extension.h>

#include <array>

namespace kittens::py {

template <typename Global>
Global tensor_to_gl(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(tensor.dim() <= 4, "Tensor rank must not exceed four");
    std::array<int, 4> shape = {1, 1, 1, 1};
    for (int index = 0; index < tensor.dim(); ++index) {
        shape[4 - tensor.dim() + index] = tensor.size(index);
    }
    return kittens::make_gl<Global>(
        reinterpret_cast<uint64_t>(tensor.data_ptr()),
        shape[0], shape[1], shape[2], shape[3]
    );
}

}  // namespace kittens::py
