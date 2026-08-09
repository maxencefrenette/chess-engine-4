#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <pybind11/pybind11.h>

#define MOE_COMPUTE_CAPABILITY_MAJOR 10
#define MOE_COMPUTE_CAPABILITY_MINOR 0
#define MOE_ARCHITECTURE_NAME "SM100"

#define MOE_D_MODEL 128
#define MOE_NAMESPACE chess_engine_4::sm100::moe_d128
#include "../common/moe_impl.cuh"
#undef MOE_NAMESPACE
#undef MOE_D_MODEL

#define MOE_D_MODEL 256
#define MOE_NAMESPACE chess_engine_4::sm100::moe_d256
#include "../common/moe_impl.cuh"
#undef MOE_NAMESPACE
#undef MOE_D_MODEL

#define MOE_D_MODEL 512
#define MOE_NAMESPACE chess_engine_4::sm100::moe_d512
#include "../common/moe_impl.cuh"
#undef MOE_NAMESPACE
#undef MOE_D_MODEL

void bind_sm100_moe_kernels(pybind11::module_ &module) {
    module.def("sm100_moe_d128_forward", &chess_engine_4::sm100::moe_d128::forward);
    module.def(
        "sm100_moe_d128_training_forward",
        &chess_engine_4::sm100::moe_d128::training_forward
    );
    module.def("sm100_moe_d128_backward", &chess_engine_4::sm100::moe_d128::backward);
    module.def("sm100_moe_d256_forward", &chess_engine_4::sm100::moe_d256::forward);
    module.def(
        "sm100_moe_d256_training_forward",
        &chess_engine_4::sm100::moe_d256::training_forward
    );
    module.def("sm100_moe_d256_backward", &chess_engine_4::sm100::moe_d256::backward);
    module.def("sm100_moe_d512_forward", &chess_engine_4::sm100::moe_d512::forward);
    module.def(
        "sm100_moe_d512_training_forward",
        &chess_engine_4::sm100::moe_d512::training_forward
    );
    module.def("sm100_moe_d512_backward", &chess_engine_4::sm100::moe_d512::backward);
}
