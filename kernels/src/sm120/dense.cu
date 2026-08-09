#define DENSE_ARCH_NAMESPACE chess_engine_4::sm120
#define DENSE_COMPUTE_CAPABILITY_MAJOR 12
#define DENSE_COMPUTE_CAPABILITY_MINOR 0
#define DENSE_ARCHITECTURE_NAME "SM120"

#include "../common/dense_bf16_gemm.cuh"
#include "../common/dense_bf16_impl.cuh"

void bind_sm120_dense_kernels(pybind11::module_& module) {
    module.def("sm120_dense_bf16_gemm", &chess_engine_4::sm120::dense::Bf16Gemm);
    module.def("sm120_dense_rmsnorm_forward", &chess_engine_4::sm120::dense::RmsNormForward);
    module.def("sm120_dense_swiglu_forward", &chess_engine_4::sm120::dense::SwiGluForward);
    module.def("sm120_dense_residual_add", &chess_engine_4::sm120::dense::ResidualAdd);
    module.def("sm120_dense_swiglu_backward", &chess_engine_4::sm120::dense::SwiGluBackward);
    module.def("sm120_dense_rmsnorm_backward", &chess_engine_4::sm120::dense::RmsNormBackward);
}
