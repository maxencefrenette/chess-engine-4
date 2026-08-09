#define DENSE_ARCH_NAMESPACE chess_engine_4::sm80
#define DENSE_COMPUTE_CAPABILITY_MAJOR 8
#define DENSE_COMPUTE_CAPABILITY_MINOR 0
#define DENSE_ARCHITECTURE_NAME "SM80"

#include "../common/dense_bf16_gemm.cuh"
#include "../common/dense_bf16_impl.cuh"

void bind_sm80_dense_kernels(pybind11::module_& module) {
    module.def("sm80_dense_bf16_gemm", &chess_engine_4::sm80::dense::Bf16Gemm);
    module.def("sm80_dense_rmsnorm_forward", &chess_engine_4::sm80::dense::RmsNormForward);
    module.def("sm80_dense_swiglu_forward", &chess_engine_4::sm80::dense::SwiGluForward);
    module.def("sm80_dense_residual_add", &chess_engine_4::sm80::dense::ResidualAdd);
    module.def("sm80_dense_swiglu_backward", &chess_engine_4::sm80::dense::SwiGluBackward);
    module.def("sm80_dense_rmsnorm_backward", &chess_engine_4::sm80::dense::RmsNormBackward);
}
