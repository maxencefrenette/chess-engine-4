#define DENSE_ARCH_NAMESPACE chess_engine_4::sm90
#define DENSE_COMPUTE_CAPABILITY_MAJOR 9
#define DENSE_COMPUTE_CAPABILITY_MINOR 0
#define DENSE_ARCHITECTURE_NAME "SM90"

// Hopper supports the same warp-tiled BF16 implementation as Ampere. Keep the
// first SM90 path deliberately simple until an end-to-end benchmark justifies
// adopting a more specialized WGMMA schedule.
#include "../common/dense_bf16_gemm.cuh"
#include "../common/dense_bf16_impl.cuh"

void bind_sm90_dense_kernels(pybind11::module_& module) {
    module.def("sm90_dense_bf16_gemm", &chess_engine_4::sm90::dense::Bf16Gemm);
    module.def("sm90_dense_rmsnorm_forward", &chess_engine_4::sm90::dense::RmsNormForward);
    module.def("sm90_dense_swiglu_forward", &chess_engine_4::sm90::dense::SwiGluForward);
    module.def("sm90_dense_residual_add", &chess_engine_4::sm90::dense::ResidualAdd);
    module.def("sm90_dense_swiglu_backward", &chess_engine_4::sm90::dense::SwiGluBackward);
    module.def("sm90_dense_rmsnorm_backward", &chess_engine_4::sm90::dense::RmsNormBackward);
}
