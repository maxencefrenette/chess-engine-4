#include <pybind11/pybind11.h>

void bind_sm100_dense_kernels(pybind11::module_ &module);
void bind_sm100_moe_kernels(pybind11::module_ &module);
void bind_sm120_moe_kernels(pybind11::module_ &module);
void bind_sm80_dense_kernels(pybind11::module_ &module);
void bind_sm80_moe_kernels(pybind11::module_ &module);

PYBIND11_MODULE(_chess_engine_4_kernels, module) {
    bind_sm100_dense_kernels(module);
    bind_sm100_moe_kernels(module);
    bind_sm120_moe_kernels(module);
    bind_sm80_dense_kernels(module);
    bind_sm80_moe_kernels(module);
}
