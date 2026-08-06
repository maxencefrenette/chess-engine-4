#include <pybind11/pybind11.h>

void bind_dense_kernels(pybind11::module_ &module);

PYBIND11_MODULE(_chess_engine_4_kernels, module) {
    bind_dense_kernels(module);
}
