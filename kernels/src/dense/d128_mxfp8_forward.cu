// Compile ThunderKittens' reference Blackwell MXFP8 quantizer and GEMM as a
// project-owned extension. The dense-block composition lives in Python so this
// first baseline can be checked and benchmarked before replacing it with a
// persistent fused kernel.
#define _C _chess_engine_4_kernels
#include "../../../third_party/ThunderKittens/kernels/gemm/mxfp8_b200/mxfp8_b200_gemm.cu"
