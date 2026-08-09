#define CE4_BF16_GEMM_NAMESPACE chess_engine_4::sm90
#define CE4_DENSE_NAMESPACE chess_engine_4::sm90::dense
#define CE4_DENSE_GEMM_NAMESPACE chess_engine_4::sm90
#define CE4_DENSE_COMPUTE_CAPABILITY_MAJOR 9
#define CE4_DENSE_COMPUTE_CAPABILITY_MINOR 0
#define CE4_DENSE_ARCHITECTURE_NAME "SM90"
#define CE4_DENSE_BIND_FUNCTION bind_sm90_dense_kernels
#define CE4_DENSE_BIND_PREFIX "sm90_"

// Hopper supports the same warp-tiled BF16 implementation as Ampere. Keep the
// first SM90 path deliberately simple until an end-to-end benchmark justifies
// adopting a more specialized WGMMA schedule.
#include "../sm80/dense.cu"
