#include "tk_bf16.h"

#include "kittens.cuh"

#define main thunderkittens_bf16_benchmark_main
#include "../../../third_party/ThunderKittens/kernels/gemm/bf16_b200/bf16_b200_gemm.cu"
#undef main

namespace chess_engine_4::inference {
namespace {

template <int OutputTile, int ReductionTile>
using Config = ::config<
    256,
    OutputTile,
    ReductionTile,
    4,
    true,
    2,
    OutputTile / 32
>;

template <int OutputTile, int ReductionTile>
void Launch(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows,
    int columns,
    int reduction,
    cudaStream_t stream
) {
    using GemmConfig = Config<OutputTile, ReductionTile>;
    using Globals = ::globals<GemmConfig>;
    Globals globals{
        .a = typename Globals::a_gl{
            reinterpret_cast<kittens::bf16*>(const_cast<__nv_bfloat16*>(input)),
            nullptr,
            nullptr,
            static_cast<std::size_t>(rows),
            static_cast<std::size_t>(reduction)
        },
        .b = typename Globals::b_gl{
            reinterpret_cast<kittens::bf16*>(const_cast<__nv_bfloat16*>(weight)),
            nullptr,
            nullptr,
            static_cast<std::size_t>(columns),
            static_cast<std::size_t>(reduction)
        },
        .d = typename Globals::d_gl{
            reinterpret_cast<kittens::bf16*>(output),
            nullptr,
            nullptr,
            static_cast<std::size_t>(rows),
            static_cast<std::size_t>(columns)
        },
    };
    CUDACHECK(cudaFuncSetAttribute(
        ::kernel<GemmConfig>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        globals.dynamic_shared_memory()
    ));
    kittens::LaunchConfig<true, true> launch_config(
        globals.grid(),
        globals.block(),
        globals.dynamic_shared_memory(),
        stream,
        GemmConfig::CLUSTER_SIZE
    );
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        ::kernel<GemmConfig>,
        globals
    ));
}

template <int OutputTile>
void DispatchReduction(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows,
    int columns,
    int reduction,
    cudaStream_t stream
) {
    if (reduction == 32) {
        Launch<OutputTile, 32>(input, weight, output, rows, columns, reduction, stream);
    } else if (reduction == 64) {
        Launch<OutputTile, 64>(input, weight, output, rows, columns, reduction, stream);
    } else {
        Launch<OutputTile, 128>(input, weight, output, rows, columns, reduction, stream);
    }
}

}  // namespace

bool TkBf16GemmSupported(int rows, int columns, int reduction) {
    const bool columns_supported =
        columns == 32 || columns == 64 || columns == 128 || columns % 256 == 0;
    const bool reduction_supported =
        reduction == 32 || reduction == 64 || reduction % 128 == 0;
    return rows % 256 == 0 && columns_supported && reduction_supported;
}

void LaunchTkBf16Gemm(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows,
    int columns,
    int reduction,
    cudaStream_t stream
) {
    if (columns == 32) {
        DispatchReduction<32>(input, weight, output, rows, columns, reduction, stream);
    } else if (columns == 64) {
        DispatchReduction<64>(input, weight, output, rows, columns, reduction, stream);
    } else if (columns == 128) {
        DispatchReduction<128>(input, weight, output, rows, columns, reduction, stream);
    } else {
        DispatchReduction<256>(input, weight, output, rows, columns, reduction, stream);
    }
}

}  // namespace chess_engine_4::inference
