#include "bf16_gemm.h"

#include "kittens.cuh"

#include <stdexcept>

namespace chess_engine_4::inference {
namespace {

using namespace kittens;

constexpr int kTile = 16;
using Matrix = gl<bf16, 1, 1, -1, -1>;

struct GemmGlobals {
    Matrix input;
    Matrix weight;
    Matrix output;
    int rows;
    int columns;
    int reduction;
};

template <int ColumnBlock>
__global__ __launch_bounds__(256) void RowTileGemmKernel(GemmGlobals globals) {
    const int warp = threadIdx.x / 32;
    const int column_tiles = globals.columns / ColumnBlock;
    const int tasks = globals.rows / kTile * column_tiles;
    for (int task = blockIdx.x * 8 + warp;
         task < tasks;
         task += gridDim.x * 8) {
        const int row_tile = task / column_tiles;
        const int column_tile = task % column_tiles;
        rt_bf<kTile, kTile> input_tile;
        rt_bf<ColumnBlock, kTile> weight_tile;
        rt_fl<kTile, ColumnBlock> output_tile;
        warp::zero(output_tile);
        for (int reduction_tile = 0;
             reduction_tile < globals.reduction / kTile;
             ++reduction_tile) {
            warp::load(input_tile, globals.input, {0, 0, row_tile, reduction_tile});
            warp::load(
                weight_tile,
                globals.weight,
                {0, 0, column_tile, reduction_tile}
            );
            warp::mma_ABt(output_tile, input_tile, weight_tile, output_tile);
        }
        warp::store(globals.output, output_tile, {0, 0, row_tile, column_tile});
    }
}

template <int BlockRows, int ColumnBlock>
__global__ __launch_bounds__((BlockRows / kTile) * 32) void CtaGemmKernel(
    GemmGlobals globals
) {
    constexpr int Warps = BlockRows / kTile;
    extern __shared__ alignment_dummy shared[];
    shared_allocator allocator(reinterpret_cast<int*>(shared));
    auto &input_shared = allocator.allocate<st_bf<BlockRows, kTile>>();
    auto &weight_shared = allocator.allocate<st_bf<ColumnBlock, kTile>>();

    rt_bf<kTile, kTile> input_tile;
    rt_bf<ColumnBlock, kTile> weight_tile;
    rt_fl<kTile, ColumnBlock> output_tile;
    warp::zero(output_tile);

    for (int reduction_tile = 0;
         reduction_tile < globals.reduction / kTile;
         ++reduction_tile) {
        group<Warps>::load(
            input_shared,
            globals.input,
            {0, 0, static_cast<int>(blockIdx.y), reduction_tile}
        );
        group<Warps>::load(
            weight_shared,
            globals.weight,
            {0, 0, static_cast<int>(blockIdx.x), reduction_tile}
        );
        __syncthreads();
        group<Warps>::load(input_tile, input_shared);
        warp::load(weight_tile, weight_shared);
        warp::mma_ABt(output_tile, input_tile, weight_tile, output_tile);
        __syncthreads();
    }

    const int warp = threadIdx.x / 32;
    const int local_row = Warps % 4 == 0
        ? warp / 4 + (warp % 4) * (Warps / 4)
        : warp;
    warp::store(
        globals.output,
        output_tile,
        {
            0,
            0,
            static_cast<int>(blockIdx.y) * Warps + local_row,
            static_cast<int>(blockIdx.x),
        }
    );
}

template <int BlockRows, int ColumnBlock>
void LaunchCtaGemm(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows,
    int columns,
    int reduction,
    cudaStream_t stream
) {
    GemmGlobals globals{
        .input = Matrix{
            reinterpret_cast<bf16*>(const_cast<__nv_bfloat16*>(input)),
            nullptr,
            nullptr,
            static_cast<std::size_t>(rows),
            static_cast<std::size_t>(reduction),
        },
        .weight = Matrix{
            reinterpret_cast<bf16*>(const_cast<__nv_bfloat16*>(weight)),
            nullptr,
            nullptr,
            static_cast<std::size_t>(columns),
            static_cast<std::size_t>(reduction),
        },
        .output = Matrix{
            reinterpret_cast<bf16*>(output),
            nullptr,
            nullptr,
            static_cast<std::size_t>(rows),
            static_cast<std::size_t>(columns),
        },
        .rows = rows,
        .columns = columns,
        .reduction = reduction,
    };
    constexpr int Warps = BlockRows / kTile;
    constexpr std::size_t SharedBytes =
        sizeof(st_bf<BlockRows, kTile>) + sizeof(st_bf<ColumnBlock, kTile>);
    CtaGemmKernel<BlockRows, ColumnBlock><<<
        dim3(columns / ColumnBlock, rows / BlockRows),
        Warps * 32,
        SharedBytes,
        stream
    >>>(globals);
}

template <int ColumnBlock>
void DispatchRows(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows,
    int columns,
    int reduction,
    cudaStream_t stream
) {
    if (rows % 32 != 0) {
        GemmGlobals globals{
            .input = Matrix{
                reinterpret_cast<bf16*>(const_cast<__nv_bfloat16*>(input)),
                nullptr,
                nullptr,
                static_cast<std::size_t>(rows),
                static_cast<std::size_t>(reduction),
            },
            .weight = Matrix{
                reinterpret_cast<bf16*>(const_cast<__nv_bfloat16*>(weight)),
                nullptr,
                nullptr,
                static_cast<std::size_t>(columns),
                static_cast<std::size_t>(reduction),
            },
            .output = Matrix{
                reinterpret_cast<bf16*>(output),
                nullptr,
                nullptr,
                static_cast<std::size_t>(rows),
                static_cast<std::size_t>(columns),
            },
            .rows = rows,
            .columns = columns,
            .reduction = reduction,
        };
        const int tasks = rows / kTile * (columns / ColumnBlock);
        RowTileGemmKernel<ColumnBlock><<<
            (tasks + 7) / 8,
            256,
            0,
            stream
        >>>(globals);
    } else if (rows % 128 == 0) {
        LaunchCtaGemm<128, ColumnBlock>(
            input, weight, output, rows, columns, reduction, stream
        );
    } else if (rows % 64 == 0) {
        LaunchCtaGemm<64, ColumnBlock>(
            input, weight, output, rows, columns, reduction, stream
        );
    } else if (rows % 32 == 0) {
        LaunchCtaGemm<32, ColumnBlock>(
            input, weight, output, rows, columns, reduction, stream
        );
    }
}

}  // namespace

bool CustomBf16GemmSupported(int rows, int columns, int reduction) {
    return rows > 0 && rows % kTile == 0
        && columns > 0 && columns % 32 == 0
        && reduction > 0 && reduction % kTile == 0;
}

void LaunchCustomBf16Gemm(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows,
    int columns,
    int reduction,
    cudaStream_t stream
) {
    if (!CustomBf16GemmSupported(rows, columns, reduction)) {
        throw std::runtime_error("unsupported SM120 BF16 GEMM shape");
    }
    if (columns % 128 == 0) {
        DispatchRows<128>(input, weight, output, rows, columns, reduction, stream);
    } else if (columns % 64 == 0) {
        DispatchRows<64>(input, weight, output, rows, columns, reduction, stream);
    } else {
        DispatchRows<32>(input, weight, output, rows, columns, reduction, stream);
    }
}

}  // namespace chess_engine_4::inference
