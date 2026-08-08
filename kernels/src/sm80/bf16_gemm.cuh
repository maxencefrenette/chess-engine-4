#pragma once

#include "kittens.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace chess_engine_4::sm80 {

using namespace kittens;

constexpr int kGemmTile = 16;
constexpr int kGemmWarps = 8;
constexpr int kGemmThreads = kGemmWarps * 32;

using MatrixGlobal = gl<bf16, 1, 1, -1, -1>;

struct Bf16GemmGlobals {
    MatrixGlobal left;
    MatrixGlobal right;
    MatrixGlobal output;
    int rows;
    int columns;
    int reduction;
};

__global__ void Bf16GemmKernel(Bf16GemmGlobals globals) {
    const int warp = threadIdx.x / 32;
    const int column_tiles = globals.columns / kGemmTile;
    const int tasks = globals.rows / kGemmTile * column_tiles;
    for (
        int task = blockIdx.x * kGemmWarps + warp;
        task < tasks;
        task += gridDim.x * kGemmWarps
    ) {
        const int row_tile = task / column_tiles;
        const int column_tile = task % column_tiles;
        rt_bf<kGemmTile, kGemmTile> left;
        rt_bf<kGemmTile, kGemmTile> right;
        rt_fl<kGemmTile, kGemmTile> accumulator;
        warp::zero(accumulator);
        for (int k = 0; k < globals.reduction / kGemmTile; ++k) {
            warp::load(left, globals.left, {0, 0, row_tile, k});
            warp::load(right, globals.right, {0, 0, column_tile, k});
            warp::mma_ABt(accumulator, left, right, accumulator);
        }
        warp::store(globals.output, accumulator, {0, 0, row_tile, column_tile});
    }
}

inline Bf16GemmGlobals MakeBf16GemmGlobals(
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    __nv_bfloat16* output,
    int rows,
    int columns,
    int reduction
) {
    return {
        .left = MatrixGlobal{
            reinterpret_cast<bf16*>(const_cast<__nv_bfloat16*>(left)),
            nullptr,
            nullptr,
            static_cast<std::size_t>(rows),
            static_cast<std::size_t>(reduction)
        },
        .right = MatrixGlobal{
            reinterpret_cast<bf16*>(const_cast<__nv_bfloat16*>(right)),
            nullptr,
            nullptr,
            static_cast<std::size_t>(columns),
            static_cast<std::size_t>(reduction)
        },
        .output = MatrixGlobal{
            reinterpret_cast<bf16*>(output),
            nullptr,
            nullptr,
            static_cast<std::size_t>(rows),
            static_cast<std::size_t>(columns)
        },
        .rows = rows,
        .columns = columns,
        .reduction = reduction,
    };
}

inline void LaunchBf16Gemm(
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    __nv_bfloat16* output,
    int rows,
    int columns,
    int reduction,
    int multiprocessors,
    cudaStream_t stream
) {
    const int tasks = rows / kGemmTile * (columns / kGemmTile);
    const int blocks = std::min(
        (tasks + kGemmWarps - 1) / kGemmWarps,
        multiprocessors * 4
    );
    Bf16GemmKernel<<<blocks, kGemmThreads, 0, stream>>>(
        MakeBf16GemmGlobals(left, right, output, rows, columns, reduction)
    );
}

}  // namespace chess_engine_4::sm80
