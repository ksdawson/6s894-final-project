// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh"]}
// TL {"workspace_files": []}

#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "utils.cuh"

namespace gemm {
// Computes A- BB^T, current naive gemm can only handle 32x32 matrix
// N: number of blocks in triblock diagonal
// block_n: dimension of each block in triblock diagonal
// A_col_offset, A_row_offset: offset for block of interest in A in the global memory
// B_col_offset, B_row_offset: offset for block of interest in B in the global memory
// out: outputs A- BB^T
__device__ void gemm_naive_XY(
    const uint32_t N, const uint32_t block_n, float const *A, float const *B, float *out,
    uint32_t A_col_offset, uint32_t A_row_offset, uint32_t B_col_offset, uint32_t B_row_offset
) {
    const uint32_t dim = N * block_n;
    int32_t col_ID = threadIdx.x % block_n;
    int32_t row_ID = threadIdx.x / block_n;

    float sum = 0.0f;
    for (uint32_t i = 0; i < block_n; ++i) {
        sum += B[(B_col_offset + row_ID) * dim + (B_row_offset + i)] * B[(B_col_offset + col_ID) * dim + (B_row_offset + i)];
    }

    if (row_ID < block_n && col_ID < block_n) {
        out[row_ID * block_n + col_ID] = A[(A_col_offset + row_ID) * dim + (A_row_offset + col_ID)] - sum;
    }
}

struct BlockUpdate {
    float const *A; // input matrix
    float const *L; // Chol matrix
    const uint32_t n; // matrix size
    const uint32_t m; // block size
    const uint32_t i; // Lik * Ljk^T
    const uint32_t j;
    float *out; // add result to out (likely register array)
};

__device__ int32_t index (int32_t row_ID, int32_t col_ID, int32_t offset_col, int32_t offset_row, int32_t dim) {
    return (offset_col + row_ID) * dim + (offset_row + col_ID);
}

// can handle maximum block size of 64x64 due to shared memory size limit
template <int32_t T_TH, int32_t T_TW>
__device__ void block_gemm (BlockUpdate gemm_info, const uint32_t k, float *shared_mem) {

    int32_t N = gemm_info.n;
    int32_t block_n = gemm_info.m;
    float const *B1 = gemm_info.L;
    float const *B2 = gemm_info.L;
    float *out = gemm_info.out;
    int32_t B1_row_offset = k * block_n;
    int32_t B1_col_offset = gemm_info.i * block_n;
    int32_t B2_row_offset = k * block_n;
    int32_t B2_col_offset = gemm_info.j * block_n;

    for (int32_t i = 0; i < block_n * block_n; i += blockDim.x) {
        int32_t row_ID = (int32_t)((i + threadIdx.x) / block_n);
        int32_t col_ID = int32_t((i + threadIdx.x) % block_n);
        shared_mem[threadIdx.x + i] = B1[index(row_ID, col_ID, B1_col_offset, B1_row_offset, N)];
        shared_mem[threadIdx.x + i + block_n * block_n] = B2[index(row_ID, col_ID, B2_col_offset, B2_row_offset, N)];
    }
    __syncthreads();

    int32_t row_ID = (int32_t)(threadIdx.x / 32) * T_TH;
    int32_t col_ID = int32_t(threadIdx.x % 32) * T_TW;

    float sum[T_TH * T_TW];
    float b1_val[T_TH];
    float b2_val[T_TW];
    #pragma unroll
    for (int32_t i = 0; i < T_TH * T_TW; ++i) {
        sum[i] = 0.0f;
    }
    #pragma unroll
    for (int32_t i = 0; i < T_TH; ++i) {
        b1_val[i] = 0.0f;
    }
    #pragma unroll
    for (int32_t i = 0; i < T_TW; ++i) {
        b2_val[i] = 0.0f;
    }

    for (int32_t i = 0; i < block_n; ++i) {
        // computing B1 * B2^T
        for (int32_t tile_row = 0; tile_row < T_TH; ++tile_row) {
            b1_val[tile_row] = shared_mem[index(row_ID + tile_row, i, 0, 0, block_n)];
        }
        for (int32_t tile_col = 0; tile_col < T_TW; ++tile_col) {
            b2_val[tile_col] = shared_mem[index(col_ID + tile_col, i, 0, 0, block_n) + block_n * block_n];
        }

        for (int32_t tile_row = 0; tile_row < T_TH; ++tile_row) {
            for (int32_t tile_col = 0; tile_col < T_TW; ++tile_col) {
                sum[tile_row * T_TW + tile_col] += b1_val[tile_row] * b2_val[tile_col];
            }
        }


    }

    for (int32_t tile_row = 0; tile_row < T_TH; ++tile_row) {
        for (int32_t tile_col = 0; tile_col < T_TW; ++tile_col) {
            out[tile_row * T_TW + tile_col] += sum[tile_row * T_TW + tile_col];
        }
    }
    __syncthreads();
}


} // namespace gemm_naive