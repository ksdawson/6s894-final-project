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

__device__ void gemm_helper (gemm_struct &gemm_struct) {


}

__device__ void index (int32_t row_ID, int32_t col_ID, int32_t offset_col, int32_t offset_row, int32_t dim) {
    return (offset_col + row_ID) * dim + (offset_row + col_ID);
}

struct Gemm_Struct {
    int32_t N;
    int32_t block_n;
    float const *B1;
    float const *B2;
    float *out;
    int32_t B1_row_offset;
    int32_t B1_col_offset;
    int32_t B2_row_offset;
    int32_t B2_col_offset;
}

// Computes A-BB^T for one SM
// N is now the total dimension of the matrix, block_n is the dimension of the block
__device__ void gemm_one_SM(
    Block_Update &block_info,
    const uint32_t N, const uint32_t block_n, float const *A, float const *B, float *out,
    uint32_t A_col_offset, uint32_t A_row_offset, uint32_t B_col_offset, uint32_t B_row_offset
) {

    int32_t N = block_info.n;
    int32_t block_n = block_info.m;
    float const *A = block_info.A;
    float const *B = block_info.L;
    float *out = block_info.out;
    int32_t offset_j = gemm_struct.j;
    int32_t offset_i = gemm_struct.i;

    Gemm_Struct gemm_struct;
    gemm_struct.N = N;
    gemm_struct.block_n = block_n;
    gemm_struct.out = out;

    for (int32_t k = 0; k < j; ++k) {
        gemm_struct.B1 = B + offset_i * block_n * N + k * block_n;
        gemm_struct.B2 = B + k * block_n * N + offset_j * block_n;
        gemm_struct.B1_col_offset = offset_i * block_n;
        gemm_struct.B1_row_offset = k * block_n;
        gemm_struct.B2_col_offset = k * block_n;
        gemm_struct.B2_row_offset = offset_j * block_n;
        gemm_helper(gemm_struct);
    }

    int32_t col_ID = threadIdx.x % block_n;
    int32_t row_ID = threadIdx.x / block_n;
    
    // subtract out from A
    float sum = A[index(row_ID, col_ID, offset_i * block_n, offset_j * block_n, N)] = 


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
} // namespace gemm_naive