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

// // Macro to check CUDA errors
// #define CUDA_CHECK(err) \
//     if ((err) != cudaSuccess) { \
//         fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
//         exit(EXIT_FAILURE); \
//     }

////////////////////////////////////////////////////////////////////////////////
// Cholesky Naive Implementation
namespace cholesky_naive {

size_t get_workspace_size(int32_t size) {
    // Allocate enough space to hold size_i * size_j * size_k floats
    // Return size in bytes
    // Use size_t to avoid overflow on intermediate multiplication by casting early
    return 0;
}

__global__ void cholesky_naive(
    const uint32_t n, float const *in, float *out
) {
    // Iterate over all rows
    for (uint32_t i = 0; i < n; ++i) {
        // Iterate over lower triangle off-diagonal cols
        for (uint32_t j = 0; j < i; ++j) {
            // Each thread computes a piece of the sum
            float tmp = 0.0f;
            for (uint32_t k = threadIdx.x; k < j; k += 32) {
                tmp += out[i * n + k] * out[j * n + k];
            }
            // Combine the sum across the warp
            tmp = utils::warp_prefix_sum<float>(tmp);
            // Last thread handles writing it back
            if (threadIdx.x == 31) {
                out[i * n + j] = (in[i * n + j] - tmp) / out[j * n + j];
            }
        }
        // Handle diagonal col
        float tmp = 0.0f;
        for (uint32_t k = threadIdx.x; k < i; k += 32) {
            tmp += out[i * n + k] * out[i * n + k];
        }
        tmp = utils::warp_prefix_sum<float>(tmp);
        if (threadIdx.x == 31) {
            out[i * n + i] = sqrtf((in[i * n + i] - tmp));
        }
    }
}

// Computes out = in*in^T
// N: number of blocks in triblock diagonal
// block_n: dimension of each block in triblock diagonal
// dim_in: dimension of in
// out_col_offset, out_row_offset: offset for block of interest in out in the global memory
__device__ void cholesky_parallel_col_XY(
    const uint32_t dim_in, 
    const uint32_t N, const uint32_t block_n, float const *in, float *out,
    uint32_t out_col_offset, uint32_t out_row_offset) {
    
    const uint32_t dim_out = N * block_n;
    const int32_t tile_size = 1;
    float diag = 0.0f;
    float sum_list[tile_size];
    int32_t row_ID = threadIdx.x * tile_size;
    if (row_ID >= block_n) row_ID = 0;

    for (uint32_t j = 0; j < block_n; ++j) {
        diag = in[j * dim_in + j];
        for (uint32_t i = 0; i < j; ++i) {
            diag -= out[(out_col_offset + j) * dim_out + (out_row_offset + i)] * out[(out_col_offset + j) * dim_out + (out_row_offset + i)];
        }
        diag = sqrtf(diag);
        out[(out_col_offset + j) * dim_out + (out_row_offset + j)] = diag;
        //printf("out[%u*%u + %u] = %f\n", j, n, j, out[j * n + j]);

        for (uint32_t t = 0; t < tile_size; ++t) {
            sum_list[t] = in[(row_ID + t) * dim_in + (j)];

            for (uint32_t k = 0; k < j; ++k) {
                sum_list[t] -= out[(out_col_offset + j) * dim_out + (out_row_offset + k)] * out[(out_col_offset + row_ID + t) * dim_out + (out_row_offset + k)];
            }
            sum_list[t] /= diag;
        }
        
        for (uint32_t t = 0; t < tile_size; ++t) {
            if (row_ID +t > j && row_ID +t < block_n) {
                out[(out_col_offset + row_ID + t) * dim_out + (out_row_offset + j)] = sum_list[t];
                //printf("out[%u*%u + %u] = %f\n", row_ID, n, j, sum);
            }
        }
        __syncthreads();
    }
}

__device__ void cholesky_XY(
    const uint32_t dim_in, 
    const uint32_t N, const uint32_t block_n, float const *in, float *out,
    uint32_t out_col_offset, uint32_t out_row_offset) {
    
    const uint32_t dim_out = N * block_n;
    const int32_t tile_size = 4;
    float diag = 0.0f;
    float sum_list[tile_size];
    float tmp[tile_size];
    int32_t row_ID = (threadIdx.x / 32) * tile_size;
    int32_t warp_ID = threadIdx.x % 32;
    if (row_ID >= block_n) row_ID = 0;

    for (uint32_t j = 0; j < block_n; ++j) {
        // solving for the diagonal element of the jth column
        diag = in[j * dim_in + j];
        for (uint32_t i = 0; i < j; ++i) {
            diag -= out[(out_col_offset + j) * dim_out + (out_row_offset + i)] * out[(out_col_offset + j) * dim_out + (out_row_offset + i)];
        }
        diag = sqrtf(diag);
        out[(out_col_offset + j) * dim_out + (out_row_offset + j)] = diag;
        //printf("out[%u*%u + %u] = %f\n", j, n, j, out[j * n + j]);

        // solving for the off-diagonal elements of the jth column
        for (uint32_t t = 0; t < tile_size; ++t) {
            tmp[t] = 0.0f;
        }

        for (uint32_t t = 0; t < tile_size; ++t) {
            sum_list[t] = in[(row_ID+t)*dim_in + j];

            for (uint32_t k = warp_ID; k < j; k += 32) {
                tmp[t] += out[(out_col_offset + j)*dim_out + out_row_offset + k] * out[(out_col_offset + row_ID + t) *dim_out + out_row_offset + k];
            }

            tmp[t] = utils::warp_prefix_sum<float>(tmp[t]);
            if (warp_ID == 31) {
                sum_list[t] = (sum_list[t] - tmp[t]) / diag;
                if (row_ID + t > j && row_ID + t < block_n) {
                    out[(out_col_offset+row_ID+t)*dim_out + (out_row_offset + j)] = sum_list[t];
                }
            }
        }
        __syncthreads();
    }
}
} // namespace cholesky_naive
