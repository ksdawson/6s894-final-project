// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh", "gpu_block_kernel_fusion.cuh"]}
// TL {"workspace_files": []}

#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "utils.cuh"
#include "gpu_block_kernel_fusion.cuh"

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

// T_TH: tile size of each thread
// num_threads_H: number of threads per row/column in the GPU block
// X: input matrix, to perform X*X^T
// smem: shared memory to store X for computation
// reg: register array to store results of X*X^T
// N: matrix size, determines padding
// block_n: block size of triblock, determines loop condition
template <uint32_t T_TH, uint32_t num_threads_H> 
__device__ void triblock_diag_gemm_GPUblock(float *X_rows_i, float *X_rows_j, 
    float *smem1, float *smem2, float *reg, 
    const int32_t thread_tile_i, const int32_t thread_tile_j,
    const uint32_t N, const uint32_t block_n) {
    
    for (uint32_t k = 0; k < block_n; k += T_TH * num_threads_H) {

        // copy X to smem
        block_cholesky_space::gmem_to_smem(X_rows_i + k, smem1, N, T_TH * num_threads_H);
        block_cholesky_space::gmem_to_smem(X_rows_j + k, smem2, N, T_TH * num_threads_H);
        __syncthreads();

        // solve X * X^T
        // if (tile_i == tile_j) {
        //     diag_block_gemm_naive<T_TH, T_TW>(smem, reg, tile_i, tile_j);
        // } else {
        //     block_gemm_naive<T_TH, T_TW>(smem, reg, tile_i, tile_j);
        // }
        block_cholesky_space::block_gemm_naive<T_TH * num_threads_H, T_TH * num_threads_H, T_TH * num_threads_H, T_TH, T_TH>(
            smem1, smem2, reg, thread_tile_i, thread_tile_j);
        __syncthreads();
    }
}

// requires shared memory of size at least (T_TH * num_threads_H)^2
template <uint32_t T_TH, uint32_t num_threads_H>
__global__ void triblock_diagonal_gemm(float *A, float *X, const uint32_t N, const uint32_t block_n, const uint32_t smem_size_bytes) {
    extern __shared__ float smem[];

    float reg[T_TH * T_TH] = {0.0f};

    // Map rectangular to triangular tiles
    const uint32_t block_tile_i = (uint32_t)((sqrtf(8.f * blockIdx.x + 1.f) - 1.f) * 0.5f);
    const uint32_t block_tile_j = blockIdx.x - (block_tile_i * (block_tile_i + 1) / 2);

    // calculate thread tile indices
    const uint32_t thread_tile_i = threadIdx.x / num_threads_H;
    const uint32_t thread_tile_j = threadIdx.x % num_threads_H;
    float *smem1 = smem;
    float *smem2 = smem1 + smem_size_bytes/(2*sizeof(float));

    const uint32_t valid_tiles = block_n / (T_TH * num_threads_H);
    if (block_tile_i < valid_tiles && block_tile_j < valid_tiles) {

        // locate X for tile_i, tile_j
        float *X_i = X + block_tile_i * (T_TH * num_threads_H) * N;
        float *X_j = X + block_tile_j * (T_TH * num_threads_H) * N;
        float *A_ij = A + block_tile_i * (T_TH * num_threads_H) * N + block_tile_j * (T_TH * num_threads_H);

        // solve X * X^T and store in reg iteratively in the k dimension
        triblock_diag_gemm_GPUblock<T_TH, num_threads_H>(
            X_i, X_j, smem1, smem2, reg, thread_tile_i, thread_tile_j, N, block_n);
    
        // when tile_i == tile_j, solve using diag_block_gemm_naive
        // when tile_i != tile_j, solve using block_gemm_naive
        // do later

        // move to sub tile in a GPU block

        float *A_subtile = A_ij + thread_tile_i * T_TH * N + thread_tile_j * T_TH;

        // calculate A - X * X^T 
        #pragma unroll
        for (uint32_t ti = 0; ti < T_TH; ++ti) {
            #pragma unroll
            for (uint32_t tj = 0; tj < T_TH; ++tj) {
                A_subtile[ti * N + tj] -= reg[ti * T_TH + tj];
            }
        }
        __syncthreads();
    }
}

// // T_TH: tile size of each thread
// // num_threads_H: number of threads per row/column in the GPU block
// // X: input matrix, to perform X*X^T
// // smem: shared memory to store X for computation
// // reg: register array to store results of X*X^T
// // N: matrix size, determines padding
// // block_n: block size of triblock, determines loop condition
// template <uint32_t T_TH, uint32_t num_threads_H> 
// __device__ void triblock_gemm_GPUblock(float *X_rows_i, float *X_rows_j, 
//     float *smem1, float *smem2, float *reg, 
//     const int32_t thread_tile_i, const int32_t thread_tile_j,
//     const uint32_t N, const uint32_t block_n) {
    
//     for (uint32_t k = 0; k < block_n; k += T_TH * num_threads_H) {

//         // copy X to smem
//         block_cholesky_space::gmem_to_smem(X_rows_i + k, smem1, N, T_TH * num_threads_H);
//         block_cholesky_space::gmem_to_smem(X_rows_j + k, smem2, N, T_TH * num_threads_H);
//         __syncthreads();

//         // solve X * X^T
//         // if (tile_i == tile_j) {
//         //     diag_block_gemm_naive<T_TH, T_TW>(smem, reg, tile_i, tile_j);
//         // } else {
//         //     block_gemm_naive<T_TH, T_TW>(smem, reg, tile_i, tile_j);
//         // }
//         block_cholesky_space::block_gemm_naive<T_TH * num_threads_H, T_TH * num_threads_H, T_TH * num_threads_H, T_TH, T_TH>(
//             smem1, smem2, reg, thread_tile_i, thread_tile_j);
//         __syncthreads();
//     }
// }

// template <uint32_t T_TH, uint32_t num_threads_H>
// __device__ void triblock_gemm_GPUblock(float *X, float *A,float *smem1, float *smem2, float *reg,
//     const uint32_t block_tile_i, const uint32_t block_tile_j,
//     const uint32_t N, const uint32_t block_n) {

//     // calculate thread tile indices
//     const uint32_t thread_tile_i = threadIdx.x / num_threads_H;
//     const uint32_t thread_tile_j = threadIdx.x % num_threads_H;

//     const uint32_t valid_tiles = block_n / (T_TH * num_threads_H);
//     if (block_tile_i < valid_tiles && block_tile_j < valid_tiles) {

//         // locate X for tile_i, tile_j
//         float *X_i = X + block_tile_i * (T_TH * num_threads_H) * N;
//         float *X_j = X + block_tile_j * (T_TH * num_threads_H) * N;
//         float *A_ij = A + block_tile_i * (T_TH * num_threads_H) * N + block_tile_j * (T_TH * num_threads_H);

//         // solve X * X^T and store in reg iteratively in the k dimension
//         for (uint32_t k = 0; k < block_n; k += T_TH * num_threads_H) {

//             // copy X to smem
//             block_cholesky_space::gmem_to_smem(X_i + k, smem1, N, T_TH * num_threads_H);
//             block_cholesky_space::gmem_to_smem(X_j + k, smem2, N, T_TH * num_threads_H);
//             __syncthreads();
    
//             block_cholesky_space::block_gemm_naive<T_TH * num_threads_H, T_TH * num_threads_H, T_TH * num_threads_H, T_TH, T_TH>(
//                 smem1, smem2, reg, thread_tile_i, thread_tile_j);
//             __syncthreads();
//         }

//         // move to sub tile in a GPU block
//         float *A_subtile = A_ij + thread_tile_i * T_TH * N + thread_tile_j * T_TH;

//         // calculate A - X * X^T 
//         #pragma unroll
//         for (uint32_t ti = 0; ti < T_TH; ++ti) {
//             #pragma unroll
//             for (uint32_t tj = 0; tj < T_TH; ++tj) {
//                 A_subtile[ti * N + tj] -= reg[ti * T_TH + tj];
//             }
//         }
//         __syncthreads();
//     }
// }

// // requires shared memory of size at least (T_TH * num_threads_H)^2
// template <uint32_t T_TH, uint32_t num_threads_H>
// __global__ void triblock_diagonal_gemm(float *A, float *X, const uint32_t N, const uint32_t block_n, const uint32_t smem_size_bytes) {
//     extern __shared__ float smem[];

//     float reg[T_TH * T_TH] = {0.0f};

//     // Map rectangular to triangular tiles
//     const uint32_t block_tile_i = (uint32_t)((sqrtf(8.f * blockIdx.x + 1.f) - 1.f) * 0.5f);
//     const uint32_t block_tile_j = blockIdx.x - (block_tile_i * (block_tile_i + 1) / 2);

//     if (block_tile_i == block_tile_j) {
//         triblock_gemm_GPUblock<T_TH, num_threads_H>(X, A, smem1, smem2, reg, block_tile_i, block_tile_j, N, block_n);
//     } else {
//         triblock_diag_gemm_GPUblock<T_TH, num_threads_H>(X, A, smem1, smem2, reg, block_tile_i, block_tile_j, N, block_n);
//     }
// }



template <uint32_t m, uint32_t T_TH, uint32_t T_TW>
__device__ void diagonal_block_update(float *A, float *L,
    const uint32_t n,
    const uint32_t i, const uint32_t j,
    float *smem1, float *smem2
) {
    // Accumulate update results in registers w/ each thread getting a subtile
    float reg[T_TH * T_TW] = {0.0f}; // zero-init
    
    // Map rectangular to triangular tiles
    const uint32_t tile_i = (uint32_t)((sqrtf(8.f * threadIdx.x + 1.f) - 1.f) * 0.5f);
    const uint32_t tile_j = threadIdx.x - (tile_i * (tile_i + 1) / 2);

    // Only compute if valid tile
    constexpr uint32_t N = m / T_TH;
    if (tile_i < N && tile_j < N) {
        // Compute Lij * Lij^T
        block_cholesky_space::diagonal_block_gemm_naive<m, m, T_TH, T_TW>(smem2, reg, tile_i, tile_j);

        // Move A to Aii
        float *Aii = block_cholesky_space::get_block(A, i, i, n, m);

        // Move to subtile
        float *_Aii = Aii + tile_i * T_TH * n + tile_j * T_TW;
        float *_Aii_p = smem1 + tile_i * T_TH * m + tile_j * T_TW;

        // Compute Aii - Lij * Lij^T
        #pragma unroll
        for (uint32_t ti = 0; ti < T_TH; ++ti) {
            #pragma unroll
            for (uint32_t tj = 0; tj < (tile_i == tile_j ? ti+1 : T_TW); ++tj) {
                _Aii_p[ti * m + tj] = _Aii[ti * n + tj] - reg[ti * T_TW + tj];
            }
        }
    }

    // Wait for the entire block to finish
    __syncthreads();
}

} // namespace gemm_naive