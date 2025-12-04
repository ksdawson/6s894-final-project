// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["trsm_small.cuh", "cholesky_small.cuh"]}
// TL {"workspace_files": []}

#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include "trsm_small.cuh"
#include "cholesky_small.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper functions

namespace block_cholesky_space {

size_t get_workspace_size(int32_t size) {
    return 0;
}

__device__ float* get_block(float *A, const uint32_t i, const uint32_t j, const uint32_t n, const uint32_t m) { return A + i * m * n + j * m; }
__device__ const float* get_block(const float *A, const uint32_t i, const uint32_t j, const uint32_t n, const uint32_t m) { return A + i * m * n + j * m; }

////////////////////////////////////////////////////////////////////////////////
// Device functions

struct BlockUpdate {
    const float *A; // input matrix
    float *L; // Chol matrix
    const uint32_t n; // matrix size
    const uint32_t m; // block size
    const uint32_t i; // Lik * Ljk^T
    const uint32_t j;
    float *reg; // add result to reg
    float *smem; // use for read-only data reuse
};

template <uint32_t T_TH, uint32_t T_TW>
__device__ void block_gemm_naive(float *A, float *B, float* C,
    const uint A_n, const uint32_t B_n, const uint32_t r
) {
    // Move to subtile
    const uint32_t tile_i = threadIdx.x / (r / T_TW);
    const uint32_t tile_j = threadIdx.x % (r / T_TW);
    float *_A = A + tile_i * T_TH * A_n;
    float *_B = B + tile_j * T_TH * B_n;

    // Each thread handles a tile
    #pragma unroll
    for (uint32_t ti = 0; ti < T_TH; ++ti) {
        #pragma unroll
        for (uint32_t tj = 0; tj < T_TW; ++tj) {
            for (uint32_t tk = 0; tk < r; ++tk) {
                C[ti * T_TW + tj] += _A[ti * A_n + tk] * _B[tj * B_n + tk];
            }
        }
    }

    // Make sure every thread is done
    __syncthreads();
}

template <uint32_t T_TH, uint32_t T_TW>
__device__ void block_update(const float *A, float *L,
    const uint32_t n, const uint32_t m,
    const uint32_t i, const uint32_t j,
    float *smem
) {
    // Accumulate update results in registers w/ each thread getting a subtile
    float reg[T_TH * T_TW] = {0.0f}; // zero-init

    // Sum Lik * Ljk^T
    for (uint32_t k = 0; k < j; ++k) {
        float *Lik = get_block(L, i, k, n, m);
        float *Ljk = get_block(L, j, k, n, m);
        block_gemm_naive<T_TH, T_TW>(Lik, Ljk, reg, n, n, m);
    }

    // Move A to Aij 
    const float *Aij = get_block(A, i, j, n, m);

    // Move to subtile
    const uint32_t tile_i = threadIdx.x / (m / T_TW);
    const uint32_t tile_j = threadIdx.x % (m / T_TW);
    const float *_Aij = Aij + tile_i * T_TH * n + tile_j * T_TW;
    float *_Aij_p = smem + tile_i * T_TH * m + tile_j * T_TW;

    // Compute Aij - sum
    #pragma unroll
    for (uint32_t ti = 0; ti < T_TH; ++ti) {
        #pragma unroll
        for (uint32_t tj = 0; tj < T_TW; ++tj) {
            _Aij_p[ti * m + tj] = _Aij[ti * n + tj] - reg[ti * T_TW + tj];
        }
    }

    // Wait for the entire block to finish
    __syncthreads();
}

template <uint32_t T_TH, uint32_t T_TW>
__global__ void block_kernel(const float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, const uint32_t m, // matrix size, block size
    const uint32_t j // block col
) {
    // Setup smem
    extern __shared__ float smem[];

    // Each SM gets a block
    for (uint32_t i = j + 1 + blockIdx.x; i < n / m; i += gridDim.x) {
        // Update
        block_update<T_TH, T_TW>(A, L, n, m, i, j, smem);

        // TRSM
        float *Lij = get_block(L, i, j, n, m);
        float *Ljj = get_block(L, j, j, n, m);
        float *Aij = smem;
        trsm_small::block_trsm(Ljj, Lij, Aij, n, n, m, m); // A, X, B
    }
}

template <uint32_t T_TH, uint32_t T_TW>
__global__ void chol_kernel(const float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, const uint32_t m, // matrix size, block size
    const uint32_t j // block col
) {
    // Only 1 SM participates

    // Setup smem
    extern __shared__ float smem[];

    // Update (all threads participate)
    block_update<T_TH, T_TW>(A, L, n, m, j, j, smem);

    // Chol (only first warp participates)
    float *Ajj = smem;
    float *Ljj = get_block(L, j, j, n, m);
    cholesky_small::block_cholesky(Ajj, Ljj, m, n, m);
}

////////////////////////////////////////////////////////////////////////////////
// Host functions

void launch_block_cholesky(
    const uint32_t n, float const *in, float *out, void *workspace
) {
    // Divide the grid into blocks
    constexpr uint32_t m = 64;

    // Setup smem
    constexpr int smem_size_bytes = m * m * 2 * sizeof(float); // need to store 2 blocks in smem
    cudaFuncSetAttribute(
        chol_kernel<4, 4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes
    );
    cudaFuncSetAttribute(
        block_kernel<4, 4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes
    );

    // Iterate over block cols launching a kernel for each step
    for (uint32_t j = 0; j < n / m; ++j) {
        // Step 1: Chol(update) diagonal block
        chol_kernel<4, 4><<<1, 8*32, smem_size_bytes>>>(in, out, n, m, j);

        // Step 2: Trsm(update) all other blocks
        block_kernel<4, 4><<<48, 8*32, smem_size_bytes>>>(in, out, n, m, j);
    }
}

}