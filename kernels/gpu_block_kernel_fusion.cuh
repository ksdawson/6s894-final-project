// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["trsm.cuh", "gpu_naive.cuh"]}
// TL {"workspace_files": []}

#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include "trsm.cuh"
#include "gpu_naive.cuh"

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
__device__ void block_gemm_naive(BlockUpdate input, const uint32_t k) {
    auto [A, L, n, m, i, j, reg, smem] = input;

    // Matrices to multiply
    float *Lik = L + i * m * n + k * m;
    float *Ljk = L + j * m * n + k * m;

    // Move to subtile
    const uint32_t tile_i = threadIdx.x / (m / T_TW);
    const uint32_t tile_j = threadIdx.x % (m / T_TW);
    float *Lik_subtile = Lik + tile_i * T_TH * n + tile_j * T_TW;
    float *Ljk_subtile = Ljk + tile_i * T_TH * n + tile_j * T_TW;

    // Each thread handles a tile
    for (uint32_t ti = 0; ti < T_TH; ++ti) {
        for (uint32_t tj = 0; tj < T_TW; ++tj) {
            for (uint32_t tk = 0; tk < m; ++tk) {
                reg[ti * T_TW + tj] += Lik_subtile[ti * n + tk] * Ljk_subtile[tj * n + tk];
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
    BlockUpdate input = {A, L, n, m, i, j, reg, smem};
    for (uint32_t k = 0; k < j; ++k) {
        block_gemm_naive<T_TH, T_TW>(input, k);
    }

    // Move A to Aij 
    const float *Aij = A + i * m * n + j * m;

    // Move to subtile
    const uint32_t tile_i = threadIdx.x / (m / T_TW);
    const uint32_t tile_j = threadIdx.x % (m / T_TW);
    const float *Aij_subtile = Aij + tile_i * T_TH * n + tile_j * T_TW;
    float *smem_subtile = smem + tile_i * T_TH * m + tile_j * T_TW;

    // Compute Aij - sum
    #pragma unroll
    for (uint32_t ti = 0; ti < T_TH; ++ti) {
        // Vectorize
        const float4 *a4 = reinterpret_cast<const float4*>(Aij_subtile + ti * n);
        float4 *reg4 = reinterpret_cast<float4*>(reg + ti * T_TW);
        float4 *smem4 = reinterpret_cast<float4*>(smem_subtile + ti * m);
        #pragma unroll
        for (uint32_t tj = 0; tj < T_TW / 4; ++tj) {
            // A - sum
            float4 a = a4[tj];
            float4 o = reg4[tj];

            // Write back to SMEM
            smem4[tj] = {a.x - o.x, a.y - o.y, a.z - o.z, a.w - o.w};
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
        float *Lij = L + i * m * n + j * m;
        float *Ljj = L + j * m * n + j * m;
        float *Aij = smem;
        block_trsm(Ljj, Lij, Aij, n, m); // A, X, B
    }
}

template <uint32_t T_TH, uint32_t T_TW>
__global__ void chol_kernel(const float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, const uint32_t m, // matrix size, block size
    const uint32_t j // block col
) {
    // Only uses 1 SM

    // Setup smem
    extern __shared__ float smem[];

    // Update (uses all threads)
    block_update<T_TH, T_TW>(A, L, n, m, j, j, smem);

    // Chol (only uses first warp)
    const float *Ajj = A + j * m * n + j * m;
    float *Ljj = L + j * m * n + j * m;
    block_cholesky(Ajj, Ljj, n, m);
}

////////////////////////////////////////////////////////////////////////////////
// Host functions

void launch_block_cholesky(
    const uint32_t n, float const *in, float *out
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