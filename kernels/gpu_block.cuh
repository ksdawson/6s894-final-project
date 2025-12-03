// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["trsm.cuh", "gpu_naive.cuh", "gpu_block_kernel_fusion.cuh"]}
// TL {"workspace_files": []}

#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include "trsm.cuh"
#include "gpu_naive.cuh"
#include "gpu_block_kernel_fusion.cuh"

////////////////////////////////////////////////////////////////////////////////
// Device functions

namespace default_chol {

template <uint32_t T_TH, uint32_t T_TW>
__device__ void block_gemm_naive(BlockUpdate input) {
    auto [A, L, n, m, i, j, reg, smem] = input;

    // Matrix to multiply
    float *Lij = smem;

    // Move to subtile
    const uint32_t tile_i = threadIdx.x / (m / T_TW);
    const uint32_t tile_j = threadIdx.x % (m / T_TW);
    float *_Lij_row = Lij + tile_i * T_TH * m;
    float *_Lij_col = Lij + tile_j * T_TH * m;

    // Each thread handles a tile
    #pragma unroll
    for (uint32_t ti = 0; ti < T_TH; ++ti) {
        #pragma unroll
        for (uint32_t tj = 0; tj < T_TW; ++tj) {
            for (uint32_t tk = 0; tk < m; ++tk) {
                reg[ti * T_TW + tj] += _Lij_row[ti * m + tk] * _Lij_col[tj * m + tk];
            }
        }
    }

    // Make sure every thread is done
    __syncthreads();
}

template <uint32_t T_TH, uint32_t T_TW>
__device__ void block_update(float *A, float *L,
    const uint32_t n, const uint32_t m,
    const uint32_t i, const uint32_t j,
    float *smem
) {
    // Accumulate update results in registers w/ each thread getting a subtile
    float reg[T_TH * T_TW] = {0.0f}; // zero-init

    // Compute Lij * Lij^T
    BlockUpdate input = {A, L, n, m, i, j, reg, smem};
    block_gemm_naive<T_TH, T_TW>(input);

    // Move A to Aii
    float *Aii = get_block(A, i, i, n, m);

    // Move to subtile
    const uint32_t tile_i = threadIdx.x / (m / T_TW);
    const uint32_t tile_j = threadIdx.x % (m / T_TW);
    float *_Aii = Aii + tile_i * T_TH * n + tile_j * T_TW;

    // Compute Aii - Lij * Lij^T
    #pragma unroll
    for (uint32_t ti = 0; ti < T_TH; ++ti) {
        #pragma unroll
        for (uint32_t tj = 0; tj < T_TW; ++tj) {
            _Aii[ti * n + tj] -= reg[ti * T_TW + tj];
        }
    }

    // Wait for the entire block to finish
    __syncthreads();
}

template <uint32_t T_TH, uint32_t T_TW>
__global__ void block_kernel(float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, const uint32_t m, // matrix size, block size
    const uint32_t j // block col
) {
    // Setup smem
    extern __shared__ float smem[];

    // Each SM gets a block
    for (uint32_t i = j + 1 + blockIdx.x; i < n / m; i += gridDim.x) {
        float *Ljj = get_block(L, j, j, n, m);
        float *Aij = get_block(A, i, j, n, m);
        block_trsm(Ljj, smem, Aij, n, m, n, m); // A, X, B (Ljj * Lij^T = Aij^T)

        // TRSM
        // float *Ljj = get_block(L, j, j, n, m);
        // float *Aij = get_block(A, i, j, n, m);
        // block_trsm(Ljj, smem, Aij, n, m, n, m); // A, X, B (Ljj * Lij^T = Aij^T)

        // // Move L to Lij 
        // float *Lij = get_block(L, i, j, n, m);

        // // Write back Lij
        // for (uint32_t idx = threadIdx.x; idx < m * m; idx += blockDim.x) {
        //     const uint32_t ti = idx / m;
        //     const uint32_t tj = idx % m;
        //     Lij[ti * n + tj] = smem[idx];
        // }
        // __syncthreads();

        // Update Aii
        block_update<T_TH, T_TW>(A, L, n, m, i, j, smem);
    }
}

template <uint32_t T_TH, uint32_t T_TW>
__global__ void chol_kernel(const float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, const uint32_t m, // matrix size, block size
    const uint32_t j // block col
) {
    // Only 1 SM participates

    // Chol (only first warp participates)
    const float *Ajj = get_block(A, j, j, n, m);
    float *Ljj = get_block(L, j, j, n, m);
    block_cholesky(Ajj, Ljj, n, n, m);
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
        block_kernel<4, 4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes
    );

    // Iterate over block cols launching a kernel for each step
    for (uint32_t j = 0; j < n / m; ++j) {
        // Step 1: Chol diagonal block
        chol_kernel<4, 4><<<1, 32>>>(in, out, n, m, j);

        // Step 2: Trsm then update
        block_kernel<4, 4><<<48, 8*32, smem_size_bytes>>>(const_cast<float*>(in), out, n, m, j);
    }
}

} // namespace default_chol