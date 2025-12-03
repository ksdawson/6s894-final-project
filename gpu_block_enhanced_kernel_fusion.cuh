// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["trsm_small.cuh", "cholesky_small.cuh", "gpu_block_kernel_fusion.cuh"]}
// TL {"workspace_files": []}

#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include "trsm_small.cuh"
#include "cholesky_small.cuh"
#include "gpu_block_kernel_fusion.cuh"

////////////////////////////////////////////////////////////////////////////////
// Device functions

namespace alt_kernel_fusion {

size_t get_workspace_size(int32_t size) {
    return 0;
}

template <uint32_t T_TH, uint32_t T_TW>
__device__ void diagonal_block_gemm_naive(block_cholesky_space::BlockUpdate input) {
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
__device__ void diagonal_block_update(float *A, float *L,
    const uint32_t n, const uint32_t m,
    const uint32_t i, const uint32_t j,
    float *smem
) {
    // Accumulate update results in registers w/ each thread getting a subtile
    float reg[T_TH * T_TW] = {0.0f}; // zero-init

    // Compute Lij * Lij^T
    block_cholesky_space::BlockUpdate input = {A, L, n, m, i, j, reg, smem};
    diagonal_block_gemm_naive<T_TH, T_TW>(input);

    // Move A to Aii
    float *Aii = block_cholesky_space::get_block(A, i, i, n, m);

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
    float *smem2 = smem + m * m;

    // Each SM gets a block
    for (uint32_t i = j + 1 + blockIdx.x; i < n / m; i += gridDim.x) {
        // Update
        block_cholesky_space::block_update<T_TH, T_TW>(A, L, n, m, i, j, smem);

        // TRSM
        float *Lij = smem2;
        float *Ljj = block_cholesky_space::get_block(L, j, j, n, m);
        float *Aij = smem;
        trsm_small::block_trsm(Ljj, Lij, Aij, n, m, m, m); // A, X, B

        // Write back Lij
        Lij = block_cholesky_space::get_block(L, i, j, n, m);
        for (uint32_t idx = threadIdx.x; idx < m * m; idx += blockDim.x) {
            const uint32_t ti = idx / m;
            const uint32_t tj = idx % m;
            Lij[ti * n + tj] = smem2[idx];
        }
        __syncthreads();

        // Update Aii
        diagonal_block_update<T_TH, T_TW>(A, L, n, m, i, j, smem2);
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
    float *smem2 = smem + m * m;

    // Chol (only first warp participates)
    const float *Ajj = block_cholesky_space::get_block(A, j, j, n, m);
    float *Ljj = smem2;
    cholesky_small::block_cholesky(Ajj, Ljj, n, m, m);

    // Write back Ljj
    Ljj = block_cholesky_space::get_block(L, j, j, n, m);
    for (uint32_t idx = threadIdx.x; idx < m * m; idx += blockDim.x) {
        const uint32_t ti = idx / m;
        const uint32_t tj = idx % m;
        Ljj[ti * n + tj] = smem2[idx];
    }
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
        // Step 1: Chol diagonal block
        chol_kernel<4, 4><<<1, 32, smem_size_bytes>>>(in, out, n, m, j);

        // Step 2: Trsm then update
        block_kernel<4, 4><<<48, 8*32, smem_size_bytes>>>(const_cast<float*>(in), out, n, m, j);
    }
}

} // namespace alt_kernel_fusion