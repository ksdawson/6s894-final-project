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

template <uint32_t m, uint32_t T_TH, uint32_t T_TW>
__device__ void diagonal_block_update(float *A, float *L,
    const uint32_t n,
    const uint32_t i, const uint32_t j,
    float *smem
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
        block_cholesky_space::diagonal_block_gemm_naive<m, m, T_TH, T_TW>(smem, reg, tile_i, tile_j);

        // Move A to Aii
        float *Aii = block_cholesky_space::get_block(A, i, i, n, m);

        // Move to subtile
        float *_Aii = Aii + tile_i * T_TH * n + tile_j * T_TW;

        // Compute Aii - Lij * Lij^T
        #pragma unroll
        for (uint32_t ti = 0; ti < T_TH; ++ti) {
            #pragma unroll
            for (uint32_t tj = 0; tj < (tile_i == tile_j ? ti+1 : T_TW); ++tj) {
                _Aii[ti * n + tj] -= reg[ti * T_TW + tj];
            }
        }
    }

    // Wait for the entire block to finish
    __syncthreads();
}

template <uint32_t m, uint32_t W, uint32_t T_TH, uint32_t T_TW>
__launch_bounds__(W*32)
__global__ void block_kernel(float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, // matrix size
    const uint32_t j // block col
) {
    // Setup smem
    extern __shared__ float smem[];
    float *smem2 = smem + m * m;
    float *smem3 = smem2 + m * m;

    // Each SM gets a block
    for (uint32_t i = j + 1 + blockIdx.x; i < n / m; i += gridDim.x) {
        // Update
        block_cholesky_space::block_update<m, T_TH, T_TW>(A, L, n, i, j, smem, smem2);

        // Load Ljj into smem
        float *Ljj = block_cholesky_space::get_block(L, j, j, n, m);
        block_cholesky_space::gmem_to_smem(Ljj, smem3, n, m);
        Ljj = smem3;

        // TRSM
        float *Lij = smem2;
        float *Aij = smem;
        trsm_small::block_trsm(Ljj, Lij, Aij, m, m, m, m); // A, X, B

        // Write back Lij
        Lij = block_cholesky_space::get_block(L, i, j, n, m);
        block_cholesky_space::smem_to_gmem(Lij, smem2, n, m);

        // Update Aii
        diagonal_block_update<m, T_TH, T_TW>(A, L, n, i, j, smem2);
    }
}

template <uint32_t m>
__launch_bounds__(1024)
__global__ void chol_kernel(const float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, // matrix size
    const uint32_t j // block col
) {
    // Only 1 SM participates

    // Setup smem
    extern __shared__ float smem[];

    // Chol
    const float *Ajj = block_cholesky_space::get_block(A, j, j, n, m);
    float *Ljj = smem;
    cholesky_small::block_col_cholesky(Ajj, Ljj, n, m, m);

    // Write back Ljj
    Ljj = block_cholesky_space::get_block(L, j, j, n, m);
    block_cholesky_space::smem_to_gmem(Ljj, smem, n, m);
}

__launch_bounds__(1024)
__global__ void chol_kernel(const float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, const uint32_t m, // matrix size, block size
    const uint32_t j // block col
) {
    // Only 1 SM participates

    // Setup smem
    extern __shared__ float smem[];

    // Chol
    const float *Ajj = block_cholesky_space::get_block(A, j, j, n, m);
    float *Ljj = smem;
    cholesky_small::block_col_cholesky(Ajj, Ljj, n, m, m);

    // Write back Ljj
    Ljj = block_cholesky_space::get_block(L, j, j, n, m);
    block_cholesky_space::smem_to_gmem(Ljj, smem, n, m);
}

////////////////////////////////////////////////////////////////////////////////
// Host functions

template <uint32_t m, uint32_t T_TS, uint32_t W>
void launch_specialized_kernel(const uint32_t n, float const *in, float *out) {
    // Setup smem
    constexpr int smem_size_bytes = m * m * sizeof(float);
    cudaFuncSetAttribute(
        chol_kernel<m>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes
    );
    cudaFuncSetAttribute(
        block_kernel<m, W, T_TS, T_TS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes * 3 // need to store 3 blocks in smem
    );

    // Iterate over block cols launching a kernel for each step
    for (uint32_t j = 0; j < n / m; ++j) {
        // Step 1: Chol diagonal block
        chol_kernel<m><<<1, 32*32, smem_size_bytes>>>(in, out, n, j);

        // Step 2: Trsm then update
        block_kernel<m, W, T_TS, T_TS><<<48, W*32, smem_size_bytes*3>>>(const_cast<float*>(in), out, n, j);
    }
}

void launch_block_cholesky(
    const uint32_t n, float const *in, float *out, void *workspace
) {
    // Divide the grid into blocks
    if (n < 2048) {
        launch_specialized_kernel<16, 1, 8>(n, in, out);
    } else if (n < 4096) {
        launch_specialized_kernel<32, 2, 8>(n, in, out);
    } else {
        launch_specialized_kernel<64, 2, 32>(n, in, out);
    }
}

} // namespace alt_kernel_fusion