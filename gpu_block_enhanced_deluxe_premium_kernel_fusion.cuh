// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["trsm_small.cuh", "cholesky_small.cuh", "gpu_block_kernel_fusion.cuh", "gpu_block_enhanced_kernel_fusion.cuh", "gpu_block_enhanced_deluxe_kernel_fusion.cuh"]}
// TL {"workspace_files": []}

#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include "trsm_small.cuh"
#include "cholesky_small.cuh"
#include "gpu_block_kernel_fusion.cuh"
#include "gpu_block_enhanced_kernel_fusion.cuh"
#include "gpu_block_enhanced_deluxe_kernel_fusion.cuh"

////////////////////////////////////////////////////////////////////////////////
// Device functions

namespace prem_deluxe_alt_kernel_fusion {

size_t get_workspace_size(int32_t size) {
    return 0;
}

template <uint32_t m, uint32_t W, uint32_t T_TH, uint32_t T_TW>
__launch_bounds__(W*32)
__global__ void block_kernel(float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, // matrix size
    const uint32_t j, const uint32_t end_j // block col, last col
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
        block_cholesky_space::gmem_to_smem_async<m>(Ljj, smem3, n);
        Ljj = smem3;

        // TRSM
        float *Lij = smem2;
        float *Aij = smem;
        trsm_small::block_trsm_reuse<W, m, m, m, m>(Ljj, Lij, Aij);

        // Write back Lij
        Lij = block_cholesky_space::get_block(L, i, j, n, m);
        block_cholesky_space::smem_to_gmem(Lij, smem2, n, m);

        // Update Aii
        if (i < end_j) { // Don't update matrix that will have smaller block size
            if (i == j + 1) {
                // Update Aii
                deluxe_alt_kernel_fusion::diagonal_block_update<m, T_TH, T_TW>(A, L, n, i, j, smem, smem2);
                
                // Chol Aii
                float *Aii = smem;
                float *Lii = smem2;
                cholesky_small::block_col_cholesky<m, m, m>(Aii, Lii);

                // Write back Lii
                Lii = block_cholesky_space::get_block(L, i, i, n, m);
                block_cholesky_space::smem_to_gmem(Lii, smem2, n, m);
            } else {
                // Write back to A
                alt_kernel_fusion::diagonal_block_update<m, T_TH, T_TW>(A, L, n, i, j, smem2);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host functions

template <uint32_t m, uint32_t T_TS, uint32_t W>
void launch_specialized_kernel(const uint32_t n, float const *in, float *out, const uint32_t start_j) {
    // Setup chol kernel smem
    constexpr int smem_size_bytes = m * m * sizeof(float);
    cudaFuncSetAttribute(
        block_cholesky_space::chol_kernel<m, W, T_TS, T_TS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes * 2
    );

    // Setup block kernel smem
    cudaFuncSetAttribute(
        deluxe_alt_kernel_fusion::block_kernel<m, W, T_TS, T_TS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes * 3 // need to store 3 blocks in smem
    );

    // Chol first diagonal block
    // alt_kernel_fusion::chol_kernel<m><<<1, 32*32, smem_size_bytes>>>(in, out, n, start_j);
    block_cholesky_space::chol_kernel<m, W, T_TS, T_TS><<<1, W*32, smem_size_bytes * 2>>>(in, out, n, start_j);

    // Iterate over block cols launching a kernel for each step
    for (uint32_t j = start_j; j < n / m - 1; ++j) {
        // Trsm then update w/ first off diagonal computing next Chol diagonal block
        deluxe_alt_kernel_fusion::block_kernel<m, W, T_TS, T_TS><<<48, W*32, smem_size_bytes*3>>>(const_cast<float*>(in), out, n, j);
    }
}

template <uint32_t m, uint32_t T_TS, uint32_t W>
void launch_specialized_kernel_dynamic_block(const uint32_t n, float const *in, float *out,
    const uint32_t start_j, const uint32_t end_j
) {
    // Setup chol kernel smem
    constexpr int smem_size_bytes = m * m * sizeof(float);
    cudaFuncSetAttribute(
        block_cholesky_space::chol_kernel<m, W, T_TS, T_TS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes * 2
    );

    // Setup block kernel smem
    cudaFuncSetAttribute(
        block_kernel<m, W, T_TS, T_TS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes * 3 // need to store 3 blocks in smem
    );

    // Chol first diagonal block
    // alt_kernel_fusion::chol_kernel<m><<<1, 32*32, smem_size_bytes>>>(in, out, n, start_j);
    block_cholesky_space::chol_kernel<m, W, T_TS, T_TS><<<1, W*32, smem_size_bytes * 2>>>(in, out, n, start_j);

    // Iterate over block cols launching a kernel for each step
    for (uint32_t j = start_j; j < end_j; ++j) {
        // Trsm then update w/ first off diagonal computing next Chol diagonal block
        block_kernel<m, W, T_TS, T_TS><<<48, W*32, smem_size_bytes*3>>>(const_cast<float*>(in), out, n, j, end_j);
    }
}

void launch_block_cholesky(
    const uint32_t n, float const *in, float *out, void *workspace
) {
    // Make sure # blocks never falls below 1/2 # SMs (w/ block size btwn 16 and 64)
    if (n > 1536 + 64) {
        launch_specialized_kernel_dynamic_block<64, 2, 32>(n, in, out, 0, (n-1536)/64);
        launch_specialized_kernel_dynamic_block<32, 2, 8>(n, in, out, (n-1536)/32, (n-768)/32);
        launch_specialized_kernel<16, 1, 8>(n, in, out, (n-768)/16);
    } else if (n > 768 + 32) {
        launch_specialized_kernel_dynamic_block<32, 2, 8>(n, in, out, 0, (n-768)/32);
        launch_specialized_kernel<16, 1, 8>(n, in, out, (n-768)/16);
    } else {
        launch_specialized_kernel<16, 1, 8>(n, in, out, 0);
    }
}

} // namespace prem_deluxe_alt_kernel_fusion