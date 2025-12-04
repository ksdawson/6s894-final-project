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
__device__ void diagonal_block_gemm_naive(float *A, float* C,
    const uint A_n, const uint32_t r
) {
    // TODO: AA^T is symmetric so only need to compute lower triangle

    // Move to subtile
    const uint32_t tile_i = threadIdx.x / (r / T_TW);
    const uint32_t tile_j = threadIdx.x % (r / T_TW);
    float *_A = A + tile_i * T_TH * A_n;
    float *_B = A + tile_j * T_TH * A_n;

    // Each thread handles a tile
    for (uint32_t tk = 0; tk < r; tk += 4) {
        #pragma unroll
        for (uint32_t ti = 0; ti < T_TH; ++ti) {
            const float4 a = *(reinterpret_cast<float4*>(_A + ti * A_n + tk));
            #pragma unroll
            for (uint32_t tj = 0; tj < T_TW; ++tj) {
                if ((_A + ti * A_n + tk) == (_B + tj * A_n + tk)) {
                    // If i==j reuse a
                    C[ti * T_TW + tj] += (a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
                    continue;
                }
                const float4 b = *(reinterpret_cast<float4*>(_B + tj * A_n + tk));
                C[ti * T_TW + tj] += (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
            }
        }
    }
    // Handle tail
    for (uint32_t tk = (r / 4) * 4; tk < r; ++tk) {
        #pragma unroll
        for (uint32_t ti = 0; ti < T_TH; ++ti) {
            const float a = _A[ti * A_n + tk];
            #pragma unroll
            for (uint32_t tj = 0; tj < T_TW; ++tj) {
                C[ti * T_TW + tj] += a * _B[tj * A_n + tk];
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
    diagonal_block_gemm_naive<T_TH, T_TW>(smem, reg, m, m);

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
__launch_bounds__(256)
__global__ void block_kernel(float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, const uint32_t m, // matrix size, block size
    const uint32_t j // block col
) {
    // Setup smem
    extern __shared__ float smem[];
    float *smem2 = smem + m * m;
    float *smem3 = smem2 + m * m;

    // Each SM gets a block
    for (uint32_t i = j + 1 + blockIdx.x; i < n / m; i += gridDim.x) {
        // Update
        block_cholesky_space::block_update<T_TH, T_TW>(A, L, n, m, i, j, smem, smem2);

        // Load Ljj into smem
        float *Ljj = block_cholesky_space::get_block(L, j, j, n, m);
        for (uint32_t idx = threadIdx.x; idx < m * m; idx += blockDim.x) {
            const uint32_t ti = idx / m;
            const uint32_t tj = idx % m;
            smem3[idx] = Ljj[ti * n + tj];
        }
        Ljj = smem3;
        __syncthreads();

        // TRSM
        float *Lij = smem2;
        float *Aij = smem;
        trsm_small::block_trsm(Ljj, Lij, Aij, m, m, m, m); // A, X, B

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

__launch_bounds__(1024)
__global__ void chol_kernel(const float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, const uint32_t m, // matrix size, block size
    const uint32_t j // block col
) {
    // Only 1 SM participates

    // Setup smem
    extern __shared__ float smem[];

    // Chol (only first warp participates)
    const float *Ajj = block_cholesky_space::get_block(A, j, j, n, m);
    float *Ljj = smem;
    cholesky_small::block_cholesky(Ajj, Ljj, n, m, m);
    __syncthreads();

    // Write back Ljj (all threads participate)
    Ljj = block_cholesky_space::get_block(L, j, j, n, m);
    for (uint32_t idx = threadIdx.x; idx < m * m; idx += blockDim.x) {
        const uint32_t ti = idx / m;
        const uint32_t tj = idx % m;
        Ljj[ti * n + tj] = smem[idx];
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
    constexpr int smem_size_bytes = m * m * sizeof(float);
    cudaFuncSetAttribute(
        chol_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes
    );
    cudaFuncSetAttribute(
        block_kernel<4, 4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes * 3 // need to store 3 blocks in smem
    );

    // Iterate over block cols launching a kernel for each step
    for (uint32_t j = 0; j < n / m; ++j) {
        // Step 1: Chol diagonal block
        chol_kernel<<<1, 32*32, smem_size_bytes>>>(in, out, n, m, j);

        // Step 2: Trsm then update
        block_kernel<4, 4><<<48, 8*32, smem_size_bytes*3>>>(const_cast<float*>(in), out, n, m, j);
    }
}

} // namespace alt_kernel_fusion