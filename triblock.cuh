// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh", "trsm_small.cuh", "cholesky.cuh", "gemm.cuh", "gpu_block_kernel_fusion.cuh", "cholesky_small.cuh"]}
// TL {"workspace_files": []}

#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "utils.cuh"
#include "trsm_small.cuh"
#include "cholesky.cuh"
#include "gemm.cuh"
#include "gpu_block_kernel_fusion.cuh"
#include "cholesky_small.cuh"

// #define CUDA_CHECK(x) \
//   do { \
//       utils::cuda_check((x), __FILE__, __LINE__); \
//   } while (0)

namespace triblock {
size_t get_workspace_size(int32_t size) {
    return 0;
}

template <uint32_t T_TH, uint32_t T_TW>
__device__ void triblock_update(const float *Aii, float *smem, const uint32_t N, const uint32_t block_n, const uint32_t i) {

    // compute A_(i,i-1) - L_(i,i-1) * L_(i,i-1)^T
    float reg[T_TH * T_TW] = {0.0f};

    // Map rectangular to triangular tiles
    const uint32_t tile_i = (uint32_t)((sqrtf(8.f * threadIdx.x + 1.f) - 1.f) * 0.5f);
    const uint32_t tile_j = threadIdx.x - (tile_i * (tile_i + 1) / 2);

    // Only compute if valid tile
    const uint32_t n_valid = block_n / T_TH;
    if (tile_i < n_valid && tile_j < n_valid) {
        block_cholesky_space::diagonal_block_gemm_naive<T_TH, T_TW>(smem, reg, block_n, block_n, tile_i, tile_j);
    }

    __syncthreads();

    if (tile_i < n_valid && tile_j < n_valid) {
        // Move to subtile
        const float *_Aii = Aii + tile_i * T_TH * N + tile_j * T_TW;
        float *_Aii_p = smem + tile_i * T_TH * block_n + tile_j * T_TW;

        // Compute Aii - sum
        #pragma unroll
        for (uint32_t ti = 0; ti < T_TH; ++ti) {
            #pragma unroll
            for (uint32_t tj = 0; tj < (tile_i == tile_j ? ti+1 : T_TW); ++tj) {
                _Aii_p[ti * block_n + tj] = _Aii[ti * N + tj] - reg[ti * T_TW + tj];
            }
        }
    }

    // Wait for the entire block to finish
    __syncthreads();
}

// Works for N >=3, block_n >=2, and block_n <= 32
// Computes out = in*in^T, block Cholesky decomposition for triblock diagonal
// N: total dimension of the matrix now!!!!!!!!! different from cholesky_trsm_combined
// block_n: dimension of each block in triblock diagonal
// note: currently can only handle block_n <=32, this naive version need to be optimized with better shared memory or register reuse
template <uint32_t T_TH, uint32_t T_TW>
__global__ void triblock_2(const uint32_t N, const uint32_t block_n, float const *in, float *out) {
    extern __shared__ float shared_mem[];

    float *smem1 = shared_mem;
    float *smem2 = smem1 + block_n * block_n;
    float *smem3 = smem2 + block_n * block_n;
    const int32_t num_blocks = (int32_t)(N / block_n);

    // store A00 into smem2
    const float *gmem = in;
    block_cholesky_space::gmem_to_smem(gmem, smem2, N, block_n);
    cholesky_small::block_col_cholesky(smem2, smem1, block_n, block_n, block_n);
    // compute cholesky and store in smem1
    block_cholesky_space::smem_to_gmem(out, smem1, N, block_n);
    // if (threadIdx.x == 0) {
    //     for (uint32_t i = 0; i < block_n; ++i) {
    //         for (uint32_t j = 0; j < block_n; ++j) {
    //             printf("smem1[%u, %u] = %f\n", i, j, smem1[i * N + j]);
    //         }
    //     }
    // }

    const float *A;
    float *Lii;
    float *Lij;

    for (uint32_t i = 1; i < num_blocks; ++i) {

        // trsm
        // store A_(i,i-1) into smem2
        A = block_cholesky_space::get_block(in, i, i-1, N, block_n);
        block_cholesky_space::gmem_to_smem(A, smem2, N, block_n);

        // compute trsm and store in smem3
        trsm_small::block_trsm(smem1, smem3, smem2, block_n, block_n, block_n, block_n); // A, X, B
        Lij = block_cholesky_space::get_block(out, i, i-1, N, block_n);
        block_cholesky_space::smem_to_gmem(Lij, smem3, N, block_n);

        // gemm A_(i,i) - L_(i,i-1) * L_(i,i-1)^T
        // compute gemm and store in smem3
        A = block_cholesky_space::get_block(in, i, i, N, block_n);
        triblock_update<T_TH, T_TW>(A, smem3, N, block_n, i);


        // cholesky L_(i,i), store in smem1
        cholesky_small::block_col_cholesky(smem3, smem1, block_n, block_n, block_n);
        Lii = block_cholesky_space::get_block(out, i, i, N, block_n);
        block_cholesky_space::smem_to_gmem(Lii, smem1, N, block_n);
    }
}

// only works for block_n <= 64
void launch_triblock_small(const uint32_t N, const uint32_t block_n, float const *in, float *out, void *workspace) {
    if (block_n == 32) {
        uint32_t shared_mem_size = 64*64*3 * sizeof(float);
        CUDA_CHECK(cudaFuncSetAttribute(
            triblock_2<1, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem_size));
        triblock_2<1, 1><<<1, 32 * 32, shared_mem_size>>>(N, block_n, in, out);
    } else if (block_n == 64) {
        uint32_t shared_mem_size = 64*64*3 * sizeof(float);
        CUDA_CHECK(cudaFuncSetAttribute(
            triblock_2<2, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem_size));
        triblock_2<2, 2><<<1, 32 * 32, shared_mem_size>>>(N, block_n, in, out);
    }
    else {
        printf("block_n not supported\n");
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
}

namespace triblock_small {
size_t get_workspace_size(int32_t size) {
    return 0;
}

__device__ uint32_t calc_offset(const uint32_t block_n, const uint32_t block_idx) {
    return block_idx * block_n;
}

// Works for N >=3, block_n >=2, and block_n <= 32
// Computes out = in*in^T, block Cholesky decomposition for triblock diagonal
// N: number of blocks in triblock diagonal
// block_n: dimension of each block in triblock diagonal
// note: currently can only handle block_n <=32, this naive version need to be optimized with better shared memory or register reuse
__global__ void cholesky_trsm_combined(const uint32_t N, const uint32_t block_n, float const *in, float *out) {
    extern __shared__ float shared_mem[];

    cholesky::cholesky_XY(N*block_n, N, block_n, in, out, 0, 0);

    for (uint32_t i = 1; i < N; ++i) {
        uint32_t offset_i = calc_offset(block_n, i);
        uint32_t offset_i_minus_1 = calc_offset(block_n, i-1);

        // run trsm on block A_j, j-1
    
        trsm_small::trsm_transpose_kernel_XY(
            N, block_n, out, out, in, offset_i_minus_1, offset_i_minus_1, offset_i, offset_i_minus_1, offset_i_minus_1, offset_i);
        __syncthreads();

        gemm::gemm_naive_XY(N, block_n, in, out, shared_mem, offset_i, offset_i, offset_i, offset_i_minus_1);
        __syncthreads();

        cholesky::cholesky_XY(block_n, N, block_n, shared_mem, out, offset_i, offset_i);
        __syncthreads();
        // if (threadIdx.x == 0) {
        //     for (uint32_t i = 0; i < N*block_n; ++i) {
        //         for (uint32_t j = 0; j < N*block_n; ++j) {
        //             printf("out[%u, %u] = %f\n", i, j, out[i * N*block_n + j]);
        //         }
        //     }
        // }
    }
}

// only works for block_n <= 32
void launch_cholesky_trsm_combined(const uint32_t N, const uint32_t block_n, float const *in, float *out) {
    uint32_t shared_mem_size = 4000 * sizeof(float);
        CUDA_CHECK(cudaFuncSetAttribute(
            cholesky_trsm_combined,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem_size));
    cholesky_trsm_combined<<<1, 32 * 32, shared_mem_size>>>(N, block_n, in, out);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_triblock_small(const uint32_t N, float const *in, float *out, void *workspace) {
    int32_t block_n = 32;
    int32_t num_blocks = (int32_t) (N/ block_n);
    uint32_t shared_mem_size = 4000 * sizeof(float);
        CUDA_CHECK(cudaFuncSetAttribute(
            cholesky_trsm_combined,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem_size));
    cholesky_trsm_combined<<<1, 32 * 32, shared_mem_size>>>(num_blocks, block_n, in, out);
    CUDA_CHECK(cudaDeviceSynchronize());
}
}


