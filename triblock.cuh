// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh", "trsm_small.cuh", "cholesky.cuh", "gemm.cuh"]}
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

// #define CUDA_CHECK(x) \
//   do { \
//       utils::cuda_check((x), __FILE__, __LINE__); \
//   } while (0)

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


