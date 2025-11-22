// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh", "trsm_naive.cuh", "cholesky_naive.cuh", "gemm.cuh"]}
// TL {"workspace_files": []}

//#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "utils.cuh"
#include "trsm_naive.cuh"
#include "cholesky_naive.cuh"
#include "gemm.cuh"

#define CUDA_CHECK(x) \
  do { \
      utils::cuda_check((x), __FILE__, __LINE__); \
  } while (0)

namespace triblock {

__device__ uint32_t calc_offset(const uint32_t block_n, const uint32_t block_idx) {
    return block_idx * block_n;
}

// Works for N >=3, block_n >=2
// Computes out = in*in^T, block Cholesky decomposition for triblock diagonal
// N: number of blocks in triblock diagonal
// block_n: dimension of each block in triblock diagonal
// note: currently can only handle block_n <=32, this naive version need to be optimized with better shared memory or register reuse
__global__ void cholesky_trsm_combined(const uint32_t N, const uint32_t block_n, float const *in, float *out) {
    extern __shared__ float shared_mem[];

    cholesky_naive::cholesky_parallel_col_XY(N*block_n, N, block_n, in, out, 0, 0);

    for (uint32_t i = 1; i < N; ++i) {
        uint32_t offset_i = calc_offset(block_n, i);
        uint32_t offset_i_minus_1 = calc_offset(block_n, i-1);

        // run trsm on block A_j, j-1
    
        trsm_naive::trsm_transpose_kernel_XY(
            N, block_n, out, out, in, offset_i_minus_1, offset_i_minus_1, offset_i, offset_i_minus_1, offset_i_minus_1, offset_i);

        gemm::gemm_naive_XY(N, block_n, in, out, shared_mem, offset_i, offset_i, offset_i, offset_i_minus_1);

        cholesky_naive::cholesky_parallel_col_XY(block_n, N, block_n, shared_mem, out, offset_i, offset_i);
        // if (threadIdx.x == 0) {
        //     for (uint32_t i = 0; i < N*block_n; ++i) {
        //         for (uint32_t j = 0; j < N*block_n; ++j) {
        //             printf("out[%u, %u] = %f\n", i, j, out[i * N*block_n + j]);
        //         }
        //     }
        // }
    }
}



void launch_cholesky_trsm_combined(const uint32_t N, const uint32_t block_n, float const *in, float *out) {
    uint32_t shared_mem_size = 1000 * sizeof(float);
        CUDA_CHECK(cudaFuncSetAttribute(
            cholesky_trsm_combined,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem_size));
    cholesky_trsm_combined<<<1, 32 * 32, shared_mem_size>>>(N, block_n, in, out);
    CUDA_CHECK(cudaDeviceSynchronize());
}
}



void generate_lower_triangular(uint32_t N, uint32_t block_n, float *A, uint32_t A_col_offset, uint32_t A_row_offset) {
    for (uint32_t i = 0; i < block_n; ++i) {
        for (uint32_t j = 0; j < block_n; ++j) {
            if (j <= i) {
                A[(A_col_offset + i) * N * block_n + (A_row_offset + j)] = (float)(rand() % 9 + 1); // positive
            } else {
                A[(A_col_offset + i) * N * block_n + (A_row_offset + j)] = 0.0f;
            }
        }
    }
}

void generate_matrix(uint32_t N, uint32_t block_n, float *A, uint32_t A_col_offset, uint32_t A_row_offset) {
    for (uint32_t i = 0; i < block_n; ++i) {
        for (uint32_t j = 0; j < block_n; ++j) {
            A[(A_col_offset + i) * N * block_n + (A_row_offset + j)] = (float)(rand() % 9 + 1); // positive
        }
    }
}

void test_triblock(uint32_t N, uint32_t block_n) {
    printf("Testing triblock with N=%u and block_n=%u\n", N, block_n);

    float *A = (float *)malloc(N * block_n * N * block_n * sizeof(float));
    float *A_gpu = (float *)malloc(N * block_n * N * block_n * sizeof(float));
    float *X_true = (float *)malloc(N * block_n * N * block_n * sizeof(float));
    float *X_gpu = (float *)malloc(N * block_n * N * block_n * sizeof(float));

    // Generate random X_true matrix
    generate_lower_triangular(N, block_n, X_true, 0, 0);
    
    for (uint32_t i = 1; i < N; ++i) {
        generate_matrix(N, block_n, X_true, i*block_n, (i-1)*block_n);
        generate_lower_triangular(N, block_n, X_true, i*block_n, i*block_n);
    }

    // for (uint32_t i = 0; i < N * block_n; ++i) {
    //     for (uint32_t j = 0; j < N * block_n; ++j) {
    //         printf("X_true[%u, %u] = %f\n", i, j, X_true[i * N * block_n + j]);
    //     }
    // }

    // calculate A = X_true * X_true^T
    for (uint32_t i = 0; i < N * block_n; ++i) {
        for (uint32_t j = 0; j < N * block_n; ++j) {
            for (uint32_t k = 0; k < N * block_n; ++k) {
                A[i * N * block_n + j] += X_true[i * N * block_n + k] * X_true[j * N * block_n + k];
            }
        }
    }

    // for (uint32_t i = 0; i < N * block_n; ++i) {
    //     for (uint32_t j = 0; j < N * block_n; ++j) {
    //         printf("A[%u, %u] = %f\n", i, j, A[i * N * block_n + j]);
    //     }
    // }

    // Allocate device memory
    float *A_d, *X_d;
    CUDA_CHECK(cudaMalloc(&A_d, N * block_n * N * block_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&X_d, N * block_n * N * block_n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(A_d, A, N * block_n * N * block_n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(X_d, 0, N * block_n * N * block_n * sizeof(float)));

    // Launch kernel (1 block, multiple warps)
    triblock::launch_cholesky_trsm_combined(N, block_n, A_d, X_d);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(X_gpu, X_d, N *block_n * N *block_n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    for (uint32_t i = 0; i < N*block_n; ++i) {
        for (uint32_t j = 0; j < N*block_n; ++j) {
            for (uint32_t k = 0; k < N*block_n; ++k) {
                A_gpu[i * N*block_n + j] += X_gpu[i * N*block_n + k] * X_gpu[j * N*block_n + k];
            }
        }
    }

    bool failed = false;
    float tol = 1e-3f;
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            if (fabsf(A_gpu[i * N + j] - A[i * N + j]) > tol) {
                printf("Mismatch at (%u, %u): got %.5f, expected %.5f\n", i, j, A_gpu[i * N + j], A[i * N + j]);
                failed = true;
            }
        }
    }

    if (!failed) {
        printf("Test PASSED for N=%u\n", N);
    } else {
        printf("Test FAILED for N=%u\n", N);
    }

    free(A);
    free(A_gpu);
    free(X_true);
    free(X_gpu);
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(X_d));
}

int main() {
    srand(0);

    test_triblock(3, 2);
    //test_triblock(8, 2);
    test_triblock(4, 4);
    return 0;
}
