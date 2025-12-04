
// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh", "trsm_small.cuh", "cholesky.cuh", "gemm.cuh", "triblock.cuh"]}
// TL {"workspace_files": []}

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "utils.cuh"
#include "trsm_small.cuh"
#include "cholesky.cuh"
#include "gemm.cuh"
#include "triblock.cuh"

void generate_lower_triangular(uint32_t N, uint32_t block_n, float *A, uint32_t A_col_offset, uint32_t A_row_offset) {
    for (uint32_t i = 0; i < block_n; ++i) {
        for (uint32_t j = 0; j < block_n; ++j) {
            if (j <= i) {
                A[(A_col_offset + i) * N * block_n + (A_row_offset + j)] = (float)(rand() % 2 + 1); // positive
            } else {
                A[(A_col_offset + i) * N * block_n + (A_row_offset + j)] = 0.0f;
            }

            if (j == i) {
                A[(A_col_offset + i) * N * block_n + (A_row_offset + j)] += block_n;
            }
        }
    }
}

void generate_matrix(uint32_t N, uint32_t block_n, float *A, uint32_t A_col_offset, uint32_t A_row_offset) {
    for (uint32_t i = 0; i < block_n; ++i) {
        for (uint32_t j = 0; j < block_n; ++j) {
            A[(A_col_offset + i) * N * block_n + (A_row_offset + j)] = (float)(rand() % 2 + 1); // positive

            if (j == i) {
                A[(A_col_offset + i) * N * block_n + (A_row_offset + j)] += block_n;
            }
        }
    }
}

void test_triblock(uint32_t N, uint32_t block_n) {
    printf("Testing triblock with N=%u and block_n=%u\n", N, block_n);

    float *A = (float *)malloc(N * block_n * N * block_n * sizeof(float));
    float *A_gpu = (float *)malloc(N * block_n * N * block_n * sizeof(float));
    float *X_true = (float *)malloc(N * block_n * N * block_n * sizeof(float));
    float *X_gpu = (float *)malloc(N * block_n * N * block_n * sizeof(float));
    float *A_rep = (float *)malloc(N * block_n * N * block_n * sizeof(float));

    // Generate random X_true matrix
    for (uint32_t i = 0; i < N * block_n; ++i) {
        for (uint32_t j = 0; j < N * block_n; ++j) {
            X_true[i * N * block_n + j] = 0.0f;
        }
    }
    //memset(X_true, 0, N * block_n * N * block_n * sizeof(float));

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
            A[i * N * block_n + j] = 0.0f;
            A_rep[i * N * block_n + j] = 0.0f;
            for (uint32_t k = 0; k < N * block_n; ++k) {
                A[i * N * block_n + j] += X_true[i * N * block_n + k] * X_true[j * N * block_n + k];
                A_rep[i * N * block_n + j] += X_true[i * N * block_n + k] * X_true[j * N * block_n + k];
            }
        }
    }

    // for (uint32_t i = 0; i < N * block_n; ++i) {
    //     for (uint32_t j = 0; j < N * block_n; ++j) {
    //         printf("A[%u, %u] = %f\n", i, j, A[i * N * block_n + j]);
    //     }
    // }

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
    triblock_small::launch_cholesky_trsm_combined(N, block_n, A_d, X_d);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(X_gpu, X_d, N *block_n * N *block_n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results

    for (uint32_t i = 0; i < N*block_n; ++i) {
        for (uint32_t j = 0; j < N*block_n; ++j) {
            A_gpu[i * N*block_n + j] = 0.0f;
            for (uint32_t k = 0; k < N*block_n; ++k) {
                A_gpu[i * N*block_n + j] += X_gpu[i * N*block_n + k] * X_gpu[j * N*block_n + k];
            }
        }
    }

    bool failed = false;
    float tol = 1e-3f;

    // for (uint32_t i = 0; i < N*block_n; ++i) {
    //     for (uint32_t j = 0; j <= i; ++j) {
    //         printf("X_gpu[%u, %u] = %f, X_true[%u, %u] = %f\n", i, j, X_gpu[i * N*block_n + j], i, j, X_true[i * N*block_n + j]);
    //     }
    // }

    // for (uint32_t i = 0; i < N*block_n; ++i) {
    //     for (uint32_t j = 0; j < N*block_n; ++j) {
    //         if (fabsf(A[i * N*block_n + j] - A_rep[i * N*block_n + j]) > tol) {
    //             printf("A and A_rep mismatch at (%u, %u): got %.5f, expected %.5f\n", i, j, A[i * N*block_n + j], A_rep[i * N*block_n + j]);
    //             failed = true;
    //         }
    //     }
    // }
    
    for (uint32_t i = 0; i < N * block_n; ++i) {
        for (uint32_t j = 0; j < N * block_n; ++j) {
            //printf("A_gpu[%u, %u] = %f, A[%u, %u] = %f\n", i, j, A_gpu[i * N * block_n + j], i, j, A[i * N * block_n + j]);
            if (fabsf(A_gpu[i * N * block_n + j] - A[i * N * block_n + j]) < tol) {
                continue;
            } else {
                //printf("Mismatch at (%u, %u): got %.5f, expected %.5f\n", i, j, A_gpu[i * N * block_n + j], A[i * N * block_n + j]);
                failed = true;
            }
        }
    }

    if (!failed) {
        printf("Test PASSED for N=%u, block_n=%u\n", N, block_n);
    } else {
        printf("Test FAILED for N=%u, block_n=%u\n", N, block_n);
    }

    // for (uint32_t i = 0; i < N*block_n; ++i) {
    //     for (uint32_t j = 0; j < N*block_n; ++j) {
    //         printf("A_gpu[%u, %u] = %f, A[%u, %u] = %f\n", i, j, A_gpu[i * N*block_n + j], i, j, A[i * N*block_n + j]);
    //     }
    // }

    // for (uint32_t i = 0; i < N*block_n; ++i) {
    //     for (uint32_t j = 0; j < N*block_n; ++j) {
    //         printf("X_gpu[%u, %u] = %f, X_true[%u, %u] = %f\n", i, j, X_gpu[i * N*block_n + j], i, j, X_true[i * N*block_n + j]);
    //     }
    // }

    free(A);
    free(A_gpu);
    free(X_true);
    free(X_gpu);
    free(A_rep);
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(X_d));
    CUDA_CHECK(cudaDeviceReset());
}

int main() {
    srand(0);

    test_triblock(1, 32);
    test_triblock(2, 32);
    test_triblock(4, 32);
    test_triblock(8, 32);
    test_triblock(16, 32);
    test_triblock(32, 32); // 32, 32 does not work
    //test_triblock(10, 32);
    return 0;
}
