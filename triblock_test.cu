
// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh", "trsm_small.cuh", "cholesky.cuh", "gemm.cuh", "triblock.cuh", "gpu_block_kernel_fusion.cuh", "cholesky_small.cuh", "triblock_helper.cuh", "gpu_block_enhanced_deluxe_kernel_fusion.cuh", "gpu_block_enhanced_kernel_fusion.cuh"]}
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
#include "gpu_block_kernel_fusion.cuh"
#include "cholesky_small.cuh"
#include "triblock_helper.cuh"
#include "gpu_block_enhanced_deluxe_kernel_fusion.cuh"
#include "gpu_block_enhanced_kernel_fusion.cuh"

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
    uint32_t num_blocks = (uint32_t)(N / block_n);
    printf("Testing triblock with N=%u and block_n=%u\n", N, block_n);

    float *A = (float *)malloc(N*N* sizeof(float));
    float *A_gpu = (float *)malloc(N*N * sizeof(float));
    float *X_true = (float *)malloc(N*N * sizeof(float));
    float *X_gpu = (float *)malloc(N*N * sizeof(float));
    float *A_rep = (float *)malloc(N*N * sizeof(float));

    // Generate random X_true matrix
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            X_true[i * N + j] = 0.0f;
        }
    }
    //memset(X_true, 0, N * block_n * N * block_n * sizeof(float));

    generate_lower_triangular(num_blocks, block_n, X_true, 0, 0);
    
    for (uint32_t i = 1; i < num_blocks; ++i) {
        generate_matrix(num_blocks, block_n, X_true, i*block_n, (i-1)*block_n);
        generate_lower_triangular(num_blocks, block_n, X_true, i*block_n, i*block_n);
    }

    // for (uint32_t i = 0; i < N * block_n; ++i) {
    //     for (uint32_t j = 0; j < N * block_n; ++j) {
    //         printf("X_true[%u, %u] = %f\n", i, j, X_true[i * N * block_n + j]);
    //     }
    // }

    // calculate A = X_true * X_true^T
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            A[i * N + j] = 0.0f;
            A_rep[i * N + j] = 0.0f;
            for (uint32_t k = 0; k < N; ++k) {
                A[i * N + j] += X_true[i * N + k] * X_true[j * N + k];
                A_rep[i * N + j] += X_true[i * N + k] * X_true[j * N + k];
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
    CUDA_CHECK(cudaMalloc(&A_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&X_d, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(X_d, 0, N * N * sizeof(float)));

    // Launch kernel (1 block, multiple warps)
    //triblock_small::launch_cholesky_trsm_combined(N, block_n, A_d, X_d);
    //triblock::launch_triblock_small(N, block_n, A_d, X_d, nullptr);
    triblock::launch_triblock(N, block_n, A_d, X_d, nullptr);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(X_gpu, X_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    bool failed = false;
    float tol = 1e-4f;
    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            float diff = X_gpu[i * N + j] - X_true[i * N + j];
            mse += diff * diff;
            if (fabsf(diff) > tol) {
                printf("Mismatch at (%u, %u): got %.5f, expected %.5f\n", i, j, X_gpu[i * N + j], X_true[i * N + j]);
                failed = true;
            }
            ref_mean_square += X_true[i * N + j] * X_true[i * N + j];
        }
    }
    mse /= N * N;
    ref_mean_square /= N * N;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);
    printf("RMSE = %f, REL_RMSE = %f\n", rmse, rel_rmse);


    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            printf("X_gpu[%u, %u] = %f, X_true[%u, %u] = %f\n", i, j, X_gpu[i * N + j], i, j, X_true[i * N + j]);
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

    // test_triblock(32, 32);
    // test_triblock(64, 32);
    // test_triblock(128, 32);
    // test_triblock(256, 32);
    // test_triblock(512, 32);
    // test_triblock(1024, 32);
    // test_triblock(64, 64);
    // test_triblock(128, 64);
    // test_triblock(256, 64);
    // test_triblock(512, 64);
    // test_triblock(64, 64);
    // test_triblock(1024, 64);
    test_triblock(1024, 128);
    // test_triblock(1024, 256);
    // test_triblock(1024, 512);
    // test_triblock(1024, 1024);

    //test_triblock(2048, 32);
    return 0;
}
