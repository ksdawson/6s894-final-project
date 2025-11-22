// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh"]}
// TL {"workspace_files": []}

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "utils.cuh"

// Macro to check CUDA errors
#define CUDA_CHECK(x) \
  do { \
      utils::cuda_check((x), __FILE__, __LINE__); \
  } while (0)

////////////////////////////////////////////////////////////////////////////////
// current naive gemm can only handle 32x32 matrix
template <uint32_t A_col_offset, uint32_t A_row_offset, uint32_t B_col_offset, uint32_t B_row_offset>
__global__ void gemm_kernel_XY(
    const uint32_t n, float const *A, float const *B, float *out
) {
    int32_t col_ID = threadIdx.x % n;
    int32_t row_ID = threadIdx.x / n;

    float sum = A[(A_col_offset + row_ID) * n + (A_row_offset + col_ID)];
    for (uint32_t i = 0; i < n; ++i) {
        sum -= B[(B_col_offset + row_ID) * n + (B_row_offset + i)] * B[(B_col_offset + col_ID) * n + (B_row_offset + i)];
    }

    if (row_ID < n && col_ID < n) {
        out[row_ID * n + col_ID] = sum;
    }
}

void launch_gemm_naive(
    const uint32_t n, float const *A, float const *B, float *out
) {
    gemm_kernel_XY<0, 0, 0, 0><<<1, 32*32>>>(n, A, B, out);
}

////////////////////////////////////////////////////////////////////////////////
// Test harness

void generate_lower_triangular(uint32_t N, float *A) {
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            if (j <= i) {
                A[i * N + j] = (float)(rand() % 9 + 1); // positive
            } else {
                A[i * N + j] = 0.0f;
            }
        }
    }
}

void test_gemm(uint32_t N) {
    printf("Testing gemm with N=%u\n", N);

    float *L = (float *)malloc(N * N * sizeof(float));
    float *A = (float *)malloc(N * N * sizeof(float));
    float *X_true = (float *)malloc(N * N * sizeof(float));
    float *X_gpu = (float *)malloc(N * N * sizeof(float));

    generate_lower_triangular(N, L);


    // Generate random A matrix
    for (uint32_t i = 0; i < N * N; ++i) {
        A[i] = (float)(rand() % 10 + 1);
    }

    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            X_true[i * N + j] = A[i * N + j];
            for (uint32_t k = 0; k < N; ++k) {
                X_true[i * N + j] -= L[i * N + k] * L[j * N + k];
            }
        }
    }

    // Allocate device memory
    float *A_d, *L_d, *X_d;
    CUDA_CHECK(cudaMalloc(&L_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&A_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&X_d, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(L_d, L, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(X_d, 0, N * N * sizeof(float)));

    // Launch kernel (1 block, multiple warps)
    gemm_kernel_XY<0, 0, 0, 0><<<1, 32 * 32>>>(N, A_d, L_d, X_d);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(X_gpu, X_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    bool failed = false;
    float tol = 1e-3f;
    for (uint32_t i = 0; i < N * N; ++i) {
        if (fabsf(X_gpu[i] - X_true[i]) > tol) {
            printf("Mismatch at (%u): got %.5f, expected %.5f\n", i, X_gpu[i],
                    X_true[i]);
            failed = true;
        }
    }

    if (!failed) {
        printf("Test PASSED for N=%u\n", N);
    } else {
        printf("Test FAILED for N=%u\n", N);
    }

    free(A);
    free(L);
    free(X_true);
    free(X_gpu);
    CUDA_CHECK(cudaFree(L_d));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(X_d));
}

int main() {
    srand(0);

    test_gemm(2);
    test_gemm(4);
    test_gemm(8);
    return 0;
}
