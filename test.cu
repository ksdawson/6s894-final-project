// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["cholesky_small.cuh", "cpu.cuh", "utils.cuh", "trsm_small.cuh", "gpu_block_kernel_fusion.cuh", "gpu_block_enhanced_kernel_fusion.cuh"]}
// TL {"workspace_files": []}

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "cpu.cuh"
#include "cholesky_small.cuh"
#include "utils.cuh"
#include "trsm_small.cuh"
#include "gpu_block_kernel_fusion.cuh"
#include "gpu_block_enhanced_kernel_fusion.cuh"

////////////////////////////////////////////////////////////////////////////////
// Cholesky test harness

void test_case_3x3_cpu() {
    // Test case
    const uint32_t n = 3;

    // Allocate host memory
    float *in_cpu = static_cast<float*>(malloc(n * n * sizeof(float)));
    float *out_cpu = static_cast<float*>(malloc(n * n * sizeof(float)));

    // Fill in test data on host
    in_cpu[0] = 4.0f;
    in_cpu[1] = 12.0f;
    in_cpu[2] = -16.0f;
    in_cpu[3] = 12.0f;
    in_cpu[4] = 37.0f;
    in_cpu[5] = -43.0f;
    in_cpu[6] = -16.0f;
    in_cpu[7] = -43.0f;
    in_cpu[8] = 98.0f;

    // Run Cholesky decomposition
    cholesky_cpu_naive(n, in_cpu, out_cpu);

    // Verify output
    bool test_failed = false;
    // Verify upper triangle
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = i + 1; j < n; ++j) {
            if (out_cpu[i * n + j] != 0.0f) {
                printf("Test 3x3 failed: upper triangle at (%u, %u) should be 0\n", i, j);
                test_failed = true;
                break;
            }
        }
    }
    // Verify lower triangle
    if (out_cpu[0] != 2.0f) {
        printf("Test 3x3 failed: lower triangle at (0, 0) should be 2\n");
        test_failed = true;
    } else if (out_cpu[3] != 6.0f) {
        printf("Test 3x3 failed: lower triangle at (1, 0) should be 6\n");
        test_failed = true;
    } else if (out_cpu[4] != 1.0f) {
        printf("Test 3x3 failed: lower triangle at (1, 1) should be 1\n");
        test_failed = true;
    } else if (out_cpu[6] != -8.0f) {
        printf("Test 3x3 failed: lower triangle at (2, 0) should be -8\n");
        test_failed = true;
    } else if (out_cpu[7] != 5.0f) {
        printf("Test 3x3 failed: lower triangle at (2, 1) should be 5\n");
        test_failed = true;
    } else if (out_cpu[8] != 3.0f) {
        printf("Test 3x3 failed: lower triangle at (2, 2) should be 3\n");
        test_failed = true;
    }

    if (!test_failed) {
        // Test passed
        printf("Test 3x3 passed\n");
    }

    // Clean up memory
    free(in_cpu);
    free(out_cpu);
}

void test_case_3x3_gpu() {
    // Test case
    constexpr uint32_t n = 3;

    // Allocate device memory
    float *in_gpu;
    float *out_gpu;
    CUDA_CHECK(cudaMalloc(&in_gpu, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_gpu, n * n * sizeof(float)));

    // Test data on host
    float cpu[n*n] = {
        4.0f, 12.0f, -16.0f,
        12.0f, 37.0f, -43.0f,
        -16.0f, -43.0f, 98.0f
    };

    // Copy from host to device
    CUDA_CHECK(cudaMemcpy(in_gpu, cpu, n * n * sizeof(float), cudaMemcpyHostToDevice));

    // Run Cholesky decomposition
    void *workspace = nullptr;
    cholesky_small::launch_cholesky(n, in_gpu, out_gpu, workspace);

    // Verify output
    CUDA_CHECK(cudaMemcpy(cpu, out_gpu, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    bool test_failed = false;
    // Verify upper triangle
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = i + 1; j < n; ++j) {
            if (cpu[i * n + j] != 0.0f) {
                printf("Test 3x3 failed: upper triangle at (%u, %u) should be 0\n", i, j);
                test_failed = true;
                break;
            }
        }
    }
    // Verify lower triangle
    if (cpu[0] != 2.0f) {
        printf("Test 3x3 failed: lower triangle at (0, 0) should be 2\n");
        test_failed = true;
    } else if (cpu[3] != 6.0f) {
        printf("Test 3x3 failed: lower triangle at (1, 0) should be 6\n");
        test_failed = true;
    } else if (cpu[4] != 1.0f) {
        printf("Test 3x3 failed: lower triangle at (1, 1) should be 1\n");
        test_failed = true;
    } else if (cpu[6] != -8.0f) {
        printf("Test 3x3 failed: lower triangle at (2, 0) should be -8\n");
        test_failed = true;
    } else if (cpu[7] != 5.0f) {
        printf("Test 3x3 failed: lower triangle at (2, 1) should be 5\n");
        test_failed = true;
    } else if (cpu[8] != 3.0f) {
        printf("Test 3x3 failed: lower triangle at (2, 2) should be 3\n");
        test_failed = true;
    }

    if (!test_failed) {
        // Test passed
        printf("Test 3x3 passed\n");
    }

    // Clean up memory
    CUDA_CHECK(cudaFree(in_gpu));
    CUDA_CHECK(cudaFree(out_gpu));
}

// Generate a random SPD matrix of size N x N
void generate_spd_matrix(uint32_t N, float* A) {
    float* L = (float*)malloc(N * N * sizeof(float));

    // Fill L lower-triangular with random positive numbers
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            L[i*N + j] = (float)(rand() % 10 + 1);
        }
        for (uint32_t j = i+1; j < N; ++j) {
            L[i*N + j] = 0.0f;
        }
    }

    // Compute A = L * L^T
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k <= (i<j?i:j); ++k) {
                sum += L[i*N + k] * L[j*N + k];
            }
            A[i*N + j] = sum;
        }
    }

    free(L);
}

// Test case for any size
void test_case_gpu(uint32_t N,
    void (*chol)(const uint32_t n, float const *in, float *out, void *workspace)
) {
    printf("Testing Cholesky %ux%u\n", N, N);

    float *in_cpu  = (float*)malloc(N * N * sizeof(float));
    float *out_cpu = (float*)malloc(N * N * sizeof(float));

    float *in_gpu, *out_gpu;
    CUDA_CHECK(cudaMalloc(&in_gpu, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_gpu, N * N * sizeof(float)));

    // Generate SPD input
    generate_spd_matrix(N, in_cpu);

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(in_gpu, in_cpu, N * N * sizeof(float), cudaMemcpyHostToDevice));

    // Run Cholesky
    void *workspace = nullptr;
    chol(N, in_gpu, out_gpu, workspace);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(out_cpu, out_gpu, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: L * L^T ≈ original matrix
    bool test_failed = false;
    float tol = 1e-2f;

    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k <= (i<j?i:j); ++k) {
                sum += out_cpu[i*N + k] * out_cpu[j*N + k];
            }
            if (fabsf(sum - in_cpu[i*N + j]) > tol) {
                printf("Mismatch at (%u,%u): computed %f, expected %f\n", i, j, sum, in_cpu[i*N + j]);
                test_failed = true;
            }
        }
    }

    if (!test_failed) {
        printf("Test %ux%u passed\n", N, N);
    } else {
        printf("Test %ux%u FAILED\n", N, N);
    }

    free(in_cpu);
    free(out_cpu);
    CUDA_CHECK(cudaFree(in_gpu));
    CUDA_CHECK(cudaFree(out_gpu));
}

////////////////////////////////////////////////////////////////////////////////
// TRSM test harness

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

void test_forward_substitution(uint32_t N) {
  printf("Testing forward substitution with N=%u\n", N);

  float *A = (float *)malloc(N * N * sizeof(float));
  float *x_true = (float *)malloc(N * sizeof(float));
  float *b = (float *)malloc(N * sizeof(float));
  float *x_gpu = (float *)malloc(N * sizeof(float));

  generate_lower_triangular(N, A);

  // Generate random true solution
  for (uint32_t i = 0; i < N; ++i) {
    x_true[i] = (float)(rand() % 10 + 1);
  }

  // Compute b = A * x_true
  for (uint32_t i = 0; i < N; ++i) {
    float sum = 0.0f;
    for (uint32_t j = 0; j <= i; ++j) {
      sum += A[i * N + j] * x_true[j];
    }
    b[i] = sum;
  }

  // Allocate device memory
  float *A_d, *b_d, *x_d;
  CUDA_CHECK(cudaMalloc(&A_d, N * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b_d, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&x_d, N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_d, b, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(x_d, 0, N * sizeof(float)));

  // Launch with one warp (since function assumes warp-level sum)
  trsm_small::forward_substitution_kernel<<<1, 32>>>(N, A_d, x_d, b_d);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(x_gpu, x_d, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Verify results
  bool failed = false;
  float tol = 1e-3f;
  for (uint32_t i = 0; i < N; ++i) {
    if (fabsf(x_gpu[i] - x_true[i]) > tol) {
      printf("Mismatch at %u: got %.5f, expected %.5f\n", i, x_gpu[i],
             x_true[i]);
      failed = true;
    }
  }

  if (!failed)
    printf("Test PASSED for N=%u\n", N);
  else
    printf("Test FAILED for N=%u\n", N);

  free(A);
  free(x_true);
  free(b);
  free(x_gpu);
  CUDA_CHECK(cudaFree(A_d));
  CUDA_CHECK(cudaFree(b_d));
  CUDA_CHECK(cudaFree(x_d));
}

void test_trsm(uint32_t N) {
  printf("Testing trsm with N=%u\n", N);

  float *L = (float *)malloc(N * N * sizeof(float));
  float *X_true = (float *)malloc(N * N * sizeof(float));
  float *B = (float *)malloc(N * N * sizeof(float));
  float *X_gpu = (float *)malloc(N * N * sizeof(float));

  generate_lower_triangular(N, L);

  // Generate random true solution
  for (uint32_t i = 0; i < N * N; ++i) {
    X_true[i] = (float)(rand() % 10 + 1);
  }

  // Compute B = X_true * L^T (since trsm solves L * X^T = B, so B = X_true *
  // L^T) Alternatively, if trsm is solving row-wise L*x = b, then B = X_true *
  // L^T still fits.
  for (uint32_t i = 0; i < N; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < N; ++k) {
        sum += X_true[i * N + k] * L[j * N + k]; // row-wise L^T multiply
      }
      B[j * N + i] = sum;
    }
  }

  // Allocate device memory
  float *L_d, *B_d, *X_d;
  CUDA_CHECK(cudaMalloc(&L_d, N * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&B_d, N * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&X_d, N * N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(L_d, L, N * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(X_d, 0, N * N * sizeof(float)));

  // Launch kernel (1 block, multiple warps)
  trsm_small::trsm_kernel<<<1, 32 * 32>>>(N, L_d, X_d, B_d);
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

  free(L);
  free(B);
  free(X_true);
  free(X_gpu);
  CUDA_CHECK(cudaFree(L_d));
  CUDA_CHECK(cudaFree(B_d));
  CUDA_CHECK(cudaFree(X_d));
}

////////////////////////////////////////////////////////////////////////////////
// Entry point

int main(int argc, char **argv) {
    printf("Testing CPU naive\n");
    test_case_3x3_cpu();
    printf("\n");

    printf("Testing GPU naive\n");
    test_case_3x3_gpu();
    test_case_gpu(50, cholesky_small::launch_cholesky);
    printf("\n");

    printf("Testing TRSM naive\n");
    srand(0);
    // Test forward substitution
    test_forward_substitution(4);
    test_forward_substitution(8);
    test_forward_substitution(16);
    test_forward_substitution(1024);
    // Test trsm
    test_trsm(2);
    test_trsm(4);
    test_trsm(8);
    test_trsm(16);
    printf("\n");

    printf("Testing GPU block w/ kernel fusion\n");
    printf("1x1 block Cholesky\n");
    test_case_gpu(64, block_cholesky_space::launch_block_cholesky);
    printf("2x2 block Cholesky\n");
    test_case_gpu(128, block_cholesky_space::launch_block_cholesky);
    printf("4x4 block Cholesky\n");
    test_case_gpu(256, block_cholesky_space::launch_block_cholesky);
    printf("8x8 block Cholesky\n");
    test_case_gpu(512, block_cholesky_space::launch_block_cholesky);
    printf("\n");

    printf("Testing GPU block w/ enhanced kernel fusion\n");
    printf("1x1 block Cholesky\n");
    test_case_gpu(64, alt_kernel_fusion::launch_block_cholesky);
    printf("2x2 block Cholesky\n");
    test_case_gpu(128, alt_kernel_fusion::launch_block_cholesky);
    printf("4x4 block Cholesky\n");
    test_case_gpu(256, alt_kernel_fusion::launch_block_cholesky);
    printf("8x8 block Cholesky\n");
    test_case_gpu(512, alt_kernel_fusion::launch_block_cholesky);
    printf("\n");

    return 0;
}