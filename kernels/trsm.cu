// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh"]}
// TL {"workspace_files": []}

#include "utils.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Macro to check CUDA errors
#define CUDA_CHECK(x) \
  do { \
      utils::cuda_check((x), __FILE__, __LINE__); \
  } while (0)

////////////////////////////////////////////////////////////////////////////////
// Substitution methods

template <bool x_row, bool b_row>
__device__ void forward_substitution(const uint32_t n, float const *A, float *x,
                                     float const *b) {
  // Use local thread idx as this is done at the warp level
  const uint32_t thread_idx = threadIdx.x % 32;
  for (uint32_t i = 0; i < n; ++i) {
    // Each thread computes a piece of the sum
    float partial_sum = 0.0f;
    for (uint32_t j = thread_idx; j < i; j += 32) {
      if constexpr (x_row)
        partial_sum += A[i * n + j] * x[j];
      else
        partial_sum += A[i * n + j] * x[j * n];
    }
    // Combine the sum across the warp
    float sum = utils::warp_prefix_sum<float>(partial_sum);
    // Last thread handles writing it back
    if (thread_idx == 31) {
      float xi = 0.0f;
      if constexpr (b_row)
        xi = (b[i] - sum) / A[i * n + i];
      else
        xi = (b[i * n] - sum) / A[i * n + i];
      if constexpr (x_row)
        x[i] = xi;
      else
        x[i * n] = xi;
    }
    // All threads need this iteration to be done
    __syncwarp();
  }
}

__global__ void forward_substitution_kernel(uint32_t n, const float *A,
                                            float *x, const float *b) {
  forward_substitution<true, true>(n, A, x, b);
}

////////////////////////////////////////////////////////////////////////////////
// TRSM
// In Cholesky, we want to solve L_ik = A_ik * L_kk^-T, but inverse is
// expensive. Instead solve L_kk * L_ik^T = A_ik, which can be done with TRSM.
// TRSM uses forward substitution in each row. It is sequential in a row, but
// all rows are independent.

__device__ void trsm(const uint32_t n, float const *A, float *X,
                     float const *B) {
  // Assumes we're solving for X in A * X^T = B, so we can use rows of X instead
  // of cols Get grid-level warp idx
  const uint32_t warps = blockDim.x / 32;
  const uint32_t warp_idx = warps * blockIdx.x + threadIdx.x / 32;

  // Iterate over rows of X,B with each row handled by one warp
  for (uint32_t i = warp_idx; i < n; i += warps * gridDim.x) {
    float *x = X + i * n;   // row
    float const *b = B + i; // col
    forward_substitution<true, false>(n, A, x, b);
  }
}

__global__ void trsm_kernel(uint32_t n, const float *A, float *X,
                            const float *B) {
  trsm(n, A, X, B);
}

////////////////////////////////////////////////////////////////////////////////
// TRSM solves A * X^T = B, so we need to transpose the kernel
__global__ void trsm_transpose_kernel_XY(const uint32_t n, float const *A, float *X, float const *B) {

  int32_t col_ID = threadIdx.x;
  for (uint32_t i = 0; i < n; ++i) {

    float sum = 0;
    //B[i*n + col_ID];

    for (uint32_t k = 0; k < i; ++k) {
      sum += A[i*n+k] * X[col_ID*n + k];
    }
    sum = (B[i*n + col_ID] - sum) / A[i*n+i];
    if (col_ID < n) {
      X[col_ID*n + i] = sum;
    }
    __syncthreads();
  }
}

// TRSM solves A * X = B
__global__ void trsm_kernel_XY(const uint32_t n, float const *A, float *X, float const *B) {

  int32_t col_ID = threadIdx.x;
  for (uint32_t i = 0; i < n; ++i) {

    float sum = B[i*n + col_ID];

    for (uint32_t k = 0; k < i; ++k) {
      sum -= A[i*n+k] * X[k*n + col_ID];
    }
    sum /= A[i*n+i];
    if (col_ID < n) {
      X[i*n + col_ID] = sum;
    }
    __syncthreads();
  }

  // if (threadIdx.x == 31) {
  //   for (uint32_t i = 0; i < n*n; ++i) {
  //     printf("A[%u] = %f\n", i, A[i]);
  //   }
  //   for (uint32_t i = 0; i < n*n; ++i) {
  //     printf("B[%u] = %f\n", i, B[i]);
  //   }
  // }
}

void launch_trsm_kernel_XY(const uint32_t n, float const *A, float *X, float const *B) {
  trsm_transpose_kernel_XY<<<1, 32>>>(n, A, X, B);
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
  forward_substitution_kernel<<<1, 32>>>(N, A_d, x_d, b_d);
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
  trsm_kernel<<<1, 32 * 32>>>(N, L_d, X_d, B_d);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(X_gpu, X_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));

  // Verify results
  bool failed = false;
  float tol = 1e-3f;
  for (uint32_t i = 0; i < N * N; ++i) {
    printf("X_gpu[%u] = %f, X_true[%u] = %f\n", i, X_gpu[i], i, X_true[i]);
    if (fabsf(X_gpu[i] - X_true[i]) < tol) {
      continue;
    } else {
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

int main() {
  srand(0);
  // Test forward substitution
  // test_forward_substitution(4);
  // test_forward_substitution(8);
  // test_forward_substitution(16);
  // test_forward_substitution(1024);
  // Test trsm
  test_trsm(2);
  test_trsm(4);
  test_trsm(32);
  //test_trsm(16);
  return 0;
}
