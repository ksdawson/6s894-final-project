// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh"]}
// TL {"workspace_files": []}

#include "utils.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <math.h>
#include <stdio.h>
#include <vector>

// Macro to check CUDA errors
#define CUDA_CHECK(err)                                                        \
  if ((err) != cudaSuccess) {                                                  \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
            cudaGetErrorString(err));                                          \
    exit(EXIT_FAILURE);                                                        \
  }

////////////////////////////////////////////////////////////////////////////////
// Substitution methods

__device__ void index2rowcol(uint32_t *row, uint32_t *col, uint32_t k) {
  uint32_t r = (uint32_t)floorf((sqrtf(8.0f * k + 1.0f) - 1.0f) * 0.5f);
  uint32_t tr = r * (r + 1) / 2;

  *row = r;
  *col = k - tr;
}

template <uint32_t blocksize>
__device__ void blockSolve(uint32_t n, float const *A, float *x,
                           float const *b) {

  constexpr uint32_t numel = (blocksize * (blocksize + 1)) / 2;
  uint32_t tid = threadIdx.x;
  uint32_t bdim = blockDim.x;

  __shared__ float sh_A[numel];
  __shared__ float sh_x[blocksize];

  for (uint32_t k = tid; k < numel; k += bdim) {
    uint32_t i, j;
    index2rowcol(&i, &j, k);
    sh_A[k] = A[i * n + j];
  }

  for (uint32_t k = tid; k < blocksize; k += bdim) {
    sh_x[k] = b[k];
  }

  __syncthreads();

  for (uint32_t i = 0; i < blocksize; ++i) {

    if (tid == 0) {
      uint32_t diag_idx = i * (i + 1) / 2 + i;
      float val = sh_x[i] / sh_A[diag_idx];
      sh_x[i] = val;
    }

    __syncthreads();

    float x_i = sh_x[i];

    for (uint32_t j = i + 1 + tid; j < blocksize; j += bdim) {
      uint32_t A_ji_idx = j * (j + 1) / 2 + i;
      sh_x[j] -= sh_A[A_ji_idx] * x_i;
    }
    __syncthreads();
  }

  for (uint32_t k = tid; k < blocksize; k += bdim) {
    x[k] = sh_x[k];
  }
}

template <uint32_t blocksize>
__device__ void blockSubtract(uint32_t n, float *A, float *x, float *b) {
  constexpr uint32_t numel = blocksize * blocksize;
  uint32_t tid = threadIdx.x;
  uint32_t bdim = blockDim.x;

  __shared__ float sh_A[numel];
  __shared__ float sh_x[blocksize];

  for (uint32_t k = tid; k < numel; k += bdim) {
    uint32_t i = k / blocksize;
    uint32_t j = k % blocksize;
    sh_A[k] = A[i * n + j];
  }
  for (uint32_t k = tid; k < blocksize; k += bdim) {
    sh_x[k] = b[k];
  }
  __syncthreads();

  for (uint32_t i = tid; i < blocksize; i += bdim) {

    float dot = 0.0f;

#pragma unroll
    for (uint32_t j = 0; j < i + 1; ++j) {
      uint32_t offset = i * (i + 1) / 2;
      dot += sh_A[offset + j] * sh_x[j];
    }

    b[i] -= dot;
  }
}

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

template <uint32_t blocksize>
__global__ void blockSolve_kernel(uint32_t n, float const *A, float *x,
                                  float const *b) {
  blockSolve<blocksize>(n, A, x, b);
}

template <uint32_t blocksize>
__global__ void blockSubtract_kernel(uint32_t n, float *A, float *x, float *b) {
  blockSubtract<blocksize>(n, A, x, b);
}

template <uint32_t blocksize>
void buildTriangularSolverGraph(cudaGraph_t &graph, int num_blocks,
                                int n_stride, float *d_A, float *d_x,
                                float *d_b) {

  std::vector<cudaGraphNode_t> solve_nodes(num_blocks);
  std::vector<std::vector<cudaGraphNode_t>> subtract_nodes(num_blocks);

  for (int i = 0; i < num_blocks; i++)
    subtract_nodes[i].resize(i + 1);

  dim3 gridDim(1);
  dim3 blockDim(blocksize * blocksize > 1024
                    ? 1024
                    : blocksize * blocksize); // NOTE: may be unnecessary

  void *kernelArgs[4];

  for (int col = 0; col < num_blocks; ++col) {
    float *block_A = d_A + (col * blocksize * n_stride) + (col * blocksize);
    float *block_x = d_x + (col * blocksize);
    float *block_b = d_b + (col * blocksize);

    cudaKernelNodeParams solve_params = {0};
    solve_params.func = (void *)blockSolve_kernel<blocksize>;
    solve_params.gridDim = gridDim;
    solve_params.blockDim = blockDim;
    solve_params.sharedMemBytes = 0;

    kernelArgs[0] = &n_stride;
    kernelArgs[1] = &block_A;
    kernelArgs[2] = &block_x;
    kernelArgs[3] = &block_b;
    solve_params.kernelParams = kernelArgs;
    solve_params.extra = NULL;

    cudaGraphAddKernelNode(&solve_nodes[col], graph, NULL, 0, &solve_params);

    if (col > 0) {
      cudaGraphAddDependencies(graph, &subtract_nodes[col][col - 1],
                               &solve_nodes[col], NULL, 1);
    }

    for (int row = col + 1; row < num_blocks; ++row) {

      float *sub_A = d_A + (row * blocksize * n_stride) + (col * blocksize);
      float *sub_x = d_x + (row * blocksize);
      float *sub_b = d_b + (col * blocksize);

      cudaKernelNodeParams sub_params = {0};
      sub_params.func = (void *)blockSubtract_kernel<blocksize>;
      sub_params.gridDim = gridDim;
      sub_params.blockDim = blockDim;
      sub_params.sharedMemBytes = 0;

      kernelArgs[0] = &n_stride;
      kernelArgs[1] = &sub_A;
      kernelArgs[2] = &sub_x;
      kernelArgs[3] = &sub_b;
      sub_params.kernelParams = kernelArgs;
      sub_params.extra = NULL;

      cudaGraphAddKernelNode(&subtract_nodes[row][col], graph, NULL, 0,
                             &sub_params);

      cudaGraphAddDependencies(graph, &solve_nodes[col],
                               &subtract_nodes[row][col], NULL, 1);

      if (col > 0) {
        cudaGraphAddDependencies(graph, &subtract_nodes[row][col - 1],
                                 &subtract_nodes[row][col], NULL, 1);
      }
    }
  }
}

__global__ void forward_substitution_kernel(uint32_t n, const float *A,
                                            float *x, const float *b) {
  // forward_substitution<true, true>(n, A, x, b);
  blockSolve<16>(n, A, x, b);
}

////////////////////////////////////////////////////////////////////////////////
// TRSM
// In Cholesky, we want to solve L_ik = A_ik * L_kk^-T, but inverse is
// expensive. Instead solve L_kk * L_ik^T = A_ik, which can be done with TRSM.
// TRSM uses forward substitution in each row. It is sequential in a row, but
// all rows are independent.

__device__ void trsm(const uint32_t n, float const *A, float *X,
                     float const *B) {
  // Assumes we're solving for X in A * X^T = B, so we can use rows of X
  // instead of cols Get grid-level warp idx
  const uint32_t warps = blockDim.x / 32;
  const uint32_t warp_idx = warps * blockIdx.x + threadIdx.x / 32;

  // Iterate over rows of X,B with each row handled by one warp
  for (uint32_t i = warp_idx; i < n; i += warps * gridDim.x) {
    float *x = X + i * n;   // row
    float const *b = B + i; // col
    forward_substitution<true, false>(n, A, x, b);
  }
}

__global__ void trsm_kernel(uint32_t n, float *A, float *X, float *B) {
  trsm(n, A, X, B);
}

void trsmGraphLaunch(uint32_t n, float *A, float *X, float *B) {

  CUDA_CHECK(cudaMalloc(&A_d, N * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b_d, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&x_d, N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_d, b, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(x_d, 0, N * sizeof(float)));

  constexpr uint32_t blocksize = 32;
  uint32_t numblocks = n / blocksize;

  cudaGraph_t graph;
  CUDA_CHECK(cudaGraphCreate(&graph, 0));

  buildTriangularSolverGraph<blocksize>(graph, numblocks, n, A_d, X_d, B_d);

  cudaGraphExec_t graphExec;
  CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  CUDA_CHECK(cudaGraphLaunch(graphExec, 0));

  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
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
  // L^T) Alternatively, if trsm is solving row-wise L*x = b, then B = X_true
  // * L^T still fits.
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

int main() {
  srand(0);
  // Test forward substitution
  // test_forward_substitution(4);
  // test_forward_substitution(8);
  test_forward_substitution(16);
  // test_forward_substitution(1024);
  // Test trsm
  // test_trsm(2);
  // test_trsm(4);
  // test_trsm(8);
  // test_trsm(16);
  return 0;
}
