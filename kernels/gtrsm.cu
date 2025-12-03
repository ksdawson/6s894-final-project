#include "utils.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <math.h>
#include <random>
#include <stdio.h>
#include <vector>

#define CUDA_CHECK(err)                                                        \
  if ((err) != cudaSuccess) {                                                  \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
            cudaGetErrorString(err));                                          \
    exit(EXIT_FAILURE);                                                        \
  }

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
__device__ void blockSubtract(uint32_t n, float const *A, float *x, float *b) {
  constexpr uint32_t numel = blocksize * blocksize;
  uint32_t tid = threadIdx.x;
  uint32_t bdim = blockDim.x;
  __shared__ float sh_A[numel];
  __shared__ float sh_x[blocksize];

  for (uint32_t k = tid; k < numel; k += bdim) {
    uint32_t r = k / blocksize;
    uint32_t c = k % blocksize;
    sh_A[k] = A[r * n + c];
  }

  for (uint32_t k = tid; k < blocksize; k += bdim) {
    sh_x[k] = x[k];
  }
  __syncthreads();

  for (uint32_t i = tid; i < blocksize; i += bdim) {
    float dot = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < blocksize; ++j) {
      dot += sh_A[i * blocksize + j] * sh_x[j];
    }
    b[i] -= dot;
  }
}
template <uint32_t blocksize>
__global__ void blockSolve_kernel(uint32_t n, float *A, float *x, float *b) {
  blockSolve<blocksize>(n, A, x, b);
}

template <uint32_t blocksize>
__global__ void blockSubtract_kernel(uint32_t n, float *A, float *x, float *b) {
  blockSubtract<blocksize>(n, A, x, b);
}

template <uint32_t blocksize>
void buildTriangularSolverGraph(cudaGraph_t &graph, int num_blocks,
                                int n_stride, int d_dim, float *d_A, float *d_x,
                                float *d_b) {

  std::vector<cudaGraphNode_t> solve_nodes(num_blocks);
  std::vector<std::vector<cudaGraphNode_t>> subtract_nodes(num_blocks - 1);

  for (int i = 0; i < num_blocks - 1; i++)
    subtract_nodes[i].resize(i + 1);

  dim3 gridDim(1);
  dim3 blockDim(blocksize * blocksize > 1024
                    ? 1024
                    : blocksize * blocksize); // NOTE: may be unnecessary

  for (int col = 0; col < num_blocks; ++col) {
    float *block_A = d_A + (col * blocksize * n_stride) + (col * blocksize);
    float *block_x = d_x + (col * blocksize);
    float *block_b = d_b + (col * blocksize);

    cudaKernelNodeParams solve_params = {0};
    solve_params.func = (void *)blockSolve_kernel<blocksize>;
    solve_params.gridDim = gridDim;
    solve_params.blockDim = blockDim;
    solve_params.sharedMemBytes = 0;

    void *kernelArgs[4];
    kernelArgs[0] = &n_stride;
    kernelArgs[1] = &block_A;
    kernelArgs[2] = &block_x;
    kernelArgs[3] = &block_b;
    solve_params.kernelParams = kernelArgs;
    solve_params.extra = NULL;

    cudaGraphAddKernelNode(&solve_nodes[col], graph, NULL, 0, &solve_params);

    for (int row = col + 1; row < num_blocks; ++row) {

      float *sub_A = d_A + (row * blocksize * n_stride) + (col * blocksize);
      float *sub_x = d_x + (col * blocksize);
      float *sub_b = d_b + (row * blocksize);

      cudaKernelNodeParams sub_params = {0};
      sub_params.func = (void *)blockSubtract_kernel<blocksize>;
      sub_params.gridDim = gridDim;
      sub_params.blockDim = blockDim;
      sub_params.sharedMemBytes = 0;

      void *kernelArgs[4];
      kernelArgs[0] = &n_stride;
      kernelArgs[1] = &sub_A;
      kernelArgs[2] = &sub_x;
      kernelArgs[3] = &sub_b;
      sub_params.kernelParams = kernelArgs;
      sub_params.extra = NULL;

      cudaGraphAddKernelNode(&subtract_nodes[row - 1][col], graph, NULL, 0,
                             &sub_params);

      cudaGraphAddDependencies(graph, &solve_nodes[col],
                               &subtract_nodes[row - 1][col], NULL, 1);
    }

    if (col > 0) {
      for (int row = 0; row < col; ++row) {
        cudaGraphAddDependencies(graph, &subtract_nodes[col - 1][row],
                                 &solve_nodes[col], NULL, 1);
      }
    }
  }
}

void trsmGraphLaunch(uint32_t n, float *A, float *b, float *x) {

  float *A_d, *b_d, *x_d;
  CUDA_CHECK(cudaMalloc(&A_d, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b_d, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&x_d, n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(A_d, A, n * n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_d, b, n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(x_d, 0, n * sizeof(float)));

  constexpr uint32_t blocksize = 32;
  uint32_t numblocks = n / blocksize;

  cudaGraph_t graph;
  CUDA_CHECK(cudaGraphCreate(&graph, 0));

  buildTriangularSolverGraph<blocksize>(graph, numblocks, n, 1, A_d, x_d, b_d);

  cudaGraphExec_t graphExec;
  CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  CUDA_CHECK(cudaGraphLaunch(graphExec, 0));

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(x, x_d, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(b, b_d, n * sizeof(float), cudaMemcpyDeviceToHost));

  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
}

/////////////////////////////////////////////////////
/// Test
////////////////////////////////////////////////////

void verify_result(int n, const std::vector<float> &x,
                   const std::vector<float> &x_true) {
  float max_err = 0.0f;
  for (int i = 0; i < n; ++i) {
    float sum = 0.0f;
    // std::cout << x[i] << '\t' << x_true[i] << std::endl;
    float err = abs(x_true[i] - x[i]);
    if (err > max_err)
      max_err = err;
  }
  std::cout << "Max Error: " << max_err << std::endl;
  if (max_err < 1e-4)
    std::cout << "Test PASSED" << std::endl;
  else
    std::cout << "Test FAILED" << std::endl;
}

int main() {
  const int BLOCK_SIZE = 32;
  const int NUM_BLOCKS = 16;
  const int N = BLOCK_SIZE * NUM_BLOCKS;

  std::cout << "Running Block Triangular Solver with Graph..." << std::endl;
  std::cout << "Matrix Size: " << N << "x" << N << std::endl;
  std::cout << "Block Size:  " << BLOCK_SIZE << std::endl;
  std::cout << "Number of Blocks:" << NUM_BLOCKS << std::endl;

  std::vector<float> h_A(N * N);
  std::vector<float> h_b(N);
  std::vector<float> h_x(N); // Will hold result

  // Initialize A (Lower Triangular) and x (Random), then compute b
  std::default_random_engine gen(42);
  std::uniform_real_distribution<float> dist(0.1f, 1.0f);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j <= i; ++j) {
      h_A[i * N + j] = dist(gen);
      if (i == j)
        h_A[i * N + j] += 2.0f; // Diagonal dominance
    }
  }

  std::vector<float> h_x_true(N);
  for (int i = 0; i < N; i++)
    h_x_true[i] = dist(gen);

  // Compute b = A * x_true
  for (int i = 0; i < N; ++i) {
    float sum = 0.0f;
    for (int j = 0; j <= i; ++j) {
      sum += h_A[i * N + j] * h_x_true[j];
    }
    h_b[i] = sum;
  }

  float *d_A, *d_x, *d_b;
  cudaMalloc(&d_A, N * N * sizeof(float));
  cudaMalloc(&d_x, N * sizeof(float)); // will act as in/out
  cudaMalloc(&d_b,
             N * sizeof(float)); // not strictly needed if we init d_x with b

  cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  cudaGraph_t graph;
  cudaGraphCreate(&graph, 0);

  buildTriangularSolverGraph<BLOCK_SIZE>(graph, NUM_BLOCKS, N, 1, d_A, d_x,
                                         d_b);

  cudaGraphExec_t graphExec;
  cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

  cudaGraphLaunch(graphExec, 0);
  cudaDeviceSynchronize();

  // 6. Verify
  cudaMemcpy(h_x.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b.data(), d_b, N * sizeof(float), cudaMemcpyDeviceToHost);
  verify_result(N, h_x, h_x_true);

  // Cleanup
  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
  cudaFree(d_A);
  cudaFree(d_x);
  cudaFree(d_b);

  return 0;
}
