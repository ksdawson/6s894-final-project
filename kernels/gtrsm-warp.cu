// TL+ {"compile_flags": ["-lcuda", "-lcublas"]}
// TL+ {"header_files": ["utils.cuh"]}
// TL {"workspace_files": []}

#include "utils.cuh"
#include <cstdint>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <istream>
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

#define CUBLAS_CHECK(err)                                                      \
  if ((err) != CUBLAS_STATUS_SUCCESS) {                                        \
    fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__);            \
    exit(EXIT_FAILURE);                                                        \
  }

template <uint32_t blocksize>
__forceinline__ __device__ void
blockSolve_warp(uint32_t col_id, uint32_t n, float const *A_sh, float *x_sh) {

  float x_val = x_sh[col_id];
#pragma unroll
  for (int i = 0; i < blocksize; i++) {
    float pivot = x_val;
    if (col_id == i) {
      float diag = A_sh[i * blocksize + i];
      pivot /= diag;
      x_val = pivot;
    }

    float pivot_bcast = __shfl_sync(0xffffffff, pivot, i);

    if (col_id > i) {
      x_val -= A_sh[col_id * blocksize + i] * pivot_bcast;
    }
  }

  x_sh[col_id] = x_val;
}

template <uint32_t blocksize>
__forceinline__ __device__ void blockSubtract_warp(uint32_t col_id, uint32_t n,
                                                   float const *A_sh,
                                                   float *x_sh, float *b_sh) {
  float delta = 0.0f;
#pragma unroll
  for (int i = 0; i < blocksize; i++) {
    delta += A_sh[col_id * blocksize + i] * x_sh[i];
  }
  b_sh[col_id] -= delta;
}

template <uint32_t blocksize>
__forceinline__ __device__ void load_A(uint32_t n, float const *A,
                                       float *A_sh) {
  uint32_t tid = threadIdx.x;
  uint32_t bdim = blockDim.x;

  for (uint32_t i = tid; i < blocksize * blocksize; i += bdim) {
    A_sh[i] = A[(i / blocksize) * n + i % blocksize];
  }
}

template <uint32_t blocksize>
__forceinline__ __device__ void load_B(uint32_t n, float *B, float *B_sh) {
  uint32_t tid = threadIdx.x;
  uint32_t bdim = blockDim.x;

  for (uint32_t i = tid; i < blocksize * blocksize; i += bdim) {
    B_sh[i] = B[(i / blocksize) * n + i % blocksize];
  }
}

template <uint32_t blocksize>
__forceinline__ __device__ void load_X(uint32_t n, float *X, float *X_sh) {
  uint32_t tid = threadIdx.x;
  uint32_t bdim = blockDim.x;

  for (uint32_t i = tid; i < blocksize * blocksize; i += bdim) {
    X_sh[i] = X[(i / blocksize) * n + i % blocksize];
  }
}

template <uint32_t blocksize>
__forceinline__ __device__ void store_X(uint32_t n, float *X_sh, float *X) {
  uint32_t tid = threadIdx.x;
  uint32_t bdim = blockDim.x;

  for (uint32_t i = tid; i < blocksize * blocksize; i += bdim) {
    X[(i / blocksize) * n + i % blocksize] = X_sh[i];
  }
}

template <uint32_t blocksize>
__forceinline__ __device__ void store_B(uint32_t n, float *B_sh, float *B) {
  uint32_t tid = threadIdx.x;
  uint32_t bdim = blockDim.x;

  for (uint32_t i = tid; i < blocksize * blocksize; i += bdim) {
    B[(i / blocksize) * n + i % blocksize] = B_sh[i];
  }
}

template <uint32_t blocksize>
__global__ void blockSolve_kernel(uint32_t n, uint32_t k, float const *A,
                                  float *X, float *B) {
  extern __shared__ float shmem[];
  float *A_sh = shmem;
  float *X_sh = shmem + blocksize * blocksize;

  uint32_t bid = blockIdx.x;

  load_A<blocksize>(n, A, A_sh);
  load_B<blocksize>(n, B + bid * blocksize * n, X_sh);

  __syncthreads();

  uint32_t tid = threadIdx.x;
  uint32_t col_idx = tid % blocksize;
  uint32_t row_idx = tid / blocksize;

  float *x_row = X_sh + row_idx * blocksize;

  blockSolve_warp<blocksize>(col_idx, n, A_sh, x_row);

  __syncthreads();

  store_X<blocksize>(n, X_sh, X + bid * blocksize * n);
}

template <uint32_t blocksize>
__global__ void blockSubtract_kernel(uint32_t n, uint32_t k, float const *A,
                                     float *X, float *B) {
  extern __shared__ float shmem[];
  float *A_sh = shmem;
  float *X_sh = shmem + blocksize * blocksize;

  uint32_t bid = blockIdx.x;

  load_A<blocksize>(n, A, A_sh);
  load_X<blocksize>(n, X + bid * blocksize * n, X_sh);

  __syncthreads();

  uint32_t tid = threadIdx.x;
  uint32_t col_idx = tid % blocksize;
  uint32_t row_idx = tid / blocksize;

  float *x_row = X_sh + row_idx * blocksize;
  float *b_row = B + bid * blocksize * n + row_idx * n;

  blockSubtract_warp<blocksize>(col_idx, n, A_sh, x_row, b_row);
}

template <uint32_t blocksize>
void buildTriangularSolverGraph(cudaGraph_t &graph, int num_blocks,
                                int n_stride, int k_stride, float const *d_A,
                                float *d_x, float *d_b) {

  std::vector<cudaGraphNode_t> solve_nodes(num_blocks);
  std::vector<std::vector<cudaGraphNode_t>> subtract_nodes(num_blocks - 1);

  for (int i = 0; i < num_blocks - 1; i++)
    subtract_nodes[i].resize(i + 1);

  dim3 gridDim(k_stride / blocksize);
  dim3 blockDim(blocksize * blocksize > 1024
                    ? 1024
                    : blocksize * blocksize); // NOTE: may be unnecessary

  for (int col = 0; col < num_blocks; ++col) {
    float const *sol_A = d_A + (col * blocksize * n_stride) + (col * blocksize);
    float *sol_x = d_x + (col * blocksize);
    float *sol_b = d_b + (col * blocksize);

    cudaKernelNodeParams solve_params = {0};
    solve_params.func = (void *)blockSolve_kernel<blocksize>;
    solve_params.gridDim = gridDim;
    solve_params.blockDim = blockDim;
    solve_params.sharedMemBytes = 2 * sizeof(float) * blocksize * blocksize;

    void *kernelArgs[5];
    kernelArgs[0] = &n_stride;
    kernelArgs[1] = &k_stride;
    kernelArgs[2] = &sol_A;
    kernelArgs[3] = &sol_x;
    kernelArgs[4] = &sol_b;
    solve_params.kernelParams = kernelArgs;
    solve_params.extra = NULL;

    cudaGraphAddKernelNode(&solve_nodes[col], graph, NULL, 0, &solve_params);

    for (int row = col + 1; row < num_blocks; ++row) {

      float const *sub_A =
          d_A + (row * blocksize * n_stride) + (col * blocksize);
      float *sub_x = d_x + (col * blocksize);
      float *sub_b = d_b + (row * blocksize);

      cudaKernelNodeParams sub_params = {0};
      sub_params.func = (void *)blockSubtract_kernel<blocksize>;
      sub_params.gridDim = gridDim;
      sub_params.blockDim = blockDim;
      sub_params.sharedMemBytes = 2 * sizeof(float) * blocksize * blocksize;

      void *kernelArgs[5];
      kernelArgs[0] = &n_stride;
      kernelArgs[1] = &k_stride;
      kernelArgs[2] = &sub_A;
      kernelArgs[3] = &sub_x;
      kernelArgs[4] = &sub_b;
      sub_params.kernelParams = kernelArgs;
      sub_params.extra = NULL;

      cudaGraphAddKernelNode(&subtract_nodes[row - 1][col], graph, NULL, 0,
                             &sub_params);

      cudaGraphAddDependencies(graph, &solve_nodes[col],
                               &subtract_nodes[row - 1][col], 1);
    }

    if (col > 0) {
      for (int row = 0; row < col; ++row) {
        cudaGraphAddDependencies(graph, &subtract_nodes[col - 1][row],
                                 &solve_nodes[col], 1);
      }
    }
  }
}

void trsmGraphLaunch(uint32_t n, uint32_t k, float const *A_d, float *b_d,
                     float *x_d, float *h_x_result) {

  constexpr uint32_t blocksize = 32;
  uint32_t numblocks = n / blocksize;

  cudaGraph_t graph;
  CUDA_CHECK(cudaGraphCreate(&graph, 0));

  buildTriangularSolverGraph<blocksize>(graph, numblocks, n, k, A_d, x_d, b_d);

  cudaGraphExec_t graphExec;
  CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  // Launch the graph
  CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Copy result (x_d) back to host
  CUDA_CHECK(cudaMemcpy(h_x_result, x_d, n * k * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Cleanup
  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("Custom Graph TRSM Time: %.3f ms\n", milliseconds);
}

void trsmCublasLaunch(cublasHandle_t handle, uint32_t n, uint32_t k, float *A_d,

                      float *b_d, float *h_x_result) {

  float alpha = 1.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  CUBLAS_CHECK(cublasStrsm(handle, CUBLAS_SIDE_LEFT,
                           CUBLAS_FILL_MODE_UPPER, // Read as Upper
                           CUBLAS_OP_T,            // Transpose it
                           CUBLAS_DIAG_NON_UNIT, n, k, &alpha, A_d, n, b_d, n));

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  CUDA_CHECK(cudaMemcpy(h_x_result, b_d, n * k * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("cuBLAS TRSM Time:      %.3f ms\n", milliseconds);
}

void flush_gpu() {
  void *flush_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&flush_gpu, 1024 * 1024 * 64));
  CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024 * 1024 * 64));
}

void verify_result(int N, int K, const std::vector<float> &X,
                   const std::vector<float> &X_true, const std::string &label) {
  float max_err = 0.0f;

  for (int col = 0; col < K; ++col)
    for (int i = 0; i < N; ++i)
      max_err =
          std::max(max_err, std::abs(X[col * N + i] - X_true[col * N + i]));

  std::cout << label << " max error = " << max_err << "\n";
}

int main() {
  const int BLOCK_SIZE = 32;
  const int NUM_BLOCKS = 32;
  const int N = BLOCK_SIZE * NUM_BLOCKS;
  const int K = N;

  std::cout << "--- TRSM Comparison (Graph vs. cuBLAS) ---\n";
  std::cout << "A Size: " << N << "x" << N << std::endl;
  std::cout << "X Size: " << N << "x" << K << std::endl;
  std::cout << "Block Size:  " << BLOCK_SIZE << std::endl;

  // 2. Host Setup
  std::vector<float> h_A(N * N);
  std::vector<float> h_B_original(N * K);
  std::vector<float> h_X_true(N * K);   // true solution
  std::vector<float> h_X_graph(N * K);  // graph result
  std::vector<float> h_X_cublas(N * K); // cublas result

  std::default_random_engine gen(42);
  std::uniform_real_distribution<float> dist(0.1f, 1.0f);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j <= i; ++j) {
      h_A[i * N + j] = dist(gen);
      if (i == j)
        h_A[i * N + j] += (float)N;
    }
  }

  for (int col = 0; col < K; ++col) {
    for (int i = 0; i < N; ++i) {
      h_X_true[col * N + i] = dist(gen);

      float sum = 0.0f;
      for (int j = 0; j <= i; ++j)
        sum += h_A[i * N + j] * h_X_true[col * N + j];

      h_B_original[col * N + i] = sum;
    }
  }

  float *d_A, *d_X_graph, *d_B_graph, *d_B_cublas;

  CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaMalloc(&d_X_graph, N * K * sizeof(float));
  cudaMalloc(&d_B_graph, N * K * sizeof(float));
  cudaMalloc(&d_B_cublas, N * K * sizeof(float));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  cudaMemcpy(d_B_graph, h_B_original.data(), N * K * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_B_cublas, h_B_original.data(), N * K * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemset(d_X_graph, 0, N * K * sizeof(float));

  trsmGraphLaunch(N, K, d_A, d_B_graph, d_X_graph, h_X_graph.data());
  verify_result(N, K, h_X_graph, h_X_true, "Graph");

  CUDA_CHECK(cudaMemcpy(d_B_cublas, h_B_original.data(), N * K * sizeof(float),
                        cudaMemcpyHostToDevice));

  trsmCublasLaunch(handle, N, K, d_A, d_B_cublas, h_X_cublas.data());
  verify_result(N, K, h_X_cublas, h_X_true, "cuBLAS");

  CUBLAS_CHECK(cublasDestroy(handle));
  cudaFree(d_A);
  cudaFree(d_X_graph);
  cudaFree(d_B_graph);
  cudaFree(d_B_cublas);

  return 0;
}
