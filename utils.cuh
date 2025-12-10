#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>
#include <stdio.h>

namespace utils {

// Macro to check CUDA errors
#define CUDA_CHECK(err) \
  if ((err) != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
  }

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUSPARSE_CHECK(err)                                                                        \
    do {                                                                                           \
        cusparseStatus_t err_ = (err);                                                             \
        if (err_ != CUSPARSE_STATUS_SUCCESS) {                                                     \
            printf("cusparse error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusparse error");                                            \
        }                                                                                          \
    } while (0)
template <typename T> __device__ T warp_prefix_sum(T val) {
  // Computes parallel prefix on 32 elements using Hillis Steele Scan w/ warp
  // shuffle
  const uint32_t thread_idx = threadIdx.x % 32;
  uint32_t idx = 1;
  #pragma unroll
  for (uint32_t step = 0; step < 5; ++step) { // log2(32) = 5
    // Load prefix from register
    T tmp = __shfl_up_sync(0xffffffff, val, idx);
    tmp = (thread_idx >= idx) ? tmp : 0; // Mask out

    // Update prefix in register
    val += tmp;

    // Multiply idx by 2
    idx <<= 1;
  }

  return val;
}

static cudaGraphExec_t instance = nullptr;
static uint32_t last_n = 0;
static uint32_t last_block_n = 0;

void launch_cuda_graph(
  void (*kernel_launcher)(const uint32_t n, float const *in, float *out, void *workspace),
  const uint32_t n, float const *in, float *out, void *workspace
) {
    // Invalidate graph if matrix size changes
    if (instance != nullptr && n != last_n) {
        cudaGraphExecDestroy(instance);
        instance = nullptr;
    }

    // If no graph exists, capture one
    if (instance == nullptr) {
        cudaGraph_t graph;
        
        // Start recording on the default stream
        cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal);

        // This code records nodes into the graph instead of launching
        kernel_launcher(n, in, out, workspace);

        // Stop recording
        cudaStreamEndCapture(0, &graph);

        // Create an executable graph from the recording
        // (This performs validation and sets up the launch structures)
        cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

        // Clean up the template graph (the Exec object owns the data now)
        cudaGraphDestroy(graph);
        
        last_n = n;
    }

    // Launch the cached graph
    // This issues all kernels in a single driver call
    cudaGraphLaunch(instance, 0);
}

// void set_cuda_graph_triblock(void (*kernel_launcher)(const uint32_t N, const uint32_t block_n, float const *in, float *out, void *workspace),
// const uint32_t N, const uint32_t block_n, float const *in, float *out, void *workspace,
// cudaGraph_t &graph, cudaGraphExec_t &
// )

void set_cuda_graph_triblock(
  void (*kernel_launcher)(const uint32_t N, const uint32_t block_n, float const *in, float *out, void *workspace),
  const uint32_t N, const uint32_t block_n, float const *in, float *out, void *workspace,
  cudaGraphExec_t *instance_ptr
) {
    // If no graph exists, capture one
    
    cudaGraph_t graph;
    // cudaGraphCreate(&graph, 0); // Removed redundant creation
    
    // Start recording on the default stream
    CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal));

    // This code records nodes into the graph instead of launching
    kernel_launcher(N, block_n, in, out, workspace);

    // Stop recording
    CUDA_CHECK(cudaStreamEndCapture(0, &graph));

    // Create an executable graph from the recording
    // (This performs validation and sets up the launch structures)
    CUDA_CHECK(cudaGraphInstantiate(instance_ptr, graph, nullptr, nullptr, 0));

    // Clean up the template graph (the Exec object owns the data now)
    CUDA_CHECK(cudaGraphDestroy(graph));
}

void launch_cuda_graph_triblock(cudaGraphExec_t *instance) {
  // Launch the cached graph
    // This issues all kernels in a single driver call
    cudaGraphLaunch(*instance, 0);
}

void invalidate_cuda_graph_triblock(cudaGraphExec_t *instance) {
  cudaGraphExecDestroy(*instance);
}


void launch_cuda_graph_triblock(
  void (*kernel_launcher)(const uint32_t N, const uint32_t block_n, float const *in, float *out, void *workspace),
  const uint32_t N, const uint32_t block_n, float const *in, float *out, void *workspace
) {
    // Invalidate graph if matrix size changes
    if (instance != nullptr && (N != last_n || block_n != last_block_n)) {
        cudaGraphExecDestroy(instance);
        instance = nullptr;
    }

    // If no graph exists, capture one
    if (instance == nullptr) {
        cudaGraph_t graph;
        
        // Start recording on the default stream
        cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal);

        // This code records nodes into the graph instead of launching
        kernel_launcher(N, block_n, in, out, workspace);

        // Stop recording
        cudaStreamEndCapture(0, &graph);

        // Create an executable graph from the recording
        // (This performs validation and sets up the launch structures)
        cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

        // Clean up the template graph (the Exec object owns the data now)
        cudaGraphDestroy(graph);
        
        last_n = N;
        last_block_n = block_n;
    }

    // Launch the cached graph
    // This issues all kernels in a single driver call
    cudaGraphLaunch(instance, 0);
}

} // namespace utils