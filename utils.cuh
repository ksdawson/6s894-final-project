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

void launch_cuda_graph(
  void (*launch_block_cholesky)(const uint32_t n, float const *in, float *out, void *workspace),
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
        launch_block_cholesky(n, in, out, workspace);

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

} // namespace utils