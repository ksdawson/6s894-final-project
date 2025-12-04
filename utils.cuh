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
} // namespace utils
