#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace utils {
  
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

void cuda_check(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
      std::cerr << "CUDA error at " << file << ":" << line << ": "
                << cudaGetErrorString(code) << std::endl;
      exit(1);
  }
}

#define CUDA_CHECK(x) \
  do { \
      utils::cuda_check((x), __FILE__, __LINE__); \
  } while (0)
} // namespace utils
