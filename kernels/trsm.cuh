#include <cuda_runtime.h>
#include <cstdint>

__device__ void block_trsm(float const *A, float *X, float const *B, const uint32_t n, const uint32_t m);