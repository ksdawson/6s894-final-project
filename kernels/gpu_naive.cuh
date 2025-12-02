#include <cuda_runtime.h>
#include <cstdint>

__device__ void cholesky(const uint32_t n, float const *in, float *out);