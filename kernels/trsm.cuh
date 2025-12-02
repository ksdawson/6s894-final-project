#include <cuda_runtime.h>
#include <cstdint>

__device__ void forward_substitution(const uint32_t n, float const *A, float *x, float const *b);
__device__ void trsm(const uint32_t n, float const *A, float *X, float const *B);