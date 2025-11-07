#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// GPU Naive Block Implementation

__global__ void cholesky_gpu_naive(
    const uint32_t n, float const *in, float *out
) {
    return;
}

void launch_cholesky_gpu_naive(
    const uint32_t n, float const *in, float *out
) {
    // Divide the grid into blocks
    const uint32_t block_n = n / ;

    // Iterate over block cols launching a kernel for each step
    for (uint32_t k = 0; k < n / block_n; ++k) {
        // Move buffers
        const uint32_t offset = k * block_n * (n + 1);
        float *block_in = in + offset;
        float *block_out = out + offset;

        // Step 1: Cholesky factorize diagonal block
        cholesky_gpu_naive<<<1, 1*32>>>(block_n, in, out);

        // Step 2: TRSM all other blocks in the col

        // Step 3: SYRK all other blocks in i, j > k
    }
}