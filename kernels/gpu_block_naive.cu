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

// Block algorithm:
// (0) Divide square A into square blocks
// (1) Iterate over blocks columns of A
// (2) Start at the diagonal block A_kk
// (3) Solve A_kk := L_kk = Chol(A_kk)
// (4) Iterate over the rest of the block col
// (5) Solve A_ik := L_ik = A_ik * L_kk^-T (use TRSM)
// (6) Iterate over the rest of the block cols, rows
// (6) Solve A_ij := A_ij - L_i(k-1) * L_i(k-1)^T

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