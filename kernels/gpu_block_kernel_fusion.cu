#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// Device functions

struct BlockUpdate {
    float const *A; // input matrix
    float *L; // Chol matrix
    const uint32_t n; // matrix size
    const uint32_t m; // block size
    const uint32_t i; // Lik * Ljk^T
    const uint32_t j;
    float *out; // add result to out (likely register array)
};

__device__ void block_gemm(BlockUpdate input, const uint32_t k) {

}

__device__ void block_update(BlockUpdate input) {
    // Sum Lik * Ljk^T
    for (uint32_t k = 0; k < input.j - 1; ++k) {
        block_gemm(input, k);
    }

    // TODO: Compute Aij - sum
}

__global__ void block_kernel() {
    // TODO: Update

    // TODO: TRSM
}

__global__ void chol_kernel() {
    // TODO: Update

    // TODO: Chol
}

////////////////////////////////////////////////////////////////////////////////
// Host functions

void launch_block_cholesky(
    const uint32_t n, float const *in, float *out
) {
    // Divide the grid into blocks
    constexpr uint32_t block_n = 128;

    // Iterate over block cols launching a kernel for each step
    for (uint32_t k = 0; k < n / block_n; ++k) {
        // Step 1: Chol(update) diagonal block

        // Step 2: Trsm(update) all other blocks
    }
}