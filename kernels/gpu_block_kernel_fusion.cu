#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// Device functions

struct BlockUpdate {
    const float *A; // input matrix
    float *L; // Chol matrix
    const uint32_t n; // matrix size
    const uint32_t m; // block size
    const uint32_t i; // Lik * Ljk^T
    const uint32_t j;
    float *out; // add result to out (likely register array)
};

template <uint32_t T_TH, uint32_t T_TW>
__device__ void block_gemm(BlockUpdate input, const uint32_t k) {

}

template <uint32_t T_TH, uint32_t T_TW>
__device__ void block_update(const float *A, float *L,
    const uint32_t n, const uint32_t m,
    const uint32_t i, const uint32_t j
) {
    // Accumulate update results in registers w/ each thread getting a subtile
    float out[T_TH * T_TW]; // initialized to zero

    // Sum Lik * Ljk^T
    BlockUpdate input = {A, L, n, m, i, j, out};
    for (uint32_t k = 0; k < j - 1; ++k) {
        block_gemm<T_TH, T_TW>(input, k);
    }

    // Move A to Aij 
    const float *Aij = A + i * m * n + j * m;

    // Move to subtile
    const uint32_t tile_i = threadIdx.x / (m / T_TW);
    const uint32_t tile_j = threadIdx.x % (m / T_TW);
    const float *Aij_subtile = Aij + tile_i * T_TH * n + tile_j * T_TW;

    // Compute Aij - sum
    #pragma unroll
    for (const uint32_t ti; ti < T_TH; ++ti) {
        // Vectorize
        const float4 *a4 = reinterpret_cast<const float4*>(Aij_subtile + ti * n);
        float4 *out4 = reinterpret_cast<float4*>(out + ti * T_TW);
        #pragma unroll
        for (const uint32_t tj; tj < T_TW / 4; ++tj) {
            out4[tj] = a4[tj] - out4[tj];
        }
    }
}

template <uint32_t T_TH, uint32_t T_TW>
__global__ void block_kernel(const float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, const uint32_t m, // matrix size, block size
    const uint32_t j // block col
) {
    // Each SM gets a block
    for (uint32_t i = j + 1 + blockIdx.x; i < n / m; ++i) {
        // Update
        block_update<T_TH, T_TW>(A, L, n, m, i, j);

        // TODO: TRSM
    }
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
    constexpr uint32_t m = 64;

    // Iterate over block cols launching a kernel for each step
    for (uint32_t j = 0; j < n / m; ++j) {
        // Step 1: Chol(update) diagonal block
        block_kernel<4, 4><<<1, 8*32>>>(in, out, n, m, j);

        // Step 2: Trsm(update) all other blocks
        block_kernel<4, 4><<<48, 8*32>>>(in, out, n, m, j);
    }
}