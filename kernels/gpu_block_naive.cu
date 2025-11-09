#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>

// Macro to check CUDA errors
#define CUDA_CHECK(err) \
    if ((err) != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// Helpers
template <typename T>
__device__ T warp_prefix_sum(T val) {
    // Computes parallel prefix on 32 elements using Hillis Steele Scan w/ warp shuffle
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

template <bool x_row, bool b_row>
__device__ void forward_substitution(
    const uint32_t n, float const *A, float *x, float const *b
) {
    // Use local thread idx as this is done at the warp level
    const uint32_t thread_idx = threadIdx.x % 32;
    for (uint32_t i = 0; i < n; ++i) {
        // Each thread computes a piece of the sum
        float partial_sum = 0.0f;
        for (uint32_t j = thread_idx; j < i; j += 32) {
            if constexpr (x_row) partial_sum += A[i * n + j] * x[j];
            else partial_sum += A[i * n + j] * x[j * n];
        }
        // Combine the sum across the warp
        float sum = warp_prefix_sum<float>(partial_sum);
        // Last thread handles writing it back
        if (thread_idx == 31) {
            float xi = 0.0f;
            if constexpr (b_row) xi = (b[i] - sum) / A[i * n + i];
            else xi = (b[i * n] - sum) / A[i * n + i];
            if constexpr (x_row) x[i] = xi;
            else x[i * n] = xi;
        }
        // All threads need this iteration to be done
        __syncwarp();
    }
}
__device__ void trsm(
    const uint32_t n, float const *A, float *X, float const *B
) {
    // Assumes we're solving for X in A * X^T = B, so we can use rows of X instead of cols
    // Get block-level warp idx
    const uint32_t warps = blockDim.x / 32;
    const uint32_t warp_idx = threadIdx.x / 32;

    // Iterate over rows of X,B with each row handled by one warp
    for (uint32_t i = warp_idx; i < n; i += warps) {
        float *x = X + i * n; // row
        float const *b = B + i; // col
        forward_substitution<true, false>(n, A, x, b);
    }
}

////////////////////////////////////////////////////////////////////////////////
// GPU Naive Block Implementation

__global__ void cholesky_gpu_naive(
    const uint32_t n, float const *in, float *out
) {
    // Iterate over all rows
    for (uint32_t i = 0; i < n; ++i) {
        // Iterate over lower triangle off-diagonal cols
        for (uint32_t j = 0; j < i; ++j) {
            // Each thread computes a piece of the sum
            float tmp = 0.0f;
            for (uint32_t k = threadIdx.x; k < j; k += 32) {
                tmp += out[i * n + k] * out[j * n + k];
            }
            // Combine the sum across the warp
            tmp = warp_prefix_sum<float>(tmp);
            // Last thread handles writing it back
            if (threadIdx.x == 31) {
                out[i * n + j] = (in[i * n + j] - tmp) / out[j * n + j];
            }
        }
        // Handle diagonal col
        float tmp = 0.0f;
        for (uint32_t k = threadIdx.x; k < i; k += 32) {
            tmp += out[i * n + k] * out[i * n + k];
        }
        tmp = warp_prefix_sum<float>(tmp);
        if (threadIdx.x == 31) {
            out[i * n + i] = sqrtf((in[i * n + i] - tmp));
        }
    }
}

__global__ void cholesky_trsm(
    const uint32_t n, const uint32_t num_blocks, const uint32_t block_n,
    float const *A, float *L
) {
    // Iterate over blocks in block col
    float *L_kk = L;
    L += block_n * n;
    for (uint32_t i = blockIdx.x; i < num_blocks; i += gridDim.x) {
        // Move buffers
        const uint32_t offset = i * block_n * n;
        float const *A_ik = A + offset;
        float *L_ik = L + offset;

        // Each block is handled by an SM
        trsm(block_n, A_ik, L_ik, L_kk);
    }
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
    const uint32_t num_blocks = n / block_n;

    // Iterate over block cols launching a kernel for each step
    for (uint32_t k = 0; k < num_blocks; ++k) {
        // Move buffers
        const uint32_t offset = k * block_n * (n + 1);
        float *block_in = in + offset;
        float *block_out = out + offset;

        // Step 1: Cholesky factorize diagonal block
        cholesky_gpu_naive<<<1, 1*32>>>(block_n, block_in, block_out);

        // Step 2: TRSM all other blocks in the col
        block_in += block_n * n;
        block_out += block_n * n;
        cholesky_trsm<<<48, 32*32>>>(n, num_blocks - k, block_n, block_in, block_out);

        // Step 3: SYRK all other blocks in i, j > k
    }
}