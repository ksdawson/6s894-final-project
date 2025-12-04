// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh"]}
// TL {"workspace_files": []}

#pragma once
#include "utils.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
// GPU Naive Implementation
namespace cholesky_small {

size_t get_workspace_size(int32_t size) {
    return 0;
}

__device__ void cholesky(
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
            tmp = utils::warp_prefix_sum<float>(tmp);
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
        tmp = utils::warp_prefix_sum<float>(tmp);
        if (threadIdx.x == 31) {
            out[i * n + i] = sqrtf((in[i * n + i] - tmp));
        }
    }
}

__global__ void cholesky_gpu_naive(
    const uint32_t n, float const *in, float *out
) {
    cholesky(n, in, out);
}

////////////////////////////////////////////////////////////////////////////////
// Block methods

__device__ void block_cholesky(float const *A, float *L,
    const uint32_t A_n, const uint32_t L_n,
    const uint32_t r
) {
    if (threadIdx.x < 32) { // Ensure only one warp participates
        // Iterate over all rows
        for (uint32_t i = 0; i < r; ++i) {
            // Iterate over lower triangle off-diagonal cols
            for (uint32_t j = 0; j < i; ++j) {
                // Each thread computes a piece of the sum
                float tmp = 0.0f;
                for (uint32_t k = threadIdx.x; k < j; k += 32) {
                    tmp += L[i * L_n + k] * L[j * L_n + k];
                }
                // Combine the sum across the warp
                tmp = utils::warp_prefix_sum<float>(tmp);
                // Last thread handles writing it back
                if (threadIdx.x == 31) {
                    L[i * L_n + j] = (A[i * A_n + j] - tmp) / L[j * L_n + j];
                }
            }
            // Handle diagonal col
            float tmp = 0.0f;
            for (uint32_t k = threadIdx.x; k < i; k += 32) {
                tmp += L[i * L_n + k] * L[i * L_n + k];
            }
            tmp = utils::warp_prefix_sum<float>(tmp);
            if (threadIdx.x == 31) {
                L[i * L_n + i] = sqrtf((A[i * A_n + i] - tmp));
            }
        }
    }
}

__device__ void block_col_cholesky(float const *A, float *L,
    const uint32_t A_n, const uint32_t L_n,
    const uint32_t r
) {
    // Use local thread idx as this is done at the warp level
    const uint32_t thread_idx = threadIdx.x % 32;

    // Iterate over cols
    for (uint32_t j = 0; j < r; ++j) {
        // Compute diagonal
        float ajj = 0.0f;
        for (uint32_t k = thread_idx; k < j; k += 32) {
            const float ljk = L[j * L_n + k];
            ajj += ljk * ljk;
        }
        ajj = A[j * A_n + j] - utils::warp_prefix_sum<float>(ajj);
        const float ljj = sqrtf(ajj);

        // Compute off diagonals
        for (uint32_t i = j + 1 + threadIdx.x / 32; i < r; i += blockDim.x / 32) {
            float aij = 0.0f;
            for (uint32_t k = thread_idx; k < j; k += 32) {
                aij += L[i * L_n + k] * L[j * L_n + k];
            }
            aij = A[i * A_n + j] - utils::warp_prefix_sum<float>(aij);
            const float lij = aij / ljj;

            // Last thread handles writing lij back
            if (thread_idx == 31) {
                L[i * L_n + j] = lij;
            }
        }

        // Last thread handles writing ljj back
        if (threadIdx.x == blockDim.x - 1) {
            L[j * L_n + j] = ljj;
        }
        __syncthreads();
    }
}

void launch_cholesky(
    const uint32_t n, float const *in, float *out, void *workspace
) {
    // Cholesky only using 1 warp and parallelizing over the inner sum
    cholesky_gpu_naive<<<1, 1*32>>>(n, in, out);
}

}