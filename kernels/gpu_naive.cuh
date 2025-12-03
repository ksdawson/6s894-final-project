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

void launch_cholesky_gpu_naive(
    const uint32_t n, float const *in, float *out
) {
    // Cholesky only using 1 warp and parallelizing over the inner sum
    cholesky_gpu_naive<<<1, 1*32>>>(n, in, out);
}

////////////////////////////////////////////////////////////////////////////////
// Block methods

__device__ void block_cholesky(float const *A, float *L,
    const uint32_t n, const uint32_t m
) {
    if (threadIdx.x < 32) { // Ensure only one warp participates
        // Iterate over all rows
        for (uint32_t i = 0; i < m; ++i) {
            // Iterate over lower triangle off-diagonal cols
            for (uint32_t j = 0; j < i; ++j) {
                // Each thread computes a piece of the sum
                float tmp = 0.0f;
                for (uint32_t k = threadIdx.x; k < j; k += 32) {
                    tmp += L[i * n + k] * L[j * n + k];
                }
                // Combine the sum across the warp
                tmp = utils::warp_prefix_sum<float>(tmp);
                // Last thread handles writing it back
                if (threadIdx.x == 31) {
                    L[i * n + j] = (A[i * m + j] - tmp) / L[j * n + j];
                }
            }
            // Handle diagonal col
            float tmp = 0.0f;
            for (uint32_t k = threadIdx.x; k < i; k += 32) {
                tmp += L[i * n + k] * L[i * n + k];
            }
            tmp = utils::warp_prefix_sum<float>(tmp);
            if (threadIdx.x == 31) {
                L[i * n + i] = sqrtf((A[i * m + i] - tmp));
            }
        }
    }
}