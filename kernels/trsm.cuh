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
// Substitution methods

template <bool x_row, bool b_row>
__device__ void forward_substitution(const uint32_t n, float const *A, float *x,
                                     float const *b) {
  // Use local thread idx as this is done at the warp level
  const uint32_t thread_idx = threadIdx.x % 32;
  for (uint32_t i = 0; i < n; ++i) {
    // Each thread computes a piece of the sum
    float partial_sum = 0.0f;
    for (uint32_t j = thread_idx; j < i; j += 32) {
      if constexpr (x_row)
        partial_sum += A[i * n + j] * x[j];
      else
        partial_sum += A[i * n + j] * x[j * n];
    }
    // Combine the sum across the warp
    float sum = utils::warp_prefix_sum<float>(partial_sum);
    // Last thread handles writing it back
    if (thread_idx == 31) {
      float xi = 0.0f;
      if constexpr (b_row)
        xi = (b[i] - sum) / A[i * n + i];
      else
        xi = (b[i * n] - sum) / A[i * n + i];
      if constexpr (x_row)
        x[i] = xi;
      else
        x[i * n] = xi;
    }
    // All threads need this iteration to be done
    __syncwarp();
  }
}

template <bool x_row, bool b_row>
__device__ void block_forward_substitution(float const *A, float *x, float const *b,
  const uint32_t n, const uint32_t m
) {
  // Use local thread idx as this is done at the warp level
  const uint32_t thread_idx = threadIdx.x % 32;
  for (uint32_t i = 0; i < m; ++i) {
    // Each thread computes a piece of the sum
    float partial_sum = 0.0f;
    for (uint32_t j = thread_idx; j < i; j += 32) {
      if constexpr (x_row)
        partial_sum += A[i * n + j] * x[j];
      else
        partial_sum += A[i * n + j] * x[j * n];
    }
    // Combine the sum across the warp
    float sum = utils::warp_prefix_sum<float>(partial_sum);
    // Last thread handles writing it back
    if (thread_idx == 31) {
      float xi = 0.0f;
      if constexpr (b_row)
        xi = (b[i] - sum) / A[i * n + i];
      else
        xi = (b[i * m] - sum) / A[i * n + i];
      if constexpr (x_row)
        x[i] = xi;
      else
        x[i * n] = xi;
    }
    // All threads need this iteration to be done
    __syncwarp();
  }
}

__global__ void forward_substitution_kernel(uint32_t n, const float *A,
                                            float *x, const float *b) {
  forward_substitution<true, true>(n, A, x, b);
}

////////////////////////////////////////////////////////////////////////////////
// TRSM
// In Cholesky, we want to solve L_ik = A_ik * L_kk^-T, but inverse is
// expensive. Instead solve L_kk * L_ik^T = A_ik, which can be done with TRSM.
// TRSM uses forward substitution in each row. It is sequential in a row, but
// all rows are independent.

__device__ void trsm(const uint32_t n, float const *A, float *X,
                     float const *B) {
  // Assumes we're solving for X in A * X^T = B, so we can use rows of X instead
  // of cols
  // Get grid-level warp idx
  const uint32_t warps = blockDim.x / 32;
  const uint32_t warp_idx = warps * blockIdx.x + threadIdx.x / 32;

  // Iterate over rows of X,B with each row handled by one warp
  for (uint32_t i = warp_idx; i < n; i += warps * gridDim.x) {
    float *x = X + i * n;   // row
    float const *b = B + i; // col
    forward_substitution<true, false>(n, A, x, b);
  }
}

__device__ void block_trsm(float const *A, float *X, float const *B,
  const uint32_t n, const uint32_t m
) {
  const uint32_t warps = blockDim.x / 32;
  const uint32_t warp_idx = warps * blockIdx.x + threadIdx.x / 32;

  for (uint32_t i = warp_idx; i < m; i += warps * gridDim.x) {
    float *x = X + i * n;   // row
    // float const *b = B + i; // col
    // block_forward_substitution<true, false>(A, x, b, n, m);
    float const *b = B + i * m; // row
    block_forward_substitution<true, true>(A, x, b, n, m);
  }

  // Wait for everything to be done
  __syncthreads();
}

__global__ void trsm_kernel(uint32_t n, const float *A, float *X,
                            const float *B) {
  trsm(n, A, X, B);
}