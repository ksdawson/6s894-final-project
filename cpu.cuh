#pragma once
#include <cstdint>
#include <cstdio>
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)

void cholesky_cpu_naive(
    const uint32_t n, float const *in, float *out
) {
    // Iterate over all rows
    for (uint32_t i = 0; i < n; ++i) {
        // Iterate over lower triangle off-diagonal cols
        for (uint32_t j = 0; j < i; ++j) {
            float l_ij = in[i * n + j];
            for (uint32_t k = 0; k < j; ++k) {
                l_ij -= out[i * n + k] * out[j * n + k];
            }
            out[i * n + j] = l_ij / out[j * n + j];
        }
        // Handle diagonal col
        float l_ij = in[i * n + i];
        for (uint32_t k = 0; k < i; ++k) {
            l_ij -= out[i * n + k] * out[i * n + k];
        }
        out[i * n + i] = sqrtf(l_ij);
    }
}