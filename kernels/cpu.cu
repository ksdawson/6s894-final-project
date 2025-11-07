#include <cstdint>
#include <cstdio>
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)

void cholesky_cpu_naive(
    uint32_t size_i, uint32_t size_j,
    float const *in, float *out
) {
    // Iterate over all rows
    for (uint32_t i = 0; i < size_i; ++i) {
        // Iterate over lower triangle off-diagonal cols
        for (uint32_t j = 0; j < i; ++j) {
            float l_ij = in[i * size_j + j];
            for (uint32_t k = 0; k < j; ++k) {
                l_ij -= out[i * size_j + k] * out[j * size_j + k];
            }
            out[i * size_j + j] = l_ij / out[j * size_j + j];
        }
        // Handle diagonal col
        float l_ij = in[i * size_j + i];
        for (uint32_t k = 0; k < i; ++k) {
            l_ij -= out[i * size_j + k] * out[i * size_j + k];
        }
        out[i * size_j + i] = sqrtf(l_ij);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Test harness

void test_case_3x3() {
    // Test case
    const uint32_t size_i = 3;
    const uint32_t size_j = 3;

    // Allocate host memory
    float *in_cpu = static_cast<float*>(malloc(size_i * size_j * sizeof(float)));
    float *out_cpu = static_cast<float*>(malloc(size_i * size_j * sizeof(float)));

    // Fill in test data on host
    in_cpu[0] = 4.0f;
    in_cpu[1] = 12.0f;
    in_cpu[2] = -16.0f;
    in_cpu[3] = 12.0f;
    in_cpu[4] = 37.0f;
    in_cpu[5] = -43.0f;
    in_cpu[6] = -16.0f;
    in_cpu[7] = -43.0f;
    in_cpu[8] = 98.0f;

    // Run Cholesky decomposition
    cholesky_cpu_naive(size_i, size_j, in_cpu, out_cpu);

    // Verify output
    bool test_failed = false;
    // Verify upper triangle
    for (uint32_t i = 0; i < size_i; ++i) {
        for (uint32_t j = i + 1; j < size_j; ++j) {
            if (out_cpu[i * size_j + j] != 0.0f) {
                printf("Test 3x3 failed: upper triangle at (%u, %u) should be 0\n", i, j);
                test_failed = true;
                break;
            }
        }
    }
    // Verify lower triangle
    if (out_cpu[0] != 2.0f) {
        printf("Test 3x3 failed: lower triangle at (0, 0) should be 2\n");
        test_failed = true;
    } else if (out_cpu[3] != 6.0f) {
        printf("Test 3x3 failed: lower triangle at (1, 0) should be 6\n");
        test_failed = true;
    } else if (out_cpu[4] != 1.0f) {
        printf("Test 3x3 failed: lower triangle at (1, 1) should be 1\n");
        test_failed = true;
    } else if (out_cpu[6] != -8.0f) {
        printf("Test 3x3 failed: lower triangle at (2, 0) should be -8\n");
        test_failed = true;
    } else if (out_cpu[7] != 5.0f) {
        printf("Test 3x3 failed: lower triangle at (2, 1) should be 5\n");
        test_failed = true;
    } else if (out_cpu[8] != 3.0f) {
        printf("Test 3x3 failed: lower triangle at (2, 2) should be 3\n");
        test_failed = true;
    }

    if (!test_failed) {
        // Test passed
        printf("Test 3x3 passed\n");
    }

    // Clean up memory
    free(in_cpu);
    free(out_cpu);
}

int main(int argc, char **argv) {
    printf("Testing CPU naive\n");
    test_case_3x3();
}