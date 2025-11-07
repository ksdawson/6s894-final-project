#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// GPU Naive Block Implementation

__global__ void cholesky_gpu_naive(
    uint32_t size_i, uint32_t size_j,
    float const *in, float *out
) {
    return;
}

void launch_cholesky_gpu_naive(
    uint32_t size_i, uint32_t size_j,
    float const *in, float *out
) {
    cholesky_gpu_naive<<<48, 32*32>>>(size_i, size_j, in, out);
}

////////////////////////////////////////////////////////////////////////////////
// Test harness

void test_case_3x3() {
    // Test case
    const uint32_t size_i = 3;
    const uint32_t size_j = 3;

    // Allocate device memory
    float *in_gpu;
    float *out_gpu
    CUDA_CHECK(cudaMalloc(&in_gpu, size_i * size_j * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_gpu, size_i * size_j * sizeof(float)));

    // Fill in test data on host
    in_gpu[0] = 4.0f;
    in_gpu[1] = 12.0f;
    in_gpu[2] = -16.0f;
    in_gpu[3] = 12.0f;
    in_gpu[4] = 37.0f;
    in_gpu[5] = -43.0f;
    in_gpu[6] = -16.0f;
    in_gpu[7] = -43.0f;
    in_gpu[8] = 98.0f;

    // Copy from host to device
    CUDA_CHECK(cudaMemcpy(in_gpu, in_gpu, size_i * size_j * sizeof(float), cudaMemcpyHostToDevice));

    // Run Cholesky decomposition
    launch_cholesky_cpu_naive(size_i, size_j, in_cpu, out_cpu);

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

// Generate a random SPD matrix of size N x N
void generate_spd_matrix(uint32_t N, float* A) {
    float* L = (float*)malloc(N * N * sizeof(float));

    // Fill L lower-triangular with random positive numbers
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            L[i*N + j] = (float)(rand() % 10 + 1);
        }
        for (uint32_t j = i+1; j < N; ++j) {
            L[i*N + j] = 0.0f;
        }
    }

    // Compute A = L * L^T
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k <= (i<j?i:j); ++k) {
                sum += L[i*N + k] * L[j*N + k];
            }
            A[i*N + j] = sum;
        }
    }

    free(L);
}

// Test case for any size
void test_case(uint32_t N) {
    printf("Testing Cholesky %ux%u\n", N, N);

    float *in_cpu  = (float*)malloc(N * N * sizeof(float));
    float *out_cpu = (float*)malloc(N * N * sizeof(float));

    float *in_gpu, *out_gpu;
    CUDA_CHECK(cudaMalloc(&in_gpu, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_gpu, N * N * sizeof(float)));

    // Generate SPD input
    generate_spd_matrix(N, in_cpu);

    // Copy input to device (if needed)
    CUDA_CHECK(cudaMemcpy(in_gpu, in_cpu, N * N * sizeof(float), cudaMemcpyHostToDevice));

    // Run Cholesky (CPU version here, replace with GPU kernel if desired)
    cholesky_cpu_naive(N, N, in_cpu, out_cpu);

    // Verify: L * L^T ≈ original matrix
    bool test_failed = false;
    float tol = 1e-5f;

    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k <= (i<j?i:j); ++k) {
                sum += out_cpu[i*N + k] * out_cpu[j*N + k];
            }
            if (fabsf(sum - in_cpu[i*N + j]) > tol) {
                printf("Mismatch at (%u,%u): computed %f, expected %f\n", i, j, sum, in_cpu[i*N + j]);
                test_failed = true;
            }
        }
    }

    if (!test_failed) {
        printf("Test %ux%u passed\n", N, N);
    } else {
        printf("Test %ux%u FAILED\n", N, N);
    }

    free(in_cpu);
    free(out_cpu);
    CUDA_CHECK(cudaFree(in_gpu));
    CUDA_CHECK(cudaFree(out_gpu));
}

int main(int argc, char **argv) {
    printf("Testing GPU naive\n");
    test_case_3x3();
}