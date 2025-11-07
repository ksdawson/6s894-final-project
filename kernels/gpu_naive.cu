#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

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

////////////////////////////////////////////////////////////////////////////////
// GPU Naive Implementation

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

void launch_cholesky_gpu_naive(
    const uint32_t n, float const *in, float *out
) {
    // Cholesky only using 1 warp and parallelizing over the inner sum
    cholesky_gpu_naive<<<1, 1*32>>>(n, in, out);
}

////////////////////////////////////////////////////////////////////////////////
// Test harness

void test_case_3x3() {
    // Test case
    constexpr uint32_t n = 3;

    // Allocate device memory
    float *in_gpu;
    float *out_gpu;
    CUDA_CHECK(cudaMalloc(&in_gpu, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_gpu, n * n * sizeof(float)));

    // Test data on host
    float cpu[n*n] = {
        4.0f, 12.0f, -16.0f,
        12.0f, 37.0f, -43.0f,
        -16.0f, -43.0f, 98.0f
    };

    // Copy from host to device
    CUDA_CHECK(cudaMemcpy(in_gpu, cpu, n * n * sizeof(float), cudaMemcpyHostToDevice));

    // Run Cholesky decomposition
    launch_cholesky_gpu_naive(n, in_gpu, out_gpu);

    // Verify output
    CUDA_CHECK(cudaMemcpy(cpu, out_gpu, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    bool test_failed = false;
    // Verify upper triangle
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = i + 1; j < n; ++j) {
            if (cpu[i * n + j] != 0.0f) {
                printf("Test 3x3 failed: upper triangle at (%u, %u) should be 0\n", i, j);
                test_failed = true;
                break;
            }
        }
    }
    // Verify lower triangle
    if (cpu[0] != 2.0f) {
        printf("Test 3x3 failed: lower triangle at (0, 0) should be 2\n");
        test_failed = true;
    } else if (cpu[3] != 6.0f) {
        printf("Test 3x3 failed: lower triangle at (1, 0) should be 6\n");
        test_failed = true;
    } else if (cpu[4] != 1.0f) {
        printf("Test 3x3 failed: lower triangle at (1, 1) should be 1\n");
        test_failed = true;
    } else if (cpu[6] != -8.0f) {
        printf("Test 3x3 failed: lower triangle at (2, 0) should be -8\n");
        test_failed = true;
    } else if (cpu[7] != 5.0f) {
        printf("Test 3x3 failed: lower triangle at (2, 1) should be 5\n");
        test_failed = true;
    } else if (cpu[8] != 3.0f) {
        printf("Test 3x3 failed: lower triangle at (2, 2) should be 3\n");
        test_failed = true;
    }

    if (!test_failed) {
        // Test passed
        printf("Test 3x3 passed\n");
    }

    // Clean up memory
    CUDA_CHECK(cudaFree(in_gpu));
    CUDA_CHECK(cudaFree(out_gpu));
}

// // Generate a random SPD matrix of size N x N
// void generate_spd_matrix(uint32_t N, float* A) {
//     float* L = (float*)malloc(N * N * sizeof(float));

//     // Fill L lower-triangular with random positive numbers
//     for (uint32_t i = 0; i < N; ++i) {
//         for (uint32_t j = 0; j <= i; ++j) {
//             L[i*N + j] = (float)(rand() % 10 + 1);
//         }
//         for (uint32_t j = i+1; j < N; ++j) {
//             L[i*N + j] = 0.0f;
//         }
//     }

//     // Compute A = L * L^T
//     for (uint32_t i = 0; i < N; ++i) {
//         for (uint32_t j = 0; j < N; ++j) {
//             float sum = 0.0f;
//             for (uint32_t k = 0; k <= (i<j?i:j); ++k) {
//                 sum += L[i*N + k] * L[j*N + k];
//             }
//             A[i*N + j] = sum;
//         }
//     }

//     free(L);
// }

// // Test case for any size
// void test_case(uint32_t N) {
//     printf("Testing Cholesky %ux%u\n", N, N);

//     float *in_gpu  = (float*)malloc(N * N * sizeof(float));
//     float *out_gpu = (float*)malloc(N * N * sizeof(float));

//     CUDA_CHECK(cudaMalloc(&in_gpu, N * N * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&out_gpu, N * N * sizeof(float)));

//     // Generate SPD input
//     generate_spd_matrix(N, in_gpu);

//     // Copy input to device (if needed)
//     CUDA_CHECK(cudaMemcpy(in_gpu, in_gpu, N * N * sizeof(float), cudaMemcpyHostToDevice));

//     // Run Cholesky (gpu version here, replace with GPU kernel if desired)
//     launch_cholesky_gpu_naive(N, in_gpu, out_gpu);

//     // Verify: L * L^T ≈ original matrix
//     bool test_failed = false;
//     float tol = 1e-5f;

//     for (uint32_t i = 0; i < N; ++i) {
//         for (uint32_t j = 0; j < N; ++j) {
//             float sum = 0.0f;
//             for (uint32_t k = 0; k <= (i<j?i:j); ++k) {
//                 sum += out_gpu[i*N + k] * out_gpu[j*N + k];
//             }
//             if (fabsf(sum - in_gpu[i*N + j]) > tol) {
//                 printf("Mismatch at (%u,%u): computed %f, expected %f\n", i, j, sum, in_gpu[i*N + j]);
//                 test_failed = true;
//             }
//         }
//     }

//     if (!test_failed) {
//         printf("Test %ux%u passed\n", N, N);
//     } else {
//         printf("Test %ux%u FAILED\n", N, N);
//     }

//     free(in_gpu);
//     free(out_gpu);
//     CUDA_CHECK(cudaFree(in_gpu));
//     CUDA_CHECK(cudaFree(out_gpu));
// }

int main(int argc, char **argv) {
    printf("Testing GPU naive\n");
    test_case_3x3();
}