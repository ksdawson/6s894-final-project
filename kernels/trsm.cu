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
// Substitution methods

__device__ void forward_substitution(
    const uint32_t n, float const *A, float *x, float const *b
) {
    for (uint32_t i = 0; i < n; ++i) {
        // Each thread computes a piece of the sum
        float partial_sum = 0.0f;
        for (uint32_t j = threadIdx.x; j < i; j += 32) {
            partial_sum += A[i * n + j] * x[j];
        }
        // Combine the sum across the warp
        float sum = warp_prefix_sum<float>(partial_sum);
        // Last thread handles writing it back
        if (threadIdx.x == 31) {
            x[i] = (b[i] - sum) / A[i * n + i];
        }
        // All threads need this iteration to be done
        __syncwarp();
    }
}

__global__ void forward_substitution_kernel(uint32_t n, const float* A, float* x, const float* b) {
    forward_substitution(n, A, x, b);
}

////////////////////////////////////////////////////////////////////////////////
// TRSM

// TRSM algorithm:
// We want to solve L_ik = A_ik * L_kk^-T, but inverse is expensive
// Instead solve L_kk * L_ik^T = A_ik, which can be done row-by-row as
// L_kk * L_ik_x^T = A_ik_x^T, which is TRSM -> forward substitution

////////////////////////////////////////////////////////////////////////////////
// Test harness

void generate_lower_triangular(uint32_t N, float* A) {
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            if (j <= i) A[i*N + j] = (float)(rand() % 9 + 1);  // positive
            else A[i*N + j] = 0.0f;
        }
    }
}

void test_forward_substitution(uint32_t N) {
    printf("Testing forward substitution with N=%u\n", N);

    float *A = (float*)malloc(N * N * sizeof(float));
    float *x_true = (float*)malloc(N * sizeof(float));
    float *b = (float*)malloc(N * sizeof(float));
    float *x_gpu = (float*)malloc(N * sizeof(float));

    generate_lower_triangular(N, A);

    // Generate random true solution
    for (uint32_t i = 0; i < N; ++i)
        x_true[i] = (float)(rand() % 10 + 1);

    // Compute b = A * x_true
    for (uint32_t i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (uint32_t j = 0; j <= i; ++j)
            sum += A[i*N + j] * x_true[j];
        b[i] = sum;
    }

    // Allocate device memory
    float *A_d, *b_d, *x_d;
    CUDA_CHECK(cudaMalloc(&A_d, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_d, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&x_d, N*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(A_d, A, N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(x_d, 0, N*sizeof(float)));

    // Launch with one warp (since function assumes warp-level sum)
    forward_substitution_kernel<<<1, 32>>>(N, A_d, x_d, b_d);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(x_gpu, x_d, N*sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    bool failed = false;
    float tol = 1e-3f;
    for (uint32_t i = 0; i < N; ++i) {
        if (fabsf(x_gpu[i] - x_true[i]) > tol) {
            printf("Mismatch at %u: got %.5f, expected %.5f\n", i, x_gpu[i], x_true[i]);
            failed = true;
        }
    }

    if (!failed)
        printf("Test PASSED for N=%u\n", N);
    else
        printf("Test FAILED for N=%u\n", N);

    free(A);
    free(x_true);
    free(b);
    free(x_gpu);
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(x_d));
}

int main() {
    srand(0);
    test_forward_substitution(4);
    test_forward_substitution(8);
    test_forward_substitution(16);
    test_forward_substitution(1024);
    return 0;
}