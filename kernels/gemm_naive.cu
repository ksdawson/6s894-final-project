// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh"]}
// TL {"workspace_files": []}

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "utils.cuh"

// Macro to check CUDA errors
#define CUDA_CHECK(x) \
  do { \
      utils::cuda_check((x), __FILE__, __LINE__); \
  } while (0)

////////////////////////////////////////////////////////////////////////////////
// current naive gemm can only handle 32x32 matrix
template <uint32_t A_col_offset, uint32_t A_row_offset, uint32_t B_col_offset, uint32_t B_row_offset>
__global__ void gemm_kernel_XY(
    const uint32_t n, float const *A, float const *B, float *out
) {
    int32_t col_ID = threadIdx.x % n;
    int32_t row_ID = threadIdx.x / n;

    float sum = 0;
    //A[(A_col_offset + row_ID) * n + (A_row_offset + col_ID)];
    for (uint32_t i = 0; i < n; ++i) {
        sum += B[(B_col_offset + row_ID) * n + (B_row_offset + i)] * B[(B_col_offset + col_ID) * n + (B_row_offset + i)];
    }

    if (row_ID < n && col_ID < n) {
        out[row_ID * n + col_ID] = A[(A_col_offset + row_ID) * n + (A_row_offset + col_ID)] -sum;
    }
}

struct BlockUpdate {
    float const *A; // input matrix
    float const *L; // Chol matrix
    const uint32_t n; // matrix size
    const uint32_t m; // block size
    const uint32_t i; // Lik * Ljk^T
    const uint32_t j;
    float *out; // add result to out (likely register array)
};

__device__ int32_t index (int32_t row_ID, int32_t col_ID, int32_t offset_col, int32_t offset_row, int32_t dim) {
    return (offset_col + row_ID) * dim + (offset_row + col_ID);
}

template <int32_t T_TH, int32_t T_TW>
__device__ void block_gemm (BlockUpdate gemm_info, const uint32_t k, float *shared_mem) {

    int32_t N = gemm_info.n;
    int32_t block_n = gemm_info.m;
    float const *B1 = gemm_info.L;
    float const *B2 = gemm_info.L;
    float *out = gemm_info.out;
    int32_t B1_row_offset = k * block_n;
    int32_t B1_col_offset = gemm_info.i * block_n;
    int32_t B2_row_offset = k * block_n;
    int32_t B2_col_offset = gemm_info.j * block_n;

    for (int32_t i = 0; i < block_n * block_n; i += blockDim.x) {
        int32_t row_ID = (int32_t)((i + threadIdx.x) / block_n);
        int32_t col_ID = int32_t((i + threadIdx.x) % block_n);
        shared_mem[threadIdx.x + i] = B1[index(row_ID, col_ID, B1_col_offset, B1_row_offset, N)];
        shared_mem[threadIdx.x + i + block_n * block_n] = B2[index(row_ID, col_ID, B2_col_offset, B2_row_offset, N)];
    }
    __syncthreads();

    int32_t row_ID = (int32_t)(threadIdx.x / 32) * T_TH;
    int32_t col_ID = int32_t(threadIdx.x % 32) * T_TW;

    float sum[T_TH * T_TW];
    float b1_val[T_TH];
    float b2_val[T_TW];
    #pragma unroll
    for (int32_t i = 0; i < T_TH * T_TW; ++i) {
        sum[i] = 0.0f;
    }
    #pragma unroll
    for (int32_t i = 0; i < T_TH; ++i) {
        b1_val[i] = 0.0f;
    }
    #pragma unroll
    for (int32_t i = 0; i < T_TW; ++i) {
        b2_val[i] = 0.0f;
    }

    for (int32_t i = 0; i < block_n; ++i) {
        // computing B1 * B2^T
        for (int32_t tile_row = 0; tile_row < T_TH; ++tile_row) {
            b1_val[tile_row] = shared_mem[index(row_ID + tile_row, i, 0, 0, block_n)];
        }
        for (int32_t tile_col = 0; tile_col < T_TW; ++tile_col) {
            b2_val[tile_col] = shared_mem[index(col_ID + tile_col, i, 0, 0, block_n) + block_n * block_n];
        }

        for (int32_t tile_row = 0; tile_row < T_TH; ++tile_row) {
            for (int32_t tile_col = 0; tile_col < T_TW; ++tile_col) {
                sum[tile_row * T_TW + tile_col] += b1_val[tile_row] * b2_val[tile_col];
            }
        }


    }

    for (int32_t tile_row = 0; tile_row < T_TH; ++tile_row) {
        for (int32_t tile_col = 0; tile_col < T_TW; ++tile_col) {
            out[tile_row * T_TW + tile_col] += sum[tile_row * T_TW + tile_col];
        }
    }
    __syncthreads();
}

template <int32_t T_TH, int32_t T_TW>
__global__ void gemm_kernel(
    const uint32_t n, float const *A, float const *L, float *out_global,
    const uint32_t m, const uint32_t i, const uint32_t j,
    const uint32_t k
) {
    extern __shared__ float shared_mem[];
    float out[T_TH * T_TW];
    for (int32_t i = 0; i < T_TH * T_TW; ++i) {
        out[i] = 0.0f;
    }
    BlockUpdate gemm_info = {A, L, n, m, i, j, out};

    block_gemm<T_TH, T_TW>(gemm_info, k, shared_mem);

    int32_t row_ID = (int32_t)(threadIdx.x / 32) * T_TH;
    int32_t col_ID = int32_t(threadIdx.x % 32) * T_TW;

    for (int32_t tile_row = 0; tile_row < T_TH; ++tile_row) {
        for (int32_t tile_col = 0; tile_col < T_TW; ++tile_col) {
            out_global[(row_ID + tile_row) * m + (col_ID + tile_col)] = out[tile_row * T_TW + tile_col];
        }
    }
    __syncthreads();

}

template <int32_t T_TH, int32_t T_TW>
void launch_gemm_naive(
    const uint32_t n, float const *A, float const *B, float *out,
    const uint32_t m, const uint32_t i, const uint32_t j,
    const uint32_t k
) {
    int32_t shared_mem_size = 2 * m * m * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(
        gemm_kernel<T_TH, T_TW>,  // VEC_SIZE=4
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size));
    gemm_kernel<T_TH, T_TW><<<1, 32*32, shared_mem_size>>>(n, A, B, out, m, i, j, k);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
}



////////////////////////////////////////////////////////////////////////////////
// Test harness



void generate_lower_triangular(uint32_t N, float *A) {
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            if (j <= i) {
                A[i * N + j] = (float)(rand() % 9 + 1); // positive
            } else {
                A[i * N + j] = 0.0f;
            }
        }
    }
}

void generate_random_matrix(uint32_t N, float *A) {
    for (uint32_t i = 0; i < N * N; ++i) {
        A[i] = (float)(rand() % 10 + 1);
    }
}

void test_gemm_block(uint32_t N, uint32_t n) {
    printf("Testing gemm with N=%u\n", N);

    float *L = (float *)malloc(N * N * sizeof(float));
    float *B1 = (float *)malloc(n * n * sizeof(float));
    float *B2 = (float *)malloc(n * n * sizeof(float));
    float *A = (float *)malloc(N * N * sizeof(float));
    float *X_true = (float *)malloc(n * n * sizeof(float));
    float *X_gpu = (float *)malloc(n * n * sizeof(float));

    generate_random_matrix(N, A);
    generate_random_matrix(N, L);

    uint32_t i = 1;
    uint32_t j = 1;
    uint32_t k = 2;

    for (uint32_t row = 0; row < n; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            B1[row * n + col] = L[(row + i * n) * N + (col + k * n)];
            B2[row * n + col] = L[(row + j * n) * N + (col + k * n)];
            //printf("B1[%u, %u] = %f, B2[%u, %u] = %f\n", row, col, B1[row * n + col], row, col, B2[row * n + col]);
        }
    }

    for (uint32_t row = 0; row < n; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            X_true[row * n + col] = 0;
            for (uint32_t ind = 0; ind < n; ++ind) {
                X_true[row * n + col] += B1[row * n + ind] * B2[col * n + ind];
            }
            
        }
    }

    // Allocate device memory
    float *A_d, *L_d, *X_d;
    CUDA_CHECK(cudaMalloc(&L_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&A_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&X_d, n * n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(L_d, L, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(X_d, 0, n * n * sizeof(float)));

    // Launch kernel (1 block, multiple warps)
    launch_gemm_naive<1, 1>(
        N, A_d, L_d, X_d,
        n, i, j, k);


    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(X_gpu, X_d, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    bool failed = false;
    float tol = 1e-3f;
    for (uint32_t i = 0; i < n * n; ++i) {
        if (fabsf(X_gpu[i] - X_true[i]) < tol) {
            continue;
        } else {
            printf("Mismatch at (%u): got %.5f, expected %.5f\n", i, X_gpu[i],
                    X_true[i]);
            failed = true;
        }
    }

    // for (uint32_t i = 0; i < N * N; ++i) {
    //     printf("X_gpu[%u] = %f, X_true[%u] = %f\n", i, X_gpu[i], i, X_true[i]);
    // }

    if (!failed) {
        printf("Test PASSED for N=%u\n", N);
    } else {
        printf("Test FAILED for N=%u\n", N);
    }

    free(A);
    free(L);
    free(B1);
    free(B2);
    free(X_true);
    free(X_gpu);
    CUDA_CHECK(cudaFree(L_d));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(X_d));
}

void test_gemm(uint32_t N) {
    printf("Testing gemm with N=%u\n", N);

    float *L = (float *)malloc(N * N * sizeof(float));
    float *A = (float *)malloc(N * N * sizeof(float));
    float *X_true = (float *)malloc(N * N * sizeof(float));
    float *X_gpu = (float *)malloc(N * N * sizeof(float));

    generate_lower_triangular(N, L);


    // Generate random A matrix
    for (uint32_t i = 0; i < N * N; ++i) {
        A[i] = (float)(rand() % 10 + 1);
    }

    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            X_true[i * N + j] = A[i * N + j];
            for (uint32_t k = 0; k < N; ++k) {
                X_true[i * N + j] -= L[i * N + k] * L[j * N + k];
            }
        }
    }

    // Allocate device memory
    float *A_d, *L_d, *X_d;
    CUDA_CHECK(cudaMalloc(&L_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&A_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&X_d, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(L_d, L, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(X_d, 0, N * N * sizeof(float)));

    // Launch kernel (1 block, multiple warps)
    gemm_kernel_XY<0, 0, 0, 0><<<1, 32 * 32>>>(N, A_d, L_d, X_d);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(X_gpu, X_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    bool failed = false;
    float tol = 1e-3f;
    for (uint32_t i = 0; i < N * N; ++i) {
        if (fabsf(X_gpu[i] - X_true[i]) < tol) {
            continue;
        } else {
            printf("Mismatch at (%u): got %.5f, expected %.5f\n", i, X_gpu[i],
                    X_true[i]);
            failed = true;
        }
    }

    // for (uint32_t i = 0; i < N * N; ++i) {
    //     printf("X_gpu[%u] = %f, X_true[%u] = %f\n", i, X_gpu[i], i, X_true[i]);
    // }

    if (!failed) {
        printf("Test PASSED for N=%u\n", N);
    } else {
        printf("Test FAILED for N=%u\n", N);
    }

    free(A);
    free(L);
    free(X_true);
    free(X_gpu);
    CUDA_CHECK(cudaFree(L_d));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(X_d));
}

int main() {
    srand(0);
    test_gemm_block(128, 32);

    // test_gemm(2);
    // test_gemm(4);
    // test_gemm(32);
    return 0;
}
