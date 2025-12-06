// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["trsm_small.cuh", "cholesky_small.cuh"]}
// TL {"workspace_files": []}

#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include "trsm_small.cuh"
#include "cholesky_small.cuh"

namespace block_cholesky_space {

size_t get_workspace_size(int32_t size) {
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Helper functions

__device__ float* get_block(float *A, const uint32_t i, const uint32_t j, const uint32_t n, const uint32_t m) { return A + i * m * n + j * m; }
__device__ const float* get_block(const float *A, const uint32_t i, const uint32_t j, const uint32_t n, const uint32_t m) { return A + i * m * n + j * m; }

// This is goated -- Xiaomian
// input: 
// gmem: pointer to start of global memory to be written
// smem: pointer to start of shared memory to be read
// gmem_w: width of the entire matrix
// smem_w: width of a block in the matrix to be handled, writing a total of smem_w * smem_w elements
__device__ void smem_to_gmem(float *gmem, float*smem,
    const uint32_t gmem_w, const uint32_t smem_w
) {
    // Handle vectors
    float4 *gmem4 = reinterpret_cast<float4*>(gmem);
    float4 *smem4 = reinterpret_cast<float4*>(smem);
    const uint32_t gmem4_w = gmem_w / 4;
    const uint32_t smem4_w = smem_w / 4;
    for (uint32_t idx = threadIdx.x; idx < smem_w * smem_w / 4; idx += blockDim.x) {
        const uint32_t i = idx / smem4_w;
        const uint32_t j = idx % smem4_w;
        gmem4[i * gmem4_w + j] = smem4[idx];
    }

    // Handle tail
    for (uint32_t idx = (smem_w * smem_w / 4) * 4 + threadIdx.x; idx < smem_w * smem_w; idx += blockDim.x) {
        const uint32_t i = idx / smem_w;
        const uint32_t j = idx % smem_w;
        gmem[i * gmem_w + j] = smem[idx];
    }

    __syncthreads();
}

// This is goated -- Xiaomian
// input: 
// gmem: pointer to start of global memory to be read
// smem: pointer to start of shared memory to be written
// gmem_w: width of the entire matrix
// smem_w: width of a block in the matrix to be handled, writing a total of smem_w * smem_w elements
__device__ void gmem_to_smem(float *gmem, float*smem,
    const uint32_t gmem_w, const uint32_t smem_w
) {
    // Handle vectors
    float4 *gmem4 = reinterpret_cast<float4*>(gmem);
    float4 *smem4 = reinterpret_cast<float4*>(smem);
    const uint32_t gmem4_w = gmem_w / 4;
    const uint32_t smem4_w = smem_w / 4;
    for (uint32_t idx = threadIdx.x; idx < smem_w * smem_w / 4; idx += blockDim.x) {
        const uint32_t i = idx / smem4_w;
        const uint32_t j = idx % smem4_w;
        smem4[idx] = gmem4[i * gmem4_w + j];
    }

    // Handle tail
    for (uint32_t idx = (smem_w * smem_w / 4) * 4 + threadIdx.x; idx < smem_w * smem_w; idx += blockDim.x) {
        const uint32_t i = idx / smem_w;
        const uint32_t j = idx % smem_w;
        smem[idx] = gmem[i * gmem_w + j];
    }

    __syncthreads();
}

__device__ void gmem_to_smem(const float *gmem, float*smem,
    const uint32_t gmem_w, const uint32_t smem_w
) {
    // Handle vectors
    const float4 *gmem4 = reinterpret_cast<const float4*>(gmem);
    float4 *smem4 = reinterpret_cast<float4*>(smem);
    const uint32_t gmem4_w = gmem_w / 4;
    const uint32_t smem4_w = smem_w / 4;
    for (uint32_t idx = threadIdx.x; idx < smem_w * smem_w / 4; idx += blockDim.x) {
        const uint32_t i = idx / smem4_w;
        const uint32_t j = idx % smem4_w;
        smem4[idx] = gmem4[i * gmem4_w + j];
    }

    // Handle tail
    for (uint32_t idx = (smem_w * smem_w / 4) * 4 + threadIdx.x; idx < smem_w * smem_w; idx += blockDim.x) {
        const uint32_t i = idx / smem_w;
        const uint32_t j = idx % smem_w;
        smem[idx] = gmem[i * gmem_w + j];
    }

    __syncthreads();
}


__device__ void gmem_to_smem(float *gmem1, float *gmem2,
    float*smem1, float*smem2,
    const uint32_t gmem_w, const uint32_t smem_w
) {
    // Handle vectors
    float4 *gmem1_4 = reinterpret_cast<float4*>(gmem1);
    float4 *gmem2_4 = reinterpret_cast<float4*>(gmem2);
    float4 *smem1_4 = reinterpret_cast<float4*>(smem1);
    float4 *smem2_4 = reinterpret_cast<float4*>(smem2);
    const uint32_t gmem4_w = gmem_w / 4;
    const uint32_t smem4_w = smem_w / 4;
    for (uint32_t idx = threadIdx.x; idx < smem_w * smem_w / 4; idx += blockDim.x) {
        const uint32_t i = idx / smem4_w;
        const uint32_t j = idx % smem4_w;
        smem1_4[idx] = gmem1_4[i * gmem4_w + j];
        smem2_4[idx] = gmem2_4[i * gmem4_w + j];
    }

    // Handle tail
    for (uint32_t idx = (smem_w * smem_w / 4) * 4 + threadIdx.x; idx < smem_w * smem_w; idx += blockDim.x) {
        const uint32_t i = idx / smem_w;
        const uint32_t j = idx % smem_w;
        smem1[idx] = gmem1[i * gmem_w + j];
        smem2[idx] = gmem2[i * gmem_w + j];
    }

    __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// Device functions

template <uint32_t A_n, uint32_t r, uint32_t T_TH, uint32_t T_TW>
__device__ void diagonal_block_gemm_naive(float *A, float* C,
    const uint32_t tile_i, const uint32_t tile_j
) {
    // Move to subtile
    float *_A = A + tile_i * T_TH * A_n;
    float *_B = A + tile_j * T_TH * A_n;

    // Each thread handles a tile
    for (uint32_t tk = 0; tk < r; tk += 4) {
        #pragma unroll
        for (uint32_t ti = 0; ti < T_TH; ++ti) {
            const float4 a = *(reinterpret_cast<float4*>(_A + ti * A_n + tk));
            #pragma unroll
            for (uint32_t tj = 0; tj < (tile_i == tile_j ? ti+1 : T_TW); ++tj) {
                if ((_A + ti * A_n + tk) == (_B + tj * A_n + tk)) {
                    // If i==j reuse a
                    C[ti * T_TW + tj] += (a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
                    continue;
                }
                const float4 b = *(reinterpret_cast<float4*>(_B + tj * A_n + tk));
                C[ti * T_TW + tj] += (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
            }
        }
    }
    // Handle tail
    for (uint32_t tk = (r / 4) * 4; tk < r; ++tk) {
        #pragma unroll
        for (uint32_t ti = 0; ti < T_TH; ++ti) {
            const float a = _A[ti * A_n + tk];
            #pragma unroll
            for (uint32_t tj = 0; tj < (tile_i == tile_j ? ti+1 : T_TW); ++tj) {
                C[ti * T_TW + tj] += a * _B[tj * A_n + tk];
            }
        }
    }
}

template <uint32_t A_n, uint32_t B_n, uint32_t r, uint32_t T_TH, uint32_t T_TW>
__device__ void block_gemm_naive(float *A, float *B, float* C,
    const uint32_t tile_i, const uint32_t tile_j
) {
    // Move to subtile
    float *_A = A + tile_i * T_TH * A_n;
    float *_B = B + tile_j * T_TH * B_n;

    // Each thread handles a tile
    for (uint32_t tk = 0; tk < r; tk += 4) {
        #pragma unroll
        for (uint32_t ti = 0; ti < T_TH; ++ti) {
            const float4 a = *(reinterpret_cast<float4*>(_A + ti * A_n + tk));
            #pragma unroll
            for (uint32_t tj = 0; tj < T_TW; ++tj) {
                const float4 b = *(reinterpret_cast<float4*>(_B + tj * B_n + tk));
                C[ti * T_TW + tj] += (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
            }
        }
    }
    // Handle tail
    for (uint32_t tk = (r / 4) * 4; tk < r; ++tk) {
        #pragma unroll
        for (uint32_t ti = 0; ti < T_TH; ++ti) {
            const float a = _A[ti * A_n + tk];
            #pragma unroll
            for (uint32_t tj = 0; tj < T_TW; ++tj) {
                C[ti * T_TW + tj] += a * _B[tj * B_n + tk];
            }
        }
    }
}

template <uint32_t m, uint32_t T_TH, uint32_t T_TW>
__device__ void block_update(const float *A, float *L,
    const uint32_t n,
    const uint32_t i, const uint32_t j,
    float *smem1, float*smem2
) {
    // Accumulate update results in registers w/ each thread getting a subtile
    float reg[T_TH * T_TW] = {0.0f}; // zero-init
    const uint32_t tile_i = threadIdx.x / (m / T_TW);
    const uint32_t tile_j = threadIdx.x % (m / T_TW);

    // Only compute if valid tile
    constexpr uint32_t N = m / T_TH;

    // Sum Lik * Ljk^T
    for (uint32_t k = 0; k < j; ++k) {
        // Load Lik, Ljk into smem
        float *Lik = get_block(L, i, k, n, m);
        float *Ljk = get_block(L, j, k, n, m);
        gmem_to_smem(Lik, Ljk, smem1, smem2, n, m);

        if (tile_i < N && tile_j < N) {
            block_gemm_naive<m, m, m, T_TH, T_TW>(smem1, smem2, reg, tile_i, tile_j);
        }

        __syncthreads();
    }

    if (tile_i < N && tile_j < N) {
        // Move A to Aij 
        const float *Aij = get_block(A, i, j, n, m);

        // Move to subtile
        const float *_Aij = Aij + tile_i * T_TH * n + tile_j * T_TW;
        float *_Aij_p = smem1 + tile_i * T_TH * m + tile_j * T_TW;

        // Compute Aij - sum
        #pragma unroll
        for (uint32_t ti = 0; ti < T_TH; ++ti) {
            #pragma unroll
            for (uint32_t tj = 0; tj < T_TW; ++tj) {
                _Aij_p[ti * m + tj] = _Aij[ti * n + tj] - reg[ti * T_TW + tj];
            }
        }
    }

    // Wait for the entire block to finish
    __syncthreads();
}

template <uint32_t m, uint32_t T_TH, uint32_t T_TW>
__device__ void diagonal_block_update(const float *A, float *L,
    const uint32_t n,
    const uint32_t i, const uint32_t j,
    float *smem
) {
    // Accumulate update results in registers w/ each thread getting a subtile
    float reg[T_TH * T_TW] = {0.0f}; // zero-init
    
    // Map rectangular to triangular tiles
    const uint32_t tile_i = (uint32_t)((sqrtf(8.f * threadIdx.x + 1.f) - 1.f) * 0.5f);
    const uint32_t tile_j = threadIdx.x - (tile_i * (tile_i + 1) / 2);

    // Only compute if valid tile
    constexpr uint32_t N = m / T_TH;

    // Sum Lik * Lik^T
    for (uint32_t k = 0; k < j; ++k) {
        // Load Lik into smem
        float *Lik = get_block(L, i, k, n, m);
        gmem_to_smem(Lik, smem, n, m);

        if (tile_i < N && tile_j < N) {
            diagonal_block_gemm_naive<m, m, T_TH, T_TW>(smem, reg, tile_i, tile_j);
        }

        __syncthreads();
    }

    if (tile_i < N && tile_j < N) {
        // Move A to Aii
        const float *Aii = get_block(A, i, i, n, m);

        // Move to subtile
        const float *_Aii = Aii + tile_i * T_TH * n + tile_j * T_TW;
        float *_Aii_p = smem + tile_i * T_TH * m + tile_j * T_TW;

        // Compute Aii - sum
        #pragma unroll
        for (uint32_t ti = 0; ti < T_TH; ++ti) {
            #pragma unroll
            for (uint32_t tj = 0; tj < (tile_i == tile_j ? ti+1 : T_TW); ++tj) {
                _Aii_p[ti * m + tj] = _Aii[ti * n + tj] - reg[ti * T_TW + tj];
            }
        }
    }

    // Wait for the entire block to finish
    __syncthreads();
}

template <uint32_t m, uint32_t W, uint32_t T_TH, uint32_t T_TW>
__launch_bounds__(W*32)
__global__ void block_kernel(const float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, // matrix size
    const uint32_t j // block col
) {
    // Setup smem
    extern __shared__ float smem[];
    float *smem2 = smem + m * m;

    // Each SM gets a block
    for (uint32_t i = j + 1 + blockIdx.x; i < n / m; i += gridDim.x) {
        // Update
        block_update<m, T_TH, T_TW>(A, L, n, i, j, smem, smem2);

        // TRSM
        float *Lij = get_block(L, i, j, n, m);
        float *Ljj = get_block(L, j, j, n, m);
        float *Aij = smem;
        trsm_small::block_trsm(Ljj, Lij, Aij, n, n, m, m); // A, X, B
    }
}

template <uint32_t m, uint32_t W, uint32_t T_TH, uint32_t T_TW>
__launch_bounds__(W*32)
__global__ void chol_kernel(const float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, // matrix size
    const uint32_t j // block col
) {
    // Only 1 SM participates

    // Setup smem
    extern __shared__ float smem[];
    float *smem2 = smem + m * m;

    // Update (all threads participate)
    // diagonal_block_update<T_TH, T_TW>(A, L, n, m, j, j, smem);
    block_update<m, T_TH, T_TW>(A, L, n, j, j, smem, smem2);

    // Chol
    float *Ajj = smem;
    float *Ljj = smem2;
    cholesky_small::block_col_cholesky(Ajj, Ljj, m, m, m);

    // Write back Ljj
    Ljj = block_cholesky_space::get_block(L, j, j, n, m);
    smem_to_gmem(Ljj, smem2, n, m);
}

////////////////////////////////////////////////////////////////////////////////
// Host functions

template <uint32_t m, uint32_t T_TS, uint32_t W>
void launch_specialized_kernel(const uint32_t n, float const *in, float *out) {
    // Setup smem
    constexpr int smem_size_bytes = m * m * 2 * sizeof(float); // need to store 2 blocks in smem
    cudaFuncSetAttribute(
        chol_kernel<m, W, T_TS, T_TS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes
    );
    cudaFuncSetAttribute(
        block_kernel<m, W, T_TS, T_TS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes
    );

    // Iterate over block cols launching a kernel for each step
    for (uint32_t j = 0; j < n / m; ++j) {
        // Step 1: Chol(update) diagonal block
        chol_kernel<m, W, T_TS, T_TS><<<1, W*32, smem_size_bytes>>>(in, out, n, j);

        // Step 2: Trsm(update) all other blocks
        block_kernel<m, W, T_TS, T_TS><<<48, W*32, smem_size_bytes>>>(in, out, n, j);
    }
}

template <uint32_t m, uint32_t T_TS, uint32_t W>
void launch_specialized_kernel_dynamic_block(const uint32_t n, float const *in, float *out,
    const uint32_t start_j, const uint32_t end_j
) {
    // Setup smem
    constexpr int smem_size_bytes = m * m * 2 * sizeof(float); // need to store 2 blocks in smem
    cudaFuncSetAttribute(
        chol_kernel<m, W, T_TS, T_TS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes
    );
    cudaFuncSetAttribute(
        block_kernel<m, W, T_TS, T_TS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes
    );

    // Iterate over block cols launching a kernel for each step
    for (uint32_t j = start_j; j < end_j; ++j) {
        // Step 1: Chol(update) diagonal block
        chol_kernel<m, W, T_TS, T_TS><<<1, W*32, smem_size_bytes>>>(in, out, n, j);

        // Step 2: Trsm(update) all other blocks
        block_kernel<m, W, T_TS, T_TS><<<48, W*32, smem_size_bytes>>>(in, out, n, j);
    }
}

void launch_block_cholesky(
    const uint32_t n, float const *in, float *out, void *workspace
) {
    // Divide the grid into blocks
    // if (n < 2048) {
    //     launch_specialized_kernel<16, 1, 8>(n, in, out);
    // } else if (n < 4096) {
    //     launch_specialized_kernel<32, 2, 8>(n, in, out);
    // } else {
    //     launch_specialized_kernel<64, 4, 8>(n, in, out);
    // }

    // Find an m >= 16 that makes n/m = 64
    uint32_t _n = n;
    uint32_t _m = 64;
    while (64 > _n / _m && _m >= 16) {
        _m /= 2;
    }
    if (_m < 16) {
        _m = 16;
    }

    // Every time n halves, halve m until m = 16
    if (_m == 64) {
        launch_specialized_kernel_dynamic_block<64, 4, 8>(n, in, out, 0, (n/2)/64);
        launch_specialized_kernel_dynamic_block<32, 2, 8>(n, in, out, (n/2)/32, (3*n/4)/32);
        launch_specialized_kernel_dynamic_block<16, 1, 8>(n, in, out, (3*n/4)/16, n/16);
    } else if (_m == 32) {
        launch_specialized_kernel_dynamic_block<32, 2, 8>(n, in, out, 0, (n/2)/32);
        launch_specialized_kernel_dynamic_block<16, 1, 8>(n, in, out, (n/2)/16, n/16);
    } else {
        launch_specialized_kernel<16, 1, 8>(n, in, out);
    }
}

}