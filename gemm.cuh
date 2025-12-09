// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh", "gpu_block_kernel_fusion.cuh"]}
// TL {"workspace_files": []}

#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "utils.cuh"
#include "gpu_block_kernel_fusion.cuh"

namespace gemm {
// Computes A- BB^T, current naive gemm can only handle 32x32 matrix
// N: number of blocks in triblock diagonal
// block_n: dimension of each block in triblock diagonal
// A_col_offset, A_row_offset: offset for block of interest in A in the global memory
// B_col_offset, B_row_offset: offset for block of interest in B in the global memory
// out: outputs A- BB^T
__device__ void gemm_naive_XY(
    const uint32_t N, const uint32_t block_n, float const *A, float const *B, float *out,
    uint32_t A_col_offset, uint32_t A_row_offset, uint32_t B_col_offset, uint32_t B_row_offset
) {
    const uint32_t dim = N * block_n;
    int32_t col_ID = threadIdx.x % block_n;
    int32_t row_ID = threadIdx.x / block_n;

    float sum = 0.0f;
    for (uint32_t i = 0; i < block_n; ++i) {
        sum += B[(B_col_offset + row_ID) * dim + (B_row_offset + i)] * B[(B_col_offset + col_ID) * dim + (B_row_offset + i)];
    }

    if (row_ID < block_n && col_ID < block_n) {
        out[row_ID * block_n + col_ID] = A[(A_col_offset + row_ID) * dim + (A_row_offset + col_ID)] - sum;
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

// can handle maximum block size of 64x64 due to shared memory size limit
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

template <uint32_t A_n, uint32_t B_n, uint32_t r, uint32_t T_TH, uint32_t T_TW>
__device__ void diagonal_block_gemm_naive(float *A, float *B, float *C,
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
            for (uint32_t tj = 0; tj < (tile_i == tile_j ? ti+1 : T_TW); ++tj) {
                if ((_A + ti * A_n + tk) == (_B + tj * B_n + tk)) {
                    // If i==j reuse a
                    C[ti * T_TW + tj] += (a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
                    continue;
                }
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
            for (uint32_t tj = 0; tj < (tile_i == tile_j ? ti+1 : T_TW); ++tj) {
                C[ti * T_TW + tj] += a * _B[tj * B_n + tk];
            }
        }
    }
}

template <uint32_t A_n, uint32_t B_n, uint32_t T_TH, uint32_t T_TW>
__device__ void block_gemm_naive(float *A, float *B, float* C,
    const uint32_t tile_i, const uint32_t tile_j, const uint32_t r
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

__device__ void gemm_tensor_copytoreg(
    float *smem1, float *smem2, 
    uint32_t *reg_A, uint32_t *reg_B, 
    const uint32_t padding) {
    
    const uint32_t thread_ID = threadIdx.x % 32;

    const uint32_t thread_Ai = thread_ID / 4;
    const uint32_t thread_Aj = thread_ID % 4;
    reg_A[0] = __float_as_uint(smem1[thread_Ai * padding + thread_Aj]);
    reg_A[1] = __float_as_uint(smem1[thread_Ai * padding + thread_Aj + 8*padding]);
    reg_A[2] = __float_as_uint(smem1[thread_Ai * padding + thread_Aj + 4]);
    reg_A[3] = __float_as_uint(smem1[thread_Ai * padding + thread_Aj + 8*padding + 4]);

    const uint32_t thread_Bi = thread_ID / 4;
    const uint32_t thread_Bj = thread_ID % 4;
    reg_B[0] = __float_as_uint(smem2[thread_Bi * padding + thread_Bj]);
    reg_B[1] = __float_as_uint(smem2[thread_Bi * padding + thread_Bj + 4]);
}

__device__ void gemm_tensor_copytomem(
    float *mem, float *reg, const uint32_t padding) {
    
    const uint32_t thread_ID = threadIdx.x % 32;
    
    const uint32_t thread_i = thread_ID / 4;
    const uint32_t thread_j = thread_ID % 4;

    // change this back to -= after DEBUGGING!!!!!!!!
    mem[thread_i * padding + thread_j * 2] -= reg[0];
    mem[thread_i * padding + thread_j * 2 + 1] -= reg[1];
    mem[thread_i * padding + thread_j * 2 + 8*padding] -= reg[2];
    mem[thread_i * padding + thread_j * 2 + 8*padding + 1] -= reg[3];
}

__device__ void gemm_tensor_copytomem(
    float *memto, const float*memsub, float *reg, 
    const uint32_t paddingto, const uint32_t paddingsub) {
    
    const uint32_t thread_ID = threadIdx.x % 32;
    
    const uint32_t thread_i = thread_ID / 4;
    const uint32_t thread_j = thread_ID % 4;

    // change this back to -= after DEBUGGING!!!!!!!!
    memto[thread_i * paddingto + thread_j * 2] = memsub[thread_i * paddingsub + thread_j * 2] - reg[0];
    memto[thread_i * paddingto + thread_j * 2 + 1] = memsub[thread_i * paddingsub + thread_j * 2 + 1] - reg[1];
    memto[thread_i * paddingto + thread_j * 2 + 8*paddingto] = memsub[thread_i * paddingsub + thread_j * 2 + 8*paddingsub] - reg[2];
    memto[thread_i * paddingto + thread_j * 2 + 8*paddingto + 1] = memsub[thread_i * paddingsub + thread_j * 2 + 8*paddingsub + 1] - reg[3];
}

template <uint32_t W_TH, uint32_t num_threads_H>
__device__ void gemm_tensor_warp(float *smem1, float *smem2, uint32_t *reg_A, uint32_t *reg_B, float *reg_C,
    const uint32_t warp_tile_i, const uint32_t warp_tile_j, const uint32_t padding) {
    
    constexpr uint32_t warp_W = 8;
    constexpr uint32_t warp_H = 16;
    // if (threadIdx.x == 0) {
    //     printf("padding = %u\n", padding);
    // }
    for (uint32_t k_strides = 0; k_strides < padding; k_strides += 8) {
        // pointer to the right warp for smem1 and smem2
        float *A = smem1 + warp_tile_i * warp_H * W_TH * padding + k_strides;
        float *B = smem2 + warp_tile_j * warp_W * W_TH * padding + k_strides;
        // if (threadIdx.x == 0 && k_strides == 0) {
        //     for (uint32_t i = 0; i < padding; ++i) {
        //         for (uint32_t j = 0; j < padding; ++j) {
        //             if (i == j && (A[i*padding + j] != 1.0f || B[i*padding + j] != 1.0f)) {
        //                 printf("gemm_tensor: A[%u, %u] = %f, B[%u, %u] = %f\n", i, j, A[i * padding + j], i, j, B[i * padding + j]);
        //             }

        //             if (i != j && (A[i*padding + j] != 0.0f || B[i*padding + j] != 0.0f)) {
        //                 printf("gemm_tensor: A[%u, %u] = %f, B[%u, %u] = %f\n", i, j, A[i * padding + j], i, j, B[i * padding + j]);
        //             }
        //         }
        //     }
        // }
        // copy smem to registers
        for (uint32_t i = 0; i < W_TH; ++i) {
            float *A_i = A + i * warp_H * padding;
            float *B_i = B + i * warp_W * padding;
            gemm_tensor_copytoreg(A_i, B_i, 
                reg_A + i * 4, reg_B + i * 2, padding);
        }
        
        // perform tensor core operation
        for (uint32_t wi = 0; wi < W_TH; ++wi) {
            for (uint32_t wj = 0; wj < W_TH; ++wj) {
                int c_ind = (wi * W_TH + wj) * 4;
                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%10, %11, %12, %13};"
                    : "=f"(reg_C[c_ind]), "=f"(reg_C[c_ind + 1]), "=f"(reg_C[c_ind + 2]), "=f"(reg_C[c_ind + 3])
                    : "r"(reg_A[wi * 4]), "r"(reg_A[wi * 4 + 1]), "r"(reg_A[wi * 4 + 2]), "r"(reg_A[wi * 4 + 3]),
                        "r"(reg_B[wj * 2]), "r"(reg_B[wj * 2 + 1]),
                        "f"(reg_C[c_ind]), "f"(reg_C[c_ind + 1]), "f"(reg_C[c_ind + 2]), "f"(reg_C[c_ind + 3])
                );
                
            }
        }
        // if (k_strides == 0) {
        //     printf("threadIdx.x = %u, reg_A[0] = %f\n", threadIdx.x, reg_A[0]);
        //     printf("threadIdx.x = %u, reg_B[0] = %f\n", threadIdx.x, reg_B[0]);
        //     printf("threadIdx.x = %u, reg_C[0] = %f\n", threadIdx.x, reg_C[0]);
        // }
    } 
}

// GEMM with tensor core, each block works on 64x64 tiles with 8 warps
template <uint32_t W_TH, uint32_t num_threads_H>
__device__ void gemm_tensor(float *X, float *A, float *smem1, float *smem2, float *reg,
    const uint32_t block_tile_i, const uint32_t block_tile_j, const uint32_t N, const uint32_t block_n) {
    
    // if (threadIdx.x == 0) {
    //     printf("gemm_tensor called\n");
    // }

    constexpr uint32_t warp_H = 16;
    constexpr uint32_t warp_W = 8;
    constexpr uint32_t warp_rows = 2;
    const uint32_t block_size_H = warp_H * W_TH * warp_rows;
    // if (threadIdx.x == 0) {
    //     printf("block_size_H = %u\n", block_size_H);
    // }
    
    const uint32_t warp_ID = threadIdx.x / 32;
    const uint32_t warp_tile_i = warp_ID / 4;
    const uint32_t warp_tile_j = warp_ID % 4; // assuming 8 warps, so 2x4 tiles in block

    // maybe specialize writeback for diagonal blocks?
    // const bool diagonal_write = (block_tile_i == block_tile_j);
    const bool valid_block = (block_tile_i < block_n / block_size_H) && (block_tile_j < block_n / block_size_H);
    
    if (valid_block) {
        float *X_i = X + block_tile_i * block_size_H * N;
        float *X_j = X + block_tile_j * block_size_H * N;
        float *A_ij = A + block_tile_i * block_size_H * N + block_tile_j * block_size_H;

        uint32_t *reg_Xi = reinterpret_cast<uint32_t*>(reg);
        uint32_t *reg_Xj = reinterpret_cast<uint32_t*>(reg + 4 * W_TH);
        float *reg_Aij = reg + 6 * W_TH;

        for (uint32_t k = 0; k < block_n; k += block_size_H) {

            // copy X to smem
            block_cholesky_space::gmem_to_smem(X_i + k, X_j + k, smem1, smem2, N, block_size_H);
            __syncthreads();

            // solve matrix using tensor core
            gemm_tensor_warp<W_TH, num_threads_H>(smem1, smem2, reg_Xi, reg_Xj, reg_Aij, warp_tile_i, warp_tile_j, block_size_H);
            
            __syncthreads();
        }

        // move to warp tile for A_ij
        float *A_ij_warp = A_ij + warp_tile_i * warp_H * W_TH * N + warp_tile_j * warp_W * W_TH;

        // copy register values back to gmem
        for (uint32_t wi = 0; wi < W_TH; ++wi) {
            for (uint32_t wj = 0; wj < W_TH; ++wj) {
                int c_ind = (wi * W_TH + wj) * 4;
                gemm_tensor_copytomem(A_ij_warp + wi * warp_H * N + wj * warp_W, reg_Aij + c_ind, N);
            }
        }
        __syncthreads();
    }
}

// requires shared memory of size at least (T_TH * num_threads_H)^2
template <uint32_t W_TH, uint32_t num_threads_H>
__global__ void triblock_tensor_gemm(float *Out, float *In, const uint32_t N, const uint32_t block_n, const uint32_t smem_size_bytes) {
    extern __shared__ float smem[];

    float reg[4 * W_TH + 2*W_TH + 4*W_TH*W_TH] = {0.0f};

    // Map rectangular to triangular tiles
    const uint32_t block_tile_i = (uint32_t)((sqrtf(8.f * blockIdx.x + 1.f) - 1.f) * 0.5f);
    const uint32_t block_tile_j = blockIdx.x - (block_tile_i * (block_tile_i + 1) / 2);

    float *smem1 = smem;
    float *smem2 = smem + smem_size_bytes / (2 * sizeof(float));
    uint32_t smem_size = smem_size_bytes / sizeof(float);
    uint32_t reg_size = 4 * W_TH + 2*W_TH + 4*W_TH*W_TH;
    // if (threadIdx.x == 0) {
    //     printf("block tile i: %u, block tile j: %u, smem size: %u, W_TH: %u, num_threads_H: %u, reg size: %u\n", block_tile_i, block_tile_j, smem_size, W_TH, num_threads_H, reg_size);
    // }

    gemm_tensor<W_TH, num_threads_H>(In, Out, smem1, smem2, reg, block_tile_i, block_tile_j, N, block_n);
}

void launch_gemm_tensor(float *X, float *A, const uint32_t N, const uint32_t smem_size_bytes) {
    cudaFuncSetAttribute(
        triblock_tensor_gemm<1, 16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        50000
    );

    triblock_tensor_gemm<1, 16><<<36, 256, 32*32*2*4>>>(A, X, N, N, 32*32*2*4);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    printf("launch done done\n");
}


template <uint32_t T_TH, uint32_t num_threads_H>
__device__ void triblock_gemm_GPUblock(float *X, float *A, float *smem1, float *smem2, float *reg,
    const uint32_t block_tile_i, const uint32_t block_tile_j,
    const uint32_t N, const uint32_t block_n) {
    
    // calculate thread tile indices
    const uint32_t thread_tile_i = threadIdx.x / num_threads_H;
    const uint32_t thread_tile_j = threadIdx.x % num_threads_H;

    const bool diagonal_write = (block_tile_i == block_tile_j) && (thread_tile_i == thread_tile_j);

    const uint32_t valid_tiles = block_n / (T_TH * num_threads_H);
    if (block_tile_i < valid_tiles && block_tile_j < valid_tiles) {

        // locate X for tile_i, tile_j
        float *X_i = X + block_tile_i * (T_TH * num_threads_H) * N;
        float *X_j = X + block_tile_j * (T_TH * num_threads_H) * N;
        float *A_ij = A + block_tile_i * (T_TH * num_threads_H) * N + block_tile_j * (T_TH * num_threads_H);

        // solve X * X^T and store in reg iteratively in the k dimension
        for (uint32_t k = 0; k < block_n; k += T_TH * num_threads_H) {

            // copy X to smem
            block_cholesky_space::gmem_to_smem(X_i + k, smem1, N, T_TH * num_threads_H);
            block_cholesky_space::gmem_to_smem(X_j + k, smem2, N, T_TH * num_threads_H);
            __syncthreads();

            // X is a dense off-diagonal block, so we need to sum over all k
            constexpr uint32_t tile_size = T_TH * num_threads_H;
            block_cholesky_space::block_gemm_naive<tile_size, tile_size, tile_size, T_TH, T_TH>(
                smem1, smem2, reg, thread_tile_i, thread_tile_j);
            __syncthreads();
        }

        // move to sub tile in a GPU block
        float *A_subtile = A_ij + thread_tile_i * T_TH * N + thread_tile_j * T_TH;

        // calculate A - X * X^T 
        #pragma unroll
        for (uint32_t ti = 0; ti < T_TH; ++ti) {
            #pragma unroll
            for (uint32_t tj = 0; tj < (diagonal_write ? ti+1 : T_TH); ++tj) {
                A_subtile[ti * N + tj] -= reg[ti * T_TH + tj];
            }
        }
        __syncthreads();
    }
}

template <uint32_t T_TH, uint32_t num_threads_H>
__device__ void triblock_diag_gemm_GPUblock(float *X, float *A,float *smem1, float *smem2, float *reg,
    const uint32_t block_tile_i, const uint32_t block_tile_j,
    const uint32_t N, const uint32_t block_n) {

    // calculate thread tile indices (triangular mapping)
    const uint32_t thread_tile_i = (uint32_t)((sqrtf(8.f * threadIdx.x + 1.f) - 1.f) * 0.5f);
    const uint32_t thread_tile_j = threadIdx.x - (thread_tile_i * (thread_tile_i + 1) / 2);

    // Check if this thread has a valid triangular tile
    const bool valid_thread_tile = (thread_tile_i < num_threads_H);

    const uint32_t valid_tiles = block_n / (T_TH * num_threads_H);
    if (block_tile_i < valid_tiles && block_tile_j < valid_tiles) {

        // locate X for tile_i, tile_j
        float *X_i = X + block_tile_i * (T_TH * num_threads_H) * N;
        float *X_j = X + block_tile_j * (T_TH * num_threads_H) * N;
        float *A_ij = A + block_tile_i * (T_TH * num_threads_H) * N + block_tile_j * (T_TH * num_threads_H);

        // solve X * X^T and store in reg iteratively in the k dimension
        for (uint32_t k = 0; k < block_n; k += T_TH * num_threads_H) {

            // copy X to smem (all threads participate)
            block_cholesky_space::gmem_to_smem(X_i + k, smem1, N, T_TH * num_threads_H);
            block_cholesky_space::gmem_to_smem(X_j + k, smem2, N, T_TH * num_threads_H);
            __syncthreads();
    
            // Only threads with valid triangular tiles do computation
            if (valid_thread_tile) {
                gemm::diagonal_block_gemm_naive<T_TH * num_threads_H, T_TH * num_threads_H, T_TH * num_threads_H, T_TH, T_TH>(
                    smem1, smem2, reg, thread_tile_i, thread_tile_j);
            }
            __syncthreads();
        }

        // Only threads with valid triangular tiles write back
        if (valid_thread_tile) {
            // move to sub tile in a GPU block
            float *A_subtile = A_ij + thread_tile_i * T_TH * N + thread_tile_j * T_TH;

            // calculate A - X * X^T 
            #pragma unroll
            for (uint32_t ti = 0; ti < T_TH; ++ti) {
                #pragma unroll
                for (uint32_t tj = 0; tj < (thread_tile_i == thread_tile_j ? ti+1 : T_TH); ++tj) {
                    A_subtile[ti * N + tj] -= reg[ti * T_TH + tj];
                }
            }
        }
        __syncthreads();
    }
}

// requires shared memory of size at least (T_TH * num_threads_H)^2
template <uint32_t T_TH, uint32_t num_threads_H>
__global__ void triblock_diagonal_gemm(float *A, float *X, const uint32_t N, const uint32_t block_n, const uint32_t smem_size_bytes) {
    extern __shared__ float smem[];

    float reg[T_TH * T_TH] = {0.0f};

    // Map rectangular to triangular tiles
    const uint32_t block_tile_i = (uint32_t)((sqrtf(8.f * blockIdx.x + 1.f) - 1.f) * 0.5f);
    const uint32_t block_tile_j = blockIdx.x - (block_tile_i * (block_tile_i + 1) / 2);

    float *smem1 = smem;
    float *smem2 = smem + smem_size_bytes / (2 * sizeof(float));

    triblock_gemm_GPUblock<T_TH, num_threads_H>(X, A, smem1, smem2, reg, block_tile_i, block_tile_j, N, block_n);

    // if (block_tile_i == block_tile_j) {
    //     //triblock_diag_gemm_GPUblock<T_TH, num_threads_H>(X, A, smem1, smem2, reg, block_tile_i, block_tile_j, N, block_n);
    //     triblock_gemm_GPUblock<T_TH, num_threads_H>(X, A, smem1, smem2, reg, block_tile_i, block_tile_j, N, block_n);

    // } else {
    //     triblock_gemm_GPUblock<T_TH, num_threads_H>(X, A, smem1, smem2, reg, block_tile_i, block_tile_j, N, block_n);
    // }
}



template <uint32_t m, uint32_t T_TH, uint32_t T_TW>
__device__ void diagonal_block_update(float *A, float *L,
    const uint32_t n,
    const uint32_t i, const uint32_t j,
    float *smem1, float *smem2
) {
    // Accumulate update results in registers w/ each thread getting a subtile
    float reg[T_TH * T_TW] = {0.0f}; // zero-init
    
    // Map rectangular to triangular tiles
    const uint32_t tile_i = (uint32_t)((sqrtf(8.f * threadIdx.x + 1.f) - 1.f) * 0.5f);
    const uint32_t tile_j = threadIdx.x - (tile_i * (tile_i + 1) / 2);

    // Only compute if valid tile
    constexpr uint32_t N = m / T_TH;
    if (tile_i < N && tile_j < N) {
        // Compute Lij * Lij^T
        block_cholesky_space::diagonal_block_gemm_naive<m, m, T_TH, T_TW>(smem2, reg, tile_i, tile_j);

        // Move A to Aii
        float *Aii = block_cholesky_space::get_block(A, i, i, n, m);

        // Move to subtile
        float *_Aii = Aii + tile_i * T_TH * n + tile_j * T_TW;
        float *_Aii_p = smem1 + tile_i * T_TH * m + tile_j * T_TW;

        // Compute Aii - Lij * Lij^T
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

} // namespace gemm_naive