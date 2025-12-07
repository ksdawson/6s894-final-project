// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh", "trsm_small.cuh", "cholesky.cuh", "gemm.cuh", "gpu_block_kernel_fusion.cuh", "cholesky_small.cuh", "gpu_block_enhanced_deluxe_kernel_fusion.cuh", "gpu_block_enhanced_kernel_fusion.cuh"]}
// TL {"workspace_files": []}

#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "utils.cuh"
#include "trsm_small.cuh"
#include "cholesky.cuh"
#include "gemm.cuh"
#include "gpu_block_kernel_fusion.cuh"
#include "cholesky_small.cuh"
#include "gpu_block_enhanced_deluxe_kernel_fusion.cuh"
#include "gpu_block_enhanced_kernel_fusion.cuh"

struct TB {
    float const *in;
    float *out;
    const uint32_t N;
    const uint32_t block_n;
    const uint32_t m; // tiles in each block of triblock matrix
};

namespace triblock_helper {
inline float* get_block(float *A, const uint32_t i, const uint32_t j, const uint32_t n, const uint32_t m) { return A + i * m * n + j * m; }
inline const float* get_block(const float *A, const uint32_t i, const uint32_t j, const uint32_t n, const uint32_t m) { return A + i * m * n + j * m; }


// const float *triblock_get_block(const float *A, const uint32_t bi, const uint32_t bj, 
//     const uint32_t i, const uint32_t j, TB tb) {
    
//     const float *block_A = block_cholesky_space::get_block(A, bi, bj, tb.N, tb.block_n);
//     return block_cholesky_space::get_block(block_A, i, j, tb.N, tb.m);
// }

// float *triblock_get_block(float *A, const uint32_t bi, const uint32_t bj, 
//     const uint32_t i, const uint32_t j, TB tb) {
    
//     float *block_A = block_cholesky_space::get_block(A, bi, bj, tb.N, tb.block_n);
//     return block_cholesky_space::get_block(block_A, i, j, tb.N, tb.m);
// }

template <uint32_t m, uint32_t W, uint32_t T_TH, uint32_t T_TW>
__launch_bounds__(W*32)
__global__ void block_kernel(float *A, float *L, // input matrix, Chol matrix
    const uint32_t N, // matrix size
    const uint32_t block_n, // block size of triblock matrix
    const uint32_t j // block col
) {
    // Setup smem
    extern __shared__ float smem[];
    float *smem2 = smem + m * m;
    float *smem3 = smem2 + m * m;

    // Each SM gets a block
    for (uint32_t i = j + 1 + blockIdx.x; i < block_n / m; i += gridDim.x) {
        // Update
        block_cholesky_space::block_update<m, T_TH, T_TW>(A, L, N, i, j, smem, smem2);

        // Load Ljj into smem
        float *Ljj = block_cholesky_space::get_block(L, j, j, N, m);
        block_cholesky_space::gmem_to_smem(Ljj, smem3, N, m);
        Ljj = smem3;

        // TRSM
        float *Lij = smem2;
        float *Aij = smem;
        trsm_small::block_trsm(Ljj, Lij, Aij, m, m, m, m); // A, X, B

        // Write back Lij
        Lij = block_cholesky_space::get_block(L, i, j, N, m);
        block_cholesky_space::smem_to_gmem(Lij, smem2, N, m);

        // Update Aii
        if (i == j + 1) {
            // Update Aii
            deluxe_alt_kernel_fusion::diagonal_block_update<m, T_TH, T_TW>(A, L, N, i, j, smem, smem2);
            
            // Chol Aii
            float *Aii = smem;
            float *Lii = smem2;
            cholesky_small::block_col_cholesky(Aii, Lii, m, m, m);
            // if (threadIdx.x == 0) {
            //     for (uint32_t k = 0; k < m; ++k) {
            //         for (uint32_t l = 0; l < m; ++l) {
            //             printf("Lii[%u, %u] = %f\n", k, l, Lii[k * m + l]);
            //         }
            //     }
            // }

            // Write back Lii
            Lii = block_cholesky_space::get_block(L, i, i, N, m);
            block_cholesky_space::smem_to_gmem(Lii, smem2, N, m);
            
        } else {
            // Write back to A
            alt_kernel_fusion::diagonal_block_update<m, T_TH, T_TW>(A, L, N, i, j, smem2);
        }
    }
}

template <uint32_t m>
__launch_bounds__(1024)
__global__ void chol_kernel(const float *A, float *L, // input matrix, Chol matrix
    const uint32_t n, // padding
    const uint32_t j // block col
) {
    // Only 1 SM participates

    // Setup smem
    extern __shared__ float smem[];
    float *smem1 = smem;
    float *smem2 = smem + m * m;

    // store Ajj into smem1
    const float *Ajj = block_cholesky_space::get_block(A, j, j, n, m);
    float *Ajj_smem = smem1;
    block_cholesky_space::gmem_to_smem(Ajj, Ajj_smem, n, m);

    // Chol
    float *Ljj = smem2;
    cholesky_small::block_col_cholesky(Ajj_smem, Ljj, m, m, m);

    // Write back Ljj
    Ljj = block_cholesky_space::get_block(L, j, j, n, m);
    block_cholesky_space::smem_to_gmem(Ljj, smem2, n, m);
}

template <uint32_t m, uint32_t W, uint32_t T_TS, uint32_t T_TW, uint32_t smem_size_bytes>
void triblock_block_cholesky(TB tb, const uint32_t bi) {
    const float *in = tb.in;
    float *out = tb.out;
    
    const float *A = triblock_helper::get_block(in, bi, bi, tb.N, tb.block_n);
    float *L = triblock_helper::get_block(out, bi, bi, tb.N, tb.block_n);

    alt_kernel_fusion::chol_kernel<m><<<1, 32*32, smem_size_bytes>>>(A, L, tb.N, 0);

    // iterate over block cols launching a kernel for each step
    for (uint32_t j = 0; j < tb.block_n / tb.m - 1; ++j) {
        triblock_helper::block_kernel<m, W, T_TS, T_TS><<<48, W*32, smem_size_bytes*3>>>(const_cast<float*>(A), L, tb.N, tb.block_n, j);
    }
}  

// asumes A_n == X_n!!!
template <uint32_t T_TH, uint32_t r>
__device__ void triblock_trsm_update(const float *B_ij, const float *A_j, const float *X_i, 
  float *reg, 
  float *smem1, float *smem2, float *smem3, 
  const uint32_t A_n, const uint32_t X_n, const uint32_t B_n, 
  const uint32_t j) {

  const uint32_t thread_tile_i = threadIdx.x / (r / T_TH);
  const uint32_t thread_tile_j = threadIdx.x % (r / T_TH);
  const bool valid_thread = thread_tile_i < (r / T_TH) && thread_tile_j < (r / T_TH);

  for (int32_t k = 0; k < j; ++k) {
    // save A_tjj, X_tj into smem
    const float *A_jk = A_j + k * r;
    const float *X_ik = X_i + k * r;
    block_cholesky_space::gmem_to_smem(A_jk, X_ik, smem1, smem2, A_n, r);

    // call block gemm
    if (valid_thread) {
      block_cholesky_space::block_gemm_naive<r, r, r, T_TH, T_TH>(smem2, smem1, reg, thread_tile_i, thread_tile_j);
    }
    __syncthreads();
  }

  if (valid_thread) {
    const float *B_ij_subtile = B_ij + thread_tile_i * T_TH * B_n + thread_tile_j * T_TH;
    float *smem3_subtile = smem3 + thread_tile_i * T_TH * r + thread_tile_j * T_TH;

    for (int32_t ti = 0; ti < T_TH; ++ti) {
      for (int32_t tj = 0; tj < T_TH; ++tj) {
        smem3_subtile[ti * r + tj] = B_ij_subtile[ti * B_n + tj] - reg[ti * T_TH + tj];
      }
    }
  }
  __syncthreads();
}   


template <uint32_t T_TH, uint32_t r>
__global__ void triblock_block_trsm (float const *A, float *X, const float *B,
  const uint32_t A_n, const uint32_t X_n, const uint32_t B_n,
  const uint32_t block_n
) {
  
  extern __shared__ float smem[];
  float *smem1 = smem;
  float *smem2 = smem + r * r;
  float *smem3 = smem + r * r * 2;
  const bool valid_block = blockIdx.x < (block_n / r);

  if (valid_block) {
    // Calculate X, B pointers
    float *X_i = X + r * blockIdx.x * X_n; // rows of X
    const float *B_i = B + r * blockIdx.x * B_n; // rows of B

  // loop through rows in steps of r to calculate trsm
    for (uint32_t j = 0; j < (int)(block_n / r); ++j) {
      float reg[T_TH*T_TH] = {0.0f};

      // find column of X we are calculating for
      const float *B_ij = B_i + j * r; // column of B_i
      const float *A_j = A + j * r* A_n; // rows of A

      // GEMM update on B (store in smem3)
      triblock_trsm_update<T_TH, r>(B_ij, A_j, X_i, reg, smem1, smem2, smem3, A_n, X_n, B_n, j);
      
      // call block trsm
      const float *A_jj = block_cholesky_space::get_block(A, j, j, A_n, r);
      float *X_ij = X_i + j * r; // column of X_i
      block_cholesky_space::gmem_to_smem(A_jj, smem1, A_n, r);
      trsm_small::block_trsm(smem1, smem2, smem3, r, r, r, r);

      block_cholesky_space::smem_to_gmem(X_ij, smem2, X_n, r);
    }
  }
}

}
