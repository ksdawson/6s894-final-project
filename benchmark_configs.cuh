// TL+ {"compile_flags": ["-lcuda", "-lcublas", "-lcusolver"]}
// TL+ {"header_files": ["utils.cuh", "benchmark_helper.cuh", "cholesky.cuh", "trsm.cuh", "gpu_block_kernel_fusion.cuh", "cholesky_small.cuh", "trsm_small.cuh", "gpu_block_enhanced_kernel_fusion.cuh", "gtrsm.cuh", "cusolver.cuh", "cusolver_utils.cuh", "triblock.cuh", "gemm.cuh", "gpu_block_enhanced_deluxe_kernel_fusion.cuh", "triblock_helper.cuh", "gpu_block_enhanced_deluxe_premium_kernel_fusion.cuh"]}
// TL+ {"workspace_files": []}

#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include "gtrsm.cuh"
#include "utils.cuh"
#include "benchmark_helper.cuh"
#include "cholesky_small.cuh"
#include "trsm_small.cuh"
#include "gtrsm.cuh"
#include "cholesky.cuh" 
#include "gpu_block_kernel_fusion.cuh"
#include "gpu_block_enhanced_kernel_fusion.cuh"
#include "gpu_block_enhanced_deluxe_kernel_fusion.cuh"
#include "cusolver.cuh"
#include "cusolver_utils.cuh"
#include "triblock.cuh"
#include "gemm.cuh"
#include "triblock_helper.cuh"
#include "gpu_block_enhanced_deluxe_premium_kernel_fusion.cuh"

template <typename Reset, typename F>
double
benchmark_ms(double target_time_ms, int32_t num_iters_inner, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

template <typename Impl>
void run_config_trsm(
    Phase phase,
    Solver solver,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {

    auto size = config.size;
    auto block_size = config.block_size; // block_size for trsm is r, num columns to compute AX^T = B  

    auto const &a = data.a.at({size, block_size});
    auto const &c = data.c.at({size, block_size});

    float *a_gpu;
    float *c_gpu;
    float *b_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_gpu, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_gpu, size * size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size * size * sizeof(float),
        cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(c_gpu, 0, size * size * sizeof(float)));

    size_t workspace_size = Impl::get_workspace_size(size);
    void *workspace_gpu = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
    }

    //need to flush gpu caches before benchmarking each run, might have to change size based on GPU
    void *flush_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_gpu, 1024*1024*64));
    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));

    printf("  %6d  %6d", size, block_size);

    auto const &b = data.b.at({size, block_size});
    CUDA_CHECK(cudaMemcpy(b_gpu, b.data(), size * size * sizeof(float), cudaMemcpyHostToDevice));

    Impl::run(size, block_size, a_gpu, c_gpu, b_gpu, workspace_gpu);

    std::vector<float> c_out_host(size * size);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        c_gpu,
        size * size * sizeof(float),
        cudaMemcpyDeviceToHost));
    
    // need to fix tflops for vector version
    float rel_rmse = 0.0f;
    double tflops = 0.0;
    if (solver == Solver::TRSM_BLOCK || solver == Solver::TRSM_BLOCK_T) {
        tflops = tflops_trsm(size);
        rel_rmse = calc_error_trsm(c_out_host, c, size);
    } else if (solver == Solver::TRSM_VECTOR) {
        tflops = tflops_trsm_vec(size);
        rel_rmse = calc_error_trsm_vector(c_out_host, c, size);
    } else {
        tflops = tflops_trsm(size);
        rel_rmse = calc_error_trsm(c_out_host, c, size);
    }
    
    printf("  %8.02e", rel_rmse);

    // for (int32_t i = 0; i < size; ++i) {
    //     for (int32_t j = 0; j < size; ++j) {
    //         printf("c_out_host[%d][%d] = %f, c[%d][%d] = %f\n", i, j, c_out_host[i * size + j], i, j, c[i * size + j]);
    //     }
    // }

    if (rel_rmse > 1e5) {
        printf("  %9s  %7s", "-", "-");
    } else {
        // SHOULD CHANGE THIS TARGET TIME
        double target_time_ms = 40.0;
        double elapsed_ms = 0.0;
        
        elapsed_ms = benchmark_ms(
            target_time_ms,
            1,
            [&]() {
                if (workspace_size > 0) {
                    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                }
                CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));
            },
            [&]() {
                Impl::run(size, block_size, a_gpu, c_gpu, b_gpu, workspace_gpu);
            });

        results.elapsed_ms[{size, block_size}] = elapsed_ms;
        //double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
        printf("  %9.02f  %7.05f", elapsed_ms, tflops / (elapsed_ms * 1e-3));
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(flush_gpu));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
}

template <typename Impl>
void run_config_trsm_graph(
    Phase phase,
    Solver solver,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {

    auto size = config.size;
    auto block_size = config.block_size; // block_size for trsm is r, num columns to compute AX^T = B  

    auto const &a = data.a.at({size, block_size});
    auto const &c = data.c.at({size, block_size});

    float *a_gpu;
    float *c_gpu;
    float *b_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_gpu, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_gpu, size * size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size * size * sizeof(float),
        cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(c_gpu, 0, size * size * sizeof(float)));

    size_t workspace_size = Impl::get_workspace_size(size);
    void *workspace_gpu = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
    }

    //need to flush gpu caches before benchmarking each run, might have to change size based on GPU
    void *flush_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_gpu, 1024*1024*64));
    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));

    printf("  %6d  %6d", size, block_size);

    auto const &b = data.b.at({size, block_size});
    CUDA_CHECK(cudaMemcpy(b_gpu, b.data(), size * size * sizeof(float), cudaMemcpyHostToDevice));

    cudaGraphExec_t instance;
    trsm_space::set_cuda_graph_trsm(size, block_size, a_gpu, c_gpu, b_gpu, workspace_gpu, &instance);
    trsm_space::launch_cuda_graph_trsm(&instance);

    std::vector<float> c_out_host(size * size);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        c_gpu,
        size * size * sizeof(float),
        cudaMemcpyDeviceToHost));
    
    // need to fix tflops for vector version
    float rel_rmse = 0.0f;
    double tflops = 0.0;
    if (solver == Solver::TRSM_BLOCK || solver == Solver::TRSM_BLOCK_T) {
        tflops = tflops_trsm(size);
        rel_rmse = calc_error_trsm(c_out_host, c, size);
    } else if (solver == Solver::TRSM_VECTOR) {
        tflops = tflops_trsm_vec(size);
        rel_rmse = calc_error_trsm_vector(c_out_host, c, size);
    } else {
        tflops = tflops_trsm(size);
        rel_rmse = calc_error_trsm(c_out_host, c, size);
    }
    
    printf("  %8.02e", rel_rmse);

    // for (int32_t i = 0; i < size; ++i) {
    //     for (int32_t j = 0; j < size; ++j) {
    //         printf("c_out_host[%d][%d] = %f, c[%d][%d] = %f\n", i, j, c_out_host[i * size + j], i, j, c[i * size + j]);
    //     }
    // }

    if (rel_rmse > 1e5) {
        printf("  %9s  %7s", "-", "-");
    } else {
        // SHOULD CHANGE THIS TARGET TIME
        double target_time_ms = 40.0;
        double elapsed_ms = 0.0;
        
        elapsed_ms = benchmark_ms(
            target_time_ms,
            1,
            [&]() {
                if (workspace_size > 0) {
                    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                }
                CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));
            },
            [&]() {
                trsm_space::launch_cuda_graph_trsm(&instance);
            });

        results.elapsed_ms[{size, block_size}] = elapsed_ms;
        //double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
        printf("  %9.02f  %7.05f", elapsed_ms, tflops / (elapsed_ms * 1e-3));
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(flush_gpu));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
    trsm_space::invalidate_cuda_graph_trsm(&instance);
}

template <typename Impl>
void run_config_cholesky(
    Phase phase,
    Solver solver,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size = config.size;
    auto block_size = config.block_size;

    auto const &a = data.a.at({size, block_size});
    auto const &c = data.c.at({size, block_size});

    float *a_gpu;
    float *c_gpu;
    float *b_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_gpu, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_gpu, size * size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size * size * sizeof(float),
        cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(c_gpu, 0, size * size * sizeof(float)));

    size_t workspace_size = Impl::get_workspace_size(size);
    void *workspace_gpu = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
    }

    //need to flush gpu caches before benchmarking each run, might have to change size based on GPU
    void *flush_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_gpu, 1024*1024*64));
    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));

    printf("  %6d  %6d", size, block_size);

    Impl::run(size, block_size, a_gpu, c_gpu, b_gpu, workspace_gpu);

    std::vector<float> c_out_host(size * size);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        c_gpu,
        size * size * sizeof(float),
        cudaMemcpyDeviceToHost));
    
    float rel_rmse = 0.0f;
    double tflops = 0.0;
    if (solver == Solver::CHOLESKY) {
        tflops = tflops_cholesky(size);
        rel_rmse = calc_error_cholesky(c_out_host, c, size);
    } else if (solver == Solver::CHOLESKY_TRIBLOCK) {
        tflops = tflops_triblock(size, block_size);
        rel_rmse = calc_error_cholesky(c_out_host, c, size);
    } else {
        tflops = tflops_cholesky(size);
        rel_rmse = calc_error_cholesky(c_out_host, c, size);
    } 
    // printf("tflops: %f\n", tflops);

    printf("  %8.02e", rel_rmse);

    // for (int32_t i = 0; i < size; ++i) {
    //     for (int32_t j = 0; j < size; ++j) {
    //         printf("c_out_host[%d][%d] = %f, c[%d][%d] = %f\n", i, j, c_out_host[i * size + j], i, j, c[i * size + j]);
    //     }
    // }

    if (rel_rmse > 1e5) {
        printf("  %9s  %7s", "-", "-");
    } else {
        // SHOULD CHANGE THIS TARGET TIME
        double target_time_ms = 40.0;
        double elapsed_ms = 0.0;
        
        elapsed_ms = benchmark_ms(
            target_time_ms,
            1,
            [&]() {
                if (workspace_size > 0) {
                    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                }
                CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));
            },
            [&]() {
                Impl::run(size, block_size, a_gpu, c_gpu, b_gpu, workspace_gpu);
            });

        results.elapsed_ms[{size, block_size}] = elapsed_ms;
        //double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
        printf("  %9.05f  %7.02f", elapsed_ms, tflops / (elapsed_ms * 1e-3));
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(flush_gpu));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
}

template <typename Impl>
void run_config_graph(
    Phase phase,
    Solver solver,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size = config.size;
    auto block_size = config.block_size;

    auto const &a = data.a.at({size, block_size});
    auto const &c = data.c.at({size, block_size});

    float *a_gpu;
    float *c_gpu;
    float *b_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_gpu, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_gpu, size * size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size * size * sizeof(float),
        cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(c_gpu, 0, size * size * sizeof(float)));

    size_t workspace_size = Impl::get_workspace_size(size);
    void *workspace_gpu = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
    }

    //need to flush gpu caches before benchmarking each run, might have to change size based on GPU
    void *flush_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_gpu, 1024*1024*64));
    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));

    printf("  %6d  %6d", size, block_size);

    //cudaGraphExec_t instance;
    // utils::set_cuda_graph_triblock(triblock::launch_triblock_tensor, size, block_size, a_gpu, c_gpu, workspace_gpu, &instance);
    utils::launch_cuda_graph_triblock(triblock::launch_triblock_tensor, size, block_size, a_gpu, c_gpu, workspace_gpu);

    std::vector<float> c_out_host(size * size);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        c_gpu,
        size * size * sizeof(float),
        cudaMemcpyDeviceToHost));
    
    float rel_rmse = 0.0f;
    double tflops = 0.0;
    if (solver == Solver::CHOLESKY) {
        tflops = tflops_cholesky(size);
        rel_rmse = calc_error_cholesky(c_out_host, c, size);
    } else if (solver == Solver::CHOLESKY_TRIBLOCK) {
        tflops = tflops_triblock(size, block_size);
        rel_rmse = calc_error_cholesky(c_out_host, c, size);
    } else {
        tflops = tflops_cholesky(size);
        rel_rmse = calc_error_cholesky(c_out_host, c, size);
    } 

    printf("  %8.02e", rel_rmse);

    // for (int32_t i = 0; i < size; ++i) {
    //     for (int32_t j = 0; j < size; ++j) {
    //         printf("c_out_host[%d][%d] = %f, c[%d][%d] = %f\n", i, j, c_out_host[i * size + j], i, j, c[i * size + j]);
    //     }
    // }

    if (rel_rmse > 1e5) {
        printf("  %9s  %7s", "-", "-");
    } else {
        // SHOULD CHANGE THIS TARGET TIME
        double target_time_ms = 40.0;
        double elapsed_ms = 0.0;
        
        elapsed_ms = benchmark_ms(
            target_time_ms,
            1,
            [&]() {
                if (workspace_size > 0) {
                    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                }
                CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));
            },
            [&]() {
                utils::launch_cuda_graph_triblock(triblock::launch_triblock_tensor, size, block_size, a_gpu, c_gpu, workspace_gpu);
            });

        results.elapsed_ms[{size, block_size}] = elapsed_ms;
        //double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
        printf("  %9.05f  %7.02f", elapsed_ms, tflops / (elapsed_ms * 1e-3));
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(flush_gpu));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
    utils::invalidate_cuda_graph_triblock(&utils::instance);
}



void run_config_cusolver(
    Phase phase,
    Solver solver,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size = config.size;
    auto block_size = config.block_size;

    auto const &a = data.a.at({size, block_size});
    auto const &c = data.c.at({size, block_size});

    float *a_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size * size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size * size * sizeof(float),
        cudaMemcpyHostToDevice));
    
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnParams_t params = NULL;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    int *d_info = nullptr;    /* error info */
    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    void *d_work = nullptr;              /* device workspace */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void *h_work = nullptr;              /* host workspace */

    cusolver_potrf::set_potrf(&cusolverH, &params, 
        &d_info, &workspaceInBytesOnDevice, &d_work, 
        &workspaceInBytesOnHost, &h_work, size, a_gpu, uplo);
    

    //need to flush gpu caches before benchmarking each run, might have to change size based on GPU
    void *flush_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_gpu, 1024*1024*64));
    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));

    printf("  %6d  %6d", size, block_size);

    cusolver_potrf::launch_potrf(size, a_gpu, &cusolverH, 
        &params, uplo, d_info, workspaceInBytesOnDevice, 
        d_work, workspaceInBytesOnHost, h_work);

    std::vector<float> a_out_host(size * size);
    CUDA_CHECK(cudaMemcpy(
        a_out_host.data(),
        a_gpu,
        size * size * sizeof(float),
        cudaMemcpyDeviceToHost));

    double mse = 0.0;
    double ref_mean_square = 0.0;
    double tflops = 0.0;
    float rel_rmse = 0.0;
    // for (int32_t i = 0; i < size; ++i) {
    //     for (int32_t j = 0; j <= i; ++j) {
    //         float diff = a_out_host[i * size + j] - c[i * size + j];
    //         mse += diff * diff;
    //         ref_mean_square += c[i * size + j] * c[i * size + j];
    //     }
    // }
    // mse /= size * size;
    // ref_mean_square /= size * size;
    // float rmse = std::sqrt(mse);
    // float rel_rmse = rmse / std::sqrt(ref_mean_square);
    if (solver == Solver::CHOLESKY) {
        tflops = tflops_cholesky(size);
    } else {
        tflops = tflops_triblock(size, block_size);
    }

    printf("  %8.02e", rel_rmse);

    if (rel_rmse > 1e-5) {
        printf("  %9s  %7s", "-", "-");
    } else {
        // SHOULD CHANGE THIS TARGET TIME
        double target_time_ms = 40.0;
        double elapsed_ms = 0.0;
        
        elapsed_ms = benchmark_ms(
            target_time_ms,
            1,
            [&]() {
                CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));
            },
            [&]() {
                cusolver_potrf::launch_potrf(size, a_gpu, &cusolverH, 
                    &params, uplo, d_info, workspaceInBytesOnDevice, 
                    d_work, workspaceInBytesOnHost, h_work);
            });

        results.elapsed_ms[{size, block_size}] = elapsed_ms;
        //double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
        printf("  %9.02f  %7.05f", elapsed_ms, tflops / (elapsed_ms * 1e-3));
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(flush_gpu));
    cusolver_potrf::destroy_potrf(&cusolverH, &d_info, &d_work, &h_work);
}

void run_config_cublas(
    Phase phase,
    Solver solver,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size = config.size;
    auto block_size = config.block_size;

    auto const &a = data.a.at({size, block_size});
    auto const &c = data.c.at({size, block_size});
    auto const &b = data.b.at({size, block_size});

    float *a_gpu;
    float *b_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_gpu, size * size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size * size * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        b_gpu, b.data(), 
        size * size * sizeof(float), 
        cudaMemcpyHostToDevice));
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    cublasSideMode_t   side  = CUBLAS_SIDE_LEFT;       // X * op(A) = alpha * B (for row-major AX=B)
    cublasFillMode_t   uplo  = CUBLAS_FILL_MODE_UPPER; // cuBLAS sees upper (row-major lower)
    cublasOperation_t  trans = CUBLAS_OP_T;            // no transpose (cuBLAS has A^T, which is what we need)
    cublasDiagType_t   diag  = CUBLAS_DIAG_NON_UNIT;   // diagonal is not assumed to be 1
    
    if (solver == Solver::TRSM_BLOCK) {
        side  = CUBLAS_SIDE_RIGHT;      // X * op(A) = alpha * B (for row-major AX=B)
        uplo  = CUBLAS_FILL_MODE_UPPER; // cuBLAS sees upper (row-major lower)
        trans = CUBLAS_OP_N;            // no transpose (cuBLAS has A^T, which is what we need)
        diag  = CUBLAS_DIAG_NON_UNIT;   // diagonal is not assumed to be 1
    }

    int m = size;      // rows of B
    int k = (solver == Solver::TRSM_BLOCK ? size : 1);      // columns of B (using k to avoid confusion with matrix dimension n)
    int lda = size;    // leading dimension of A
    int ldb = size;    // leading dimension of B

    //need to flush gpu caches before benchmarking each run, might have to change size based on GPU
    void *flush_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_gpu, 1024*1024*64));
    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));

    printf("  %6d  %6d", size, block_size);

    cublasStrsm(
        handle,
        side,           // CUBLAS_SIDE_LEFT: op(A) * X = alpha * B
        uplo,           // CUBLAS_FILL_MODE_UPPER (because row-major lower = col-major upper)
        trans,          // CUBLAS_OP_T (transpose the col-major upper back to row-major lower)
        diag,           // CUBLAS_DIAG_NON_UNIT
        m,              // number of rows of B
        k,              // number of columns of B
        &alpha,         // scalar alpha
        a_gpu,          // device pointer to A
        lda,            // leading dimension of A
        b_gpu,          // device pointer to B
        ldb             // leading dimension of B
    );

    std::vector<float> c_out_host(size * size);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        b_gpu,
        size * size * sizeof(float),
        cudaMemcpyDeviceToHost));

    double rel_rmse = 0.0;
    // if (solver == Solver::TRSM_BLOCK) {
    //     rel_rmse = calc_error_trsm_T(c_out_host, c, size);
    // } else if (solver == Solver::TRSM_VECTOR) {
    //     rel_rmse = calc_error_trsm_vector(c_out_host, c, size);
    // }
    double tflops = tflops_trsm(size);
    if (solver == Solver::TRSM_VECTOR) {
        tflops = tflops_trsm_vec(size);
    }

    printf("  %8.02e", rel_rmse);

    if (rel_rmse > 1e-5) {
        printf("  %9s  %7s", "-", "-");
        for (int i = 0; i < size; ++i) {
            printf("c_out_host[%d] = %8.02f, c[%d] = %8.02f\n", i, c_out_host[i], i, c[i]);
        }
    } else {
        // SHOULD CHANGE THIS TARGET TIME
        double target_time_ms = 40.0;
        double elapsed_ms = 0.0;
        
        elapsed_ms = benchmark_ms(
            target_time_ms,
            1,
            [&]() {
                CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));
            },
            [&]() {
                cublasStrsm(
                    handle,
                    side,           // CUBLAS_SIDE_LEFT: op(A) * X = alpha * B
                    uplo,           // CUBLAS_FILL_MODE_UPPER (because row-major lower = col-major upper)
                    trans,          // CUBLAS_OP_T (transpose the col-major upper back to row-major lower)
                    diag,           // CUBLAS_DIAG_NON_UNIT
                    m,              // number of rows of B
                    k,              // number of columns of B
                    &alpha,         // scalar alpha
                    a_gpu,          // device pointer to A
                    lda,            // leading dimension of A
                    b_gpu,          // device pointer to B
                    ldb);           // leading dimension of B
            });

        results.elapsed_ms[{size, block_size}] = elapsed_ms;
        //double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
        printf("  %9.02f  %7.05f", elapsed_ms, tflops / (elapsed_ms * 1e-3));
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(flush_gpu));
    cublasDestroy(handle);
    
}