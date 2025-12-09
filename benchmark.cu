// TL+ {"compile_flags": ["-lcuda", "-lcublas", "-lcusolver"]}
// TL+ {"header_files": ["utils.cuh", "benchmark_helper.cuh", "cholesky.cuh", "trsm.cuh", "gpu_block_kernel_fusion.cuh", "cholesky_small.cuh", "trsm_small.cuh", "gpu_block_enhanced_kernel_fusion.cuh", "gtrsm.cuh", "cusolver.cuh", "cusolver_utils.cuh", "triblock.cuh", "gemm.cuh", "gpu_block_enhanced_deluxe_kernel_fusion.cuh", "triblock_helper.cuh", "gpu_block_enhanced_deluxe_premium_kernel_fusion.cuh"]}
// TL+ {"workspace_files": []}

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

// #define CUDA_CHECK(x) \
//   do { \
//       utils::cuda_check((x), __FILE__, __LINE__); \
//   } while (0)

// std::vector<float> read_data(std::string const &path, int32_t size) {
//     //printf("Reading data from %s\n", path.c_str());
//     std::ifstream file(path, std::ios::binary);
//     //printf("File opened\n");
//     std::vector<float> data(size);
//     file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
//     if (file.fail()) {
//         std::cerr << "Failed to read " << path << std::endl;
//         std::abort();
//     }
//     return data;
// }

enum class Solver {
    TRSM_VECTOR,
    TRSM_BLOCK,
    TRSM_BLOCK_T,
    CHOLESKY,
    CHOLESKY_TRIBLOCK
};

enum class Phase {
    CHOLESKY,
    TRSM,
    CHOLESKY_SMALL,
    TRSM_SMALL,
    TRSM_BLOCK,
    ENHANCED_CHOLESKY,
    ENHANCED_DELUXE_CHOLESKY,
    ENHANCED_DELUXE_PREMIUM_CHOLESKY,
    CUSOLVER_POTRF,
    CUBLAS_TRSM,
    TRIBLOCK_SMALL,
    TRIBLOCK
};

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t, int32_t>, double> elapsed_ms;
};

struct BenchmarkConfig {
    int32_t size;
    int32_t block_size;
};

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

struct TestData {
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> b;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> c;
};

TestData generate_test_data(
    std::vector<BenchmarkConfig> const &configs,
    Phase phase,
    Solver solver) {
    auto data = TestData{};
    
    for (auto const &config : configs) {
        if (solver == Solver::CHOLESKY) {
            auto size = config.size;
            auto block_size = config.block_size;
            data.c[{size, block_size}] = generate_lower_triangular_matrix(size);
            data.a[{size, block_size}] = chol_generate(data.c[{size, block_size}], size);
        } else if (solver == Solver::TRSM_BLOCK) {
            auto size = config.size;
            auto block_size = config.block_size;
            data.a[{size, block_size}] = generate_lower_triangular_matrix(size);
            data.c[{size, block_size}] = generate_random_matrix(size);
            data.b[{size, block_size}] = trsm_generate(data.a[{size, block_size}], data.c[{size, block_size}], size);
        } else if (solver == Solver::TRSM_BLOCK_T) {
            auto size = config.size;
            auto block_size = config.block_size;
            data.a[{size, block_size}] = generate_lower_triangular_matrix(size);
            data.c[{size, block_size}] = generate_random_matrix(size);
            data.b[{size, block_size}] = trsm_generate_T(data.a[{size, block_size}], data.c[{size, block_size}], size);
        } else if (solver == Solver::TRSM_VECTOR) {
            auto size = config.size;
            auto block_size = config.block_size;
            data.a[{size, block_size}] = generate_lower_triangular_matrix(size);
            data.c[{size, block_size}] = generate_random_vector(size);
            data.b[{size, block_size}] = trsm_vector_generate(data.a[{size, block_size}], data.c[{size, block_size}], size);
        } else if (solver == Solver::CHOLESKY_TRIBLOCK) { 
            auto size = config.size;
            auto block_size = config.block_size;           
            data.c[{size, block_size}] = generate_lower_triblock_matrix(size, block_size);
            data.a[{size, block_size}] = chol_generate(data.c[{size, block_size}], size);
        }
    }
    return data;
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

    cudaGraphExec_t instance = nullptr;
    utils::set_cuda_graph_triblock(triblock::launch_triblock, size, block_size, a_gpu, c_gpu, workspace_gpu, &instance);
    utils::launch_cuda_graph_triblock(&instance);

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
                utils::launch_cuda_graph_triblock(&instance);
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
    utils::invalidate_cuda_graph_triblock(&instance);
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
        tflops = tflops_trsm(size);
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
        printf("  %9.02f  %7.02f", elapsed_ms, tflops / (elapsed_ms * 1e-3));
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
    for (int32_t i = 0; i < size; ++i) {
        for (int32_t j = 0; j <= i; ++j) {
            float diff = a_out_host[i * size + j] - c[i * size + j];
            mse += diff * diff;
            ref_mean_square += c[i * size + j] * c[i * size + j];
        }
    }
    mse /= size * size;
    ref_mean_square /= size * size;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);
    if (size == block_size) {
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
        printf("  %9.02f  %7.02f", elapsed_ms, tflops / (elapsed_ms * 1e-3));
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
    if (solver == Solver::TRSM_BLOCK) {
        rel_rmse = calc_error_trsm_T(c_out_host, c, size);
    } else if (solver == Solver::TRSM_VECTOR) {
        rel_rmse = calc_error_trsm_vector(c_out_host, c, size);
    }
    double tflops = tflops_trsm(size);

    printf("  %8.02e", rel_rmse);

    if (rel_rmse > 1e-5) {
        printf("  %9s  %7s", "-", "-");
        for (int i = 0; i < size; ++i) {
            printf("c_out_host[%d] = %8.02f, c[%d] = %8.02f\n", i, c_out_host[i], i, c[i]);
        }
    } else {
        // SHOULD CHANGE THIS TARGET TIME
        double target_time_ms = 80.0;
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
        printf("  %9.02f  %7.02f", elapsed_ms, tflops / (elapsed_ms * 1e-3));
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(flush_gpu));
    cublasDestroy(handle);
    
}

template <typename Impl>
BenchmarkResults run_all_configs(
    Phase phase,
    Solver solver,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{Impl::name};
    
    if (phase == Phase::CUSOLVER_POTRF) {
        printf("CUSOLVER POTRF:\n\n");
    } else if (phase == Phase::CUBLAS_TRSM) {
        printf("CUBLAS TRSM:\n\n");
    } else {
        printf("%s:\n\n", Impl::name);
    }

    printf(
        "  %-6s  %-8s  %-8s  %-9s  %-7s\n",
        "size N",
        "size n",
        "RRMSE",
        "time (ms)",
        "TFLOP/s");
    printf(
        "  %-6s  %-8s  %-8s  %-9s  %-7s\n",
        "------",
        "--------",
        "--------",
        "---------",
        "-------");
    
    if (phase == Phase::CUSOLVER_POTRF) {
        for (auto const &config : configs) {
            run_config_cusolver(phase, solver, data, config, results);
        }
    } else if (phase == Phase::CUBLAS_TRSM) {
        for (auto const &config : configs) {
            run_config_cublas(phase, solver, data, config, results);
        }
    } else if (solver == Solver::TRSM_VECTOR || solver == Solver::TRSM_BLOCK || solver == Solver::TRSM_BLOCK_T) {
        for (auto const &config : configs) {
            run_config_trsm<Impl>(phase, solver, data, config, results);
        }
    } else if (phase == Phase::TRIBLOCK) {
        for (auto const &config : configs) {
            run_config_cholesky<Impl>(phase, solver, data, config, results);
        }
    } else {
        for (auto const &config : configs) {
            run_config_cholesky<Impl>(phase, solver, data, config, results);
        }
    }
    printf("\n");
    return results;
}

struct Cholesky {
    constexpr static char const *name = "cholesky";

    static size_t get_workspace_size(int32_t size) {
        return block_cholesky_space::get_workspace_size(size);
    }

    static void
    run(int32_t size,
        int32_t block_size,
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        block_cholesky_space::launch_block_cholesky(size, a, c, workspace);
    }
};

struct TrsmBlock {
    constexpr static char const *name = "trsm_block";
    
    static size_t get_workspace_size(int32_t size) {
        return triblock::get_workspace_size(size);
    }
    
    static void
    run(int32_t size,
        int32_t r,
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        triblock_helper::launch_triblock_block_trsm(size, r, a, c, b, workspace);
    }
};

struct Trsm {
    constexpr static char const *name = "trsm_cuda_graph";

    static size_t get_workspace_size(int32_t size) {
        return trsm_space::get_workspace_size(size);
    }

    static void
    run(int32_t size,
        int32_t r,
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        trsm_space::launch_trsm(size, r, a, c, b, workspace);
    }
};

struct CholeskySmall {
    constexpr static char const *name = "cholesky_small";

    static size_t get_workspace_size(int32_t size) {
        return cholesky_small::get_workspace_size(size);
    }

    static void
    run(int32_t size,
        int32_t block_size,
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        cholesky_small::launch_cholesky(size, a, c, workspace);
    }
};

struct TrsmSmall {
    constexpr static char const *name = "trsm_small";
    
    static size_t get_workspace_size(int32_t size) {
        return trsm_small::get_workspace_size(size);
    }

    static void
    run(int32_t size,
        int32_t r,
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        trsm_small::launch_trsm(size, r, a, c, b, workspace);
    }
};

struct CholeskyEnhanced {
    constexpr static char const *name = "cholesky_enhanced";

    static size_t get_workspace_size(int32_t size) {
        return alt_kernel_fusion::get_workspace_size(size);
    }

    static void
    run(int32_t size,
        int32_t block_size,
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        alt_kernel_fusion::launch_block_cholesky(size, a, c, workspace);
    }
};

struct TriblockSmall {
    constexpr static char const *name = "triblock_small";
    
    static size_t get_workspace_size(int32_t size) {
        return triblock::get_workspace_size(size);
    }

    static void 
    run(int32_t size,
        int32_t block_size,
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        
        triblock_small::launch_triblock_small(size, block_size, a, c, workspace);
    }
};

struct CholeskyEnhancedDeluxe {
    constexpr static char const *name = "cholesky_enhanced_deluxe";

    static size_t get_workspace_size(int32_t size) {
        return deluxe_alt_kernel_fusion::get_workspace_size(size);
    }

    static void
    run(int32_t size,
        int32_t block_size,
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        deluxe_alt_kernel_fusion::launch_block_cholesky(size, a, c, workspace);
    }
};

struct Triblock {
    constexpr static char const *name = "triblock";
    
    static size_t get_workspace_size(int32_t size) {
        return triblock::get_workspace_size(size);
    }
    
    static void
    run(int32_t size,
        int32_t block_size,
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        triblock::launch_triblock(size, block_size, a, c, workspace);
        //utils::launch_cuda_graph_triblock(triblock::launch_triblock, size, block_size, a, c, workspace);
    }
};

struct CholeskyEnhancedDeluxePremium {
    constexpr static char const *name = "cholesky_enhanced_deluxe_premium";

    static size_t get_workspace_size(int32_t size) {
        return prem_deluxe_alt_kernel_fusion::get_workspace_size(size);
    }

    static void
    run(int32_t size,
        int32_t block_size,
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        prem_deluxe_alt_kernel_fusion::launch_block_cholesky(size, a, c, workspace);
    }
};

// can add more structs here for other implementations of Cholesky decompositions -- XY

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    Solver solver,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
    if (phase == Phase::CHOLESKY) {
        results.push_back(run_all_configs<Cholesky>(phase, solver, data, configs));
    } else if (phase == Phase::CHOLESKY_SMALL) {
        results.push_back(run_all_configs<CholeskySmall>(phase, solver, data, configs));
    } else if (phase == Phase::TRSM_SMALL) {
        results.push_back(run_all_configs<TrsmSmall>(phase, solver, data, configs));
    } else if (phase == Phase::TRSM) {
        results.push_back(run_all_configs<Trsm>(phase, solver, data, configs));
    } else if (phase == Phase::ENHANCED_CHOLESKY) {
        results.push_back(run_all_configs<CholeskyEnhanced>(phase, solver, data, configs));
    } else if (phase == Phase::CUSOLVER_POTRF) {
        results.push_back(run_all_configs<Cholesky>(phase, solver, data, configs));
    } else if (phase == Phase::CUBLAS_TRSM) {
        results.push_back(run_all_configs<Trsm>(phase, solver, data, configs));
    } else if (phase == Phase::TRIBLOCK_SMALL) {
        results.push_back(run_all_configs<TriblockSmall>(phase, solver, data, configs));
    } else if (phase == Phase::ENHANCED_DELUXE_CHOLESKY) {
        results.push_back(run_all_configs<CholeskyEnhancedDeluxe>(phase, solver, data, configs));
    } else if (phase == Phase::TRIBLOCK) {
        results.push_back(run_all_configs<Triblock>(phase, solver, data, configs));
    } else if (phase == Phase::ENHANCED_DELUXE_PREMIUM_CHOLESKY) {
        results.push_back(run_all_configs<CholeskyEnhancedDeluxePremium>(phase, solver, data, configs));
    } else if (phase == Phase::TRSM_BLOCK) {
        results.push_back(run_all_configs<TrsmBlock>(phase, solver, data, configs));
    }
    return results;
}

// void write_json_results(
//     std::string const &path,
//     std::vector<BenchmarkResults> const &results) {
//     auto file = std::ofstream(path);
//     file << "{\n";
//     for (int32_t i = 0; i < results.size(); ++i) {
//         auto const &result = results.at(i);
//         file << "  \"" << result.name << "\": [\n";
//         int32_t j = 0;
//         for (auto const &[config, elapsed_ms] : result.elapsed_ms) {
//             auto [size] = config;
//             //double tflop = 2.0 * size * size * size * 1e-12;
//             //double tflop_per_sec = tflop / (elapsed_ms * 1e-3);
//             file << "    {\n";
//             file << "      \"size\": " << size << ",\n";
//             file << "      \"elapsed_ms\": " << elapsed_ms << ",\n";
//             // can calculate tflops later if needed -- XY
//             //file << "      \"tflop_per_sec\": " << tflop_per_sec << "\n";
//             file << "    }";
//             if (j + 1 < result.elapsed_ms.size()) {
//                 file << ",";
//             }
//             file << "\n";
//             ++j;
//         }
//         file << "  ]";
//         if (i + 1 < results.size()) {
//             file << ",";
//         }
//         file << "\n";
//     }
//     file << "}\n";
// }

int main(int argc, char **argv) {

    auto configs = std::vector<BenchmarkConfig>{
        {32, 32},
        {64, 64},
        {128, 128},
        {512, 512},
        {1024, 1024},
        {2048, 2048}
    };
    auto data_cholesky = generate_test_data(configs, Phase::CHOLESKY, Solver::CHOLESKY);
    run_all_impls(Phase::CUSOLVER_POTRF, Solver::CHOLESKY, data_cholesky, configs);
    // run_all_impls(Phase::ENHANCED_DELUXE_PREMIUM_CHOLESKY, Solver::CHOLESKY, data_cholesky, configs);
    // run_all_impls(Phase::ENHANCED_DELUXE_CHOLESKY, Solver::CHOLESKY, data_cholesky, configs);
    // run_all_impls(Phase::ENHANCED_CHOLESKY, Solver::CHOLESKY, data_cholesky, configs);
    // run_all_impls(Phase::CHOLESKY, data_cholesky, configs);
    // run_all_impls(Phase::CHOLESKY_SMALL, data_cholesky, configs);
    

    auto configs_trsm = std::vector<BenchmarkConfig> {
        {32, 32},
        {64, 64},
        {128, 128},
        {512, 512},
        {1024, 1024}
    };

    auto data_trsm = generate_test_data(configs_trsm, Phase::TRSM, Solver::TRSM_BLOCK);
    run_all_impls(Phase::CUBLAS_TRSM, Solver::TRSM_BLOCK, data_trsm, configs_trsm);
    //run_all_impls(Phase::TRSM_SMALL, Solver::TRSM_BLOCK, data_trsm, configs_trsm);

    // auto configs_trsm_T = std::vector<BenchmarkConfig> {
    //     {32, 32},
    //     {64, 64},
    //     {128, 128},
    //     {512, 512},
    //     {1024, 1024}
    // };
    // auto data_trsm_T = generate_test_data(configs_trsm_T, Phase::TRSM, Solver::TRSM_BLOCK_T);
    // run_all_impls(Phase::TRSM, Solver::TRSM_BLOCK_T, data_trsm_T, configs_trsm_T);
    // run_all_impls(Phase::TRSM_BLOCK, Solver::TRSM_BLOCK_T, data_trsm_T, configs_trsm_T);

    auto configs_trsmvec = std::vector<BenchmarkConfig> {
        {32, 1},
        {64, 1},
        {128, 1},
        {512, 1},
        {1024, 1}

    };
    auto data_trsmvec = generate_test_data(configs_trsmvec, Phase::TRSM, Solver::TRSM_VECTOR);
    run_all_impls(Phase::CUBLAS_TRSM, Solver::TRSM_VECTOR, data_trsmvec, configs_trsmvec);
    // run_all_impls(Phase::TRSM_SMALL, Solver::TRSM_VECTOR, data_trsmvec, configs_trsmvec);
    // run_all_impls(Phase::TRSM, Solver::TRSM_VECTOR, data_trsmvec, configs_trsmvec);

    // auto configs_triblock = std::vector<BenchmarkConfig>{
    //     {1024, 32},
    //     {1024, 64},
    //     {1024, 128},
    //     {1024, 256},
    //     {1024, 512},
    //     {1024, 1024}
    // };
    // auto data_triblock = generate_test_data(configs_triblock, Phase::TRIBLOCK_SMALL, Solver::CHOLESKY_TRIBLOCK);
    // run_all_impls(Phase::TRIBLOCK_SMALL, Solver::CHOLESKY_TRIBLOCK, data_triblock, configs_triblock);
    // run_all_impls(Phase::TRIBLOCK, Solver::CHOLESKY_TRIBLOCK, data_triblock, configs_triblock);
    // run_all_impls(Phase::CUSOLVER_POTRF, Solver::CHOLESKY_TRIBLOCK, data_triblock, configs_triblock);
    // run_all_impls(Phase::ENHANCED_DELUXE_CHOLESKY, Solver::CHOLESKY_TRIBLOCK, data_triblock, configs_triblock);

    //can compute speedups later if needed -- XY
    // for (int32_t j = 1; j < results.size(); ++j) {
    //     for (int32_t i = j; i > 0;) {
    //         --i;
    //         auto const &first = results.at(i);
    //         auto const &second = results.at(j);
    //         printf("\nspeedups %s -> %s:\n\n", first.name, second.name);
    //         printf("  %-6s  %-6s  %-6s  %-7s\n", "size_i", "size_j", "size_k", "speedup");
    //         printf("  %-6s  %-6s  %-6s  %-7s\n", "------", "------", "------", "-------");
    //         for (auto const &config : configs) {
    //             auto size_i = config.size_i;
    //             auto size_j = config.size_j;
    //             auto size_k = config.size_k;
    //             printf("  %6d  %6d  %6d", size_i, size_j, size_k);
    //             auto it_first = first.elapsed_ms.find({size_i, size_j, size_k});
    //             auto it_second = second.elapsed_ms.find({size_i, size_j, size_k});
    //             if (it_first != first.elapsed_ms.end() &&
    //                 it_second != second.elapsed_ms.end()) {
    //                 printf("  %6.02fx", it_first->second / it_second->second);
    //             } else {
    //                 printf("  %7s", "-");
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    // write_json_results("out/results.json", results);

    return 0;
}
