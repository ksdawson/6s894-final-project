// TL+ {"compile_flags": ["-lcuda", "-lcublas", "-lcusolver"]}
// TL+ {"header_files": ["utils.cuh", "cholesky.cuh", "trsm.cuh", "gpu_block_kernel_fusion.cuh", "cholesky_small.cuh", "trsm_small.cuh", "gpu_block_enhanced_kernel_fusion.cuh", "gtrsm.cuh", "cusolver.cuh", "cusolver_utils.cuh", "triblock.cuh", "gemm.cuh", "gpu_block_enhanced_deluxe_kernel_fusion.cuh"]}
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

enum class Phase {
    CHOLESKY,
    TRSM,
    CHOLESKY_SMALL,
    TRSM_SMALL,
    ENHANCED_CHOLESKY,
    ENHANCED_DELUXE_CHOLESKY,
    CUSOLVER_POTRF,
    CUBLAS_TRSM,
    TRIBLOCK_SMALL
};

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t>, double> elapsed_ms;
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
    std::map<std::tuple<int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t>, std::vector<float>> b;
    std::map<std::tuple<int32_t>, std::vector<float>> c;
};

std::vector<float> generate_random_matrix(int32_t size) {
    std::vector<float> matrix(size * size);
    for (int32_t i = 0; i < size; ++i) {
        for (int32_t j = 0; j < size; ++j) {
            matrix[i * size + j] = static_cast<float>(rand() % 2 + 1);
            // if (i == j) {
            //     matrix[i * size + j] += size;
            // }
        }
    }
    return matrix;
}

std::vector<float> generate_lower_triangular_matrix(int32_t size) {
    std::vector<float> matrix(size * size);
    for (int32_t i = 0; i < size; ++i) {
        for (int32_t j = 0; j < size; ++j) {
            if (j <= i) {
                matrix[i * size + j] = static_cast<float>(rand() % 2 + 1);
            } else {
                matrix[i * size + j] = 0.0f;
            }

            if (j == i) {
                matrix[i * size + j] += size;
            }
        }
    }
    return matrix;
}

std::vector<float> generate_lower_triblock_matrix(int32_t size, int32_t block_size) {
    auto result = std::vector<float>(size * size);
    for (int32_t i = 0; i < size; ++i) {
        for (int32_t j = 0; j <= i; ++j) {
            if (j >= i - block_size - i % block_size) {
                result[i * size + j] = static_cast<float>(rand() % 2 + 1);
            } else {
                result[i * size + j] = 0.0f;
            }
            
            if (j == i) {
                result[i * size + j] += block_size;
            }
        }
    }
    return result;
}

std::vector<float> chol_generate(std::vector<float> const &matrix, int32_t size) {
    auto result = std::vector<float>(size * size);
    for (int32_t i = 0; i < size; ++i) {
        for (int32_t j = 0; j < size; ++j) {
            result[i * size + j] = 0.0f;
            for (int32_t k = 0; k < size; ++k) {
                result[i * size + j] += matrix[i * size + k] * matrix[j * size + k];
            }
        }
    }
    return result;
}

std::vector<float> trsm_generate(std::vector<float> const &matrix, std::vector<float> const &b, int32_t size) {
    auto result = std::vector<float>(size * size);
    for (int32_t i = 0; i < size; ++i) {
        for (int32_t j = 0; j < size; ++j) {
            result[i * size + j] = 0.0f;
            for (int32_t k = 0; k < size; ++k) {
                result[i * size + j] += matrix[i * size + k] * b[j * size + k];
            }
        }
    }
    return result;
}

TestData generate_test_data(
    std::vector<BenchmarkConfig> const &configs,
    Phase phase) {
    auto data = TestData{};
    for (auto const &config : configs) {
        if (phase == Phase::CHOLESKY || phase == Phase::CHOLESKY_SMALL || phase == Phase::ENHANCED_CHOLESKY || phase == Phase::ENHANCED_DELUXE_CHOLESKY) {
            auto size = config.size;
            data.c[{size}] = generate_lower_triangular_matrix(size);
            data.a[{size}] = chol_generate(data.c[{size}], size);
        } else if (phase == Phase::TRSM || phase == Phase::TRSM_SMALL) {
            auto size = config.size;
            data.a[{size}] = generate_lower_triangular_matrix(size);
            data.c[{size}] = generate_random_matrix(size);
            data.b[{size}] = trsm_generate(data.a[{size}], data.c[{size}], size);
        } else if (phase == Phase::TRIBLOCK_SMALL) {
            auto size = config.size;
            auto block_size = config.block_size;
            data.c[{size}] = generate_lower_triblock_matrix(size, block_size);
            data.a[{size}] = chol_generate(data.c[{size}], size);
        }
    }
    return data;
}

// TestData read_test_data(
//     std::string const &test_data_dir,
//     std::vector<BenchmarkConfig> const &configs) {
//     auto data = TestData{};
//     for (auto const &config : configs) {
//         auto size = config.size;

//         auto path_prefix_a = test_data_dir + "/PDmatrix_";
//         auto path_prefix_c = test_data_dir + "/Cholesky_";

//         if (data.a.find({size}) == data.a.end()) {
//             printf("Reading PDmatrix from %s\n", (path_prefix_a + std::to_string(size) + "x" +
//                     std::to_string(size) + ".bin").c_str());
//             data.a[{size}] = read_data(
//                 path_prefix_a + std::to_string(size) + "x" +
//                     std::to_string(size) + ".bin",
//                 size * size);
//         }

//         if (data.c.find({size}) == data.c.end()) {
//             data.c[{size}] = read_data(
//                 path_prefix_c + std::to_string(size) + "x" +
//                     std::to_string(size) + ".bin",
//                 size * size);
//         }
//     }
//     return data;
// }

float calc_error_cholesky(std::vector<float> const &c_out_host, std::vector<float> const &c, int32_t size) {
    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < size; ++i) {
        for (int32_t j = 0; j <= i; ++j) {
            float diff = c_out_host[i * size + j] - c[i * size + j];
            mse += diff * diff;
            ref_mean_square += c[i * size + j] * c[i * size + j];
        }
    }
    mse /= size * size;
    ref_mean_square /= size * size;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);
    return rel_rmse;
}

float calc_error_trsm(std::vector<float> const &c_out_host, std::vector<float> const &c, int32_t size) {
    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < size; ++i) {
        for (int32_t j = 0; j < size; ++j) {
            float diff = c_out_host[i * size + j] - c[i * size + j];
            mse += diff * diff;
            ref_mean_square += c[i * size + j] * c[i * size + j];
        }
    }
    mse /= size * size;
    ref_mean_square /= size * size;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);
    return rel_rmse;
}

float calc_error_trsm_T(std::vector<float> const &c_out_host, std::vector<float> const &c, int32_t size) {
    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < size; ++i) {
        for (int32_t j = 0; j < size; ++j) {
            float diff = c_out_host[j * size + i] - c[i * size + j];
            mse += diff * diff;
            ref_mean_square += c[i * size + j] * c[i * size + j];
        }
    }
    mse /= size * size;
    ref_mean_square /= size * size;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);
    return rel_rmse;
}

double tflops_cholesky(int32_t size) {
    int32_t num_sqrts = size;
    int32_t num_fma = size * (size-1) * (size+1) / 3;
    int32_t num_divs = size * (size-1) / 2;

    int32_t num_ops = num_sqrts + num_fma + num_divs;
    double tflops = num_ops * 1e-12;
    return tflops;
}

double tflops_trsm(int32_t size) {
    int32_t num_divs = size * size;
    int32_t num_fma = size * size * (size-1);
    int32_t num_ops = num_divs + num_fma;
    double tflops = num_ops * 1e-12;
    return tflops;
}

double tflops_gemm(int32_t size) {
    int32_t num_fma = size * size * size * 2;
    double tflops = num_fma * 1e-12;
    return tflops;
}

double tflops_triblock(int32_t size, int32_t block_size) {
    int32_t num_blocks = (int32_t)(size / block_size);
    double tf_chol = tflops_cholesky(block_size) * num_blocks;
    double tf_trsm = tflops_trsm(block_size) * (num_blocks - 1);
    double tf_GEMMs = tflops_gemm(block_size) * (num_blocks - 1);
    double tflops = tf_chol + tf_trsm + tf_GEMMs;
    return tflops;
}

template <typename Impl>
void run_config(
    Phase phase,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size = config.size;
    auto block_size = config.block_size;

    auto const &a = data.a.at({size});
    auto const &c = data.c.at({size});

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

    printf("  %6d", size);

    if (phase == Phase::TRSM || phase == Phase::TRSM_SMALL) {
        auto const &b = data.b.at({size});
        CUDA_CHECK(cudaMemcpy(b_gpu, b.data(), size * size * sizeof(float), cudaMemcpyHostToDevice));
    }
    Impl::run(size, a_gpu, c_gpu, b_gpu, workspace_gpu);

    std::vector<float> c_out_host(size * size);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        c_gpu,
        size * size * sizeof(float),
        cudaMemcpyDeviceToHost));

    // for (int32_t i = 0; i < size; ++i) {
    //     for (int32_t j = 0; j < size; ++j) {
    //         printf("%8.02f ", c_out_host[i * size + j]);
    //     }
    //     printf("\n");
    // }
    
    
    float rel_rmse = 0.0f;
    double tflops = 0.0;
    if (phase == Phase::CHOLESKY || phase == Phase::CHOLESKY_SMALL || phase == Phase::ENHANCED_CHOLESKY) {
        rel_rmse = calc_error_cholesky(c_out_host, c, size);
        tflops = tflops_cholesky(size);
    } else if (phase == Phase::TRSM_SMALL) {
        rel_rmse = calc_error_trsm(c_out_host, c, size);
        tflops = tflops_trsm(size);
    } else if (phase == Phase::TRSM) {
        rel_rmse = calc_error_trsm_T(c_out_host, c, size);
        tflops = tflops_trsm(size);
    } else if (phase == Phase::TRIBLOCK_SMALL) {
        rel_rmse = calc_error_cholesky(c_out_host, c, size);
        tflops = tflops_triblock(size, block_size);
    }

    printf("  %8.02e", rel_rmse);

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
                Impl::run(size, a_gpu, c_gpu, b_gpu, workspace_gpu);
            });

        results.elapsed_ms[{size}] = elapsed_ms;
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
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size = config.size;

    auto const &a = data.a.at({size});
    auto const &c = data.c.at({size});

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

    printf("  %6d", size);

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
    double tflops = tflops_cholesky(size);

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

        results.elapsed_ms[{size}] = elapsed_ms;
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
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size = config.size;

    auto const &a = data.a.at({size});
    auto const &c = data.c.at({size});
    auto const &b = data.b.at({size});

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

    cublasSideMode_t   side  = CUBLAS_SIDE_RIGHT;      // X * op(A) = alpha * B (for row-major AX=B)
    cublasFillMode_t   uplo  = CUBLAS_FILL_MODE_UPPER; // cuBLAS sees upper (row-major lower)
    cublasOperation_t  trans = CUBLAS_OP_N;            // no transpose (cuBLAS has A^T, which is what we need)
    cublasDiagType_t   diag  = CUBLAS_DIAG_NON_UNIT;   // diagonal is not assumed to be 1

    int m = size;      // rows of B
    int k = size;      // columns of B (using k to avoid confusion with matrix dimension n)
    int lda = size;    // leading dimension of A
    int ldb = size;    // leading dimension of B

    //need to flush gpu caches before benchmarking each run, might have to change size based on GPU
    void *flush_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_gpu, 1024*1024*64));
    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));

    printf("  %6d", size);

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

    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < size; ++i) {
        for (int32_t j = 0; j < size; ++j) {
            float diff = c_out_host[j * size + i] - c[i * size + j];
            mse += diff * diff;
            ref_mean_square += c[i * size + j] * c[i * size + j];
        }
    }
    mse /= size * size;
    ref_mean_square /= size * size;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);
    double tflops = tflops_trsm(size);

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

        results.elapsed_ms[{size}] = elapsed_ms;
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
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{Impl::name};
    
    if (phase == Phase::CUSOLVER_POTRF) {
        printf("CUSOLVER POTRF:\n\n");
    } else if (phase == Phase::CUBLAS_TRSM) {
        printf("CUBLAS TRSM:\n\n");
    } else if (phase == Phase::TRIBLOCK_SMALL) {
        printf("TRIBLOCK SMALL (n = 32):\n\n");
    } else {
        printf("%s:\n\n", Impl::name);
    }

    printf(
        "  %-6s  %-8s  %-9s  %-7s\n",
        "size",
        "RRMSE",
        "time (ms)",
        "TFLOP/s");
    printf(
        "  %-6s  %-8s  %-9s  %-7s\n",
        "------",
        "--------",
        "---------",
        "-------");
    
    if (phase == Phase::CUSOLVER_POTRF) {
        for (auto const &config : configs) {
            run_config_cusolver(phase, data, config, results);
        }
    } else if (phase == Phase::CUBLAS_TRSM) {
        for (auto const &config : configs) {
            run_config_cublas(phase, data, config, results);
        }
    } else {
        for (auto const &config : configs) {
            run_config<Impl>(phase, data, config, results);
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
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        block_cholesky_space::launch_block_cholesky(size, a, c, workspace);
    }
};

struct Trsm {
    constexpr static char const *name = "trsm";

    static size_t get_workspace_size(int32_t size) {
        return trsm_space::get_workspace_size(size);
    }

    static void
    run(int32_t size,
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        trsm_space::launch_trsm(size, a, c, b, workspace);
    }
};

struct CholeskySmall {
    constexpr static char const *name = "cholesky_small";

    static size_t get_workspace_size(int32_t size) {
        return cholesky_small::get_workspace_size(size);
    }

    static void
    run(int32_t size,
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
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        trsm_small::launch_trsm(size, a, c, b, workspace);
    }
};

struct CholeskyEnhanced {
    constexpr static char const *name = "cholesky_enhanced";

    static size_t get_workspace_size(int32_t size) {
        return alt_kernel_fusion::get_workspace_size(size);
    }

    static void
    run(int32_t size,
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
        return triblock_small::get_workspace_size(size);
    }

    static void 
    run(int32_t size,
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        triblock_small::launch_triblock_small(size, a, c, workspace);
    }
};

struct CholeskyEnhancedDeluxe {
    constexpr static char const *name = "cholesky_enhanced_deluxe";

    static size_t get_workspace_size(int32_t size) {
        return deluxe_alt_kernel_fusion::get_workspace_size(size);
    }

    static void
    run(int32_t size,
        float const *a,
        float *c,
        float *b,
        void *workspace) {
        deluxe_alt_kernel_fusion::launch_block_cholesky(size, a, c, workspace);
    }
};

// can add more structs here for other implementations of Cholesky decompositions -- XY

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
    if (phase == Phase::CHOLESKY) {
        results.push_back(run_all_configs<Cholesky>(phase, data, configs));
    } else if (phase == Phase::CHOLESKY_SMALL) {
        results.push_back(run_all_configs<CholeskySmall>(phase, data, configs));
    } else if (phase == Phase::TRSM_SMALL) {
        results.push_back(run_all_configs<TrsmSmall>(phase, data, configs));
    } else if (phase == Phase::TRSM) {
        results.push_back(run_all_configs<Trsm>(phase, data, configs));
    } else if (phase == Phase::ENHANCED_CHOLESKY) {
        results.push_back(run_all_configs<CholeskyEnhanced>(phase, data, configs));
    } else if (phase == Phase::CUSOLVER_POTRF) {
        results.push_back(run_all_configs<Cholesky>(phase, data, configs));
    } else if (phase == Phase::CUBLAS_TRSM) {
        results.push_back(run_all_configs<Trsm>(phase, data, configs));
    } else if (phase == Phase::TRIBLOCK_SMALL) {
        results.push_back(run_all_configs<TriblockSmall>(phase, data, configs));
    } else if (phase == Phase::ENHANCED_DELUXE_CHOLESKY) {
        results.push_back(run_all_configs<CholeskyEnhancedDeluxe>(phase, data, configs));
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

    std::string test_data_dir = ".";
    auto configs = std::vector<BenchmarkConfig>{
        {32, 32},
        {64, 32},
        {128, 32},
        {512, 32},
        {1024, 32},
        // {2048, 32},
        // {4096, 32}
    };
    auto data_cholesky = generate_test_data(configs, Phase::CHOLESKY);
    run_all_impls(Phase::CUSOLVER_POTRF, data_cholesky, configs);
    run_all_impls(Phase::ENHANCED_DELUXE_CHOLESKY, data_cholesky, configs);
    run_all_impls(Phase::ENHANCED_CHOLESKY, data_cholesky, configs);
    run_all_impls(Phase::CHOLESKY, data_cholesky, configs);
    run_all_impls(Phase::CHOLESKY_SMALL, data_cholesky, configs);
    

    auto data_trsm = generate_test_data(configs, Phase::TRSM);
    run_all_impls(Phase::CUBLAS_TRSM, data_trsm, configs);
    run_all_impls(Phase::TRSM_SMALL, data_trsm, configs);
    // run_all_impls(Phase::TRSM, data_trsm, configs);

    auto data_triblock = generate_test_data(configs, Phase::TRIBLOCK_SMALL);
    run_all_impls(Phase::TRIBLOCK_SMALL, data_triblock, configs);
    run_all_impls(Phase::CUSOLVER_POTRF, data_triblock, configs);

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
