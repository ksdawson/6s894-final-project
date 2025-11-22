// TL+ {"compile_flags": ["-lcuda"]}
// TL+ {"header_files": ["utils.cuh", "cholesky_naive.cuh"]}
// TL {"workspace_files": ["PDmatrix_16x3072.bin"]}
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
#include "cholesky_naive.cuh"
#include <filesystem>



std::vector<float> read_data(std::string const &path, int32_t size) {
    //printf("Reading data from %s\n", path.c_str());
    std::ifstream file(path, std::ios::binary);
    //printf("File opened\n");
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

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

struct BenchmarkConfig {
    int32_t size;
};

struct TestData {
    std::map<std::tuple<int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t>, std::vector<float>> c;
};

TestData read_test_data(
    std::string const &test_data_dir,
    std::vector<BenchmarkConfig> const &configs) {
    auto data = TestData{};
    for (auto const &config : configs) {
        auto size = config.size;

        auto path_prefix_a = test_data_dir + "/PDmatrix_";
        auto path_prefix_c = test_data_dir + "/Cholesky_";

        if (data.a.find({size}) == data.a.end()) {
            printf("Reading PDmatrix from %s\n", (path_prefix_a + std::to_string(size) + "x" +
                    std::to_string(size) + ".bin").c_str());
            data.a[{size}] = read_data(
                path_prefix_a + std::to_string(size) + "x" +
                    std::to_string(size) + ".bin",
                size * size);
        }

        if (data.c.find({size}) == data.c.end()) {
            data.c[{size}] = read_data(
                path_prefix_c + std::to_string(size) + "x" +
                    std::to_string(size) + ".bin",
                size * size);
        }
    }
    return data;
}

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t>, double> elapsed_ms;
};

enum class Phase {
    WARMUP,
    BENCHMARK,
};


template <typename Impl>
void run_config(
    Phase phase,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size = config.size;

    auto const &a = data.a.at({size});
    auto const &c = data.c.at({size});

    float *a_gpu;
    float *c_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_gpu, size * size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size * size * sizeof(float),
        cudaMemcpyHostToDevice));

    size_t workspace_size = Impl::get_workspace_size(size);
    void *workspace_gpu = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
    }

    //not really sure what flush_gpu is for, but keeping it here for now -- XY
    void *flush_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_gpu, 1024*1024*64));
    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));

    if (phase == Phase::BENCHMARK) {
        printf("  %6d", size);
    } else {
        printf("  warmup %6d", size);
    }

    Impl::run(size, a_gpu, c_gpu, workspace_gpu);

    std::vector<float> c_out_host(size * size);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        c_gpu,
        size * size * sizeof(float),
        cudaMemcpyDeviceToHost));

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

    if (phase == Phase::BENCHMARK) {
        printf("  %8.02e", rel_rmse);
    }

    if (rel_rmse > 1e-5) {
        if (phase == Phase::BENCHMARK) {
            printf("  %9s  %7s", "-", "-");
        }
    } else {
        // SHOULD CHANGE THIS TARGET TIME
        double target_time_ms = 40.0;
        double elapsed_ms = 0.0;
        if (phase == Phase::BENCHMARK) {
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
                    Impl::run(size, a_gpu, c_gpu, workspace_gpu);
                });
        } else {
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
                    Impl::run(size, a_gpu, c_gpu, workspace_gpu);
                }); 
        }

        if (phase == Phase::BENCHMARK) {
            //double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            //printf("  %9.02f  %7.02f", elapsed_ms, tflop / (elapsed_ms * 1e-3));

            results.elapsed_ms[{size}] = elapsed_ms;
        }
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(flush_gpu));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
}

template <typename Impl>
BenchmarkResults run_all_configs(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{Impl::name};
    if (phase == Phase::WARMUP) {
        printf("warmup %s:\n\n", Impl::name);
    } else {
        printf("%s:\n\n", Impl::name);
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
    }
    for (auto const &config : configs) {
        run_config<Impl>(phase, data, config, results);
    }
    printf("\n");
    return results;
}

struct CholeskyNaive {
    constexpr static char const *name = "cholesky_naive";

    static size_t get_workspace_size(int32_t size) {
        return cholesky_naive::get_workspace_size(size);
    }

    static void
    run(int32_t size,
        float const *a,
        float *c,
        void *workspace) {
        cholesky_naive::launch_cholesky_naive(size, a, c, workspace);
    }
};

// can add more structs here for other implementations of Cholesky decompositions -- XY

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
    results.push_back(run_all_configs<CholeskyNaive>(phase, data, configs));
    return results;
}

void write_json_results(
    std::string const &path,
    std::vector<BenchmarkResults> const &results) {
    auto file = std::ofstream(path);
    file << "{\n";
    for (int32_t i = 0; i < results.size(); ++i) {
        auto const &result = results.at(i);
        file << "  \"" << result.name << "\": [\n";
        int32_t j = 0;
        for (auto const &[config, elapsed_ms] : result.elapsed_ms) {
            auto [size] = config;
            //double tflop = 2.0 * size * size * size * 1e-12;
            //double tflop_per_sec = tflop / (elapsed_ms * 1e-3);
            file << "    {\n";
            file << "      \"size\": " << size << ",\n";
            file << "      \"elapsed_ms\": " << elapsed_ms << ",\n";
            // can calculate tflops later if needed -- XY
            //file << "      \"tflop_per_sec\": " << tflop_per_sec << "\n";
            file << "    }";
            if (j + 1 < result.elapsed_ms.size()) {
                file << ",";
            }
            file << "\n";
            ++j;
        }
        file << "  ]";
        if (i + 1 < results.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "}\n";
}

int main(int argc, char **argv) {
    // std::string current_dir = std::filesystem::current_path().string();
    // std::cout << "Current directory: " << current_dir << std::endl;
    // std::string test_data_dir = ".";
    std::ifstream file("./PDmatrix_16x3072.bin", std::ios::binary);
    std::vector<float> data(16*3072);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << "PDmatrix_16x3072.bin" << std::endl;
        std::abort();
    }

    // std::string test_data_dir = ".";
    // auto configs = std::vector<BenchmarkConfig>{
    //     {64},
    //     // {128},
    //     // {256},
    //     // {512},
    //     // {1024},
    //     // {2048},
    // };
    // auto data = read_test_data(test_data_dir, configs);
    // run_all_impls(Phase::WARMUP, data, configs);
    // auto results = run_all_impls(Phase::BENCHMARK, data, configs);

    // //can compute speedups later if needed -- XY
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
