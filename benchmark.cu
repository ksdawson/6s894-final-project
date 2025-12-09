// TL+ {"compile_flags": ["-lcuda", "-lcublas", "-lcusolver"]}
// TL+ {"header_files": ["utils.cuh", "benchmark_helper.cuh", "benchmark_configs.cuh", "cholesky.cuh", "trsm.cuh", "gpu_block_kernel_fusion.cuh", "cholesky_small.cuh", "trsm_small.cuh", "gpu_block_enhanced_kernel_fusion.cuh", "gtrsm.cuh", "cusolver.cuh", "cusolver_utils.cuh", "triblock.cuh", "gemm.cuh", "gpu_block_enhanced_deluxe_kernel_fusion.cuh", "triblock_helper.cuh", "gpu_block_enhanced_deluxe_premium_kernel_fusion.cuh"]} 
// TL+ {"workspace_files": []}

#include "benchmark_configs.cuh"
#include "benchmark_helper.cuh"
#include "cholesky.cuh"
#include "cholesky_small.cuh"
#include "cusolver.cuh"
#include "cusolver_utils.cuh"
#include "gemm.cuh"
#include "gpu_block_enhanced_deluxe_kernel_fusion.cuh"
#include "gpu_block_enhanced_deluxe_premium_kernel_fusion.cuh"
#include "gpu_block_enhanced_kernel_fusion.cuh"
#include "gpu_block_kernel_fusion.cuh"
#include "gtrsm.cuh"
#include "triblock.cuh"
#include "triblock_helper.cuh"
#include "trsm_small.cuh"
#include "utils.cuh"
#include <chrono>
#include <cstdlib>
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

// #define CUDA_CHECK(x) \
//   do { \
//       utils::cuda_check((x), __FILE__, __LINE__); \
//   } while (0)

// std::vector<float> read_data(std::string const &path, int32_t size) {
//     //printf("Reading data from %s\n", path.c_str());
//     std::ifstream file(path, std::ios::binary);
//     //printf("File opened\n");
//     std::vector<float> data(size);
//     file.read(reinterpret_cast<char *>(data.data()), data.size() *
//     sizeof(float)); if (file.fail()) {
//         std::cerr << "Failed to read " << path << std::endl;
//         std::abort();
//     }
//     return data;
// }

TestData generate_test_data(std::vector<BenchmarkConfig> const &configs,
                            Phase phase, Solver solver) {
  auto data = TestData{};

  for (auto const &config : configs) {
    if (solver == Solver::CHOLESKY) {
      auto size = config.size;
      auto block_size = config.block_size;
      data.c[{size, block_size}] = generate_lower_triangular_matrix(size);
      data.a[{size, block_size}] =
          chol_generate(data.c[{size, block_size}], size);
    } else if (solver == Solver::TRSM_BLOCK) {
      auto size = config.size;
      auto block_size = config.block_size;
      data.a[{size, block_size}] = generate_lower_triangular_matrix(size);
      data.c[{size, block_size}] = generate_random_matrix(size);
      data.b[{size, block_size}] = trsm_generate(
          data.a[{size, block_size}], data.c[{size, block_size}], size);
    } else if (solver == Solver::TRSM_BLOCK_T) {
      auto size = config.size;
      auto block_size = config.block_size;
      data.a[{size, block_size}] = generate_lower_triangular_matrix(size);
      data.c[{size, block_size}] = generate_random_matrix(size);
      data.b[{size, block_size}] = trsm_generate_T(
          data.a[{size, block_size}], data.c[{size, block_size}], size);
    } else if (solver == Solver::TRSM_VECTOR) {
      auto size = config.size;
      auto block_size = config.block_size;
      data.a[{size, block_size}] = generate_lower_triangular_matrix(size);
      data.c[{size, block_size}] = generate_random_vector(size);
      data.b[{size, block_size}] = trsm_vector_generate(
          data.a[{size, block_size}], data.c[{size, block_size}], size);
    } else if (solver == Solver::CHOLESKY_TRIBLOCK) {
      auto size = config.size;
      auto block_size = config.block_size;
      data.c[{size, block_size}] =
          generate_lower_triblock_matrix(size, block_size);
      data.a[{size, block_size}] =
          chol_generate(data.c[{size, block_size}], size);
    }
  }
  return data;
}

template <typename Impl>
BenchmarkResults run_all_configs(Phase phase, Solver solver,
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

  printf("  %-6s  %-8s  %-8s  %-9s  %-7s\n", "size N", "size n", "RRMSE",
         "time (ms)", "TFLOP/s");
  printf("  %-6s  %-8s  %-8s  %-9s  %-7s\n", "------", "--------", "--------",
         "---------", "-------");

  if (phase == Phase::CUSOLVER_POTRF) {
    for (auto const &config : configs) {
      run_config_cusolver(phase, solver, data, config, results);
    }
  } else if (phase == Phase::CUBLAS_TRSM) {
    for (auto const &config : configs) {
      run_config_cublas(phase, solver, data, config, results);
    }
  } else if (phase == Phase::TRSM) {
    for (auto const &config : configs) {
      run_config_trsm_graph<Impl>(phase, solver, data, config, results);
    }
  } else if (solver == Solver::TRSM_VECTOR || solver == Solver::TRSM_BLOCK ||
             solver == Solver::TRSM_BLOCK_T) {
    for (auto const &config : configs) {
      run_config_trsm<Impl>(phase, solver, data, config, results);
    }
  } else if (phase == Phase::TRIBLOCK_TENSOR_GRAPH) {
    for (auto const &config : configs) {
      run_config_graph<Impl>(phase, solver, data, config, results);
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

  static void run(int32_t size, int32_t block_size, float const *a, float *c,
                  float *b, void *workspace) {
    block_cholesky_space::launch_block_cholesky(size, a, c, workspace);
  }
};

struct TrsmBlock {
  constexpr static char const *name = "trsm_block";

  static size_t get_workspace_size(int32_t size) {
    return triblock::get_workspace_size(size);
  }

  static void run(int32_t size, int32_t r, float const *a, float *c, float *b,
                  void *workspace) {
    triblock_helper::launch_triblock_block_trsm(size, r, a, c, b, workspace);
  }
};

struct Trsm {
  constexpr static char const *name = "trsm_cuda_graph";

  static size_t get_workspace_size(int32_t size) {
    return trsm_space::get_workspace_size(size);
  }

  static void run(int32_t size, int32_t r, float const *a, float *c, float *b,
                  void *workspace) {
    trsm_space::launch_trsm(size, r, a, c, b, workspace);
  }
};

struct CholeskySmall {
  constexpr static char const *name = "cholesky_small";

  static size_t get_workspace_size(int32_t size) {
    return cholesky_small::get_workspace_size(size);
  }

  static void run(int32_t size, int32_t block_size, float const *a, float *c,
                  float *b, void *workspace) {
    cholesky_small::launch_cholesky(size, a, c, workspace);
  }
};

struct TrsmSmall {
  constexpr static char const *name = "trsm_small";

  static size_t get_workspace_size(int32_t size) {
    return trsm_small::get_workspace_size(size);
  }

  static void run(int32_t size, int32_t r, float const *a, float *c, float *b,
                  void *workspace) {
    trsm_small::launch_trsm(size, r, a, c, b, workspace);
  }
};

struct CholeskyEnhanced {
  constexpr static char const *name = "cholesky_enhanced";

  static size_t get_workspace_size(int32_t size) {
    return alt_kernel_fusion::get_workspace_size(size);
  }

  static void run(int32_t size, int32_t block_size, float const *a, float *c,
                  float *b, void *workspace) {
    alt_kernel_fusion::launch_block_cholesky(size, a, c, workspace);
  }
};

struct TriblockSmall {
  constexpr static char const *name = "triblock_small";

  static size_t get_workspace_size(int32_t size) {
    return triblock::get_workspace_size(size);
  }

  static void run(int32_t size, int32_t block_size, float const *a, float *c,
                  float *b, void *workspace) {

    triblock_small::launch_triblock_small(size, block_size, a, c, workspace);
  }
};

struct CholeskyEnhancedDeluxe {
  constexpr static char const *name = "cholesky_enhanced_deluxe";

  static size_t get_workspace_size(int32_t size) {
    return deluxe_alt_kernel_fusion::get_workspace_size(size);
  }

  static void run(int32_t size, int32_t block_size, float const *a, float *c,
                  float *b, void *workspace) {
    deluxe_alt_kernel_fusion::launch_block_cholesky(size, a, c, workspace);
  }
};

struct Triblock {
  constexpr static char const *name = "triblock";

  static size_t get_workspace_size(int32_t size) {
    return triblock::get_workspace_size(size);
  }

  static void run(int32_t size, int32_t block_size, float const *a, float *c,
                  float *b, void *workspace) {
    triblock::launch_triblock(size, block_size, a, c, workspace);
    // utils::launch_cuda_graph_triblock(triblock::launch_triblock, size,
    // block_size, a, c, workspace);
  }
};

struct CholeskyEnhancedDeluxePremium {
  constexpr static char const *name = "cholesky_enhanced_deluxe_premium";

  static size_t get_workspace_size(int32_t size) {
    return prem_deluxe_alt_kernel_fusion::get_workspace_size(size);
  }

  static void run(int32_t size, int32_t block_size, float const *a, float *c,
                  float *b, void *workspace) {
    prem_deluxe_alt_kernel_fusion::launch_block_cholesky(size, a, c, workspace);
  }
};

struct TriblockTensor {
  constexpr static char const *name = "triblock_tensor";

  static size_t get_workspace_size(int32_t size) {
    return triblock::get_workspace_size(size);
  }

  static void run(int32_t size, int32_t block_size, float const *a, float *c,
                  float *b, void *workspace) {
    triblock::launch_triblock_tensor(size, block_size, a, c, workspace);
  }
};

struct TriblockTensorGraph {
  constexpr static char const *name = "triblock_tensor_graph";

  static size_t get_workspace_size(int32_t size) {
    return triblock::get_workspace_size(size);
  }

  static void run(int32_t size, int32_t block_size, float const *a, float *c,
                  float *b, void *workspace) {
    triblock::launch_triblock_tensor(size, block_size, a, c, workspace);
  }
};
// can add more structs here for other implementations of Cholesky
// decompositions -- XY

std::vector<BenchmarkResults>
run_all_impls(Phase phase, Solver solver, TestData const &data,
              std::vector<BenchmarkConfig> const &configs) {
  auto results = std::vector<BenchmarkResults>{};
  if (phase == Phase::CHOLESKY) {
    results.push_back(run_all_configs<Cholesky>(phase, solver, data, configs));
  } else if (phase == Phase::CHOLESKY_SMALL) {
    results.push_back(
        run_all_configs<CholeskySmall>(phase, solver, data, configs));
  } else if (phase == Phase::TRSM_SMALL) {
    results.push_back(run_all_configs<TrsmSmall>(phase, solver, data, configs));
  } else if (phase == Phase::TRSM) {
    results.push_back(run_all_configs<Trsm>(phase, solver, data, configs));
  } else if (phase == Phase::ENHANCED_CHOLESKY) {
    results.push_back(
        run_all_configs<CholeskyEnhanced>(phase, solver, data, configs));
  } else if (phase == Phase::CUSOLVER_POTRF) {
    results.push_back(run_all_configs<Cholesky>(phase, solver, data, configs));
  } else if (phase == Phase::CUBLAS_TRSM) {
    results.push_back(run_all_configs<Trsm>(phase, solver, data, configs));
  } else if (phase == Phase::TRIBLOCK_SMALL) {
    results.push_back(
        run_all_configs<TriblockSmall>(phase, solver, data, configs));
  } else if (phase == Phase::ENHANCED_DELUXE_CHOLESKY) {
    results.push_back(
        run_all_configs<CholeskyEnhancedDeluxe>(phase, solver, data, configs));
  } else if (phase == Phase::TRIBLOCK) {
    results.push_back(run_all_configs<Triblock>(phase, solver, data, configs));
  } else if (phase == Phase::ENHANCED_DELUXE_PREMIUM_CHOLESKY) {
    results.push_back(run_all_configs<CholeskyEnhancedDeluxePremium>(
        phase, solver, data, configs));
  } else if (phase == Phase::TRSM_BLOCK) {
    results.push_back(run_all_configs<TrsmBlock>(phase, solver, data, configs));
  } else if (phase == Phase::TRIBLOCK_TENSOR) {
    results.push_back(
        run_all_configs<TriblockTensor>(phase, solver, data, configs));
  } else if (phase == Phase::TRIBLOCK_TENSOR_GRAPH) {
    results.push_back(
        run_all_configs<TriblockTensorGraph>(phase, solver, data, configs));
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
//   auto configs = std::vector<BenchmarkConfig>{
//       {32, 32},     {64, 64},     {128, 128},  {512, 512},
//       {1024, 1024}, {2048, 2048}, {4096, 4096}};
//   auto data_cholesky =
//       generate_test_data(configs, Phase::CHOLESKY, Solver::CHOLESKY);
//   run_all_impls(Phase::CUSOLVER_POTRF, Solver::CHOLESKY, data_cholesky,
//                 configs);
//   run_all_impls(Phase::ENHANCED_DELUXE_PREMIUM_CHOLESKY, Solver::CHOLESKY,
//                 data_cholesky, configs);
//   run_all_impls(Phase::ENHANCED_DELUXE_CHOLESKY, Solver::CHOLESKY,
//                 data_cholesky, configs);
//   run_all_impls(Phase::ENHANCED_CHOLESKY, Solver::CHOLESKY, data_cholesky,
//                 configs);
//   run_all_impls(Phase::CHOLESKY, Solver::CHOLESKY, data_cholesky, configs);
//   run_all_impls(Phase::CHOLESKY_SMALL, Solver::CHOLESKY, data_cholesky,
//                 configs);

//   auto configs_trsm = std::vector<BenchmarkConfig>{
//       {32, 32},     {64, 64},     {128, 128},  {512, 512},
//       {1024, 1024}, {2048, 2048}, {4096, 4096}};

//   auto data_trsm =
//       generate_test_data(configs_trsm, Phase::TRSM, Solver::TRSM_BLOCK);
//   run_all_impls(Phase::CUBLAS_TRSM, Solver::TRSM_BLOCK, data_trsm,
//                 configs_trsm);
//   run_all_impls(Phase::TRSM_SMALL, Solver::TRSM_BLOCK, data_trsm, configs_trsm);

//   auto configs_trsm_T = std::vector<BenchmarkConfig>{
//       {32, 32},     {64, 64},     {128, 128},  {512, 512},
//       {1024, 1024}, {2048, 2048}, {4096, 4096}};
//   auto data_trsm_T =
//       generate_test_data(configs_trsm_T, Phase::TRSM, Solver::TRSM_BLOCK_T);
//   run_all_impls(Phase::TRSM, Solver::TRSM_BLOCK_T, data_trsm_T, configs_trsm_T);
//   run_all_impls(Phase::TRSM_BLOCK, Solver::TRSM_BLOCK_T, data_trsm_T,
//                 configs_trsm_T);

//   auto configs_trsmvec =
//       std::vector<BenchmarkConfig>{{32, 1},   {64, 1},   {128, 1}, {512, 1},
//                                    {1024, 1}, {2048, 1}, {4096, 1}

//       };
//   auto data_trsmvec =
//       generate_test_data(configs_trsmvec, Phase::TRSM, Solver::TRSM_VECTOR);
//   run_all_impls(Phase::CUBLAS_TRSM, Solver::TRSM_VECTOR, data_trsmvec,
//                 configs_trsmvec);
//   run_all_impls(Phase::TRSM_SMALL, Solver::TRSM_VECTOR, data_trsmvec,
//                 configs_trsmvec);
//   run_all_impls(Phase::TRSM, Solver::TRSM_VECTOR, data_trsmvec,
//                 configs_trsmvec);

  auto configs_triblock =
      std::vector<BenchmarkConfig>{{2048, 128}}; //,  {1024, 64},  {1024, 128},
                                  // {1024, 256}, {1024, 512}, {1024, 1024}
  auto data_triblock = generate_test_data(
      configs_triblock, Phase::TRIBLOCK_SMALL, Solver::CHOLESKY_TRIBLOCK);
  // run_all_impls(Phase::TRIBLOCK_SMALL, Solver::CHOLESKY_TRIBLOCK,
  // data_triblock, configs_triblock);
//   run_all_impls(Phase::TRIBLOCK, Solver::CHOLESKY_TRIBLOCK, data_triblock,
//                 configs_triblock);
//   run_all_impls(Phase::TRIBLOCK_TENSOR, Solver::CHOLESKY_TRIBLOCK,
//                 data_triblock, configs_triblock);
  run_all_impls(Phase::TRIBLOCK_TENSOR_GRAPH, Solver::CHOLESKY_TRIBLOCK,
                data_triblock, configs_triblock);
  run_all_impls(Phase::TRIBLOCK_TENSOR_GRAPH, Solver::CHOLESKY_TRIBLOCK,
                data_triblock, configs_triblock);
  run_all_impls(Phase::TRIBLOCK_TENSOR_GRAPH, Solver::CHOLESKY_TRIBLOCK,
                data_triblock, configs_triblock);
//   run_all_impls(Phase::TRIBLOCK_TENSOR_GRAPH, Solver::CHOLESKY_TRIBLOCK,
//                 data_triblock, configs_triblock);
//   run_all_impls(Phase::CUSOLVER_POTRF, Solver::CHOLESKY_TRIBLOCK, data_triblock,
//                 configs_triblock);
//   run_all_impls(Phase::ENHANCED_DELUXE_CHOLESKY, Solver::CHOLESKY_TRIBLOCK,
//                 data_triblock, configs_triblock);
//   run_all_impls(Phase::ENHANCED_DELUXE_PREMIUM_CHOLESKY,
//                 Solver::CHOLESKY_TRIBLOCK, data_triblock, configs_triblock);

  // can compute speedups later if needed -- XY
  //  for (int32_t j = 1; j < results.size(); ++j) {
  //      for (int32_t i = j; i > 0;) {
  //          --i;
  //          auto const &first = results.at(i);
  //          auto const &second = results.at(j);
  //          printf("\nspeedups %s -> %s:\n\n", first.name, second.name);
  //          printf("  %-6s  %-6s  %-6s  %-7s\n", "size_i", "size_j", "size_k",
  //          "speedup"); printf("  %-6s  %-6s  %-6s  %-7s\n", "------",
  //          "------", "------", "-------"); for (auto const &config : configs)
  //          {
  //              auto size_i = config.size_i;
  //              auto size_j = config.size_j;
  //              auto size_k = config.size_k;
  //              printf("  %6d  %6d  %6d", size_i, size_j, size_k);
  //              auto it_first = first.elapsed_ms.find({size_i, size_j,
  //              size_k}); auto it_second = second.elapsed_ms.find({size_i,
  //              size_j, size_k}); if (it_first != first.elapsed_ms.end() &&
  //                  it_second != second.elapsed_ms.end()) {
  //                  printf("  %6.02fx", it_first->second / it_second->second);
  //              } else {
  //                  printf("  %7s", "-");
  //              }
  //              printf("\n");
  //          }
  //      }
  //  }

  // write_json_results("out/results.json", results);

  return 0;
}
