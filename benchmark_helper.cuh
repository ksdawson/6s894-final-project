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

std::vector<float> generate_random_vector(int32_t size) {
    std::vector<float> matrix(size);
    for (int32_t i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand() % 2 + 1);
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

std::vector<float> trsm_generate_T(std::vector<float> const &matrix, std::vector<float> const &b, int32_t size) {
    auto result = std::vector<float>(size * size);
    for (int32_t i = 0; i < size; ++i) {
        for (int32_t j = 0; j < size; ++j) {
            result[j * size + i] = 0.0f;
            for (int32_t k = 0; k < size; ++k) {
                result[j * size + i] += matrix[i * size + k] * b[j * size + k];
            }
        }
    }
    return result;
}

std::vector<float> trsm_vector_generate(std::vector<float> const &matrix, std::vector<float> const &b, int32_t size) {
    auto result = std::vector<float>(size);
    for (int32_t i = 0; i < size; ++i) {
        result[i] = 0.0f;
        for (int32_t k = 0; k < size; ++k) {
            result[i] += matrix[i * size + k] * b[k];
        }
    }
    return result;
}

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

float calc_error_trsm_vector(std::vector<float> const &c_out_host, std::vector<float> const &c, int32_t size) {
    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < size; ++i) {
        float diff = c_out_host[i] - c[i];
        mse += diff * diff;
        ref_mean_square += c[i] * c[i];
    }
    mse /= size;
    ref_mean_square /= size;
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