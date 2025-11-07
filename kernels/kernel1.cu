
void test_case_3x3() {
    // Test case
    const uint32_t size_i = 3;
    const uint32_t size_j = 3;
    // Allocate device memory
    float *in_gpu;
    float *out_gpu
    CUDA_CHECK(cudaMalloc(&in_gpu, size_i * size_j * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_gpu, size_i * size_j * sizeof(float)));
    // Fill in test data on host
    in_gpu[0] = 4.0f;
    in_gpu[1] = 12.0f;
    in_gpu[2] = -16.0f;
    in_gpu[3] = 12.0f;
    in_gpu[4] = 37.0f;
    in_gpu[5] = -43.0f;
    in_gpu[6] = -16.0f;
    in_gpu[7] = -43.0f;
    in_gpu[8] = 98.0f;
    // Copy from host to device
    CUDA_CHECK(cudaMemcpy(in_gpu, in_gpu, size_i * size_j * sizeof(float), cudaMemcpyHostToDevice));
    // Run Cholesky decomposition
    cholesky_cpu_naive(size_i, size_j, )
}

int main(int argc, char **argv) {

}