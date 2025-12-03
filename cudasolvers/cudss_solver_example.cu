#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cudss.h>

// Helper macro for checking CUDA/cuDSS errors
#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        return -1; \
    } \
}

#define CHECK_CUDSS(func) { \
    cudssStatus_t status = (func); \
    if (status != CUDSS_STATUS_SUCCESS) { \
        std::cerr << "cuDSS Error: " << status << std::endl; \
        return -1; \
    } \
}

int main() {
    // -------------------------------------------------------------------------
    // 1. Define a Simple Sparse SPD Matrix (4x4)
    // -------------------------------------------------------------------------
    // Matrix A (Symmetric Positive Definite):
    // [ 4  1  0  0 ]
    // [ 1  4  1  0 ]
    // [ 0  1  4  1 ]
    // [ 0  0  1  4 ]
    //
    // Stored in CSR (Compressed Sparse Row) format.
    // Note: For symmetric matrices, cuDSS usually expects the full matrix 
    // or specific triangular parts depending on config. We provide full CSR here.
    
    int n = 4;
    int nnz = 10; // Number of non-zeros
    
    std::vector<int> csr_row_ptr = {0, 2, 5, 8, 10};
    std::vector<int> csr_col_ind = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> csr_values = {4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0};

    // RHS vector b (such that x = {1, 1, 1, 1})
    // b = A * x = {5, 6, 6, 5}
    std::vector<double> h_b = {5.0, 6.0, 6.0, 5.0};
    std::vector<double> h_x(n, 0.0); // Solution holder

    // -------------------------------------------------------------------------
    // 2. Allocate Device Memory
    // -------------------------------------------------------------------------
    int *d_row_ptr, *d_col_ind;
    double *d_values, *d_b, *d_x;

    CHECK_CUDA(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_ind, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_row_ptr, csr_row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, csr_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, csr_values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // 3. Initialize cuDSS
    // -------------------------------------------------------------------------
    cudssHandle_t handle;
    CHECK_CUDSS(cudssCreate(&handle));

    cudssConfig_t config;
    CHECK_CUDSS(cudssConfigCreate(&config));

    cudssData_t solver_data;
    CHECK_CUDSS(cudssDataCreate(handle, &solver_data));

    cudssMatrix_t matrix_A, vector_b, vector_x;

    // -------------------------------------------------------------------------
    // 4. Configure Matrix and Solver
    // -------------------------------------------------------------------------
    
    // Create Matrix Wrapper for A
    CHECK_CUDSS(cudssMatrixCreateCsr(&matrix_A, n, n, nnz, 
                                     d_row_ptr, NULL, d_col_ind, d_values, 
                                     CUDA_R_32I, CUDA_R_64F, CUDSS_BASE_ZERO, 
                                     CUDA_R_64F, CUDA_R_64F));

    // Create Wrappers for Vectors b and x
    CHECK_CUDSS(cudssMatrixCreateDn(&vector_b, n, 1, n, d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    CHECK_CUDSS(cudssMatrixCreateDn(&vector_x, n, 1, n, d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

    // CRITICAL: Set Matrix Type to Symmetric Positive Definite for Cholesky
    CHECK_CUDSS(cudssConfigSetModelType(config, CUDSS_MT_SYMMETRIC_POSITIVE_DEFINITE));

    // -------------------------------------------------------------------------
    // 5. Phase 1: Analysis (Symbolic Factorization)
    // -------------------------------------------------------------------------
    // Reorders matrix to minimize fill-in. No numerical values used yet.
    CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, solver_data, matrix_A, vector_x, vector_b));

    // -------------------------------------------------------------------------
    // 6. Phase 2: Factorization (Numerical Factorization)
    // -------------------------------------------------------------------------
    // Performs the Cholesky decomposition A = L * L^T
    CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, solver_data, matrix_A, vector_x, vector_b));

    // -------------------------------------------------------------------------
    // 7. Phase 3: Solve
    // -------------------------------------------------------------------------
    // Solves A * x = b using the factors
    CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, solver_data, matrix_A, vector_x, vector_b));

    // -------------------------------------------------------------------------
    // 8. Retrieve Results
    // -------------------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(h_x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "Solution x:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << h_x[i] << " ";
    }
    std::cout << "\nExpected: 1.0 1.0 1.0 1.0" << std::endl;

    // -------------------------------------------------------------------------
    // 9. Cleanup
    // -------------------------------------------------------------------------
    cudssMatrixDestroy(matrix_A);
    cudssMatrixDestroy(vector_b);
    cudssMatrixDestroy(vector_x);
    cudssDataDestroy(handle, solver_data);
    cudssConfigDestroy(config);
    cudssDestroy(handle);

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_b);
    cudaFree(d_x);

    return 0;
}