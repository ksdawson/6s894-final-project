// TL+ {"compile_flags": ["-lcuda", "-lcublas", "-lcusolver"]}
// TL+ {"header_files": []}
// TL {"workspace_files": []}

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

void solve_trsm(int n, float *d_A, float *d_B) {
    // d_A: device pointer to n×n lower triangular matrix (row-major)
    // d_B: device pointer to n×n RHS matrix (row-major), overwritten with solution X

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Parameters for AX = B with row-major lower triangular A
    // Row-major AX = B  ↔  col-major X^T * A^T = B^T  ↔  col-major X * A = B (SIDE_RIGHT)
    cublasSideMode_t   side  = CUBLAS_SIDE_RIGHT;      // X * op(A) = alpha * B (for row-major AX=B)
    cublasFillMode_t   uplo  = CUBLAS_FILL_MODE_UPPER; // cuBLAS sees upper (row-major lower)
    cublasOperation_t  trans = CUBLAS_OP_N;            // no transpose (cuBLAS has A^T, which is what we need)
    cublasDiagType_t   diag  = CUBLAS_DIAG_NON_UNIT;   // diagonal is not assumed to be 1

    int m = n;      // rows of B
    int k = n;      // columns of B (using k to avoid confusion with matrix dimension n)
    int lda = n;    // leading dimension of A
    int ldb = n;    // leading dimension of B

    float alpha = 1.0f;  // AX = 1.0 * B

    // Solve: A * X = B  (X overwrites B)
    cublasStrsm(
        handle,
        side,           // CUBLAS_SIDE_LEFT: op(A) * X = alpha * B
        uplo,           // CUBLAS_FILL_MODE_UPPER (because row-major lower = col-major upper)
        trans,          // CUBLAS_OP_T (transpose the col-major upper back to row-major lower)
        diag,           // CUBLAS_DIAG_NON_UNIT
        m,              // number of rows of B
        k,              // number of columns of B
        &alpha,         // scalar alpha
        d_A, lda,       // A and its leading dimension
        d_B, ldb        // B and its leading dimension (solution X overwrites B)
    );

    cublasDestroy(handle);
}

int main() {
    int n = 3;

    // Row-major lower triangular A:
    // [ 2  0  0 ]
    // [ 1  3  0 ]
    // [ 4  2  5 ]
    float h_A[] = {
        2, 0, 0,
        1, 3, 0,
        4, 2, 5
    };

    // B matrix (row-major):
    // [ 4  2  6 ]
    // [ 7  6  9 ]
    // [18 12  25]
    // For this example using n×n, let's use:
    float h_B[] = {
        4,  2,  6,
        7,  6,  9,
        18, 12, 25
    };

    // Allocate device memory
    float *d_A, *d_B;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;

    // Solve AX = B (row-major)
    // Row-major: AX = B  is equivalent to  col-major: X^T * A^T = B^T  i.e., X * A = B (right side)
    cublasStrsm(
        handle,
        CUBLAS_SIDE_RIGHT,       // X * op(A) = alpha * B (right side for row-major AX=B)
        CUBLAS_FILL_MODE_UPPER,  // row-major lower = col-major upper
        CUBLAS_OP_N,             // no transpose (A^T is already what we need)
        CUBLAS_DIAG_NON_UNIT,
        n, n,                    // m, n dimensions of B
        &alpha,
        d_A, n,                  // A, lda
        d_B, n                   // B, ldb (overwritten with X)
    );

    // Copy result back
    float h_X[9];
    cudaMemcpy(h_X, d_B, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print solution
    printf("Solution X:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.4f ", h_X[i * n + j]);
        }
        printf("\n");
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}