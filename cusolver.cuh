// TL+ {"compile_flags": ["-lcuda", "-lcublas", "-lcusolver"]}
// TL+ {"header_files": ["cusolver_utils.cuh"]}
// TL {"workspace_files": []}

#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.cuh"

namespace cusolver_potrf{
size_t get_workspace_size(int32_t size) {
    return 0;
}

void set_potrf(cusolverDnHandle_t *cusolverH_ptr, 
    cusolverDnParams_t *params_ptr, 
    int **d_info_ptr, 
    size_t* workspaceInBytesOnDevice_ptr, 
    void **d_work_ptr, 
    size_t* workspaceInBytesOnHost_ptr, 
    void **h_work_ptr,
    int32_t size,
    float *a,
    cublasFillMode_t uplo) {

    using data_type = float;

    CUSOLVER_CHECK(cusolverDnCreate(cusolverH_ptr));
    CUSOLVER_CHECK(cusolverDnCreateParams(params_ptr));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(d_info_ptr), sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
        *cusolverH_ptr, *params_ptr, uplo, size, traits<data_type>::cuda_data_type, a, size,
        traits<data_type>::cuda_data_type, workspaceInBytesOnDevice_ptr, workspaceInBytesOnHost_ptr));

    CUDA_CHECK(cudaMalloc(d_work_ptr, *workspaceInBytesOnDevice_ptr));

    if (0 < *workspaceInBytesOnHost_ptr) {
        *h_work_ptr = reinterpret_cast<void *>(malloc(*workspaceInBytesOnHost_ptr));
        if (*h_work_ptr == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }
}


void launch_potrf(int32_t size, float *a, 
    cusolverDnHandle_t *cusolverH_ptr, 
    cusolverDnParams_t *params_ptr, 
    cublasFillMode_t uplo,
    int *d_info, 
    size_t workspaceInBytesOnDevice,
    void *d_work, 
    size_t workspaceInBytesOnHost,
    void *h_work) {

    using data_type = float;

    CUSOLVER_CHECK(cusolverDnXpotrf(*cusolverH_ptr, *params_ptr, uplo, size, traits<data_type>::cuda_data_type,
        a, size, traits<data_type>::cuda_data_type, d_work, workspaceInBytesOnDevice,
        h_work, workspaceInBytesOnHost, d_info));

}

void destroy_potrf(cusolverDnHandle_t *cusolverH_ptr, 
    int **d_info_ptr, 
    void **d_work_ptr, 
    void **h_work_ptr) {
     
    CUSOLVER_CHECK(cusolverDnDestroy(*cusolverH_ptr));

    CUDA_CHECK(cudaFree(*d_info_ptr));
    CUDA_CHECK(cudaFree(*d_work_ptr));
    free(*h_work_ptr);
}

void launch_potrf_backup(int32_t size, float *a, float *c, float *b, void *workspace) {
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnParams_t params = NULL;

    using data_type = float;

    int *d_info = nullptr;    /* error info */
    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    void *d_work = nullptr;              /* device workspace */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void *h_work = nullptr;              /* host workspace */

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;


    /* step 1: create cusolver handle */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
        cusolverH, params, uplo, size, traits<data_type>::cuda_data_type, a, size,
        traits<data_type>::cuda_data_type, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));

    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    /* step 4: Cholesky factorization */
    CUSOLVER_CHECK(cusolverDnXpotrf(cusolverH, params, uplo, size, traits<data_type>::cuda_data_type,
        a, size, traits<data_type>::cuda_data_type, d_work, workspaceInBytesOnDevice,
        h_work, workspaceInBytesOnHost, d_info));
    
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    free(h_work);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
}
}