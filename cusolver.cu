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

void set_up_cusolver(cusolverDnHandle_t *cusolverH) {
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    cusolverDnParams_t params = NULL;
}


void launch_potrf(int32_t size, float *a, float *c, float *b, void *workspace) {
    cusolver_potrf(a, c, b, workspace);
}
}