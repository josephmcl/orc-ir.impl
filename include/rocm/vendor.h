#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hiprand/hiprand.h>

struct gpu_context {
    rocblas_handle     blas_handle;
    hiprandGenerator_t rand_gen;
    hipStream_t        stream;

    void init();
    void destroy();
};
