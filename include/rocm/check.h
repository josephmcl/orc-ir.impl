#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <hiprand/hiprand.h>

#include <iostream>
#include <cstdlib>

#define HIP_CHECK(call) do {                                           \
    hipError_t err = (call);                                           \
    if (err != hipSuccess) {                                           \
        std::cerr << "hip error at " << __FILE__ << ":" << __LINE__   \
                  << ": " << hipGetErrorString(err) << std::endl;     \
        exit(1);                                                       \
    }                                                                  \
} while (0)

#define ROCBLAS_CHECK(call) do {                                       \
    rocblas_status stat = (call);                                      \
    if (stat != rocblas_status_success) {                              \
        std::cerr << "rocblas error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << (int)stat << std::endl;                  \
        exit(1);                                                       \
    }                                                                  \
} while (0)

#define HIPRAND_CHECK(call) do {                                       \
    hiprandStatus_t stat = (call);                                     \
    if (stat != HIPRAND_STATUS_SUCCESS) {                              \
        std::cerr << "hiprand error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << (int)stat << std::endl;                  \
        exit(1);                                                       \
    }                                                                  \
} while (0)
