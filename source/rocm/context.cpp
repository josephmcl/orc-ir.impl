#include "rocm/vendor.h"
#include "rocm/check.h"

/* gpu_context implementation */

void gpu_context::init() {
    HIP_CHECK(hipStreamCreate(&stream));
    ROCBLAS_CHECK(rocblas_create_handle(&blas_handle));
    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));
    HIPRAND_CHECK(hiprandCreateGenerator(&rand_gen,
                                         HIPRAND_RNG_PSEUDO_DEFAULT));
    HIPRAND_CHECK(hiprandSetStream(rand_gen, stream));
}

void gpu_context::destroy() {
    HIPRAND_CHECK(hiprandDestroyGenerator(rand_gen));
    ROCBLAS_CHECK(rocblas_destroy_handle(blas_handle));
    HIP_CHECK(hipStreamDestroy(stream));
}
