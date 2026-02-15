#include "rocm/vendor.h"
#include "rocm/check.h"

/* LU factorization (PA = LU) */

template <typename T>
void factorize_lu(gpu_context &ctx, T *d_a, size_t n,
                  rocblas_int *d_ipiv, rocblas_int *d_info, size_t batch);

template <>
void factorize_lu<double>(gpu_context &ctx, double *d_a, size_t n,
                          rocblas_int *d_ipiv, rocblas_int *d_info,
                          size_t batch) {
    rocblas_int rn = (rocblas_int)n;
    if (batch == 1) {
        ROCBLAS_CHECK(rocsolver_dgetrf(
            ctx.blas_handle, rn, rn, d_a, rn, d_ipiv, d_info));
    } else {
        rocblas_stride stride_a    = (rocblas_stride)n * n;
        rocblas_stride stride_ipiv = (rocblas_stride)n;
        ROCBLAS_CHECK(rocsolver_dgetrf_strided_batched(
            ctx.blas_handle, rn, rn,
            d_a, rn, stride_a,
            d_ipiv, stride_ipiv,
            d_info, (rocblas_int)batch));
    }
}

template <>
void factorize_lu<float>(gpu_context &ctx, float *d_a, size_t n,
                         rocblas_int *d_ipiv, rocblas_int *d_info,
                         size_t batch) {
    rocblas_int rn = (rocblas_int)n;
    if (batch == 1) {
        ROCBLAS_CHECK(rocsolver_sgetrf(
            ctx.blas_handle, rn, rn, d_a, rn, d_ipiv, d_info));
    } else {
        rocblas_stride stride_a    = (rocblas_stride)n * n;
        rocblas_stride stride_ipiv = (rocblas_stride)n;
        ROCBLAS_CHECK(rocsolver_sgetrf_strided_batched(
            ctx.blas_handle, rn, rn,
            d_a, rn, stride_a,
            d_ipiv, stride_ipiv,
            d_info, (rocblas_int)batch));
    }
}

/* Cholesky factorization (A = L L^T) */

template <typename T>
void factorize_cholesky(gpu_context &ctx, T *d_a, size_t n,
                        rocblas_int *d_info, size_t batch);

template <>
void factorize_cholesky<double>(gpu_context &ctx, double *d_a, size_t n,
                                rocblas_int *d_info, size_t batch) {
    rocblas_int rn = (rocblas_int)n;
    if (batch == 1) {
        ROCBLAS_CHECK(rocsolver_dpotrf(
            ctx.blas_handle, rocblas_fill_lower, rn, d_a, rn, d_info));
    } else {
        rocblas_stride stride_a = (rocblas_stride)n * n;
        ROCBLAS_CHECK(rocsolver_dpotrf_strided_batched(
            ctx.blas_handle, rocblas_fill_lower, rn,
            d_a, rn, stride_a,
            d_info, (rocblas_int)batch));
    }
}

template <>
void factorize_cholesky<float>(gpu_context &ctx, float *d_a, size_t n,
                               rocblas_int *d_info, size_t batch) {
    rocblas_int rn = (rocblas_int)n;
    if (batch == 1) {
        ROCBLAS_CHECK(rocsolver_spotrf(
            ctx.blas_handle, rocblas_fill_lower, rn, d_a, rn, d_info));
    } else {
        rocblas_stride stride_a = (rocblas_stride)n * n;
        ROCBLAS_CHECK(rocsolver_spotrf_strided_batched(
            ctx.blas_handle, rocblas_fill_lower, rn,
            d_a, rn, stride_a,
            d_info, (rocblas_int)batch));
    }
}
