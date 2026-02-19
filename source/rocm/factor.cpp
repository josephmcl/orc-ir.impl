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

/* LU solve (given packed LU and ipiv from getrf,
   solve in-place: d_b becomes solution) */

template <typename T>
void solve_lu(gpu_context &ctx, T *d_lu, size_t n,
              rocblas_int *d_ipiv, T *d_b,
              size_t nrhs, size_t batch);

template <>
void solve_lu<double>(gpu_context &ctx, double *d_lu,
                      size_t n, rocblas_int *d_ipiv,
                      double *d_b, size_t nrhs,
                      size_t batch) {
    rocblas_int rn    = (rocblas_int)n;
    rocblas_int rnrhs = (rocblas_int)nrhs;
    if (batch == 1) {
        ROCBLAS_CHECK(rocsolver_dgetrs(
            ctx.blas_handle,
            rocblas_operation_none,
            rn, rnrhs, d_lu, rn, d_ipiv, d_b, rn));
    } else {
        rocblas_stride stride_a    = (rocblas_stride)n * n;
        rocblas_stride stride_ipiv = (rocblas_stride)n;
        rocblas_stride stride_b    = (rocblas_stride)n * nrhs;
        ROCBLAS_CHECK(rocsolver_dgetrs_strided_batched(
            ctx.blas_handle,
            rocblas_operation_none,
            rn, rnrhs,
            d_lu, rn, stride_a,
            d_ipiv, stride_ipiv,
            d_b, rn, stride_b,
            (rocblas_int)batch));
    }
}

template <>
void solve_lu<float>(gpu_context &ctx, float *d_lu,
                     size_t n, rocblas_int *d_ipiv,
                     float *d_b, size_t nrhs,
                     size_t batch) {
    rocblas_int rn    = (rocblas_int)n;
    rocblas_int rnrhs = (rocblas_int)nrhs;
    if (batch == 1) {
        ROCBLAS_CHECK(rocsolver_sgetrs(
            ctx.blas_handle,
            rocblas_operation_none,
            rn, rnrhs, d_lu, rn, d_ipiv, d_b, rn));
    } else {
        rocblas_stride stride_a    = (rocblas_stride)n * n;
        rocblas_stride stride_ipiv = (rocblas_stride)n;
        rocblas_stride stride_b    = (rocblas_stride)n * nrhs;
        ROCBLAS_CHECK(rocsolver_sgetrs_strided_batched(
            ctx.blas_handle,
            rocblas_operation_none,
            rn, rnrhs,
            d_lu, rn, stride_a,
            d_ipiv, stride_ipiv,
            d_b, rn, stride_b,
            (rocblas_int)batch));
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

/* Cholesky solve (given L from potrf, solve
   in-place: d_b becomes solution) */

template <typename T>
void solve_cholesky(gpu_context &ctx, T *d_l,
                    size_t n, T *d_b,
                    size_t nrhs, size_t batch);

template <>
void solve_cholesky<double>(gpu_context &ctx,
                            double *d_l, size_t n,
                            double *d_b, size_t nrhs,
                            size_t batch) {
    rocblas_int rn    = (rocblas_int)n;
    rocblas_int rnrhs = (rocblas_int)nrhs;
    if (batch == 1) {
        ROCBLAS_CHECK(rocsolver_dpotrs(
            ctx.blas_handle, rocblas_fill_lower,
            rn, rnrhs, d_l, rn, d_b, rn));
    } else {
        rocblas_stride stride_a = (rocblas_stride)n * n;
        rocblas_stride stride_b = (rocblas_stride)n * nrhs;
        ROCBLAS_CHECK(rocsolver_dpotrs_strided_batched(
            ctx.blas_handle, rocblas_fill_lower,
            rn, rnrhs,
            d_l, rn, stride_a,
            d_b, rn, stride_b,
            (rocblas_int)batch));
    }
}

template <>
void solve_cholesky<float>(gpu_context &ctx,
                           float *d_l, size_t n,
                           float *d_b, size_t nrhs,
                           size_t batch) {
    rocblas_int rn    = (rocblas_int)n;
    rocblas_int rnrhs = (rocblas_int)nrhs;
    if (batch == 1) {
        ROCBLAS_CHECK(rocsolver_spotrs(
            ctx.blas_handle, rocblas_fill_lower,
            rn, rnrhs, d_l, rn, d_b, rn));
    } else {
        rocblas_stride stride_a = (rocblas_stride)n * n;
        rocblas_stride stride_b = (rocblas_stride)n * nrhs;
        ROCBLAS_CHECK(rocsolver_spotrs_strided_batched(
            ctx.blas_handle, rocblas_fill_lower,
            rn, rnrhs,
            d_l, rn, stride_a,
            d_b, rn, stride_b,
            (rocblas_int)batch));
    }
}
