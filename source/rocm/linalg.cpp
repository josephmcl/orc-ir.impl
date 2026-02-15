#include "rocm/vendor.h"
#include "rocm/check.h"

/* diagonal shift kernel: A[b][i][i] += value
   for SPD construction */

template <typename T>
__global__ void add_diagonal(T *matrices, size_t n, T value, size_t batch) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * batch) return;
    size_t b = idx / n;
    size_t i = idx % n;
    matrices[b * n * n + i * n + i] += value;
}

/* make_spd: A = R^T R + n I */

template <typename T>
void make_spd(gpu_context &ctx, T *d_a, T *d_work, size_t n, size_t batch);

template <>
void make_spd<double>(gpu_context &ctx, double *d_a, double *d_work,
                      size_t n, size_t batch) {
    double alpha = 1.0;
    double beta  = 0.0;
    rocblas_int rn = (rocblas_int)n;

    if (batch == 1) {
        ROCBLAS_CHECK(rocblas_dgemm(
            ctx.blas_handle,
            rocblas_operation_transpose, rocblas_operation_none,
            rn, rn, rn, &alpha, d_a, rn, d_a, rn, &beta, d_work, rn));
    } else {
        rocblas_stride stride = (rocblas_stride)n * n;
        ROCBLAS_CHECK(rocblas_dgemm_strided_batched(
            ctx.blas_handle,
            rocblas_operation_transpose, rocblas_operation_none,
            rn, rn, rn, &alpha,
            d_a, rn, stride,
            d_a, rn, stride,
            &beta, d_work, rn, stride,
            (rocblas_int)batch));
    }

    size_t total = n * batch;
    size_t block = 256;
    size_t grid  = (total + block - 1) / block;
    add_diagonal<<<grid, block>>>(d_work, n, (double)n, batch);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(d_a, d_work, batch * n * n * sizeof(double),
                        hipMemcpyDeviceToDevice));
}

template <>
void make_spd<float>(gpu_context &ctx, float *d_a, float *d_work,
                     size_t n, size_t batch) {
    float alpha = 1.0f;
    float beta  = 0.0f;
    rocblas_int rn = (rocblas_int)n;

    if (batch == 1) {
        ROCBLAS_CHECK(rocblas_sgemm(
            ctx.blas_handle,
            rocblas_operation_transpose, rocblas_operation_none,
            rn, rn, rn, &alpha, d_a, rn, d_a, rn, &beta, d_work, rn));
    } else {
        rocblas_stride stride = (rocblas_stride)n * n;
        ROCBLAS_CHECK(rocblas_sgemm_strided_batched(
            ctx.blas_handle,
            rocblas_operation_transpose, rocblas_operation_none,
            rn, rn, rn, &alpha,
            d_a, rn, stride,
            d_a, rn, stride,
            &beta, d_work, rn, stride,
            (rocblas_int)batch));
    }

    size_t total = n * batch;
    size_t block = 256;
    size_t grid  = (total + block - 1) / block;
    add_diagonal<<<grid, block>>>(d_work, n, (float)n, batch);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(d_a, d_work, batch * n * n * sizeof(float),
                        hipMemcpyDeviceToDevice));
}

/* gemv_batch: b = A * x for each batch element
   and rhs column. Uses GEMM with nrhs columns
   instead of looping over GEMV. */

template <typename T>
void gemv_batch(gpu_context &ctx, size_t n, size_t nrhs, size_t batch,
                const T *d_a, const T *d_x, T *d_b);

template <>
void gemv_batch<double>(gpu_context &ctx, size_t n, size_t nrhs, size_t batch,
                        const double *d_a, const double *d_x, double *d_b) {
    double alpha = 1.0;
    double beta  = 0.0;
    rocblas_int rn    = (rocblas_int)n;
    rocblas_int rnrhs = (rocblas_int)nrhs;

    if (batch == 1) {
        ROCBLAS_CHECK(rocblas_dgemm(
            ctx.blas_handle,
            rocblas_operation_none, rocblas_operation_none,
            rn, rnrhs, rn, &alpha, d_a, rn, d_x, rn, &beta, d_b, rn));
    } else {
        rocblas_stride stride_a = (rocblas_stride)n * n;
        rocblas_stride stride_v = (rocblas_stride)n * nrhs;
        ROCBLAS_CHECK(rocblas_dgemm_strided_batched(
            ctx.blas_handle,
            rocblas_operation_none, rocblas_operation_none,
            rn, rnrhs, rn, &alpha,
            d_a, rn, stride_a,
            d_x, rn, stride_v,
            &beta, d_b, rn, stride_v,
            (rocblas_int)batch));
    }
}

template <>
void gemv_batch<float>(gpu_context &ctx, size_t n, size_t nrhs, size_t batch,
                       const float *d_a, const float *d_x, float *d_b) {
    float alpha = 1.0f;
    float beta  = 0.0f;
    rocblas_int rn    = (rocblas_int)n;
    rocblas_int rnrhs = (rocblas_int)nrhs;

    if (batch == 1) {
        ROCBLAS_CHECK(rocblas_sgemm(
            ctx.blas_handle,
            rocblas_operation_none, rocblas_operation_none,
            rn, rnrhs, rn, &alpha, d_a, rn, d_x, rn, &beta, d_b, rn));
    } else {
        rocblas_stride stride_a = (rocblas_stride)n * n;
        rocblas_stride stride_v = (rocblas_stride)n * nrhs;
        ROCBLAS_CHECK(rocblas_sgemm_strided_batched(
            ctx.blas_handle,
            rocblas_operation_none, rocblas_operation_none,
            rn, rnrhs, rn, &alpha,
            d_a, rn, stride_a,
            d_x, rn, stride_v,
            &beta, d_b, rn, stride_v,
            (rocblas_int)batch));
    }
}
