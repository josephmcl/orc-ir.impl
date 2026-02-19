#include "common/cli.h"
#include "common/artifact.h"
#include "rocm/vendor.h"
#include "rocm/check.h"

#include <iostream>
#include <type_traits>
#include <vector>

/* forward declarations from other translation
   units */

template <typename T>
void fill_random(gpu_context &ctx, T *d_ptr, size_t count);

template <typename T>
void make_spd(gpu_context &ctx, T *d_a, T *d_work, size_t n, size_t batch);

template <typename T>
void gemv_batch(gpu_context &ctx, size_t n, size_t nrhs, size_t batch,
                const T *d_a, const T *d_x, T *d_b);

template <typename T>
void factorize_lu(gpu_context &ctx, T *d_a, size_t n,
                  rocblas_int *d_ipiv, rocblas_int *d_info, size_t batch);

template <typename T>
void factorize_cholesky(gpu_context &ctx, T *d_a, size_t n,
                        rocblas_int *d_info, size_t batch);

/* LU extraction kernels: split packed LU into
   separate L and U */

template <typename T>
__global__ void extract_l(const T *lu, T *l, size_t n, size_t batch) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * n * n;
    if (idx >= total) return;

    size_t b   = idx / (n * n);
    size_t rem = idx % (n * n);
    size_t col = rem / n;
    size_t row = rem % n;

    T val;
    if (row > col)
        val = lu[b * n * n + col * n + row];
    else if (row == col)
        val = (T)1;
    else
        val = (T)0;

    l[b * n * n + col * n + row] = val;
}

template <typename T>
__global__ void extract_u(const T *lu, T *u, size_t n, size_t batch) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * n * n;
    if (idx >= total) return;

    size_t b   = idx / (n * n);
    size_t rem = idx % (n * n);
    size_t col = rem / n;
    size_t row = rem % n;

    T val;
    if (row <= col)
        val = lu[b * n * n + col * n + row];
    else
        val = (T)0;

    u[b * n * n + col * n + row] = val;
}

/* Cholesky extraction: zero the upper triangle */

template <typename T>
__global__ void extract_lower(const T *a, T *l, size_t n, size_t batch) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * n * n;
    if (idx >= total) return;

    size_t b   = idx / (n * n);
    size_t rem = idx % (n * n);
    size_t col = rem / n;
    size_t row = rem % n;

    T val = (row >= col) ? a[b * n * n + col * n + row] : (T)0;
    l[b * n * n + col * n + row] = val;
}

/* memory check */

static void check_memory(size_t required) {
    size_t free_mem, total_mem;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));

    auto gb = [](size_t bytes) { return (double)bytes / (1ull << 30); };

    std::cerr << "gpu memory: " << gb(free_mem) << " GB free / "
              << gb(total_mem) << " GB total\n";
    if (required > free_mem) {
        std::cerr << "error: need " << gb(required) << " GB but only "
                  << gb(free_mem) << " GB available\n";
        exit(1);
    }
}

/* run_generate: the main generation workflow */

template <typename T>
void run_generate(gpu_context &ctx, const problem_descriptor &desc) {
    size_t n     = desc.n;
    size_t batch = desc.batch;
    size_t nrhs  = desc.nrhs;

    size_t mat_elems = batch * n * n;
    size_t rhs_elems = batch * n * nrhs;

    /* estimate total device memory needed:
       a + work + factor_out, x + b, ipiv, info */
    size_t needed = mat_elems * sizeof(T) * 3
                  + rhs_elems * sizeof(T) * 2
                  + batch * n * sizeof(rocblas_int)
                  + batch * sizeof(rocblas_int);
    if (desc.factor == factor_type::lu)
        needed += mat_elems * sizeof(T);

    check_memory(needed);

    T *d_a, *d_x, *d_b, *d_work;
    rocblas_int *d_ipiv, *d_info;

    HIP_CHECK(hipMalloc(&d_a,    mat_elems * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_work, mat_elems * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_x,    rhs_elems * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_b,    rhs_elems * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_ipiv, batch * n * sizeof(rocblas_int)));
    HIP_CHECK(hipMalloc(&d_info, batch * sizeof(rocblas_int)));

    std::cerr << "generating random A (" << n << " x " << n
              << ", batch " << batch << ")...\n";
    fill_random(ctx, d_a, mat_elems);

    if (desc.factor == factor_type::cholesky) {
        std::cerr << "making A symmetric positive definite...\n";
        make_spd(ctx, d_a, d_work, n, batch);
    }

    std::cerr << "generating random x (" << n << " x " << nrhs
              << ", batch " << batch << ")...\n";
    fill_random(ctx, d_x, rhs_elems);

    std::cerr << "computing b = A * x...\n";
    gemv_batch(ctx, n, nrhs, batch, d_a, d_x, d_b);

    HIP_CHECK(hipStreamSynchronize(ctx.stream));

    std::string dir = artifact_directory(desc);
    std::cerr << "writing artifacts to " << dir << "/\n";
    write_metadata(dir, desc);

    {
        std::vector<T> host(mat_elems);
        HIP_CHECK(hipMemcpy(host.data(), d_a,
                            mat_elems * sizeof(T), hipMemcpyDeviceToHost));
        write_array(dir + "/a.bin", host.data(), mat_elems * sizeof(T));
    }
    {
        std::vector<T> host(rhs_elems);
        HIP_CHECK(hipMemcpy(host.data(), d_x,
                            rhs_elems * sizeof(T), hipMemcpyDeviceToHost));
        write_array(dir + "/x.bin", host.data(), rhs_elems * sizeof(T));

        HIP_CHECK(hipMemcpy(host.data(), d_b,
                            rhs_elems * sizeof(T), hipMemcpyDeviceToHost));
        write_array(dir + "/b.bin", host.data(), rhs_elems * sizeof(T));
    }

    std::cerr << "factorizing...\n";
    if (desc.factor == factor_type::lu) {
        factorize_lu(ctx, d_a, n, d_ipiv, d_info, batch);
    } else {
        factorize_cholesky(ctx, d_a, n, d_info, batch);
    }

    HIP_CHECK(hipStreamSynchronize(ctx.stream));

    {
        std::vector<rocblas_int> h_info(batch);
        HIP_CHECK(hipMemcpy(h_info.data(), d_info,
                            batch * sizeof(rocblas_int),
                            hipMemcpyDeviceToHost));
        for (size_t i = 0; i < batch; i++) {
            if (h_info[i] != 0) {
                std::cerr << "warning: factorization info[" << i
                          << "] = " << h_info[i] << "\n";
            }
        }
    }

    size_t block = 256;
    size_t grid  = (mat_elems + block - 1) / block;

    if (desc.factor == factor_type::lu) {
        extract_l<<<grid, block>>>(d_a, d_work, n, batch);
        HIP_CHECK(hipGetLastError());

        {
            std::vector<T> host(mat_elems);
            HIP_CHECK(hipMemcpy(host.data(), d_work,
                                mat_elems * sizeof(T),
                                hipMemcpyDeviceToHost));
            write_array(dir + "/l.bin", host.data(), mat_elems * sizeof(T));
        }

        extract_u<<<grid, block>>>(d_a, d_work, n, batch);
        HIP_CHECK(hipGetLastError());

        {
            std::vector<T> host(mat_elems);
            HIP_CHECK(hipMemcpy(host.data(), d_work,
                                mat_elems * sizeof(T),
                                hipMemcpyDeviceToHost));
            write_array(dir + "/u.bin", host.data(), mat_elems * sizeof(T));
        }

        {
            std::vector<rocblas_int> h_ipiv(batch * n);
            HIP_CHECK(hipMemcpy(h_ipiv.data(), d_ipiv,
                                batch * n * sizeof(rocblas_int),
                                hipMemcpyDeviceToHost));
            write_array(dir + "/ipiv.bin", h_ipiv.data(),
                        batch * n * sizeof(rocblas_int));
        }
    } else {
        extract_lower<<<grid, block>>>(d_a, d_work, n, batch);
        HIP_CHECK(hipGetLastError());

        {
            std::vector<T> host(mat_elems);
            HIP_CHECK(hipMemcpy(host.data(), d_work,
                                mat_elems * sizeof(T),
                                hipMemcpyDeviceToHost));
            write_array(dir + "/l.bin", host.data(), mat_elems * sizeof(T));
        }

        /* compute x_l = L^T * x_true, the true
           solution for the triangular system
           L * y = b (used for forward error in
           iterative refinement).
           d_work = L, d_x = x_true. */
        T *d_xl;
        HIP_CHECK(hipMalloc(&d_xl,
            rhs_elems * sizeof(T)));

        T one  = (T)1;
        T zero = (T)0;
        for (size_t bi = 0; bi < batch; bi++) {
            for (size_t j = 0; j < nrhs; j++) {
                if constexpr (
                    std::is_same_v<T, double>)
                {
                    ROCBLAS_CHECK(rocblas_dgemv(
                        ctx.blas_handle,
                        rocblas_operation_transpose,
                        (rocblas_int)n,
                        (rocblas_int)n,
                        &one,
                        d_work + bi * n * n,
                        (rocblas_int)n,
                        d_x + bi*n*nrhs + j*n, 1,
                        &zero,
                        d_xl + bi*n*nrhs + j*n,
                        1));
                } else {
                    ROCBLAS_CHECK(rocblas_sgemv(
                        ctx.blas_handle,
                        rocblas_operation_transpose,
                        (rocblas_int)n,
                        (rocblas_int)n,
                        &one,
                        d_work + bi * n * n,
                        (rocblas_int)n,
                        d_x + bi*n*nrhs + j*n, 1,
                        &zero,
                        d_xl + bi*n*nrhs + j*n,
                        1));
                }
            }
        }
        HIP_CHECK(hipStreamSynchronize(ctx.stream));

        {
            std::vector<T> host(rhs_elems);
            HIP_CHECK(hipMemcpy(host.data(), d_xl,
                rhs_elems * sizeof(T),
                hipMemcpyDeviceToHost));
            write_array(dir + "/x_l.bin",
                host.data(), rhs_elems * sizeof(T));
        }

        HIP_CHECK(hipFree(d_xl));
        std::cerr << "wrote " << dir
                  << "/x_l.bin\n";
    }

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_ipiv));
    HIP_CHECK(hipFree(d_info));

    std::cerr << "done.\n";
}

template void run_generate<double>(gpu_context &ctx,
                                   const problem_descriptor &desc);
template void run_generate<float>(gpu_context &ctx,
                                  const problem_descriptor &desc);

/* main */

int main(int argc, char **argv) {
    cli_args args = parse_args(argc, argv);

    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }

    gpu_context ctx;
    ctx.init();

    switch (args.desc.working_prec) {
        case precision::fp64:
            run_generate<double>(ctx, args.desc);
            break;
        case precision::fp32:
            run_generate<float>(ctx, args.desc);
            break;
        case precision::fp16:
            std::cerr << "error: fp16 generation not yet implemented\n";
            return 1;
    }

    ctx.destroy();
    return 0;
}
