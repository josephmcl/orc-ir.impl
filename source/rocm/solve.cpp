#include "common/cli.h"
#include "common/artifact.h"
#include "common/profiler.h"
#include "rocm/vendor.h"
#include "rocm/check.h"
#include "rocm/timer.h"

#include <cmath>
#include <iostream>
#include <vector>

/* forward declarations from factor.cpp */

template <typename T>
void solve_lu(gpu_context &ctx, T *d_lu, size_t n,
              rocblas_int *d_ipiv, T *d_b,
              size_t nrhs, size_t batch);

template <typename T>
void solve_cholesky(gpu_context &ctx, T *d_l,
                    size_t n, T *d_b,
                    size_t nrhs, size_t batch);

/* pack_lu: reassemble packed LAPACK LU from
   separate L (unit lower) and U (upper) */

template <typename T>
__global__ void pack_lu(const T *l, const T *u,
                        T *lu, size_t n,
                        size_t batch) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * n * n;
    if (idx >= total) return;

    size_t b   = idx / (n * n);
    size_t rem = idx % (n * n);
    size_t col = rem / n;
    size_t row = rem % n;

    size_t off = b * n * n + col * n + row;
    lu[off] = (row > col) ? l[off] : u[off];
}

/* host-side residual: ||Ax - b|| / ||b|| */

template <typename T>
static double compute_residual(
    const T *A, const T *x, const T *b,
    size_t n, size_t nrhs, size_t batch_idx
) {
    size_t mat_off = batch_idx * n * n;
    size_t rhs_off = batch_idx * n * nrhs;

    double norm_r = 0.0;
    double norm_b = 0.0;

    for (size_t j = 0; j < nrhs; j++) {
        for (size_t i = 0; i < n; i++) {
            double ax = 0.0;
            for (size_t k = 0; k < n; k++) {
                ax += (double)A[mat_off + k * n + i]
                    * (double)x[rhs_off + j * n + k];
            }
            double r = ax - (double)b[rhs_off + j * n + i];
            norm_r += r * r;
            double bi = (double)b[rhs_off + j * n + i];
            norm_b += bi * bi;
        }
    }

    return std::sqrt(norm_r) / std::sqrt(norm_b);
}

/* run_solve: profiled solve workflow */

template <typename T>
void run_solve(gpu_context &ctx,
               const profiler_config &cfg) {
    size_t n     = cfg.desc.n;
    size_t batch = cfg.desc.batch;
    size_t nrhs  = cfg.desc.nrhs;

    size_t mat_elems = batch * n * n;
    size_t rhs_elems = batch * n * nrhs;

    /* load artifact from disk */
    std::string dir = artifact_directory(cfg.desc);
    std::cerr << "loading artifact from "
              << dir << "/\n";

    std::vector<T> h_a(mat_elems);
    std::vector<T> h_b(rhs_elems);
    std::vector<T> h_l(mat_elems);

    read_array(dir + "/a.bin",
               h_a.data(), mat_elems * sizeof(T));
    read_array(dir + "/b.bin",
               h_b.data(), rhs_elems * sizeof(T));
    read_array(dir + "/l.bin",
               h_l.data(), mat_elems * sizeof(T));

    std::vector<T> h_u;
    std::vector<rocblas_int> h_ipiv;
    if (cfg.desc.factor == factor_type::lu) {
        h_u.resize(mat_elems);
        h_ipiv.resize(batch * n);
        read_array(dir + "/u.bin",
                   h_u.data(),
                   mat_elems * sizeof(T));
        read_array(dir + "/ipiv.bin",
                   h_ipiv.data(),
                   batch * n * sizeof(rocblas_int));
    }

    /* allocate device memory */
    T *d_factor, *d_b, *d_x;
    rocblas_int *d_ipiv = nullptr;

    HIP_CHECK(hipMalloc(&d_factor,
        mat_elems * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_b,
        rhs_elems * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_x,
        rhs_elems * sizeof(T)));

    if (cfg.desc.factor == factor_type::lu) {
        HIP_CHECK(hipMalloc(&d_ipiv,
            batch * n * sizeof(rocblas_int)));
    }

    /* upload b (kept pristine for re-copy) */
    HIP_CHECK(hipMemcpy(d_b, h_b.data(),
        rhs_elems * sizeof(T),
        hipMemcpyHostToDevice));

    /* upload and prepare factors */
    if (cfg.desc.factor == factor_type::lu) {
        /* upload L and U, then reassemble into
           packed LAPACK format on GPU */
        T *d_l, *d_u;
        HIP_CHECK(hipMalloc(&d_l,
            mat_elems * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_u,
            mat_elems * sizeof(T)));

        HIP_CHECK(hipMemcpy(d_l, h_l.data(),
            mat_elems * sizeof(T),
            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_u, h_u.data(),
            mat_elems * sizeof(T),
            hipMemcpyHostToDevice));

        size_t block = 256;
        size_t grid  = (mat_elems + block - 1) / block;
        pack_lu<<<grid, block>>>(
            d_l, d_u, d_factor, n, batch);
        HIP_CHECK(hipGetLastError());

        HIP_CHECK(hipFree(d_l));
        HIP_CHECK(hipFree(d_u));

        HIP_CHECK(hipMemcpy(d_ipiv, h_ipiv.data(),
            batch * n * sizeof(rocblas_int),
            hipMemcpyHostToDevice));
    } else {
        /* Cholesky: upload L directly */
        HIP_CHECK(hipMemcpy(d_factor, h_l.data(),
            mat_elems * sizeof(T),
            hipMemcpyHostToDevice));
    }

    HIP_CHECK(hipStreamSynchronize(ctx.stream));
    std::cerr << "factors loaded and ready\n";

    /* set up profiler and timers */
    profiler prof;
    prof.init(cfg, 1, 1);

    gpu_timer timer;
    timer.init();

    bool do_verify =
        cfg.mode == profile_mode::instrument_verify;

    /* buffer for verification */
    std::vector<T> h_x;
    if (do_verify)
        h_x.resize(rhs_elems);

    /* warmup */
    std::cerr << "warmup (" << cfg.warmup_runs
              << " runs)...\n";
    prof.begin_warmup();
    for (size_t w = 0; w < cfg.warmup_runs; w++) {
        HIP_CHECK(hipMemcpyAsync(d_x, d_b,
            rhs_elems * sizeof(T),
            hipMemcpyDeviceToDevice, ctx.stream));

        if (cfg.desc.factor == factor_type::lu) {
            solve_lu(ctx, d_factor, n, d_ipiv,
                     d_x, nrhs, batch);
        } else {
            solve_cholesky(ctx, d_factor, n,
                           d_x, nrhs, batch);
        }
        HIP_CHECK(hipStreamSynchronize(ctx.stream));
    }
    prof.in_warmup = false;

    /* measured runs */
    std::cerr << "profiling (" << cfg.measured_runs
              << " runs)...\n";
    for (size_t r = 0; r < cfg.measured_runs; r++) {
        prof.begin_run();

        HIP_CHECK(hipMemcpyAsync(d_x, d_b,
            rhs_elems * sizeof(T),
            hipMemcpyDeviceToDevice, ctx.stream));

        timer.start(ctx.stream);
        if (cfg.desc.factor == factor_type::lu) {
            solve_lu(ctx, d_factor, n, d_ipiv,
                     d_x, nrhs, batch);
        } else {
            solve_cholesky(ctx, d_factor, n,
                           d_x, nrhs, batch);
        }
        timer.stop(ctx.stream);
        timer.synchronize();

        prof.record_gpu_time("solve",
                             timer.elapsed_ms());

        if (do_verify) {
            HIP_CHECK(hipMemcpy(h_x.data(), d_x,
                rhs_elems * sizeof(T),
                hipMemcpyDeviceToHost));

            double max_resid = 0.0;
            for (size_t i = 0; i < batch; i++) {
                double res = compute_residual(
                    h_a.data(), h_x.data(),
                    h_b.data(), n, nrhs, i);
                if (res > max_resid) max_resid = res;
            }
            prof.record_metric("residual", max_resid);
        }

        prof.end_run();
    }

    prof.print_summary();
    prof.write_csv();

    /* cleanup */
    timer.destroy();
    prof.destroy();

    if (d_ipiv) HIP_CHECK(hipFree(d_ipiv));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_factor));

    std::cerr << "done.\n";
}

template void run_solve<double>(gpu_context &ctx,
    const profiler_config &cfg);
template void run_solve<float>(gpu_context &ctx,
    const profiler_config &cfg);

/* main */

int main(int argc, char **argv) {
    profiler_cli_args args =
        parse_profiler_args(argc, argv);

    if (args.help) {
        print_profiler_usage(argv[0]);
        return 0;
    }

    gpu_context ctx;
    ctx.init();

    switch (args.config.desc.working_prec) {
        case precision::fp64:
            run_solve<double>(ctx, args.config);
            break;
        case precision::fp32:
            run_solve<float>(ctx, args.config);
            break;
        case precision::fp16:
            std::cerr << "error: fp16 solve not "
                      << "yet implemented\n";
            return 1;
    }

    ctx.destroy();
    return 0;
}
