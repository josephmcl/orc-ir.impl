#include "common/cli.h"
#include "common/artifact.h"
#include "common/profiler.h"
#include "rocm/vendor.h"
#include "rocm/check.h"
#include "rocm/timer.h"

#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

/* -------------------------------------------------------- */
/* forward declarations from factor.cpp                     */

template <typename T>
void solve_lu(gpu_context &ctx, T *d_lu, size_t n,
              rocblas_int *d_ipiv, T *d_b,
              size_t nrhs, size_t batch);

template <typename T>
void solve_cholesky(gpu_context &ctx, T *d_l,
                    size_t n, T *d_b,
                    size_t nrhs, size_t batch);

template <typename T>
void solve_lu_trsm(gpu_context &ctx, T *d_lu,
                   size_t n, rocblas_int *d_ipiv,
                   T *d_b, size_t nrhs,
                   size_t batch);

template <typename T>
void solve_cholesky_trsm(gpu_context &ctx, T *d_l,
                          size_t n, T *d_b,
                          size_t nrhs, size_t batch);

template <typename T>
void apply_pivots(gpu_context &ctx,
                  T *d_b, size_t n,
                  rocblas_int *d_ipiv,
                  size_t nrhs, size_t batch);

template <typename T>
void trsm_step(gpu_context &ctx,
               rocblas_fill fill,
               rocblas_operation trans,
               rocblas_diagonal diag,
               T *d_a, T *d_b,
               size_t n, size_t nrhs,
               size_t batch);

/* -------------------------------------------------------- */
/* forward declarations from ir_solve.cpp                   */

void run_ir3_solve(gpu_context &ctx,
                   const profiler_config &cfg,
                   size_t max_ir_iters);

void run_ir3chol_solve(gpu_context &ctx,
                       const profiler_config &cfg,
                       size_t max_ir_iters);

void run_ir3lu_solve(gpu_context &ctx,
                     const profiler_config &cfg,
                     size_t max_ir_iters);

void run_ir3A_solve(gpu_context &ctx,
                    const profiler_config &cfg,
                    size_t max_ir_iters);

/* -------------------------------------------------------- */
/* solve variant                                            */

enum class solve_variant : uint8_t {
    rocblas,
    rocblastrsv,
    ir3,
    ir3chol,
    ir3lu,
    ir3A
};

/* -------------------------------------------------------- */
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

/* -------------------------------------------------------- */
/* host-side residual: ||Ax - b|| / ||b||
   always computed in fp64 for accuracy */

static double compute_residual(
    const double *A, const double *x, const double *b,
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
                ax += A[mat_off + k * n + i]
                    * x[rhs_off + j * n + k];
            }
            double r = ax - b[rhs_off + j * n + i];
            norm_r += r * r;
            double bi = b[rhs_off + j * n + i];
            norm_b += bi * bi;
        }
    }

    return std::sqrt(norm_r) / std::sqrt(norm_b);
}

/* -------------------------------------------------------- */
/* run_solve: profiled solve workflow

   artifacts are always fp64 on disk. the template
   parameter T sets the working precision for the
   solve. data is cast from fp64 -> T before upload.
   residual verification uses fp64 throughout. */

template <typename T>
void run_solve(gpu_context &ctx,
               const profiler_config &cfg,
               solve_variant variant) {
    size_t n     = cfg.desc.n;
    size_t batch = cfg.desc.batch;
    size_t nrhs  = cfg.desc.nrhs;

    size_t mat_elems = batch * n * n;
    size_t rhs_elems = batch * n * nrhs;

    bool do_verify =
        cfg.mode == profile_mode::instrument_verify;

    /* load fp64 artifacts from disk */
    std::string dir = artifact_directory(cfg.desc);
    std::cerr << "loading artifact from "
              << dir << "/\n";

    /* A and b kept in fp64 for verification */
    std::vector<double> h_a_fp64;
    std::vector<double> h_b_fp64(rhs_elems);

    if (do_verify)
        h_a_fp64.resize(mat_elems);
    if (do_verify)
        read_array(dir + "/a.bin",
                   h_a_fp64.data(),
                   mat_elems * sizeof(double));

    read_array(dir + "/b.bin",
               h_b_fp64.data(),
               rhs_elems * sizeof(double));

    /* cast b to working precision */
    std::vector<T> h_b(rhs_elems);
    for (size_t i = 0; i < rhs_elems; i++)
        h_b[i] = (T)h_b_fp64[i];

    /* load factors, cast to working precision.
       for T=double read directly to avoid a
       large temporary buffer */
    std::vector<T> h_l(mat_elems);
    if constexpr (std::is_same_v<T, double>) {
        read_array(dir + "/l.bin",
                   h_l.data(),
                   mat_elems * sizeof(double));
    } else {
        std::vector<double> tmp(mat_elems);
        read_array(dir + "/l.bin",
                   tmp.data(),
                   mat_elems * sizeof(double));
        for (size_t i = 0; i < mat_elems; i++)
            h_l[i] = (T)tmp[i];
    }

    std::vector<T> h_u;
    std::vector<rocblas_int> h_ipiv;
    if (cfg.desc.factor == factor_type::lu) {
        h_u.resize(mat_elems);
        h_ipiv.resize(batch * n);

        if constexpr (std::is_same_v<T, double>) {
            read_array(dir + "/u.bin",
                       h_u.data(),
                       mat_elems * sizeof(double));
        } else {
            std::vector<double> tmp(mat_elems);
            read_array(dir + "/u.bin",
                       tmp.data(),
                       mat_elems * sizeof(double));
            for (size_t i = 0; i < mat_elems; i++)
                h_u[i] = (T)tmp[i];
        }

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

    /* solve dispatch */
    auto do_solve = [&]() {
        if (cfg.desc.factor == factor_type::lu) {
            if (variant == solve_variant::rocblastrsv)
                solve_lu_trsm(ctx, d_factor, n,
                    d_ipiv, d_x, nrhs, batch);
            else
                solve_lu(ctx, d_factor, n,
                    d_ipiv, d_x, nrhs, batch);
        } else {
            if (variant == solve_variant::rocblastrsv)
                solve_cholesky_trsm(ctx, d_factor,
                    n, d_x, nrhs, batch);
            else
                solve_cholesky(ctx, d_factor, n,
                    d_x, nrhs, batch);
        }
    };

    /* set up profiler and timers */
    bool instrument_phases =
        (variant == solve_variant::rocblastrsv)
        && (cfg.mode == profile_mode::instrument
         || cfg.mode ==
              profile_mode::instrument_verify);

    size_t num_phases = 1;
    if (instrument_phases) {
        if (cfg.desc.factor == factor_type::lu)
            num_phases = 3;
        else
            num_phases = 2;
    }

    profiler prof;
    prof.init(cfg, num_phases,
              do_verify ? 1 : 0);

    gpu_timer timer;
    gpu_timer_pool timer_pool;
    if (instrument_phases)
        timer_pool.init(num_phases);
    else
        timer.init();

    /* pre-allocate verification buffers */
    std::vector<T> h_x_work;
    std::vector<double> h_x_fp64;
    if (do_verify) {
        h_x_work.resize(rhs_elems);
        h_x_fp64.resize(rhs_elems);
    }

    /* warmup */
    std::cerr << "warmup (" << cfg.warmup_runs
              << " runs)...\n";
    prof.begin_warmup();
    for (size_t w = 0; w < cfg.warmup_runs; w++) {
        HIP_CHECK(hipMemcpyAsync(d_x, d_b,
            rhs_elems * sizeof(T),
            hipMemcpyDeviceToDevice, ctx.stream));

        do_solve();
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

        if (instrument_phases) {
            timer_pool.reset();

            if (cfg.desc.factor == factor_type::lu) {
                timer_pool.start(0, ctx.stream);
                apply_pivots(ctx, d_x, n,
                    d_ipiv, nrhs, batch);
                timer_pool.stop(0, ctx.stream);

                timer_pool.start(1, ctx.stream);
                trsm_step(ctx,
                    rocblas_fill_lower,
                    rocblas_operation_none,
                    rocblas_diagonal_unit,
                    d_factor, d_x, n, nrhs,
                    batch);
                timer_pool.stop(1, ctx.stream);

                timer_pool.start(2, ctx.stream);
                trsm_step(ctx,
                    rocblas_fill_upper,
                    rocblas_operation_none,
                    rocblas_diagonal_non_unit,
                    d_factor, d_x, n, nrhs,
                    batch);
                timer_pool.stop(2, ctx.stream);
            } else {
                timer_pool.start(0, ctx.stream);
                trsm_step(ctx,
                    rocblas_fill_lower,
                    rocblas_operation_none,
                    rocblas_diagonal_non_unit,
                    d_factor, d_x, n, nrhs,
                    batch);
                timer_pool.stop(0, ctx.stream);

                timer_pool.start(1, ctx.stream);
                trsm_step(ctx,
                    rocblas_fill_lower,
                    rocblas_operation_transpose,
                    rocblas_diagonal_non_unit,
                    d_factor, d_x, n, nrhs,
                    batch);
                timer_pool.stop(1, ctx.stream);
            }

            timer_pool.synchronize_all();

            if (cfg.desc.factor == factor_type::lu) {
                prof.record_gpu_time("laswp",
                    timer_pool.elapsed_ms(0));
                prof.record_gpu_time("trsm_L",
                    timer_pool.elapsed_ms(1));
                prof.record_gpu_time("trsm_U",
                    timer_pool.elapsed_ms(2));
            } else {
                prof.record_gpu_time("trsm_L",
                    timer_pool.elapsed_ms(0));
                prof.record_gpu_time("trsm_LT",
                    timer_pool.elapsed_ms(1));
            }
        } else {
            timer.start(ctx.stream);
            do_solve();
            timer.stop(ctx.stream);
            timer.synchronize();

            prof.record_gpu_time("solve",
                                 timer.elapsed_ms());
        }

        if (do_verify) {
            HIP_CHECK(hipMemcpy(
                h_x_work.data(), d_x,
                rhs_elems * sizeof(T),
                hipMemcpyDeviceToHost));

            /* cast solution to fp64 for residual */
            for (size_t i = 0; i < rhs_elems; i++)
                h_x_fp64[i] = (double)h_x_work[i];

            double max_resid = 0.0;
            for (size_t i = 0; i < batch; i++) {
                double res = compute_residual(
                    h_a_fp64.data(),
                    h_x_fp64.data(),
                    h_b_fp64.data(),
                    n, nrhs, i);
                if (res > max_resid)
                    max_resid = res;
            }
            prof.record_metric("residual",
                               max_resid);
        }

        prof.end_run();
    }

    prof.print_summary();
    prof.write_csv();

    /* cleanup */
    if (instrument_phases)
        timer_pool.destroy();
    else
        timer.destroy();
    prof.destroy();

    if (d_ipiv) HIP_CHECK(hipFree(d_ipiv));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_factor));

    std::cerr << "done.\n";
}

template void run_solve<double>(gpu_context &ctx,
    const profiler_config &cfg,
    solve_variant variant);
template void run_solve<float>(gpu_context &ctx,
    const profiler_config &cfg,
    solve_variant variant);

/* -------------------------------------------------------- */
/* main                                                     */

int main(int argc, char **argv) {
    profiler_cli_args args =
        parse_profiler_args(argc, argv);

    if (args.help) {
        print_profiler_usage(argv[0]);
        return 0;
    }

    /* parse solver name */
    std::string solver = args.config.solver_name;
    solve_variant variant;
    bool use_fp32 = false;

    if (solver == "rocblas64") {
        variant = solve_variant::rocblas;
    } else if (solver == "rocblas32") {
        variant = solve_variant::rocblas;
        use_fp32 = true;
    } else if (solver == "rocblastrsv64") {
        variant = solve_variant::rocblastrsv;
    } else if (solver == "rocblastrsv32") {
        variant = solve_variant::rocblastrsv;
        use_fp32 = true;
    } else if (solver == "ir3") {
        variant = solve_variant::ir3;
    } else if (solver == "ir3chol") {
        variant = solve_variant::ir3chol;
    } else if (solver == "ir3lu") {
        variant = solve_variant::ir3lu;
    } else if (solver == "ir3A") {
        variant = solve_variant::ir3A;
    } else {
        std::cerr << "error: unknown solver '"
                  << solver << "'\n"
                  << "valid: rocblas64, rocblas32, "
                  << "rocblastrsv64, rocblastrsv32"
                  << ", ir3, ir3chol, ir3lu"
                  << ", ir3A\n";
        return 1;
    }

    if (use_fp32)
        args.config.desc.working_prec =
            precision::fp32;

    gpu_context ctx;
    ctx.init();

    if (variant == solve_variant::ir3) {
        run_ir3_solve(ctx, args.config,
                      args.ir_iters);
    } else if (variant == solve_variant::ir3chol) {
        run_ir3chol_solve(ctx, args.config,
                          args.ir_iters);
    } else if (variant == solve_variant::ir3lu) {
        run_ir3lu_solve(ctx, args.config,
                        args.ir_iters);
    } else if (variant == solve_variant::ir3A) {
        run_ir3A_solve(ctx, args.config,
                       args.ir_iters);
    } else if (use_fp32) {
        run_solve<float>(ctx, args.config,
                         variant);
    } else {
        run_solve<double>(ctx, args.config,
                          variant);
    }

    ctx.destroy();
    return 0;
}
