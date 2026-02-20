#include "common/cli.h"
#include "common/artifact.h"
#include "common/profiler.h"
#include "rocm/vendor.h"
#include "rocm/check.h"
#include "rocm/timer.h"
#include "rocm/ir_kernels.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

/* -------------------------------------------------------- */
/* host-side residual: ||Lx - b|| / ||b||
   L is lower triangular (column-major, zeros
   in upper triangle). computes 2-norm ratio
   for a single batch element.                              */

static double compute_residual_lower(
    const double *L, const double *x,
    const double *b,
    size_t n, size_t batch_idx)
{
    size_t mat_off = batch_idx * n * n;
    size_t rhs_off = batch_idx * n;

    double norm_r = 0.0;
    double norm_b = 0.0;

    for (size_t i = 0; i < n; i++) {
        double lx = 0.0;
        for (size_t k = 0; k <= i; k++)
            lx += L[mat_off + k * n + i]
                * x[rhs_off + k];

        double r = lx - b[rhs_off + i];
        norm_r += r * r;
        double bi = b[rhs_off + i];
        norm_b += bi * bi;
    }

    return std::sqrt(norm_r)
         / std::sqrt(norm_b);
}

/* -------------------------------------------------------- */
/* host-side forward error:
   ||x_computed - x_true||_inf / ||x_true||_inf
   for a single batch element.                              */

static double compute_forward_error(
    const double *x_computed,
    const double *x_true,
    size_t n, size_t batch_idx)
{
    size_t off = batch_idx * n;
    double max_diff = 0.0;
    double max_true = 0.0;

    for (size_t i = 0; i < n; i++) {
        double d = std::abs(
            x_computed[off + i]
            - x_true[off + i]);
        if (d > max_diff) max_diff = d;
        double t = std::abs(x_true[off + i]);
        if (t > max_true) max_true = t;
    }

    if (max_true == 0.0) return max_diff;
    return max_diff / max_true;
}

/* -------------------------------------------------------- */
/* host-side residual: ||L^T x - b|| / ||b||
   L is lower triangular; uses transposed access.
   computes 2-norm ratio for a single batch element.        */

static double compute_residual_lower_transpose(
    const double *L, const double *x,
    const double *b,
    size_t n, size_t batch_idx)
{
    size_t mat_off = batch_idx * n * n;
    size_t rhs_off = batch_idx * n;

    double norm_r = 0.0;
    double norm_b = 0.0;

    for (size_t i = 0; i < n; i++) {
        double ltx = 0.0;
        /* L^T[i,k] = L[k,i] = data[k + i*n] */
        for (size_t k = i; k < n; k++)
            ltx += L[mat_off + k + i * n]
                * x[rhs_off + k];

        double r = ltx - b[rhs_off + i];
        norm_r += r * r;
        double bi = b[rhs_off + i];
        norm_b += bi * bi;
    }

    return std::sqrt(norm_r)
         / std::sqrt(norm_b);
}

/* -------------------------------------------------------- */
/* host-side residual: ||Ux - b|| / ||b||
   U is upper triangular (column-major).
   computes 2-norm ratio for a single batch element.        */

static double compute_residual_upper(
    const double *U, const double *x,
    const double *b,
    size_t n, size_t batch_idx)
{
    size_t mat_off = batch_idx * n * n;
    size_t rhs_off = batch_idx * n;

    double norm_r = 0.0;
    double norm_b = 0.0;

    for (size_t i = 0; i < n; i++) {
        double ux = 0.0;
        /* U[i,k] = data[i + k*n], k >= i */
        for (size_t k = i; k < n; k++)
            ux += U[mat_off + i + k * n]
                * x[rhs_off + k];

        double r = ux - b[rhs_off + i];
        norm_r += r * r;
        double bi = b[rhs_off + i];
        norm_b += bi * bi;
    }

    return std::sqrt(norm_r)
         / std::sqrt(norm_b);
}

/* -------------------------------------------------------- */
/* host-side residual: ||Ax - b|| / ||b||
   A is dense (column-major). nrhs=1.
   computes 2-norm ratio for a single batch element.        */

static double compute_residual_full(
    const double *A, const double *x,
    const double *b,
    size_t n, size_t batch_idx)
{
    size_t mat_off = batch_idx * n * n;
    size_t rhs_off = batch_idx * n;

    double norm_r = 0.0;
    double norm_b = 0.0;

    for (size_t i = 0; i < n; i++) {
        double ax = 0.0;
        for (size_t k = 0; k < n; k++)
            ax += A[mat_off + k * n + i]
                * x[rhs_off + k];

        double r = ax - b[rhs_off + i];
        norm_r += r * r;
        double bi = b[rhs_off + i];
        norm_b += bi * bi;
    }

    return std::sqrt(norm_r)
         / std::sqrt(norm_b);
}

/* -------------------------------------------------------- */
/* run_ir3_solve: Higham Algorithm 4
   3-precision iterative refinement

   precisions:
     fp64: solution x, residual r = b - L*x
     fp32: TRSV accumulation (inside kernels)
     fp16: L storage, RHS demotion

   the system being solved is L*x = b where L
   is the Cholesky factor stored in the artifact.

   requires: --factor chol, nrhs=1                         */

void run_ir3_solve(gpu_context &ctx,
                   const profiler_config &cfg,
                   size_t max_ir_iters)
{
    size_t n     = cfg.desc.n;
    size_t batch = cfg.desc.batch;
    size_t nrhs  = cfg.desc.nrhs;

    if (nrhs != 1) {
        std::cerr << "error: ir3 solver requires "
                  << "nrhs=1\n";
        exit(1);
    }
    if (cfg.desc.factor != factor_type::cholesky) {
        std::cerr << "error: ir3 solver requires "
                  << "--factor chol\n";
        exit(1);
    }

    size_t mat_elems = batch * n * n;
    size_t vec_elems = batch * n;

    bool instrument =
        cfg.mode == profile_mode::instrument
     || cfg.mode == profile_mode::instrument_verify;
    bool do_verify =
        cfg.mode == profile_mode::instrument_verify;

    /* ---------------------------------------------------- */
    /* load fp64 artifacts from disk                        */

    std::string dir = artifact_directory(cfg.desc);
    std::cerr << "loading artifact from "
              << dir << "/\n";

    std::vector<double> h_l(mat_elems);
    std::vector<double> h_b(vec_elems);

    read_array(dir + "/l.bin",
               h_l.data(),
               mat_elems * sizeof(double));
    read_array(dir + "/b.bin",
               h_b.data(),
               vec_elems * sizeof(double));

    /* load true solution for forward error */
    std::vector<double> h_x_true;
    if (do_verify) {
        h_x_true.resize(vec_elems);
        read_array(dir + "/x_l.bin",
                   h_x_true.data(),
                   vec_elems * sizeof(double));
    }

    /* ---------------------------------------------------- */
    /* upload L (fp64) to GPU for dgemv residual            */

    double *d_l;
    HIP_CHECK(hipMalloc(&d_l,
        mat_elems * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_l, h_l.data(),
        mat_elems * sizeof(double),
        hipMemcpyHostToDevice));

    /* ---------------------------------------------------- */
    /* compute scale_A and demote L to fp16                 */

    std::cerr << "computing scaling factors...\n";
    double max_abs_l = gpu_compute_max_abs(
        d_l, (long long)mat_elems);
    double scale_A = compute_optimal_fp16_scale(
        max_abs_l);
    double inv_scale_A = 1.0 / scale_A;

    std::cerr << "  max(|L|) = " << max_abs_l
              << ", scale_A = " << scale_A << "\n";

    double *d_scale_A;
    HIP_CHECK(hipMalloc(&d_scale_A,
        sizeof(double)));
    HIP_CHECK(hipMemcpy(d_scale_A, &scale_A,
        sizeof(double), hipMemcpyHostToDevice));

    _Float16 *d_l16;
    HIP_CHECK(hipMalloc(&d_l16,
        mat_elems * sizeof(_Float16)));

    {
        int tpb = 256;
        long long size = (long long)mat_elems;
        int nblocks =
            (int)((size + tpb - 1) / tpb);

        if (n >= (size_t)LARGE_PROBLEM_THRESHOLD) {
            hipLaunchKernelGGL(
                (gpu_scale_and_demote_kernel<
                    long long>),
                dim3(nblocks), dim3(tpb),
                0, ctx.stream,
                d_l, d_l16, d_scale_A, size);
        } else {
            hipLaunchKernelGGL(
                (gpu_scale_and_demote_kernel<int>),
                dim3(nblocks), dim3(tpb),
                0, ctx.stream,
                d_l, d_l16, d_scale_A,
                (int)size);
        }
        HIP_CHECK(hipGetLastError());
    }

    /* ---------------------------------------------------- */
    /* allocate solver workspace                            */

    double *d_b, *d_x64, *d_r;
    _Float16 *d_x16;
    float *d_workspace, *d_rhs_workspace;
    double *d_block_maxes, *d_scale_b;
    double *d_correction_scale, *d_norm;

    HIP_CHECK(hipMalloc(&d_b,
        vec_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_x64,
        vec_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_r,
        vec_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_x16,
        vec_elems * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_workspace,
        vec_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rhs_workspace,
        vec_elems * sizeof(float)));

    int tpb = 256;
    int nblocks_r =
        (int)((vec_elems + tpb - 1) / tpb);
    size_t shmem_size = sizeof(double) * tpb;

    HIP_CHECK(hipMalloc(&d_block_maxes,
        sizeof(double) * nblocks_r));
    HIP_CHECK(hipMalloc(&d_scale_b,
        sizeof(double)));
    HIP_CHECK(hipMalloc(&d_correction_scale,
        sizeof(double)));
    HIP_CHECK(hipMalloc(&d_norm,
        sizeof(double)));

    /* upload b */
    HIP_CHECK(hipMemcpy(d_b, h_b.data(),
        vec_elems * sizeof(double),
        hipMemcpyHostToDevice));

    HIP_CHECK(hipStreamSynchronize(ctx.stream));
    std::cerr << "IR3 solver ready (iters="
              << max_ir_iters << ")\n";

    /* ---------------------------------------------------- */
    /* dgemv parameters                                     */

    int lda = (int)n;
    long long stride_a = (long long)n * n;
    long long stride_x = (long long)n;

    double alpha_neg = -1.0;
    double beta_one  =  1.0;

    /* ---------------------------------------------------- */
    /* IR phase lambdas (GPU work only, no sync)            */

    auto phase_residual = [&]() {
        for (size_t bi = 0; bi < batch; bi++) {
            HIP_CHECK(hipMemcpyAsync(
                d_r + bi * stride_x,
                d_b + bi * stride_x,
                n * sizeof(double),
                hipMemcpyDeviceToDevice,
                ctx.stream));

            ROCBLAS_CHECK(rocblas_dgemv(
                ctx.blas_handle,
                rocblas_operation_none,
                (rocblas_int)n, (rocblas_int)n,
                &alpha_neg,
                d_l + bi * stride_a, lda,
                d_x64 + bi * stride_x, 1,
                &beta_one,
                d_r + bi * stride_x, 1));
        }
    };

    auto phase_scale = [&]() {
        hipLaunchKernelGGL(
            gpu_max_norm_kernel,
            dim3(nblocks_r), dim3(tpb),
            shmem_size, ctx.stream,
            d_r, d_block_maxes,
            (int)(vec_elems));

        hipLaunchKernelGGL(
            gpu_compute_pow2_scale_kernel,
            dim3(1), dim3(1),
            0, ctx.stream,
            d_block_maxes, d_scale_b,
            d_norm, nblocks_r);

        hipLaunchKernelGGL(
            gpu_apply_scale_and_demote_kernel,
            dim3(nblocks_r), dim3(tpb),
            0, ctx.stream,
            d_r, d_x16, d_scale_b,
            (int)(vec_elems));
    };

    auto phase_trsv = [&]() {
        launch_trsv_multiCU_fp16<
            MC_TRSV_TILE_SIZE, MC_TRSV_TPB>(
            ctx.stream,
            (int)n, lda, stride_a, d_l16,
            stride_x, d_x16, (int)batch,
            d_rhs_workspace, d_workspace);
    };

    auto phase_update = [&]() {
        hipLaunchKernelGGL(
            gpu_scalar_multiply_kernel,
            dim3(1), dim3(1),
            0, ctx.stream,
            d_correction_scale,
            d_scale_b, inv_scale_A);

        hipLaunchKernelGGL(
            gpu_promote_and_add_kernel_devscale,
            dim3(nblocks_r), dim3(tpb),
            0, ctx.stream,
            d_x16, d_x64,
            d_correction_scale,
            (int)(vec_elems));
    };

    /* ---------------------------------------------------- */
    /* persistent labels for per-iteration recording        */

    std::vector<std::string> time_strs;
    std::vector<const char *> time_labels;
    std::vector<std::string> verify_strs;
    std::vector<const char *> verify_labels;

    if (instrument) {
        time_strs.resize(4 * max_ir_iters);
        time_labels.resize(4 * max_ir_iters);
        for (size_t i = 0; i < max_ir_iters; i++) {
            std::string pfx =
                "iter" + std::to_string(i) + "_";
            time_strs[4*i+0] = pfx + "residual";
            time_strs[4*i+1] = pfx + "scale";
            time_strs[4*i+2] = pfx + "trsv";
            time_strs[4*i+3] = pfx + "update";
            for (int j = 0; j < 4; j++)
                time_labels[4*i+j] =
                    time_strs[4*i+j].c_str();
        }
    }

    if (do_verify) {
        verify_strs.resize(2 * max_ir_iters);
        verify_labels.resize(2 * max_ir_iters);
        for (size_t i = 0; i < max_ir_iters; i++) {
            std::string pfx =
                "iter" + std::to_string(i) + "_";
            verify_strs[2*i+0] = pfx + "bwd_error";
            verify_strs[2*i+1] = pfx + "fwd_error";
            for (int j = 0; j < 2; j++)
                verify_labels[2*i+j] =
                    verify_strs[2*i+j].c_str();
        }
    }

    /* ---------------------------------------------------- */
    /* set up profiler and timers                           */

    size_t events_per_run = instrument
        ? 4 * max_ir_iters : 1;
    size_t verify_per_run = do_verify
        ? 2 * max_ir_iters : 0;

    profiler prof;
    prof.init(cfg, events_per_run, verify_per_run);

    gpu_timer timer;
    gpu_timer_pool timers;
    if (instrument)
        timers.init(4);
    else
        timer.init();

    std::vector<double> h_x64;
    if (do_verify)
        h_x64.resize(vec_elems);

    /* ---------------------------------------------------- */
    /* warmup                                               */

    std::cerr << "warmup (" << cfg.warmup_runs
              << " runs)...\n";
    prof.begin_warmup();
    for (size_t w = 0; w < cfg.warmup_runs; w++) {
        HIP_CHECK(hipMemsetAsync(d_x64, 0,
            vec_elems * sizeof(double),
            ctx.stream));

        for (size_t iter = 0;
             iter < max_ir_iters; iter++) {
            phase_residual();
            phase_scale();
            phase_trsv();
            phase_update();
        }

        HIP_CHECK(
            hipStreamSynchronize(ctx.stream));
    }
    prof.in_warmup = false;

    /* ---------------------------------------------------- */
    /* measured runs                                        */

    std::cerr << "profiling (" << cfg.measured_runs
              << " runs)...\n";
    for (size_t r = 0; r < cfg.measured_runs; r++) {
        prof.begin_run();

        HIP_CHECK(hipMemsetAsync(d_x64, 0,
            vec_elems * sizeof(double),
            ctx.stream));

        if (instrument) {
            HIP_CHECK(hipStreamSynchronize(
                ctx.stream));

            for (size_t iter = 0;
                 iter < max_ir_iters; iter++) {
                timers.reset();

                timers.start(0, ctx.stream);
                phase_residual();
                timers.stop(0, ctx.stream);

                timers.start(1, ctx.stream);
                phase_scale();
                timers.stop(1, ctx.stream);

                timers.start(2, ctx.stream);
                phase_trsv();
                timers.stop(2, ctx.stream);

                timers.start(3, ctx.stream);
                phase_update();
                timers.stop(3, ctx.stream);

                timers.synchronize_all();

                prof.record_gpu_time(
                    time_labels[4*iter+0],
                    timers.elapsed_ms(0));
                prof.record_gpu_time(
                    time_labels[4*iter+1],
                    timers.elapsed_ms(1));
                prof.record_gpu_time(
                    time_labels[4*iter+2],
                    timers.elapsed_ms(2));
                prof.record_gpu_time(
                    time_labels[4*iter+3],
                    timers.elapsed_ms(3));

                if (do_verify) {
                    HIP_CHECK(hipMemcpy(
                        h_x64.data(), d_x64,
                        vec_elems * sizeof(double),
                        hipMemcpyDeviceToHost));

                    double max_bwd = 0.0;
                    double max_fwd = 0.0;
                    for (size_t bi = 0;
                         bi < batch; bi++) {
                        double bwd =
                            compute_residual_lower(
                                h_l.data(),
                                h_x64.data(),
                                h_b.data(),
                                n, bi);
                        if (bwd > max_bwd)
                            max_bwd = bwd;
                        double fwd =
                            compute_forward_error(
                                h_x64.data(),
                                h_x_true.data(),
                                n, bi);
                        if (fwd > max_fwd)
                            max_fwd = fwd;
                    }

                    prof.record_metric(
                        verify_labels[2*iter+0],
                        max_bwd);
                    prof.record_metric(
                        verify_labels[2*iter+1],
                        max_fwd);
                }
            }
        } else {
            /* profile mode: single timer around
               entire IR solve */
            timer.start(ctx.stream);

            for (size_t iter = 0;
                 iter < max_ir_iters; iter++) {
                phase_residual();
                phase_scale();
                phase_trsv();
                phase_update();
            }

            timer.stop(ctx.stream);
            timer.synchronize();
            prof.record_gpu_time("ir_solve",
                timer.elapsed_ms());
        }

        prof.end_run();
    }

    prof.print_summary();
    prof.write_csv();

    /* ---------------------------------------------------- */
    /* cleanup                                              */

    if (instrument)
        timers.destroy();
    else
        timer.destroy();
    prof.destroy();

    HIP_CHECK(hipFree(d_norm));
    HIP_CHECK(hipFree(d_correction_scale));
    HIP_CHECK(hipFree(d_scale_b));
    HIP_CHECK(hipFree(d_block_maxes));
    HIP_CHECK(hipFree(d_rhs_workspace));
    HIP_CHECK(hipFree(d_workspace));
    HIP_CHECK(hipFree(d_x16));
    HIP_CHECK(hipFree(d_r));
    HIP_CHECK(hipFree(d_x64));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_l16));
    HIP_CHECK(hipFree(d_scale_A));
    HIP_CHECK(hipFree(d_l));

    std::cerr << "done.\n";
}

/* -------------------------------------------------------- */
/* run_ir3chol_solve: two-phase Higham IR for the
   full Cholesky system A = L*L^T.
     phase 1: L * y = b      (forward IR)
     phase 2: L^T * x = y    (backward IR)

   uses the same fp16 L for both phases.
   requires: --factor chol, nrhs=1                         */

void run_ir3chol_solve(gpu_context &ctx,
                       const profiler_config &cfg,
                       size_t max_ir_iters)
{
    size_t n     = cfg.desc.n;
    size_t batch = cfg.desc.batch;
    size_t nrhs  = cfg.desc.nrhs;

    if (nrhs != 1) {
        std::cerr << "error: ir3chol solver requires "
                  << "nrhs=1\n";
        exit(1);
    }
    if (cfg.desc.factor != factor_type::cholesky) {
        std::cerr << "error: ir3chol solver requires "
                  << "--factor chol\n";
        exit(1);
    }

    size_t mat_elems = batch * n * n;
    size_t vec_elems = batch * n;

    bool instrument =
        cfg.mode == profile_mode::instrument
     || cfg.mode == profile_mode::instrument_verify;
    bool do_verify =
        cfg.mode == profile_mode::instrument_verify;

    /* ---------------------------------------------------- */
    /* load fp64 artifacts from disk                        */

    std::string dir = artifact_directory(cfg.desc);
    std::cerr << "loading artifact from "
              << dir << "/\n";

    std::vector<double> h_l(mat_elems);
    std::vector<double> h_b(vec_elems);

    read_array(dir + "/l.bin",
               h_l.data(),
               mat_elems * sizeof(double));
    read_array(dir + "/b.bin",
               h_b.data(),
               vec_elems * sizeof(double));

    /* true solutions for per-phase forward error */
    std::vector<double> h_x_l;
    std::vector<double> h_x_true;
    std::vector<double> h_a;
    if (do_verify) {
        h_x_l.resize(vec_elems);
        read_array(dir + "/x_l.bin",
                   h_x_l.data(),
                   vec_elems * sizeof(double));
        h_x_true.resize(vec_elems);
        read_array(dir + "/x.bin",
                   h_x_true.data(),
                   vec_elems * sizeof(double));
        h_a.resize(mat_elems);
        read_array(dir + "/a.bin",
                   h_a.data(),
                   mat_elems * sizeof(double));
    }

    /* ---------------------------------------------------- */
    /* upload L (fp64) to GPU for dgemv residual            */

    double *d_l;
    HIP_CHECK(hipMalloc(&d_l,
        mat_elems * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_l, h_l.data(),
        mat_elems * sizeof(double),
        hipMemcpyHostToDevice));

    /* ---------------------------------------------------- */
    /* compute scale_A and demote L to fp16                 */

    std::cerr << "computing scaling factors...\n";
    double max_abs_l = gpu_compute_max_abs(
        d_l, (long long)mat_elems);
    double scale_A = compute_optimal_fp16_scale(
        max_abs_l);
    double inv_scale_A = 1.0 / scale_A;

    std::cerr << "  max(|L|) = " << max_abs_l
              << ", scale_A = " << scale_A << "\n";

    double *d_scale_A;
    HIP_CHECK(hipMalloc(&d_scale_A,
        sizeof(double)));
    HIP_CHECK(hipMemcpy(d_scale_A, &scale_A,
        sizeof(double), hipMemcpyHostToDevice));

    _Float16 *d_l16;
    HIP_CHECK(hipMalloc(&d_l16,
        mat_elems * sizeof(_Float16)));

    {
        int tpb = 256;
        long long size = (long long)mat_elems;
        int nblocks =
            (int)((size + tpb - 1) / tpb);

        if (n >= (size_t)LARGE_PROBLEM_THRESHOLD) {
            hipLaunchKernelGGL(
                (gpu_scale_and_demote_kernel<
                    long long>),
                dim3(nblocks), dim3(tpb),
                0, ctx.stream,
                d_l, d_l16, d_scale_A, size);
        } else {
            hipLaunchKernelGGL(
                (gpu_scale_and_demote_kernel<int>),
                dim3(nblocks), dim3(tpb),
                0, ctx.stream,
                d_l, d_l16, d_scale_A,
                (int)size);
        }
        HIP_CHECK(hipGetLastError());
    }

    /* ---------------------------------------------------- */
    /* allocate solver workspace                            */

    double *d_b, *d_x64, *d_r;
    _Float16 *d_x16;
    float *d_workspace, *d_rhs_workspace;
    double *d_block_maxes, *d_scale_b;
    double *d_correction_scale, *d_norm;

    HIP_CHECK(hipMalloc(&d_b,
        vec_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_x64,
        vec_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_r,
        vec_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_x16,
        vec_elems * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_workspace,
        vec_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rhs_workspace,
        vec_elems * sizeof(float)));

    int tpb = 256;
    int nblocks_r =
        (int)((vec_elems + tpb - 1) / tpb);
    size_t shmem_size = sizeof(double) * tpb;

    HIP_CHECK(hipMalloc(&d_block_maxes,
        sizeof(double) * nblocks_r));
    HIP_CHECK(hipMalloc(&d_scale_b,
        sizeof(double)));
    HIP_CHECK(hipMalloc(&d_correction_scale,
        sizeof(double)));
    HIP_CHECK(hipMalloc(&d_norm,
        sizeof(double)));

    /* upload b */
    HIP_CHECK(hipMemcpy(d_b, h_b.data(),
        vec_elems * sizeof(double),
        hipMemcpyHostToDevice));

    HIP_CHECK(hipStreamSynchronize(ctx.stream));
    std::cerr << "IR3 cholesky solver ready (iters="
              << max_ir_iters << ")\n";

    /* ---------------------------------------------------- */
    /* dgemv parameters                                     */

    int lda = (int)n;
    long long stride_a = (long long)n * n;
    long long stride_x = (long long)n;

    double alpha_neg = -1.0;
    double beta_one  =  1.0;

    /* ---------------------------------------------------- */
    /* IR phase lambdas
       parameterized by gemv operation and
       TRSV direction (set per-phase below)                 */

    rocblas_operation cur_gemv_op;

    auto phase_residual = [&]() {
        for (size_t bi = 0; bi < batch; bi++) {
            HIP_CHECK(hipMemcpyAsync(
                d_r + bi * stride_x,
                d_b + bi * stride_x,
                n * sizeof(double),
                hipMemcpyDeviceToDevice,
                ctx.stream));

            ROCBLAS_CHECK(rocblas_dgemv(
                ctx.blas_handle,
                cur_gemv_op,
                (rocblas_int)n, (rocblas_int)n,
                &alpha_neg,
                d_l + bi * stride_a, lda,
                d_x64 + bi * stride_x, 1,
                &beta_one,
                d_r + bi * stride_x, 1));
        }
    };

    auto phase_scale = [&]() {
        hipLaunchKernelGGL(
            gpu_max_norm_kernel,
            dim3(nblocks_r), dim3(tpb),
            shmem_size, ctx.stream,
            d_r, d_block_maxes,
            (int)(vec_elems));

        hipLaunchKernelGGL(
            gpu_compute_pow2_scale_kernel,
            dim3(1), dim3(1),
            0, ctx.stream,
            d_block_maxes, d_scale_b,
            d_norm, nblocks_r);

        hipLaunchKernelGGL(
            gpu_apply_scale_and_demote_kernel,
            dim3(nblocks_r), dim3(tpb),
            0, ctx.stream,
            d_r, d_x16, d_scale_b,
            (int)(vec_elems));
    };

    auto phase_trsv_forward = [&]() {
        launch_trsv_multiCU_fp16<
            MC_TRSV_TILE_SIZE, MC_TRSV_TPB>(
            ctx.stream,
            (int)n, lda, stride_a, d_l16,
            stride_x, d_x16, (int)batch,
            d_rhs_workspace, d_workspace);
    };

    auto phase_trsv_backward = [&]() {
        launch_trsv_multiCU_fp16_backward<
            MC_TRSV_TILE_SIZE, MC_TRSV_TPB,
            true>(
            ctx.stream,
            (int)n, lda, stride_a, d_l16,
            stride_x, d_x16, (int)batch,
            d_rhs_workspace, d_workspace);
    };

    auto phase_update = [&]() {
        hipLaunchKernelGGL(
            gpu_scalar_multiply_kernel,
            dim3(1), dim3(1),
            0, ctx.stream,
            d_correction_scale,
            d_scale_b, inv_scale_A);

        hipLaunchKernelGGL(
            gpu_promote_and_add_kernel_devscale,
            dim3(nblocks_r), dim3(tpb),
            0, ctx.stream,
            d_x16, d_x64,
            d_correction_scale,
            (int)(vec_elems));
    };

    /* ---------------------------------------------------- */
    /* persistent labels for per-iteration recording
       p1_iter0_residual, ..., p2_iter0_residual, ...       */

    size_t iters_total = max_ir_iters * 2;

    std::vector<std::string> time_strs;
    std::vector<const char *> time_labels;
    std::vector<std::string> verify_strs;
    std::vector<const char *> verify_labels;

    if (instrument) {
        time_strs.resize(4 * iters_total);
        time_labels.resize(4 * iters_total);
        const char *ppfx[2] = {"p1_", "p2_"};
        for (int p = 0; p < 2; p++) {
            size_t base = p * 4 * max_ir_iters;
            for (size_t i = 0;
                 i < max_ir_iters; i++) {
                std::string pfx =
                    std::string(ppfx[p])
                    + "iter" + std::to_string(i)
                    + "_";
                time_strs[base+4*i+0] =
                    pfx + "residual";
                time_strs[base+4*i+1] =
                    pfx + "scale";
                time_strs[base+4*i+2] =
                    pfx + "trsv";
                time_strs[base+4*i+3] =
                    pfx + "update";
                for (int j = 0; j < 4; j++)
                    time_labels[base+4*i+j] =
                        time_strs[base+4*i+j]
                            .c_str();
            }
        }
    }

    if (do_verify) {
        /* 2 metrics per iter per phase + 1 final */
        size_t nv = 2 * iters_total + 1;
        verify_strs.resize(nv);
        verify_labels.resize(nv);
        const char *ppfx[2] = {"p1_", "p2_"};
        for (int p = 0; p < 2; p++) {
            size_t base = p * 2 * max_ir_iters;
            for (size_t i = 0;
                 i < max_ir_iters; i++) {
                std::string pfx =
                    std::string(ppfx[p])
                    + "iter" + std::to_string(i)
                    + "_";
                verify_strs[base+2*i+0] =
                    pfx + "bwd_error";
                verify_strs[base+2*i+1] =
                    pfx + "fwd_error";
                for (int j = 0; j < 2; j++)
                    verify_labels[base+2*i+j] =
                        verify_strs[base+2*i+j]
                            .c_str();
            }
        }
        size_t fi = 2 * iters_total;
        verify_strs[fi] = "final_bwd_error";
        verify_labels[fi] =
            verify_strs[fi].c_str();
    }

    /* ---------------------------------------------------- */
    /* set up profiler and timers                           */

    size_t events_per_run = instrument
        ? 4 * iters_total : 2;
    size_t verify_per_run = do_verify
        ? 2 * iters_total + 1 : 0;

    profiler prof;
    prof.init(cfg, events_per_run, verify_per_run);

    gpu_timer timer;
    gpu_timer_pool timers;
    if (instrument)
        timers.init(4);
    else
        timer.init();

    std::vector<double> h_x64;
    std::vector<double> h_y;
    if (do_verify) {
        h_x64.resize(vec_elems);
        h_y.resize(vec_elems);
    }

    /* ---------------------------------------------------- */
    /* warmup                                               */

    std::cerr << "warmup (" << cfg.warmup_runs
              << " runs)...\n";
    prof.begin_warmup();
    for (size_t w = 0; w < cfg.warmup_runs; w++) {
        /* reset: reload original b, zero x */
        HIP_CHECK(hipMemcpyAsync(d_b, h_b.data(),
            vec_elems * sizeof(double),
            hipMemcpyHostToDevice, ctx.stream));
        HIP_CHECK(hipMemsetAsync(d_x64, 0,
            vec_elems * sizeof(double),
            ctx.stream));

        /* phase 1: L*y = b (forward) */
        cur_gemv_op = rocblas_operation_none;
        for (size_t iter = 0;
             iter < max_ir_iters; iter++) {
            phase_residual();
            phase_scale();
            phase_trsv_forward();
            phase_update();
        }

        /* transition: y -> d_b, zero d_x64 */
        HIP_CHECK(hipMemcpyAsync(d_b, d_x64,
            vec_elems * sizeof(double),
            hipMemcpyDeviceToDevice, ctx.stream));
        HIP_CHECK(hipMemsetAsync(d_x64, 0,
            vec_elems * sizeof(double),
            ctx.stream));

        /* phase 2: L^T*x = y (backward) */
        cur_gemv_op = rocblas_operation_transpose;
        for (size_t iter = 0;
             iter < max_ir_iters; iter++) {
            phase_residual();
            phase_scale();
            phase_trsv_backward();
            phase_update();
        }

        HIP_CHECK(
            hipStreamSynchronize(ctx.stream));
    }
    prof.in_warmup = false;

    /* ---------------------------------------------------- */
    /* measured runs                                        */

    std::cerr << "profiling (" << cfg.measured_runs
              << " runs)...\n";

    /* host residual function pointers per phase */
    typedef double (*residual_fn)(
        const double *, const double *,
        const double *, size_t, size_t);
    residual_fn host_resid[2] = {
        compute_residual_lower,
        compute_residual_lower_transpose
    };

    for (size_t r = 0; r < cfg.measured_runs; r++) {
        prof.begin_run();

        /* reset: reload original b, zero x */
        HIP_CHECK(hipMemcpyAsync(d_b, h_b.data(),
            vec_elems * sizeof(double),
            hipMemcpyHostToDevice, ctx.stream));
        HIP_CHECK(hipMemsetAsync(d_x64, 0,
            vec_elems * sizeof(double),
            ctx.stream));

        for (int phase = 0; phase < 2; phase++) {
            cur_gemv_op = (phase == 0)
                ? rocblas_operation_none
                : rocblas_operation_transpose;

            const double *h_true_ptr =
                (phase == 0)
                    ? h_x_l.data()
                    : h_x_true.data();
            const double *h_rhs_ptr =
                (phase == 0)
                    ? h_b.data()
                    : h_y.data();

            size_t label_base =
                phase * 4 * max_ir_iters;
            size_t vlabel_base =
                phase * 2 * max_ir_iters;

            if (instrument) {
                HIP_CHECK(hipStreamSynchronize(
                    ctx.stream));

                for (size_t iter = 0;
                     iter < max_ir_iters;
                     iter++) {
                    timers.reset();

                    timers.start(0, ctx.stream);
                    phase_residual();
                    timers.stop(0, ctx.stream);

                    timers.start(1, ctx.stream);
                    phase_scale();
                    timers.stop(1, ctx.stream);

                    timers.start(2, ctx.stream);
                    if (phase == 0)
                        phase_trsv_forward();
                    else
                        phase_trsv_backward();
                    timers.stop(2, ctx.stream);

                    timers.start(3, ctx.stream);
                    phase_update();
                    timers.stop(3, ctx.stream);

                    timers.synchronize_all();

                    size_t ti =
                        label_base + 4 * iter;
                    prof.record_gpu_time(
                        time_labels[ti+0],
                        timers.elapsed_ms(0));
                    prof.record_gpu_time(
                        time_labels[ti+1],
                        timers.elapsed_ms(1));
                    prof.record_gpu_time(
                        time_labels[ti+2],
                        timers.elapsed_ms(2));
                    prof.record_gpu_time(
                        time_labels[ti+3],
                        timers.elapsed_ms(3));

                    if (do_verify) {
                        HIP_CHECK(hipMemcpy(
                            h_x64.data(), d_x64,
                            vec_elems
                                * sizeof(double),
                            hipMemcpyDeviceToHost));

                        double max_bwd = 0.0;
                        double max_fwd = 0.0;
                        for (size_t bi = 0;
                             bi < batch; bi++) {
                            double bwd =
                                host_resid[phase](
                                    h_l.data(),
                                    h_x64.data(),
                                    h_rhs_ptr,
                                    n, bi);
                            if (bwd > max_bwd)
                                max_bwd = bwd;
                            double fwd =
                                compute_forward_error(
                                    h_x64.data(),
                                    h_true_ptr,
                                    n, bi);
                            if (fwd > max_fwd)
                                max_fwd = fwd;
                        }

                        size_t vi =
                            vlabel_base + 2 * iter;
                        prof.record_metric(
                            verify_labels[vi+0],
                            max_bwd);
                        prof.record_metric(
                            verify_labels[vi+1],
                            max_fwd);
                    }
                }
            } else {
                /* profile mode: one timer per
                   phase */
                timer.start(ctx.stream);

                for (size_t iter = 0;
                     iter < max_ir_iters;
                     iter++) {
                    phase_residual();
                    phase_scale();
                    if (phase == 0)
                        phase_trsv_forward();
                    else
                        phase_trsv_backward();
                    phase_update();
                }

                timer.stop(ctx.stream);
                timer.synchronize();
                prof.record_gpu_time(
                    (phase == 0)
                        ? "phase1" : "phase2",
                    timer.elapsed_ms());
            }

            /* transition after phase 1 */
            if (phase == 0) {
                if (do_verify) {
                    HIP_CHECK(hipMemcpy(
                        h_y.data(), d_x64,
                        vec_elems
                            * sizeof(double),
                        hipMemcpyDeviceToHost));
                }
                HIP_CHECK(hipMemcpyAsync(
                    d_b, d_x64,
                    vec_elems * sizeof(double),
                    hipMemcpyDeviceToDevice,
                    ctx.stream));
                HIP_CHECK(hipMemsetAsync(
                    d_x64, 0,
                    vec_elems * sizeof(double),
                    ctx.stream));
            }
        }

        /* final verification: ||Ax - b|| / ||b|| */
        if (do_verify) {
            HIP_CHECK(hipMemcpy(
                h_x64.data(), d_x64,
                vec_elems * sizeof(double),
                hipMemcpyDeviceToHost));

            double max_final = 0.0;
            for (size_t bi = 0;
                 bi < batch; bi++) {
                double res =
                    compute_residual_full(
                        h_a.data(),
                        h_x64.data(),
                        h_b.data(),
                        n, bi);
                if (res > max_final)
                    max_final = res;
            }
            prof.record_metric(
                verify_labels[2 * iters_total],
                max_final);
        }

        prof.end_run();
    }

    prof.print_summary();
    prof.write_csv();

    /* ---------------------------------------------------- */
    /* cleanup                                              */

    if (instrument)
        timers.destroy();
    else
        timer.destroy();
    prof.destroy();

    HIP_CHECK(hipFree(d_norm));
    HIP_CHECK(hipFree(d_correction_scale));
    HIP_CHECK(hipFree(d_scale_b));
    HIP_CHECK(hipFree(d_block_maxes));
    HIP_CHECK(hipFree(d_rhs_workspace));
    HIP_CHECK(hipFree(d_workspace));
    HIP_CHECK(hipFree(d_x16));
    HIP_CHECK(hipFree(d_r));
    HIP_CHECK(hipFree(d_x64));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_l16));
    HIP_CHECK(hipFree(d_scale_A));
    HIP_CHECK(hipFree(d_l));

    std::cerr << "done.\n";
}

/* -------------------------------------------------------- */
/* forward declaration from factor.cpp                      */

template <typename T>
void apply_pivots(gpu_context &ctx,
                  T *d_b, size_t n,
                  rocblas_int *d_ipiv,
                  size_t nrhs, size_t batch);

/* -------------------------------------------------------- */
/* run_ir3lu_solve: two-phase Higham IR for the
   full LU system PA = LU.
     phase 1: L * y = P*b    (forward IR, unit diag)
     phase 2: U * x = y      (backward IR)

   separate fp16 L and fp16 U with independent
   scaling. requires: --factor lu, nrhs=1                   */

void run_ir3lu_solve(gpu_context &ctx,
                     const profiler_config &cfg,
                     size_t max_ir_iters)
{
    size_t n     = cfg.desc.n;
    size_t batch = cfg.desc.batch;
    size_t nrhs  = cfg.desc.nrhs;

    if (nrhs != 1) {
        std::cerr << "error: ir3lu solver requires "
                  << "nrhs=1\n";
        exit(1);
    }
    if (cfg.desc.factor != factor_type::lu) {
        std::cerr << "error: ir3lu solver requires "
                  << "--factor lu\n";
        exit(1);
    }

    size_t mat_elems = batch * n * n;
    size_t vec_elems = batch * n;

    bool instrument =
        cfg.mode == profile_mode::instrument
     || cfg.mode == profile_mode::instrument_verify;
    bool do_verify =
        cfg.mode == profile_mode::instrument_verify;

    /* ---------------------------------------------------- */
    /* load fp64 artifacts from disk                        */

    std::string dir = artifact_directory(cfg.desc);
    std::cerr << "loading artifact from "
              << dir << "/\n";

    std::vector<double> h_l(mat_elems);
    std::vector<double> h_u(mat_elems);
    std::vector<double> h_b(vec_elems);
    std::vector<rocblas_int> h_ipiv(batch * n);

    read_array(dir + "/l.bin",
               h_l.data(),
               mat_elems * sizeof(double));
    read_array(dir + "/u.bin",
               h_u.data(),
               mat_elems * sizeof(double));
    read_array(dir + "/b.bin",
               h_b.data(),
               vec_elems * sizeof(double));
    read_array(dir + "/ipiv.bin",
               h_ipiv.data(),
               batch * n * sizeof(rocblas_int));

    /* true solutions for per-phase forward error */
    std::vector<double> h_x_u;
    std::vector<double> h_x_true;
    std::vector<double> h_a;
    if (do_verify) {
        h_x_u.resize(vec_elems);
        read_array(dir + "/x_u.bin",
                   h_x_u.data(),
                   vec_elems * sizeof(double));
        h_x_true.resize(vec_elems);
        read_array(dir + "/x.bin",
                   h_x_true.data(),
                   vec_elems * sizeof(double));
        h_a.resize(mat_elems);
        read_array(dir + "/a.bin",
                   h_a.data(),
                   mat_elems * sizeof(double));
    }

    /* ---------------------------------------------------- */
    /* upload L, U (fp64) to GPU                            */

    double *d_l, *d_u;
    HIP_CHECK(hipMalloc(&d_l,
        mat_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_u,
        mat_elems * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_l, h_l.data(),
        mat_elems * sizeof(double),
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_u, h_u.data(),
        mat_elems * sizeof(double),
        hipMemcpyHostToDevice));

    /* upload pivots */
    rocblas_int *d_ipiv;
    HIP_CHECK(hipMalloc(&d_ipiv,
        batch * n * sizeof(rocblas_int)));
    HIP_CHECK(hipMemcpy(d_ipiv, h_ipiv.data(),
        batch * n * sizeof(rocblas_int),
        hipMemcpyHostToDevice));

    /* ---------------------------------------------------- */
    /* demote L and U to fp16 with separate scales          */

    std::cerr << "computing scaling factors...\n";

    double max_abs_l = gpu_compute_max_abs(
        d_l, (long long)mat_elems);
    double scale_l = compute_optimal_fp16_scale(
        max_abs_l);
    double inv_scale_l = 1.0 / scale_l;

    double max_abs_u = gpu_compute_max_abs(
        d_u, (long long)mat_elems);
    double scale_u = compute_optimal_fp16_scale(
        max_abs_u);
    double inv_scale_u = 1.0 / scale_u;

    std::cerr << "  max(|L|) = " << max_abs_l
              << ", scale_L = " << scale_l << "\n"
              << "  max(|U|) = " << max_abs_u
              << ", scale_U = " << scale_u << "\n";

    double *d_scale_l, *d_scale_u;
    HIP_CHECK(hipMalloc(&d_scale_l,
        sizeof(double)));
    HIP_CHECK(hipMalloc(&d_scale_u,
        sizeof(double)));
    HIP_CHECK(hipMemcpy(d_scale_l, &scale_l,
        sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_scale_u, &scale_u,
        sizeof(double), hipMemcpyHostToDevice));

    _Float16 *d_l16, *d_u16;
    HIP_CHECK(hipMalloc(&d_l16,
        mat_elems * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_u16,
        mat_elems * sizeof(_Float16)));

    {
        int tpb = 256;
        long long size = (long long)mat_elems;
        int nblocks =
            (int)((size + tpb - 1) / tpb);

        if (n >= (size_t)LARGE_PROBLEM_THRESHOLD) {
            hipLaunchKernelGGL(
                (gpu_scale_and_demote_kernel<
                    long long>),
                dim3(nblocks), dim3(tpb),
                0, ctx.stream,
                d_l, d_l16, d_scale_l, size);
            hipLaunchKernelGGL(
                (gpu_scale_and_demote_kernel<
                    long long>),
                dim3(nblocks), dim3(tpb),
                0, ctx.stream,
                d_u, d_u16, d_scale_u, size);
        } else {
            hipLaunchKernelGGL(
                (gpu_scale_and_demote_kernel<int>),
                dim3(nblocks), dim3(tpb),
                0, ctx.stream,
                d_l, d_l16, d_scale_l,
                (int)size);
            hipLaunchKernelGGL(
                (gpu_scale_and_demote_kernel<int>),
                dim3(nblocks), dim3(tpb),
                0, ctx.stream,
                d_u, d_u16, d_scale_u,
                (int)size);
        }
        HIP_CHECK(hipGetLastError());
    }

    /* ---------------------------------------------------- */
    /* allocate solver workspace                            */

    double *d_b, *d_x64, *d_r;
    _Float16 *d_x16;
    float *d_workspace, *d_rhs_workspace;
    double *d_block_maxes, *d_scale_b;
    double *d_correction_scale, *d_norm;

    HIP_CHECK(hipMalloc(&d_b,
        vec_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_x64,
        vec_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_r,
        vec_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_x16,
        vec_elems * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_workspace,
        vec_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rhs_workspace,
        vec_elems * sizeof(float)));

    int tpb = 256;
    int nblocks_r =
        (int)((vec_elems + tpb - 1) / tpb);
    size_t shmem_size = sizeof(double) * tpb;

    HIP_CHECK(hipMalloc(&d_block_maxes,
        sizeof(double) * nblocks_r));
    HIP_CHECK(hipMalloc(&d_scale_b,
        sizeof(double)));
    HIP_CHECK(hipMalloc(&d_correction_scale,
        sizeof(double)));
    HIP_CHECK(hipMalloc(&d_norm,
        sizeof(double)));

    /* upload b and apply pivots to get Pb.
       keep h_b as original b for final verify. */
    HIP_CHECK(hipMemcpy(d_b, h_b.data(),
        vec_elems * sizeof(double),
        hipMemcpyHostToDevice));
    apply_pivots(ctx, d_b, n,
                 d_ipiv, nrhs, batch);
    HIP_CHECK(hipStreamSynchronize(ctx.stream));

    /* save Pb back to host for re-upload in
       measured runs */
    std::vector<double> h_pb(vec_elems);
    HIP_CHECK(hipMemcpy(h_pb.data(), d_b,
        vec_elems * sizeof(double),
        hipMemcpyDeviceToHost));

    std::cerr << "IR3 LU solver ready (iters="
              << max_ir_iters << ")\n";

    /* ---------------------------------------------------- */
    /* dgemv parameters                                     */

    int lda = (int)n;
    long long stride_a = (long long)n * n;
    long long stride_x = (long long)n;

    double alpha_neg = -1.0;
    double beta_one  =  1.0;

    /* ---------------------------------------------------- */
    /* IR phase lambdas
       cur_mat64 and cur_mat16 switch between L/U
       per phase. cur_inv_scale tracks the active
       scale factor.                                        */

    double *cur_mat64;
    const _Float16 *cur_mat16;
    double cur_inv_scale;

    auto phase_residual = [&]() {
        for (size_t bi = 0; bi < batch; bi++) {
            HIP_CHECK(hipMemcpyAsync(
                d_r + bi * stride_x,
                d_b + bi * stride_x,
                n * sizeof(double),
                hipMemcpyDeviceToDevice,
                ctx.stream));

            ROCBLAS_CHECK(rocblas_dgemv(
                ctx.blas_handle,
                rocblas_operation_none,
                (rocblas_int)n, (rocblas_int)n,
                &alpha_neg,
                cur_mat64 + bi * stride_a, lda,
                d_x64 + bi * stride_x, 1,
                &beta_one,
                d_r + bi * stride_x, 1));
        }
    };

    auto phase_scale = [&]() {
        hipLaunchKernelGGL(
            gpu_max_norm_kernel,
            dim3(nblocks_r), dim3(tpb),
            shmem_size, ctx.stream,
            d_r, d_block_maxes,
            (int)(vec_elems));

        hipLaunchKernelGGL(
            gpu_compute_pow2_scale_kernel,
            dim3(1), dim3(1),
            0, ctx.stream,
            d_block_maxes, d_scale_b,
            d_norm, nblocks_r);

        hipLaunchKernelGGL(
            gpu_apply_scale_and_demote_kernel,
            dim3(nblocks_r), dim3(tpb),
            0, ctx.stream,
            d_r, d_x16, d_scale_b,
            (int)(vec_elems));
    };

    auto phase_trsv_forward = [&]() {
        launch_trsv_multiCU_fp16<
            MC_TRSV_TILE_SIZE, MC_TRSV_TPB>(
            ctx.stream,
            (int)n, lda, stride_a, d_l16,
            stride_x, d_x16, (int)batch,
            d_rhs_workspace, d_workspace);
    };

    auto phase_trsv_backward = [&]() {
        launch_trsv_multiCU_fp16_backward<
            MC_TRSV_TILE_SIZE, MC_TRSV_TPB,
            false>(
            ctx.stream,
            (int)n, lda, stride_a, d_u16,
            stride_x, d_x16, (int)batch,
            d_rhs_workspace, d_workspace);
    };

    auto phase_update = [&]() {
        hipLaunchKernelGGL(
            gpu_scalar_multiply_kernel,
            dim3(1), dim3(1),
            0, ctx.stream,
            d_correction_scale,
            d_scale_b, cur_inv_scale);

        hipLaunchKernelGGL(
            gpu_promote_and_add_kernel_devscale,
            dim3(nblocks_r), dim3(tpb),
            0, ctx.stream,
            d_x16, d_x64,
            d_correction_scale,
            (int)(vec_elems));
    };

    /* ---------------------------------------------------- */
    /* persistent labels                                    */

    size_t iters_total = max_ir_iters * 2;

    std::vector<std::string> time_strs;
    std::vector<const char *> time_labels;
    std::vector<std::string> verify_strs;
    std::vector<const char *> verify_labels;

    if (instrument) {
        time_strs.resize(4 * iters_total);
        time_labels.resize(4 * iters_total);
        const char *ppfx[2] = {"p1_", "p2_"};
        for (int p = 0; p < 2; p++) {
            size_t base = p * 4 * max_ir_iters;
            for (size_t i = 0;
                 i < max_ir_iters; i++) {
                std::string pfx =
                    std::string(ppfx[p])
                    + "iter" + std::to_string(i)
                    + "_";
                time_strs[base+4*i+0] =
                    pfx + "residual";
                time_strs[base+4*i+1] =
                    pfx + "scale";
                time_strs[base+4*i+2] =
                    pfx + "trsv";
                time_strs[base+4*i+3] =
                    pfx + "update";
                for (int j = 0; j < 4; j++)
                    time_labels[base+4*i+j] =
                        time_strs[base+4*i+j]
                            .c_str();
            }
        }
    }

    if (do_verify) {
        size_t nv = 2 * iters_total + 1;
        verify_strs.resize(nv);
        verify_labels.resize(nv);
        const char *ppfx[2] = {"p1_", "p2_"};
        for (int p = 0; p < 2; p++) {
            size_t base = p * 2 * max_ir_iters;
            for (size_t i = 0;
                 i < max_ir_iters; i++) {
                std::string pfx =
                    std::string(ppfx[p])
                    + "iter" + std::to_string(i)
                    + "_";
                verify_strs[base+2*i+0] =
                    pfx + "bwd_error";
                verify_strs[base+2*i+1] =
                    pfx + "fwd_error";
                for (int j = 0; j < 2; j++)
                    verify_labels[base+2*i+j] =
                        verify_strs[base+2*i+j]
                            .c_str();
            }
        }
        size_t fi = 2 * iters_total;
        verify_strs[fi] = "final_bwd_error";
        verify_labels[fi] =
            verify_strs[fi].c_str();
    }

    /* ---------------------------------------------------- */
    /* set up profiler and timers                           */

    size_t events_per_run = instrument
        ? 4 * iters_total : 2;
    size_t verify_per_run = do_verify
        ? 2 * iters_total + 1 : 0;

    profiler prof;
    prof.init(cfg, events_per_run, verify_per_run);

    gpu_timer timer;
    gpu_timer_pool timers;
    if (instrument)
        timers.init(4);
    else
        timer.init();

    std::vector<double> h_x64;
    std::vector<double> h_y;
    if (do_verify) {
        h_x64.resize(vec_elems);
        h_y.resize(vec_elems);
    }

    /* ---------------------------------------------------- */
    /* warmup                                               */

    std::cerr << "warmup (" << cfg.warmup_runs
              << " runs)...\n";
    prof.begin_warmup();
    for (size_t w = 0; w < cfg.warmup_runs; w++) {
        /* reset: reload Pb, zero x */
        HIP_CHECK(hipMemcpyAsync(d_b, h_pb.data(),
            vec_elems * sizeof(double),
            hipMemcpyHostToDevice, ctx.stream));
        HIP_CHECK(hipMemsetAsync(d_x64, 0,
            vec_elems * sizeof(double),
            ctx.stream));

        /* phase 1: L*y = Pb (forward) */
        cur_mat64 = d_l;
        cur_mat16 = d_l16;
        cur_inv_scale = inv_scale_l;
        for (size_t iter = 0;
             iter < max_ir_iters; iter++) {
            phase_residual();
            phase_scale();
            phase_trsv_forward();
            phase_update();
        }

        /* transition: y -> d_b, zero d_x64 */
        HIP_CHECK(hipMemcpyAsync(d_b, d_x64,
            vec_elems * sizeof(double),
            hipMemcpyDeviceToDevice, ctx.stream));
        HIP_CHECK(hipMemsetAsync(d_x64, 0,
            vec_elems * sizeof(double),
            ctx.stream));

        /* phase 2: U*x = y (backward) */
        cur_mat64 = d_u;
        cur_mat16 = d_u16;
        cur_inv_scale = inv_scale_u;
        for (size_t iter = 0;
             iter < max_ir_iters; iter++) {
            phase_residual();
            phase_scale();
            phase_trsv_backward();
            phase_update();
        }

        HIP_CHECK(
            hipStreamSynchronize(ctx.stream));
    }
    prof.in_warmup = false;

    /* ---------------------------------------------------- */
    /* measured runs                                        */

    std::cerr << "profiling (" << cfg.measured_runs
              << " runs)...\n";

    /* host residual function pointers per phase */
    typedef double (*residual_fn)(
        const double *, const double *,
        const double *, size_t, size_t);
    residual_fn host_resid[2] = {
        compute_residual_lower,
        compute_residual_upper
    };
    const double *h_mat_ptrs[2] = {
        h_l.data(), h_u.data()
    };

    for (size_t r = 0; r < cfg.measured_runs; r++) {
        prof.begin_run();

        /* reset: reload Pb, zero x */
        HIP_CHECK(hipMemcpyAsync(d_b, h_pb.data(),
            vec_elems * sizeof(double),
            hipMemcpyHostToDevice, ctx.stream));
        HIP_CHECK(hipMemsetAsync(d_x64, 0,
            vec_elems * sizeof(double),
            ctx.stream));

        for (int phase = 0; phase < 2; phase++) {
            if (phase == 0) {
                cur_mat64 = d_l;
                cur_mat16 = d_l16;
                cur_inv_scale = inv_scale_l;
            } else {
                cur_mat64 = d_u;
                cur_mat16 = d_u16;
                cur_inv_scale = inv_scale_u;
            }

            const double *h_true_ptr =
                (phase == 0)
                    ? h_x_u.data()
                    : h_x_true.data();
            const double *h_rhs_ptr =
                (phase == 0)
                    ? h_pb.data()
                    : h_y.data();

            size_t label_base =
                phase * 4 * max_ir_iters;
            size_t vlabel_base =
                phase * 2 * max_ir_iters;

            if (instrument) {
                HIP_CHECK(hipStreamSynchronize(
                    ctx.stream));

                for (size_t iter = 0;
                     iter < max_ir_iters;
                     iter++) {
                    timers.reset();

                    timers.start(0, ctx.stream);
                    phase_residual();
                    timers.stop(0, ctx.stream);

                    timers.start(1, ctx.stream);
                    phase_scale();
                    timers.stop(1, ctx.stream);

                    timers.start(2, ctx.stream);
                    if (phase == 0)
                        phase_trsv_forward();
                    else
                        phase_trsv_backward();
                    timers.stop(2, ctx.stream);

                    timers.start(3, ctx.stream);
                    phase_update();
                    timers.stop(3, ctx.stream);

                    timers.synchronize_all();

                    size_t ti =
                        label_base + 4 * iter;
                    prof.record_gpu_time(
                        time_labels[ti+0],
                        timers.elapsed_ms(0));
                    prof.record_gpu_time(
                        time_labels[ti+1],
                        timers.elapsed_ms(1));
                    prof.record_gpu_time(
                        time_labels[ti+2],
                        timers.elapsed_ms(2));
                    prof.record_gpu_time(
                        time_labels[ti+3],
                        timers.elapsed_ms(3));

                    if (do_verify) {
                        HIP_CHECK(hipMemcpy(
                            h_x64.data(), d_x64,
                            vec_elems
                                * sizeof(double),
                            hipMemcpyDeviceToHost));

                        double max_bwd = 0.0;
                        double max_fwd = 0.0;
                        for (size_t bi = 0;
                             bi < batch; bi++) {
                            double bwd =
                                host_resid[phase](
                                    h_mat_ptrs[phase],
                                    h_x64.data(),
                                    h_rhs_ptr,
                                    n, bi);
                            if (bwd > max_bwd)
                                max_bwd = bwd;
                            double fwd =
                                compute_forward_error(
                                    h_x64.data(),
                                    h_true_ptr,
                                    n, bi);
                            if (fwd > max_fwd)
                                max_fwd = fwd;
                        }

                        size_t vi =
                            vlabel_base + 2 * iter;
                        prof.record_metric(
                            verify_labels[vi+0],
                            max_bwd);
                        prof.record_metric(
                            verify_labels[vi+1],
                            max_fwd);
                    }
                }
            } else {
                timer.start(ctx.stream);

                for (size_t iter = 0;
                     iter < max_ir_iters;
                     iter++) {
                    phase_residual();
                    phase_scale();
                    if (phase == 0)
                        phase_trsv_forward();
                    else
                        phase_trsv_backward();
                    phase_update();
                }

                timer.stop(ctx.stream);
                timer.synchronize();
                prof.record_gpu_time(
                    (phase == 0)
                        ? "phase1" : "phase2",
                    timer.elapsed_ms());
            }

            /* transition after phase 1 */
            if (phase == 0) {
                if (do_verify) {
                    HIP_CHECK(hipMemcpy(
                        h_y.data(), d_x64,
                        vec_elems
                            * sizeof(double),
                        hipMemcpyDeviceToHost));
                }
                HIP_CHECK(hipMemcpyAsync(
                    d_b, d_x64,
                    vec_elems * sizeof(double),
                    hipMemcpyDeviceToDevice,
                    ctx.stream));
                HIP_CHECK(hipMemsetAsync(
                    d_x64, 0,
                    vec_elems * sizeof(double),
                    ctx.stream));
            }
        }

        /* final verification: ||Ax - b|| / ||b|| */
        if (do_verify) {
            HIP_CHECK(hipMemcpy(
                h_x64.data(), d_x64,
                vec_elems * sizeof(double),
                hipMemcpyDeviceToHost));

            double max_final = 0.0;
            for (size_t bi = 0;
                 bi < batch; bi++) {
                double res =
                    compute_residual_full(
                        h_a.data(),
                        h_x64.data(),
                        h_b.data(),
                        n, bi);
                if (res > max_final)
                    max_final = res;
            }
            prof.record_metric(
                verify_labels[2 * iters_total],
                max_final);
        }

        prof.end_run();
    }

    prof.print_summary();
    prof.write_csv();

    /* ---------------------------------------------------- */
    /* cleanup                                              */

    if (instrument)
        timers.destroy();
    else
        timer.destroy();
    prof.destroy();

    HIP_CHECK(hipFree(d_norm));
    HIP_CHECK(hipFree(d_correction_scale));
    HIP_CHECK(hipFree(d_scale_b));
    HIP_CHECK(hipFree(d_block_maxes));
    HIP_CHECK(hipFree(d_rhs_workspace));
    HIP_CHECK(hipFree(d_workspace));
    HIP_CHECK(hipFree(d_x16));
    HIP_CHECK(hipFree(d_r));
    HIP_CHECK(hipFree(d_x64));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_u16));
    HIP_CHECK(hipFree(d_l16));
    HIP_CHECK(hipFree(d_scale_u));
    HIP_CHECK(hipFree(d_scale_l));
    HIP_CHECK(hipFree(d_ipiv));
    HIP_CHECK(hipFree(d_u));
    HIP_CHECK(hipFree(d_l));

    std::cerr << "done.\n";
}

/* -------------------------------------------------------- */
/* run_ir3A_solve: Higham 3-precision IR with full-system
   residual r = b - A*x (fp64) and inner approximate
   solve via two chained fp16 TRSVs.

   Cholesky (--factor chol):
     inner solve: L * y = r,  L^T * d = y
   LU (--factor lu):
     inner solve: L * y = P*r,  U * d = y

   works with both --factor chol and --factor lu.
   requires nrhs=1.                                         */

void run_ir3A_solve(gpu_context &ctx,
                    const profiler_config &cfg,
                    size_t max_ir_iters)
{
    size_t n     = cfg.desc.n;
    size_t batch = cfg.desc.batch;
    size_t nrhs  = cfg.desc.nrhs;
    bool is_lu   =
        cfg.desc.factor == factor_type::lu;

    if (nrhs != 1) {
        std::cerr << "error: ir3A solver requires "
                  << "nrhs=1\n";
        exit(1);
    }

    size_t mat_elems = batch * n * n;
    size_t vec_elems = batch * n;

    bool instrument =
        cfg.mode == profile_mode::instrument
     || cfg.mode == profile_mode::instrument_verify;
    bool do_verify =
        cfg.mode == profile_mode::instrument_verify;

    /* ---------------------------------------------------- */
    /* load fp64 artifacts from disk                        */

    std::string dir = artifact_directory(cfg.desc);
    std::cerr << "loading artifacts from "
              << dir << "/\n";

    std::vector<double> h_a(mat_elems);
    std::vector<double> h_l(mat_elems);
    std::vector<double> h_b(vec_elems);

    read_array(dir + "/a.bin",
               h_a.data(),
               mat_elems * sizeof(double));
    read_array(dir + "/l.bin",
               h_l.data(),
               mat_elems * sizeof(double));
    read_array(dir + "/b.bin",
               h_b.data(),
               vec_elems * sizeof(double));

    std::vector<double> h_u;
    std::vector<rocblas_int> h_ipiv;
    if (is_lu) {
        h_u.resize(mat_elems);
        read_array(dir + "/u.bin",
                   h_u.data(),
                   mat_elems * sizeof(double));
        h_ipiv.resize(batch * n);
        read_array(dir + "/ipiv.bin",
                   h_ipiv.data(),
                   batch * n * sizeof(rocblas_int));
    }

    /* true solution for forward error */
    std::vector<double> h_x_true;
    if (do_verify) {
        h_x_true.resize(vec_elems);
        read_array(dir + "/x.bin",
                   h_x_true.data(),
                   vec_elems * sizeof(double));
    }

    /* ---------------------------------------------------- */
    /* upload A (fp64) to GPU for residual dgemv            */

    double *d_a;
    HIP_CHECK(hipMalloc(&d_a,
        mat_elems * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_a, h_a.data(),
        mat_elems * sizeof(double),
        hipMemcpyHostToDevice));

    /* ---------------------------------------------------- */
    /* upload L (fp64), demote to fp16                      */

    double *d_l;
    HIP_CHECK(hipMalloc(&d_l,
        mat_elems * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_l, h_l.data(),
        mat_elems * sizeof(double),
        hipMemcpyHostToDevice));

    std::cerr << "computing scaling factors...\n";
    double max_abs_l = gpu_compute_max_abs(
        d_l, (long long)mat_elems);
    double scale_l = compute_optimal_fp16_scale(
        max_abs_l);
    double inv_scale_l = 1.0 / scale_l;

    std::cerr << "  max(|L|) = " << max_abs_l
              << ", scale_L = " << scale_l << "\n";

    double *d_scale_l;
    HIP_CHECK(hipMalloc(&d_scale_l,
        sizeof(double)));
    HIP_CHECK(hipMemcpy(d_scale_l, &scale_l,
        sizeof(double), hipMemcpyHostToDevice));

    _Float16 *d_l16;
    HIP_CHECK(hipMalloc(&d_l16,
        mat_elems * sizeof(_Float16)));

    {
        int tpb_d = 256;
        long long size = (long long)mat_elems;
        int nblocks =
            (int)((size + tpb_d - 1) / tpb_d);

        if (n >= (size_t)LARGE_PROBLEM_THRESHOLD) {
            hipLaunchKernelGGL(
                (gpu_scale_and_demote_kernel<
                    long long>),
                dim3(nblocks), dim3(tpb_d),
                0, ctx.stream,
                d_l, d_l16, d_scale_l, size);
        } else {
            hipLaunchKernelGGL(
                (gpu_scale_and_demote_kernel<int>),
                dim3(nblocks), dim3(tpb_d),
                0, ctx.stream,
                d_l, d_l16, d_scale_l,
                (int)size);
        }
        HIP_CHECK(hipGetLastError());
    }

    /* can free fp64 L — only needed for demotion */
    HIP_CHECK(hipFree(d_l));
    d_l = nullptr;

    /* ---------------------------------------------------- */
    /* LU: upload U (fp64), demote to fp16, upload pivots   */

    double *d_u = nullptr;
    double *d_scale_u = nullptr;
    _Float16 *d_u16 = nullptr;
    rocblas_int *d_ipiv = nullptr;
    double inv_scale_u = 0.0;

    if (is_lu) {
        HIP_CHECK(hipMalloc(&d_u,
            mat_elems * sizeof(double)));
        HIP_CHECK(hipMemcpy(d_u, h_u.data(),
            mat_elems * sizeof(double),
            hipMemcpyHostToDevice));

        double max_abs_u = gpu_compute_max_abs(
            d_u, (long long)mat_elems);
        double scale_u =
            compute_optimal_fp16_scale(max_abs_u);
        inv_scale_u = 1.0 / scale_u;

        std::cerr << "  max(|U|) = " << max_abs_u
                  << ", scale_U = " << scale_u
                  << "\n";

        HIP_CHECK(hipMalloc(&d_scale_u,
            sizeof(double)));
        HIP_CHECK(hipMemcpy(d_scale_u, &scale_u,
            sizeof(double),
            hipMemcpyHostToDevice));

        HIP_CHECK(hipMalloc(&d_u16,
            mat_elems * sizeof(_Float16)));

        {
            int tpb_d = 256;
            long long size = (long long)mat_elems;
            int nblocks =
                (int)((size + tpb_d - 1) / tpb_d);

            if (n >=
                (size_t)LARGE_PROBLEM_THRESHOLD) {
                hipLaunchKernelGGL(
                    (gpu_scale_and_demote_kernel<
                        long long>),
                    dim3(nblocks), dim3(tpb_d),
                    0, ctx.stream,
                    d_u, d_u16, d_scale_u, size);
            } else {
                hipLaunchKernelGGL(
                    (gpu_scale_and_demote_kernel<
                        int>),
                    dim3(nblocks), dim3(tpb_d),
                    0, ctx.stream,
                    d_u, d_u16, d_scale_u,
                    (int)size);
            }
            HIP_CHECK(hipGetLastError());
        }

        /* free fp64 U */
        HIP_CHECK(hipFree(d_u));
        d_u = nullptr;

        /* upload pivots */
        HIP_CHECK(hipMalloc(&d_ipiv,
            batch * n * sizeof(rocblas_int)));
        HIP_CHECK(hipMemcpy(
            d_ipiv, h_ipiv.data(),
            batch * n * sizeof(rocblas_int),
            hipMemcpyHostToDevice));
    }

    /* ---------------------------------------------------- */
    /* combined inverse-scale product for correction        */

    double inv_scale_combined = is_lu
        ? inv_scale_l * inv_scale_u
        : inv_scale_l * inv_scale_l;

    /* ---------------------------------------------------- */
    /* allocate solver workspace                            */

    double *d_b, *d_x64, *d_r;
    _Float16 *d_x16;
    float *d_workspace, *d_rhs_workspace;
    double *d_block_maxes, *d_scale_b;
    double *d_correction_scale, *d_norm;

    HIP_CHECK(hipMalloc(&d_b,
        vec_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_x64,
        vec_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_r,
        vec_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_x16,
        vec_elems * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_workspace,
        vec_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rhs_workspace,
        vec_elems * sizeof(float)));

    int tpb = 256;
    int nblocks_r =
        (int)((vec_elems + tpb - 1) / tpb);
    size_t shmem_size = sizeof(double) * tpb;

    HIP_CHECK(hipMalloc(&d_block_maxes,
        sizeof(double) * nblocks_r));
    HIP_CHECK(hipMalloc(&d_scale_b,
        sizeof(double)));
    HIP_CHECK(hipMalloc(&d_correction_scale,
        sizeof(double)));
    HIP_CHECK(hipMalloc(&d_norm,
        sizeof(double)));

    /* upload b */
    HIP_CHECK(hipMemcpy(d_b, h_b.data(),
        vec_elems * sizeof(double),
        hipMemcpyHostToDevice));

    HIP_CHECK(hipStreamSynchronize(ctx.stream));
    std::cerr << "IR3A solver ready (iters="
              << max_ir_iters << ")\n";

    /* ---------------------------------------------------- */
    /* dgemv parameters                                     */

    int lda = (int)n;
    long long stride_a = (long long)n * n;
    long long stride_x = (long long)n;

    double alpha_neg = -1.0;
    double beta_one  =  1.0;

    /* ---------------------------------------------------- */
    /* IR phase lambdas (GPU work only, no sync)            */

    auto phase_residual = [&]() {
        /* r = b - A*x64 (fp64 dgemv on full A) */
        for (size_t bi = 0; bi < batch; bi++) {
            HIP_CHECK(hipMemcpyAsync(
                d_r + bi * stride_x,
                d_b + bi * stride_x,
                n * sizeof(double),
                hipMemcpyDeviceToDevice,
                ctx.stream));

            ROCBLAS_CHECK(rocblas_dgemv(
                ctx.blas_handle,
                rocblas_operation_none,
                (rocblas_int)n, (rocblas_int)n,
                &alpha_neg,
                d_a + bi * stride_a, lda,
                d_x64 + bi * stride_x, 1,
                &beta_one,
                d_r + bi * stride_x, 1));
        }

        /* LU: apply pivots to residual r → P*r */
        if (is_lu) {
            apply_pivots(ctx, d_r, n,
                         d_ipiv, nrhs, batch);
        }
    };

    auto phase_scale = [&]() {
        hipLaunchKernelGGL(
            gpu_max_norm_kernel,
            dim3(nblocks_r), dim3(tpb),
            shmem_size, ctx.stream,
            d_r, d_block_maxes,
            (int)(vec_elems));

        hipLaunchKernelGGL(
            gpu_compute_pow2_scale_kernel,
            dim3(1), dim3(1),
            0, ctx.stream,
            d_block_maxes, d_scale_b,
            d_norm, nblocks_r);

        hipLaunchKernelGGL(
            gpu_apply_scale_and_demote_kernel,
            dim3(nblocks_r), dim3(tpb),
            0, ctx.stream,
            d_r, d_x16, d_scale_b,
            (int)(vec_elems));
    };

    auto phase_trsv_forward = [&]() {
        launch_trsv_multiCU_fp16<
            MC_TRSV_TILE_SIZE, MC_TRSV_TPB>(
            ctx.stream,
            (int)n, lda, stride_a, d_l16,
            stride_x, d_x16, (int)batch,
            d_rhs_workspace, d_workspace);
    };

    auto phase_trsv_backward = [&]() {
        if (is_lu) {
            launch_trsv_multiCU_fp16_backward<
                MC_TRSV_TILE_SIZE, MC_TRSV_TPB,
                false>(
                ctx.stream,
                (int)n, lda, stride_a, d_u16,
                stride_x, d_x16, (int)batch,
                d_rhs_workspace, d_workspace);
        } else {
            launch_trsv_multiCU_fp16_backward<
                MC_TRSV_TILE_SIZE, MC_TRSV_TPB,
                true>(
                ctx.stream,
                (int)n, lda, stride_a, d_l16,
                stride_x, d_x16, (int)batch,
                d_rhs_workspace, d_workspace);
        }
    };

    auto phase_update = [&]() {
        hipLaunchKernelGGL(
            gpu_scalar_multiply_kernel,
            dim3(1), dim3(1),
            0, ctx.stream,
            d_correction_scale,
            d_scale_b, inv_scale_combined);

        hipLaunchKernelGGL(
            gpu_promote_and_add_kernel_devscale,
            dim3(nblocks_r), dim3(tpb),
            0, ctx.stream,
            d_x16, d_x64,
            d_correction_scale,
            (int)(vec_elems));
    };

    /* ---------------------------------------------------- */
    /* persistent labels for per-iteration recording        */

    std::vector<std::string> time_strs;
    std::vector<const char *> time_labels;
    std::vector<std::string> verify_strs;
    std::vector<const char *> verify_labels;

    if (instrument) {
        time_strs.resize(5 * max_ir_iters);
        time_labels.resize(5 * max_ir_iters);
        for (size_t i = 0; i < max_ir_iters; i++) {
            std::string pfx =
                "iter" + std::to_string(i) + "_";
            time_strs[5*i+0] = pfx + "residual";
            time_strs[5*i+1] = pfx + "scale";
            time_strs[5*i+2] = pfx + "fwd_trsv";
            time_strs[5*i+3] = pfx + "bwd_trsv";
            time_strs[5*i+4] = pfx + "update";
            for (int j = 0; j < 5; j++)
                time_labels[5*i+j] =
                    time_strs[5*i+j].c_str();
        }
    }

    if (do_verify) {
        verify_strs.resize(2 * max_ir_iters);
        verify_labels.resize(2 * max_ir_iters);
        for (size_t i = 0; i < max_ir_iters; i++) {
            std::string pfx =
                "iter" + std::to_string(i) + "_";
            verify_strs[2*i+0] = pfx + "bwd_error";
            verify_strs[2*i+1] = pfx + "fwd_error";
            for (int j = 0; j < 2; j++)
                verify_labels[2*i+j] =
                    verify_strs[2*i+j].c_str();
        }
    }

    /* ---------------------------------------------------- */
    /* set up profiler and timers                           */

    size_t events_per_run = instrument
        ? 5 * max_ir_iters : 1;
    size_t verify_per_run = do_verify
        ? 2 * max_ir_iters : 0;

    profiler prof;
    prof.init(cfg, events_per_run, verify_per_run);

    gpu_timer timer;
    gpu_timer_pool timers;
    if (instrument)
        timers.init(5);
    else
        timer.init();

    std::vector<double> h_x64;
    if (do_verify)
        h_x64.resize(vec_elems);

    /* ---------------------------------------------------- */
    /* warmup                                               */

    std::cerr << "warmup (" << cfg.warmup_runs
              << " runs)...\n";
    prof.begin_warmup();
    for (size_t w = 0; w < cfg.warmup_runs; w++) {
        HIP_CHECK(hipMemsetAsync(d_x64, 0,
            vec_elems * sizeof(double),
            ctx.stream));

        for (size_t iter = 0;
             iter < max_ir_iters; iter++) {
            phase_residual();
            phase_scale();
            phase_trsv_forward();
            phase_trsv_backward();
            phase_update();
        }

        HIP_CHECK(
            hipStreamSynchronize(ctx.stream));
    }
    prof.in_warmup = false;

    /* ---------------------------------------------------- */
    /* measured runs                                        */

    std::cerr << "profiling (" << cfg.measured_runs
              << " runs)...\n";
    for (size_t r = 0; r < cfg.measured_runs; r++) {
        prof.begin_run();

        HIP_CHECK(hipMemsetAsync(d_x64, 0,
            vec_elems * sizeof(double),
            ctx.stream));

        if (instrument) {
            HIP_CHECK(hipStreamSynchronize(
                ctx.stream));

            for (size_t iter = 0;
                 iter < max_ir_iters; iter++) {
                timers.reset();

                timers.start(0, ctx.stream);
                phase_residual();
                timers.stop(0, ctx.stream);

                timers.start(1, ctx.stream);
                phase_scale();
                timers.stop(1, ctx.stream);

                timers.start(2, ctx.stream);
                phase_trsv_forward();
                timers.stop(2, ctx.stream);

                timers.start(3, ctx.stream);
                phase_trsv_backward();
                timers.stop(3, ctx.stream);

                timers.start(4, ctx.stream);
                phase_update();
                timers.stop(4, ctx.stream);

                timers.synchronize_all();

                prof.record_gpu_time(
                    time_labels[5*iter+0],
                    timers.elapsed_ms(0));
                prof.record_gpu_time(
                    time_labels[5*iter+1],
                    timers.elapsed_ms(1));
                prof.record_gpu_time(
                    time_labels[5*iter+2],
                    timers.elapsed_ms(2));
                prof.record_gpu_time(
                    time_labels[5*iter+3],
                    timers.elapsed_ms(3));
                prof.record_gpu_time(
                    time_labels[5*iter+4],
                    timers.elapsed_ms(4));

                if (do_verify) {
                    HIP_CHECK(hipMemcpy(
                        h_x64.data(), d_x64,
                        vec_elems *
                            sizeof(double),
                        hipMemcpyDeviceToHost));

                    double max_bwd = 0.0;
                    double max_fwd = 0.0;
                    for (size_t bi = 0;
                         bi < batch; bi++) {
                        double bwd =
                            compute_residual_full(
                                h_a.data(),
                                h_x64.data(),
                                h_b.data(),
                                n, 1, bi);
                        if (bwd > max_bwd)
                            max_bwd = bwd;
                        double fwd =
                            compute_forward_error(
                                h_x64.data(),
                                h_x_true.data(),
                                n, bi);
                        if (fwd > max_fwd)
                            max_fwd = fwd;
                    }

                    prof.record_metric(
                        verify_labels[2*iter+0],
                        max_bwd);
                    prof.record_metric(
                        verify_labels[2*iter+1],
                        max_fwd);
                }
            }
        } else {
            /* profile mode: single timer around
               entire IR solve */
            timer.start(ctx.stream);

            for (size_t iter = 0;
                 iter < max_ir_iters; iter++) {
                phase_residual();
                phase_scale();
                phase_trsv_forward();
                phase_trsv_backward();
                phase_update();
            }

            timer.stop(ctx.stream);
            timer.synchronize();
            prof.record_gpu_time("ir3A_solve",
                timer.elapsed_ms());
        }

        prof.end_run();
    }

    prof.print_summary();
    prof.write_csv();

    /* ---------------------------------------------------- */
    /* cleanup                                              */

    if (instrument)
        timers.destroy();
    else
        timer.destroy();
    prof.destroy();

    HIP_CHECK(hipFree(d_norm));
    HIP_CHECK(hipFree(d_correction_scale));
    HIP_CHECK(hipFree(d_scale_b));
    HIP_CHECK(hipFree(d_block_maxes));
    HIP_CHECK(hipFree(d_rhs_workspace));
    HIP_CHECK(hipFree(d_workspace));
    HIP_CHECK(hipFree(d_x16));
    HIP_CHECK(hipFree(d_r));
    HIP_CHECK(hipFree(d_x64));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_l16));
    HIP_CHECK(hipFree(d_scale_l));
    if (is_lu) {
        HIP_CHECK(hipFree(d_u16));
        HIP_CHECK(hipFree(d_scale_u));
        HIP_CHECK(hipFree(d_ipiv));
    }
    HIP_CHECK(hipFree(d_a));

    std::cerr << "done.\n";
}
