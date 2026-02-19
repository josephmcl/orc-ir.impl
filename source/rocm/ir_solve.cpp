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
