#pragma once

#include <hip/hip_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <climits>

/* -------------------------------------------------------- */
/* constants for multi-CU TRSV                              */

constexpr int MC_TRSV_TILE_SIZE       = 64;
constexpr int MC_TRSV_TPB             = 256;
constexpr int MC_TRSV_MAX_UPD_BLOCKS  = 512;

/* threshold where n*n overflows int32 */
constexpr int LARGE_PROBLEM_THRESHOLD = 46341;

/* -------------------------------------------------------- */
/* NaN-aware max for reductions
   (safe with -ffast-math)                                  */

__device__ inline bool ir_is_nan_bits(double x) {
    unsigned long long bits;
    __builtin_memcpy(&bits, &x, sizeof(double));
    unsigned long long exp_mask =
        0x7FF0000000000000ULL;
    unsigned long long mant_mask =
        0x000FFFFFFFFFFFFFULL;
    return ((bits & exp_mask) == exp_mask)
        && ((bits & mant_mask) != 0);
}

__device__ inline bool ir_is_inf_bits(double x) {
    unsigned long long bits;
    __builtin_memcpy(&bits, &x, sizeof(double));
    return bits == 0x7FF0000000000000ULL;
}

__device__ inline double
ir_nan_aware_max(double a, double b) {
    if (ir_is_nan_bits(a)) return a;
    if (ir_is_nan_bits(b)) return b;
    if (ir_is_inf_bits(a)) return a;
    if (ir_is_inf_bits(b)) return b;
    return fmax(a, b);
}

/* -------------------------------------------------------- */
/* gpu_max_norm_kernel: block-level max|data[i]|
   for residual vectors.
   requires extern shared mem of blockDim.x doubles         */

__global__ void gpu_max_norm_kernel(
    const double * __restrict__ data,
    double * __restrict__ block_maxes,
    int N)
{
    extern __shared__ double shmem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double local_max = 0.0;
    if (gid < N) local_max = fabs(data[gid]);

    shmem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shmem[tid] = ir_nan_aware_max(
                shmem[tid], shmem[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        block_maxes[blockIdx.x] = shmem[0];
}

/* -------------------------------------------------------- */
/* gpu_max_abs_kernel: grid-striding max|data[i]|
   for large arrays (one-time matrix scaling)               */

template <typename IndexType = int>
__global__ void gpu_max_abs_kernel(
    const double * __restrict__ data,
    double * __restrict__ block_maxes,
    IndexType size)
{
    constexpr int TPB = 256;
    IndexType tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    double local_max = 0.0;
    for (IndexType i = tid; i < size;
         i += (IndexType)blockDim.x * gridDim.x)
        local_max = fmax(local_max, fabs(data[i]));

    __shared__ double smem[TPB];
    smem[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = TPB / 2; s > 0; s /= 2) {
        if ((int)threadIdx.x < s)
            smem[threadIdx.x] = fmax(
                smem[threadIdx.x],
                smem[threadIdx.x + s]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        block_maxes[blockIdx.x] = smem[0];
}

/* -------------------------------------------------------- */
/* gpu_compute_pow2_scale_kernel: single-thread
   reduces block maxes to a power-of-2 scale factor.
   optionally outputs the max norm for diagnostics.         */

__global__ void gpu_compute_pow2_scale_kernel(
    double * __restrict__ block_maxes,
    double * __restrict__ scale_out,
    double * __restrict__ norm_out,
    int num_blocks)
{
    double global_max = 0.0;
    for (int i = 0; i < num_blocks; i++)
        global_max = fmax(global_max, block_maxes[i]);

    if (norm_out != nullptr)
        *norm_out = global_max;

    if (global_max == 0.0) global_max = 1.0;

    int e;
    frexp(global_max, &e);
    double scale = ldexp(1.0, e - 1);
    if (scale == 0.0) scale = 1.0;

    *scale_out = scale;
}

/* -------------------------------------------------------- */
/* gpu_apply_scale_and_demote_kernel:
   r16[i] = fp16( r64[i] / *scale_ptr )                    */

__global__ void gpu_apply_scale_and_demote_kernel(
    const double * __restrict__ r64,
    _Float16 * __restrict__ r16,
    const double * __restrict__ scale_ptr,
    int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        double scale = *scale_ptr;
        double inv = (scale == 0.0)
            ? 1.0 : 1.0 / scale;
        r16[gid] =
            (_Float16)((float)(r64[gid] * inv));
    }
}

/* -------------------------------------------------------- */
/* gpu_scale_and_demote_kernel: one-time demotion
   of fp64 matrix to fp16 with device-pointer scale.
   out16[i] = fp16( in64[i] / *scale_ptr )                 */

template <typename IndexType>
__global__ void gpu_scale_and_demote_kernel(
    const double * __restrict__ in64,
    _Float16 * __restrict__ out16,
    const double * __restrict__ scale_ptr,
    IndexType N)
{
    IndexType idx =
        (IndexType)blockIdx.x
        * (IndexType)blockDim.x
        + (IndexType)threadIdx.x;
    if (idx < N) {
        double scale = *scale_ptr;
        double inv = 1.0 / scale;
        out16[idx] =
            (_Float16)((float)(in64[idx] * inv));
    }
}

/* -------------------------------------------------------- */
/* gpu_scalar_multiply_kernel:
   *out = *a * b  (device ptr × host scalar)                */

__global__ void gpu_scalar_multiply_kernel(
    double * __restrict__ out,
    const double * __restrict__ a,
    double b)
{
    *out = *a * b;
}

/* -------------------------------------------------------- */
/* gpu_promote_and_add_kernel_devscale:
   x64[i] += fp16(delta[i]) * (*d_scale)                   */

__global__ void gpu_promote_and_add_kernel_devscale(
    const _Float16 * __restrict__ delta16,
    double * __restrict__ x64,
    const double * __restrict__ d_scale,
    int N)
{
    __shared__ double scale;
    if (threadIdx.x == 0)
        scale = *d_scale;
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        float v = (float)delta16[gid];
        if (isfinite(v))
            x64[gid] += (double)v * scale;
    }
}

/* -------------------------------------------------------- */
/* multi-CU block-column TRSV kernels
   parallelizes the lower-triangular solve
   across all GPU CUs via a host-driven
   block-column loop:
     for each tile b:
       (1) diag solve: 1 block per batch
       (2) column update: many blocks for
           remaining rows                                   */

/* init: copy fp16 RHS into fp32 workspace */

template <int TPB>
__global__ void trsv_mc_init_rhs(
    const _Float16 * __restrict__ rhs_in,
    float * __restrict__ rhs_ws,
    long long stridex,
    int n)
{
    int sys = blockIdx.y;
    for (int i = blockIdx.x * TPB + threadIdx.x;
         i < n; i += gridDim.x * TPB)
        rhs_ws[sys * stridex + i] =
            (float)rhs_in[sys * stridex + i];
}

/* finalize: convert fp32 solution to fp16 */

template <int TPB>
__global__ void trsv_mc_finalize(
    const float * __restrict__ x_ws,
    _Float16 * __restrict__ x16_out,
    long long stridex,
    int n)
{
    int sys = blockIdx.y;
    for (int i = blockIdx.x * TPB + threadIdx.x;
         i < n; i += gridDim.x * TPB)
        x16_out[sys * stridex + i] =
            (_Float16)x_ws[sys * stridex + i];
}

/* diagonal block solve: one block per batch
   element solves the TILE_SIZE x TILE_SIZE
   diagonal block using warp reductions */

template <int TILE_SIZE, int TPB>
__global__ void trsv_mc_diag_fp16(
    int n, int lda, long long strideA,
    const _Float16 * __restrict__ A16,
    int tile_idx,
    float * __restrict__ rhs_ws,
    float * __restrict__ x_ws,
    long long stridex,
    int batch)
{
    int sys = blockIdx.x;
    if (sys >= batch) return;

    const _Float16 *A0 =
        A16 + (long long)sys * strideA;
    float *rhs = rhs_ws + (long long)sys * stridex;
    float *xw  = x_ws   + (long long)sys * stridex;

    constexpr int WARP_SIZE = 64;
    __shared__ float x_cache[TILE_SIZE];
    __shared__ float warp_sums[TPB / WARP_SIZE];

    int tile_start = tile_idx * TILE_SIZE;
    int tile_end = (tile_start + TILE_SIZE < n)
        ? (tile_start + TILE_SIZE) : n;
    int tile_rows = tile_end - tile_start;

    for (int li = 0; li < tile_rows; li++) {
        int i = tile_start + li;
        float psum = 0.0f;

        for (int j = threadIdx.x; j < li;
             j += TPB) {
            int col = tile_start + j;
            long long offset =
                i + (long long)col * lda;
            psum += (float)A0[offset] * x_cache[j];
        }

        int lane_id = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;

        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0;
             off /= 2)
            psum += __shfl_down(psum, off,
                                WARP_SIZE);

        if (lane_id == 0)
            warp_sums[warp_id] = psum;
        __syncthreads();

        constexpr int num_warps = TPB / WARP_SIZE;
        if (warp_id == 0) {
            float v = (lane_id < num_warps)
                ? warp_sums[lane_id] : 0.0f;
            #pragma unroll
            for (int off = num_warps / 2; off > 0;
                 off >>= 1)
                v += __shfl_down(v, off, num_warps);
            if (lane_id == 0)
                warp_sums[0] = v;
        }
        __syncthreads();
        float total = warp_sums[0];

        if (threadIdx.x == 0) {
            float bi = rhs[i];
            long long diag_off =
                i + (long long)i * lda;
            float Lii = (float)A0[diag_off];
            float xi = (bi - total) / Lii;
            x_cache[li] = xi;
            xw[i] = xi;
        }
        __syncthreads();
    }
}

/* column update: many blocks update remaining
   rows below the diagonal tile in parallel */

template <int TILE_SIZE, int TPB>
__global__ void trsv_mc_update_fp16(
    int n, int lda, long long strideA,
    const _Float16 * __restrict__ A16,
    int tile_idx,
    const float * __restrict__ x_ws,
    float * __restrict__ rhs_ws,
    long long stridex,
    int batch)
{
    int sys = blockIdx.y;
    if (sys >= batch) return;

    const _Float16 *A0 =
        A16 + (long long)sys * strideA;
    const float *xw =
        x_ws + (long long)sys * stridex;
    float *rhs =
        rhs_ws + (long long)sys * stridex;

    int tile_start = tile_idx * TILE_SIZE;
    int tile_end = (tile_start + TILE_SIZE < n)
        ? (tile_start + TILE_SIZE) : n;
    int num_cols = tile_end - tile_start;

    __shared__ float x_tile[TILE_SIZE];
    for (int j = threadIdx.x; j < num_cols;
         j += TPB)
        x_tile[j] = xw[tile_start + j];
    __syncthreads();

    int rem_start = tile_end;
    int rem_rows  = n - rem_start;

    for (int idx = blockIdx.x * TPB + threadIdx.x;
         idx < rem_rows;
         idx += gridDim.x * TPB) {
        int row = rem_start + idx;
        float update = 0.0f;

        #pragma unroll 8
        for (int j = 0; j < num_cols; j++) {
            long long offset = row
                + (long long)(tile_start + j) * lda;
            update += (float)A0[offset] * x_tile[j];
        }

        rhs[row] -= update;
    }
}

/* host-driven launcher: orchestrates the
   init → (diag + update) loop → finalize
   sequence across all tiles */

template <int TILE_SIZE = MC_TRSV_TILE_SIZE,
          int TPB = MC_TRSV_TPB>
inline void launch_trsv_multiCU_fp16(
    hipStream_t stream,
    int n, int lda, long long strideA,
    const _Float16 *A16,
    long long stridex,
    _Float16 *x16,
    int batch,
    float *rhs_workspace,
    float *x_workspace)
{
    int num_tiles =
        (n + TILE_SIZE - 1) / TILE_SIZE;

    /* init: fp16 RHS → fp32 workspace */
    {
        int nb = std::min(256,
            (int)((n + TPB - 1) / TPB));
        dim3 grid(nb, batch);
        hipLaunchKernelGGL(
            (trsv_mc_init_rhs<TPB>),
            grid, dim3(TPB), 0, stream,
            x16, rhs_workspace, stridex, n);
    }

    /* block-column loop */
    for (int tile = 0; tile < num_tiles; tile++) {
        hipLaunchKernelGGL(
            (trsv_mc_diag_fp16<TILE_SIZE, TPB>),
            dim3(batch), dim3(TPB), 0, stream,
            n, lda, strideA, A16, tile,
            rhs_workspace, x_workspace,
            stridex, batch);

        int tile_end = std::min(
            (tile + 1) * TILE_SIZE, n);
        int remaining = n - tile_end;
        if (remaining > 0) {
            int nb = std::min(
                MC_TRSV_MAX_UPD_BLOCKS,
                (remaining + TPB - 1) / TPB);
            dim3 grid(nb, batch);
            hipLaunchKernelGGL(
                (trsv_mc_update_fp16<
                    TILE_SIZE, TPB>),
                grid, dim3(TPB), 0, stream,
                n, lda, strideA, A16, tile,
                x_workspace, rhs_workspace,
                stridex, batch);
        }
    }

    /* finalize: fp32 → fp16 */
    {
        int nb = std::min(256,
            (int)((n + TPB - 1) / TPB));
        dim3 grid(nb, batch);
        hipLaunchKernelGGL(
            (trsv_mc_finalize<TPB>),
            grid, dim3(TPB), 0, stream,
            x_workspace, x16, stridex, n);
    }
}

/* -------------------------------------------------------- */
/* backward multi-CU block-column TRSV kernels
   mirrors the forward TRSV but processes tiles
   in reverse order (last to first) and solves
   each diagonal block via backward substitution.

   TRANSPOSE template parameter controls matrix
   access:
     true  = L^T (transposed lower, for Cholesky)
     false = U  (upper triangular, for LU)            */

/* backward diagonal solve: one block per batch
   element solves the TILE_SIZE x TILE_SIZE
   diagonal block bottom-to-top */

template <int TILE_SIZE, int TPB, bool TRANSPOSE>
__global__ void trsv_mc_diag_backward(
    int n, int lda, long long strideA,
    const _Float16 * __restrict__ A16,
    int tile_idx,
    float * __restrict__ rhs_ws,
    float * __restrict__ x_ws,
    long long stridex,
    int batch)
{
    int sys = blockIdx.x;
    if (sys >= batch) return;

    const _Float16 *A0 =
        A16 + (long long)sys * strideA;
    float *rhs = rhs_ws + (long long)sys * stridex;
    float *xw  = x_ws   + (long long)sys * stridex;

    constexpr int WARP_SIZE = 64;
    __shared__ float x_cache[TILE_SIZE];
    __shared__ float warp_sums[TPB / WARP_SIZE];

    int tile_start = tile_idx * TILE_SIZE;
    int tile_end = (tile_start + TILE_SIZE < n)
        ? (tile_start + TILE_SIZE) : n;
    int tile_rows = tile_end - tile_start;

    /* backward: process rows bottom-to-top */
    for (int li = tile_rows - 1; li >= 0; li--) {
        int i = tile_start + li;
        float psum = 0.0f;

        /* sum over columns AFTER i in the tile */
        for (int j = li + 1 + (int)threadIdx.x;
             j < tile_rows; j += TPB) {
            int col = tile_start + j;
            long long offset;
            if constexpr (TRANSPOSE)
                offset = col + (long long)i * lda;
            else
                offset = i + (long long)col * lda;
            psum += (float)A0[offset]
                  * x_cache[j];
        }

        int lane_id = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;

        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0;
             off /= 2)
            psum += __shfl_down(psum, off,
                                WARP_SIZE);

        if (lane_id == 0)
            warp_sums[warp_id] = psum;
        __syncthreads();

        constexpr int num_warps = TPB / WARP_SIZE;
        if (warp_id == 0) {
            float v = (lane_id < num_warps)
                ? warp_sums[lane_id] : 0.0f;
            #pragma unroll
            for (int off = num_warps / 2; off > 0;
                 off >>= 1)
                v += __shfl_down(v, off, num_warps);
            if (lane_id == 0)
                warp_sums[0] = v;
        }
        __syncthreads();
        float total = warp_sums[0];

        if (threadIdx.x == 0) {
            float bi = rhs[i];
            long long diag_off =
                i + (long long)i * lda;
            float Aii = (float)A0[diag_off];
            float xi = (bi - total) / Aii;
            x_cache[li] = xi;
            xw[i] = xi;
        }
        __syncthreads();
    }
}

/* backward column update: many blocks update
   remaining rows ABOVE the diagonal tile */

template <int TILE_SIZE, int TPB, bool TRANSPOSE>
__global__ void trsv_mc_update_backward(
    int n, int lda, long long strideA,
    const _Float16 * __restrict__ A16,
    int tile_idx,
    const float * __restrict__ x_ws,
    float * __restrict__ rhs_ws,
    long long stridex,
    int batch)
{
    int sys = blockIdx.y;
    if (sys >= batch) return;

    const _Float16 *A0 =
        A16 + (long long)sys * strideA;
    const float *xw =
        x_ws + (long long)sys * stridex;
    float *rhs =
        rhs_ws + (long long)sys * stridex;

    int tile_start = tile_idx * TILE_SIZE;
    int tile_end = (tile_start + TILE_SIZE < n)
        ? (tile_start + TILE_SIZE) : n;
    int num_cols = tile_end - tile_start;

    __shared__ float x_tile[TILE_SIZE];
    for (int j = threadIdx.x; j < num_cols;
         j += TPB)
        x_tile[j] = xw[tile_start + j];
    __syncthreads();

    /* update rows ABOVE tile: [0, tile_start) */
    int rem_rows = tile_start;

    for (int idx = blockIdx.x * TPB + threadIdx.x;
         idx < rem_rows;
         idx += gridDim.x * TPB) {
        int row = idx;
        float update = 0.0f;

        #pragma unroll 8
        for (int j = 0; j < num_cols; j++) {
            int col = tile_start + j;
            long long offset;
            if constexpr (TRANSPOSE)
                offset = col + (long long)row * lda;
            else
                offset = row + (long long)col * lda;
            update += (float)A0[offset] * x_tile[j];
        }

        rhs[row] -= update;
    }
}

/* backward host-driven launcher: orchestrates
   init → (diag + update) loop → finalize
   in reverse tile order */

template <int TILE_SIZE = MC_TRSV_TILE_SIZE,
          int TPB = MC_TRSV_TPB,
          bool TRANSPOSE = true>
inline void launch_trsv_multiCU_fp16_backward(
    hipStream_t stream,
    int n, int lda, long long strideA,
    const _Float16 *A16,
    long long stridex,
    _Float16 *x16,
    int batch,
    float *rhs_workspace,
    float *x_workspace)
{
    int num_tiles =
        (n + TILE_SIZE - 1) / TILE_SIZE;

    /* init: fp16 RHS → fp32 workspace */
    {
        int nb = std::min(256,
            (int)((n + TPB - 1) / TPB));
        dim3 grid(nb, batch);
        hipLaunchKernelGGL(
            (trsv_mc_init_rhs<TPB>),
            grid, dim3(TPB), 0, stream,
            x16, rhs_workspace, stridex, n);
    }

    /* reverse block-column loop */
    for (int tile = num_tiles - 1;
         tile >= 0; tile--) {
        hipLaunchKernelGGL(
            (trsv_mc_diag_backward<
                TILE_SIZE, TPB, TRANSPOSE>),
            dim3(batch), dim3(TPB), 0, stream,
            n, lda, strideA, A16, tile,
            rhs_workspace, x_workspace,
            stridex, batch);

        int tile_start = tile * TILE_SIZE;
        if (tile_start > 0) {
            int nb = std::min(
                MC_TRSV_MAX_UPD_BLOCKS,
                (tile_start + TPB - 1) / TPB);
            dim3 grid(nb, batch);
            hipLaunchKernelGGL(
                (trsv_mc_update_backward<
                    TILE_SIZE, TPB, TRANSPOSE>),
                grid, dim3(TPB), 0, stream,
                n, lda, strideA, A16, tile,
                x_workspace, rhs_workspace,
                stridex, batch);
        }
    }

    /* finalize: fp32 → fp16 */
    {
        int nb = std::min(256,
            (int)((n + TPB - 1) / TPB));
        dim3 grid(nb, batch);
        hipLaunchKernelGGL(
            (trsv_mc_finalize<TPB>),
            grid, dim3(TPB), 0, stream,
            x_workspace, x16, stridex, n);
    }
}

/* -------------------------------------------------------- */
/* host utilities                                           */

/* next power of 2 >= x */
inline double next_power_of_2(double x) {
    if (x <= 0.0) return 1.0;
    int exp;
    double m = std::frexp(x, &exp);
    if (m == 0.5) return std::ldexp(1.0, exp - 1);
    return std::ldexp(1.0, exp);
}

/* optimal fp16 scale: keeps values within
   safe fp16 range after division by scale */
inline double compute_optimal_fp16_scale(
    double max_abs)
{
    constexpr double FP16_MAX_SAFE = 32768.0;
    if (max_abs < 1e-30) return 1.0;
    double required = max_abs / FP16_MAX_SAFE;
    return next_power_of_2(required);
}

/* two-stage max-abs reduction on GPU
   (allocates temporary, synchronizes) */
inline double gpu_compute_max_abs(
    const double *d_data, long long size)
{
    constexpr int TPB = 256;
    int nblocks = std::min(256,
        (int)((size + TPB - 1) / TPB));

    double *d_block_maxes;
    hipMalloc(&d_block_maxes,
              nblocks * sizeof(double));

    if (size >= (long long)INT_MAX) {
        hipLaunchKernelGGL(
            (gpu_max_abs_kernel<long long>),
            dim3(nblocks), dim3(TPB), 0, 0,
            d_data, d_block_maxes,
            (long long)size);
    } else {
        hipLaunchKernelGGL(
            (gpu_max_abs_kernel<int>),
            dim3(nblocks), dim3(TPB), 0, 0,
            d_data, d_block_maxes, (int)size);
    }

    std::vector<double> h_maxes(nblocks);
    hipMemcpy(h_maxes.data(), d_block_maxes,
              nblocks * sizeof(double),
              hipMemcpyDeviceToHost);
    hipFree(d_block_maxes);

    double global_max = 0.0;
    for (int i = 0; i < nblocks; i++)
        global_max = std::max(global_max,
                              h_maxes[i]);

    return global_max;
}
