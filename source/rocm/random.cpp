#include "rocm/vendor.h"
#include "rocm/check.h"

template <typename T>
void fill_random(gpu_context &ctx, T *d_ptr, size_t count);

template <>
void fill_random<double>(gpu_context &ctx, double *d_ptr, size_t count) {
    HIPRAND_CHECK(hiprandGenerateUniformDouble(ctx.rand_gen, d_ptr, count));
}

template <>
void fill_random<float>(gpu_context &ctx, float *d_ptr, size_t count) {
    HIPRAND_CHECK(hiprandGenerateUniform(ctx.rand_gen, d_ptr, count));
}
