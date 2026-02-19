#include "rocm/timer.h"
#include "rocm/check.h"

/* -------------------------------------------------------- */
/* gpu_timer                                                */

void gpu_timer::init() {
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));
}

void gpu_timer::destroy() {
    HIP_CHECK(hipEventDestroy(start_event));
    HIP_CHECK(hipEventDestroy(stop_event));
}

void gpu_timer::start(hipStream_t stream) {
    HIP_CHECK(hipEventRecord(start_event, stream));
}

void gpu_timer::stop(hipStream_t stream) {
    HIP_CHECK(hipEventRecord(stop_event, stream));
}

void gpu_timer::synchronize() {
    HIP_CHECK(hipEventSynchronize(stop_event));
}

float gpu_timer::elapsed_ms() {
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(
        &ms, start_event, stop_event));
    return ms;
}

/* -------------------------------------------------------- */
/* gpu_timer_pool                                           */

void gpu_timer_pool::init(size_t cap) {
    capacity = cap;
    count    = 0;
    start_events = new hipEvent_t[cap];
    stop_events  = new hipEvent_t[cap];
    for (size_t i = 0; i < cap; i++) {
        HIP_CHECK(hipEventCreate(&start_events[i]));
        HIP_CHECK(hipEventCreate(&stop_events[i]));
    }
}

void gpu_timer_pool::destroy() {
    for (size_t i = 0; i < capacity; i++) {
        HIP_CHECK(hipEventDestroy(start_events[i]));
        HIP_CHECK(hipEventDestroy(stop_events[i]));
    }
    delete[] start_events;
    delete[] stop_events;
}

void gpu_timer_pool::reset() {
    count = 0;
}

void gpu_timer_pool::start(size_t idx,
                           hipStream_t stream) {
    HIP_CHECK(hipEventRecord(
        start_events[idx], stream));
    if (idx >= count) count = idx + 1;
}

void gpu_timer_pool::stop(size_t idx,
                          hipStream_t stream) {
    HIP_CHECK(hipEventRecord(
        stop_events[idx], stream));
}

void gpu_timer_pool::synchronize_all() {
    for (size_t i = 0; i < count; i++) {
        HIP_CHECK(hipEventSynchronize(
            stop_events[i]));
    }
}

float gpu_timer_pool::elapsed_ms(size_t idx) {
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(
        &ms, start_events[idx], stop_events[idx]));
    return ms;
}
