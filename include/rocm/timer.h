#pragma once

#include <hip/hip_runtime.h>

/* single event pair for whole-operation timing */

struct gpu_timer {
    hipEvent_t start_event;
    hipEvent_t stop_event;

    void  init();
    void  destroy();
    void  start(hipStream_t stream);
    void  stop(hipStream_t stream);
    void  synchronize();
    float elapsed_ms();
};

/* pool of event pairs for per-event instrument
   mode (avoids create/destroy per event) */

struct gpu_timer_pool {
    hipEvent_t *start_events;
    hipEvent_t *stop_events;
    size_t      capacity;
    size_t      count;

    void  init(size_t cap);
    void  destroy();
    void  reset();
    void  start(size_t idx, hipStream_t stream);
    void  stop(size_t idx, hipStream_t stream);
    void  synchronize_all();
    float elapsed_ms(size_t idx);
};
