#pragma once

#include "problem.h"

#include <cstddef>
#include <cstdint>
#include <string>

/* -------------------------------------------------------- */
/* profiling mode                                           */

enum class profile_mode : uint8_t {
    profile           = 0,
    instrument        = 1,
    instrument_verify = 2
};

const char *profile_mode_name(profile_mode m);

/* -------------------------------------------------------- */
/* profiler configuration                                   */

struct profiler_config {
    profile_mode       mode;
    size_t             warmup_runs;
    size_t             measured_runs;
    problem_descriptor desc;
    const char        *environment_slug;
    const char        *solver_name;
};

/* -------------------------------------------------------- */
/* timing record: one per (run, event) pair                 */

struct time_record {
    size_t      run;
    const char *label;
    float       gpu_ms;   /* -1.0f if not measured */
    double      cpu_ms;   /* -1.0  if not measured */
};

/* -------------------------------------------------------- */
/* verification record: one per (run, metric) pair          */

struct verify_record {
    size_t      run;
    const char *metric_name;
    double      value;
};

/* -------------------------------------------------------- */
/* profiler                                                 */

struct profiler {
    profiler_config config;

    time_record   *time_log;
    size_t         time_count;
    size_t         time_capacity;

    verify_record *verify_log;
    size_t         verify_count;
    size_t         verify_capacity;

    size_t current_run;
    bool   in_warmup;

    void init(const profiler_config &cfg,
              size_t max_events_per_run,
              size_t max_verify_per_run);
    void destroy();

    /* run lifecycle */
    void begin_warmup();
    void begin_run();
    void end_run();

    /* record a timed event (skipped in warmup) */
    void record_gpu_time(const char *label,
                         float ms);
    void record_time(const char *label,
                     float gpu_ms, double cpu_ms);

    /* record a verification metric */
    void record_metric(const char *name,
                       double value);

    /* output */
    std::string output_path() const;
    void write_csv() const;
    void print_summary() const;
};

/* -------------------------------------------------------- */
/* helper: build problem slug from descriptor               */
/*   e.g. "lu.64.1.1"                                       */

std::string problem_slug(
    const problem_descriptor &desc);
