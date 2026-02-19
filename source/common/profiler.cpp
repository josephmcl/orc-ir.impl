#include "common/profiler.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

/* -------------------------------------------------------- */
/* name helpers                                             */

const char *profile_mode_name(profile_mode m) {
    switch (m) {
        case profile_mode::profile:
            return "profile";
        case profile_mode::instrument:
            return "instrument";
        case profile_mode::instrument_verify:
            return "instrument_verify";
    }
    return "unknown";
}

std::string problem_slug(
    const problem_descriptor &desc
) {
    return std::string(factor_type_name(desc.factor))
         + "." + std::to_string(desc.n)
         + "." + std::to_string(desc.batch)
         + "." + std::to_string(desc.nrhs);
}

/* -------------------------------------------------------- */
/* profiler lifecycle                                       */

void profiler::init(
    const profiler_config &cfg,
    size_t max_events_per_run,
    size_t max_verify_per_run
) {
    config = cfg;

    time_capacity = cfg.measured_runs
                  * max_events_per_run;
    time_count    = 0;
    time_log      = new time_record[time_capacity];

    verify_capacity = cfg.measured_runs
                    * max_verify_per_run;
    verify_count    = 0;
    if (verify_capacity > 0)
        verify_log = new verify_record[verify_capacity];
    else
        verify_log = nullptr;

    current_run = 0;
    in_warmup   = false;
}

void profiler::destroy() {
    delete[] time_log;
    time_log = nullptr;

    delete[] verify_log;
    verify_log = nullptr;
}

/* -------------------------------------------------------- */
/* run lifecycle                                            */

void profiler::begin_warmup() {
    in_warmup = true;
}

void profiler::begin_run() {
    /* nothing to do; current_run is advanced
       in end_run */
}

void profiler::end_run() {
    if (in_warmup) return;
    current_run++;
}

/* -------------------------------------------------------- */
/* event recording                                         */

void profiler::record_gpu_time(const char *label,
                               float ms) {
    if (in_warmup) return;
    if (time_count >= time_capacity) return;
    time_log[time_count] = {
        current_run, label, ms, -1.0
    };
    time_count++;
}

void profiler::record_time(const char *label,
                           float gpu_ms,
                           double cpu_ms) {
    if (in_warmup) return;
    if (time_count >= time_capacity) return;
    time_log[time_count] = {
        current_run, label, gpu_ms, cpu_ms
    };
    time_count++;
}

/* -------------------------------------------------------- */
/* verification metrics                                     */

void profiler::record_metric(const char *name,
                             double value) {
    if (in_warmup) return;
    if (verify_count >= verify_capacity) return;
    verify_log[verify_count] = {
        current_run, name, value
    };
    verify_count++;
}

/* -------------------------------------------------------- */
/* output path                                              */

std::string profiler::output_path() const {
    std::string slug = problem_slug(config.desc);
    return "profile/"
         + std::string(config.environment_slug)
         + "/" + slug
         + "/" + std::string(config.solver_name)
         + ".csv";
}

/* -------------------------------------------------------- */
/* CSV output                                               */

void profiler::write_csv() const {
    std::string path = output_path();
    std::string dir  = path.substr(
        0, path.rfind('/'));
    fs::create_directories(dir);

    std::ofstream out(path);
    if (!out) {
        std::cerr << "error: could not open "
                  << path << " for writing\n";
        return;
    }

    bool has_verify =
        config.mode == profile_mode::instrument_verify
        && verify_count > 0;

    /* header */
    out << "environment,factor,n,batch,nrhs,"
        << "precision,solver,mode,"
        << "warmup,measured_runs,"
        << "run,event,gpu_ms,cpu_ms";
    if (has_verify)
        out << ",metric_value";
    out << "\n";

    /* common prefix for every row */
    std::string prefix =
        std::string(config.environment_slug)
        + "," + factor_type_name(config.desc.factor)
        + "," + std::to_string(config.desc.n)
        + "," + std::to_string(config.desc.batch)
        + "," + std::to_string(config.desc.nrhs)
        + "," + precision_name(
                    config.desc.working_prec)
        + "," + config.solver_name
        + "," + profile_mode_name(config.mode)
        + "," + std::to_string(config.warmup_runs)
        + "," + std::to_string(
                    config.measured_runs);

    /* timing rows */
    for (size_t i = 0; i < time_count; i++) {
        const time_record &r = time_log[i];
        out << prefix
            << "," << r.run
            << "," << r.label
            << "," << r.gpu_ms
            << "," << r.cpu_ms;
        if (has_verify) out << ",";
        out << "\n";
    }

    /* verification rows */
    for (size_t i = 0; i < verify_count; i++) {
        const verify_record &v = verify_log[i];
        out << prefix
            << "," << v.run
            << "," << v.metric_name
            << ",,";
        if (has_verify) out << v.value;
        out << "\n";
    }

    std::cerr << "wrote " << path << "\n";
}

/* -------------------------------------------------------- */
/* summary                                                  */

void profiler::print_summary() const {
    std::cerr << "profiler summary ("
              << profile_mode_name(config.mode)
              << ", " << config.warmup_runs
              << " warmup, "
              << config.measured_runs
              << " measured):\n";

    /* collect unique labels */
    std::vector<const char *> labels;
    for (size_t i = 0; i < time_count; i++) {
        const char *l = time_log[i].label;
        bool found = false;
        for (size_t j = 0; j < labels.size(); j++) {
            if (std::string(labels[j]) == l) {
                found = true;
                break;
            }
        }
        if (!found) labels.push_back(l);
    }

    /* per-label statistics */
    for (size_t li = 0; li < labels.size(); li++) {
        std::vector<float> gpu_times;
        std::vector<double> cpu_times;

        for (size_t i = 0; i < time_count; i++) {
            if (std::string(time_log[i].label)
                != labels[li])
                continue;
            if (time_log[i].gpu_ms >= 0.0f)
                gpu_times.push_back(
                    time_log[i].gpu_ms);
            if (time_log[i].cpu_ms >= 0.0)
                cpu_times.push_back(
                    time_log[i].cpu_ms);
        }

        std::cerr << "  " << labels[li] << ":";

        if (!gpu_times.empty()) {
            std::sort(gpu_times.begin(),
                      gpu_times.end());
            float gmin = gpu_times.front();
            float gmed = gpu_times[
                gpu_times.size() / 2];
            float gmax = gpu_times.back();
            std::cerr << " gpu "
                      << gmin << " / "
                      << gmed << " / "
                      << gmax << " ms"
                      << " (min/med/max)";
        }

        if (!cpu_times.empty()) {
            std::sort(cpu_times.begin(),
                      cpu_times.end());
            double cmin = cpu_times.front();
            double cmed = cpu_times[
                cpu_times.size() / 2];
            double cmax = cpu_times.back();
            std::cerr << " cpu "
                      << cmin << " / "
                      << cmed << " / "
                      << cmax << " ms"
                      << " (min/med/max)";
        }

        std::cerr << "\n";
    }
}
