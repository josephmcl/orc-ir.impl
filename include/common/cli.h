#pragma once

#include "problem.h"
#include "profiler.h"

struct cli_args {
    problem_descriptor desc;
    bool               help;
};

cli_args parse_args(int argc, char **argv);

void print_usage(const char *program_name);

/* -------------------------------------------------------- */
/* profiler CLI (for benchmark binaries)                    */

struct profiler_cli_args {
    profiler_config config;
    bool            help;
};

profiler_cli_args parse_profiler_args(
    int argc, char **argv);

void print_profiler_usage(const char *prog);
