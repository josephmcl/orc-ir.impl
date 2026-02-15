#pragma once

#include "problem.h"

struct cli_args {
    problem_descriptor desc;
    bool               help;
};

cli_args parse_args(int argc, char **argv);

void print_usage(const char *program_name);
