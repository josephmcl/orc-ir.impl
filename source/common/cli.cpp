#include "common/cli.h"

#include <iostream>
#include <string>

void print_usage(const char *program_name) {
    std::cerr
        << "usage: " << program_name << " [options]\n"
        << "\n"
        << "options:\n"
        << "  --factor {lu|chol}       factorization type (required)\n"
        << "  --n N                    matrix dimension (required)\n"
        << "  --batch B                batch size (default: 1)\n"
        << "  --nrhs K                 number of right-hand sides (default: 1)\n"
        << "  --precision {fp64|fp32}  working precision (default: fp64)\n"
        << "  --help                   show this message\n";
}

cli_args parse_args(int argc, char **argv) {
    cli_args args{};
    args.desc.batch        = 1;
    args.desc.nrhs         = 1;
    args.desc.working_prec = precision::fp64;
    args.help              = false;

    bool have_factor = false;
    bool have_n      = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help") {
            args.help = true;
            return args;
        }

        if (i + 1 >= argc) {
            std::cerr << "error: " << arg << " requires an argument\n";
            exit(1);
        }

        std::string val = argv[++i];

        if (arg == "--factor") {
            if (val == "lu")
                args.desc.factor = factor_type::lu;
            else if (val == "chol")
                args.desc.factor = factor_type::cholesky;
            else {
                std::cerr << "error: unknown factor type '" << val << "'\n";
                exit(1);
            }
            have_factor = true;
        } else if (arg == "--n") {
            args.desc.n = std::stoul(val);
            have_n = true;
        } else if (arg == "--batch") {
            args.desc.batch = std::stoul(val);
        } else if (arg == "--nrhs") {
            args.desc.nrhs = std::stoul(val);
        } else if (arg == "--precision") {
            if (val == "fp64")
                args.desc.working_prec = precision::fp64;
            else if (val == "fp32")
                args.desc.working_prec = precision::fp32;
            else if (val == "fp16")
                args.desc.working_prec = precision::fp16;
            else {
                std::cerr << "error: unknown precision '" << val << "'\n";
                exit(1);
            }
        } else {
            std::cerr << "error: unknown option '" << arg << "'\n";
            exit(1);
        }
    }

    if (!have_factor) {
        std::cerr << "error: --factor is required\n";
        exit(1);
    }
    if (!have_n) {
        std::cerr << "error: --n is required\n";
        exit(1);
    }

    return args;
}
