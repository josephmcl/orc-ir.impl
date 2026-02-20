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

/* -------------------------------------------------------- */
/* profiler CLI                                             */

#ifndef ENVIRONMENT_SLUG
#define ENVIRONMENT_SLUG "unknown"
#endif

void print_profiler_usage(const char *prog) {
    std::cerr
        << "usage: " << prog << " [options]\n"
        << "\n"
        << "options:\n"
        << "  --factor {lu|chol}       "
           "factorization type (required)\n"
        << "  --n N                    "
           "matrix dimension (required)\n"
        << "  --batch B                "
           "batch size (default: 1)\n"
        << "  --nrhs K                 "
           "number of rhs (default: 1)\n"
        << "  --solver NAME            "
           "solver variant (required)\n"
        << "         rocblas64          "
           "rocsolver getrs/potrs, fp64\n"
        << "         rocblas32          "
           "rocsolver getrs/potrs, fp32\n"
        << "         rocblastrsv64      "
           "rocblas trsm, fp64\n"
        << "         rocblastrsv32      "
           "rocblas trsm, fp32\n"
        << "         ir3                "
           "3-precision IR, L*x=b (chol)\n"
        << "         ir3chol            "
           "3-precision IR, full A=LL^T\n"
        << "         ir3lu              "
           "3-precision IR, full PA=LU\n"
        << "         ir3A               "
           "3-precision IR, A-resid + 2 TRSVs\n"
        << "         ir3Af32            "
           "3-precision IR, A-resid + fp32 TRSM\n"
        << "  --iters N                "
           "IR iterations (default: 10)\n"
        << "  --mode {profile|instrument"
           "|instrument_verify}\n"
        << "                           "
           "profiling mode (default: profile)\n"
        << "  --warmup N               "
           "warmup iterations (default: 5)\n"
        << "  --runs N                 "
           "measured iterations (default: 20)\n"
        << "  --help                   "
           "show this message\n";
}

profiler_cli_args parse_profiler_args(
    int argc, char **argv
) {
    profiler_cli_args args{};
    args.config.mode             = profile_mode::profile;
    args.config.warmup_runs      = 5;
    args.config.measured_runs    = 20;
    args.config.desc.batch       = 1;
    args.config.desc.nrhs        = 1;
    args.config.desc.working_prec = precision::fp64;
    args.config.environment_slug = ENVIRONMENT_SLUG;
    args.config.solver_name      = nullptr;
    args.help                    = false;
    args.ir_iters                = 10;

    bool have_factor = false;
    bool have_n      = false;
    bool have_solver = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help") {
            args.help = true;
            return args;
        }

        if (i + 1 >= argc) {
            std::cerr << "error: " << arg
                      << " requires an argument\n";
            exit(1);
        }

        std::string val = argv[++i];

        if (arg == "--factor") {
            if (val == "lu")
                args.config.desc.factor =
                    factor_type::lu;
            else if (val == "chol")
                args.config.desc.factor =
                    factor_type::cholesky;
            else {
                std::cerr << "error: unknown factor "
                          << "type '" << val << "'\n";
                exit(1);
            }
            have_factor = true;
        } else if (arg == "--n") {
            args.config.desc.n = std::stoul(val);
            have_n = true;
        } else if (arg == "--batch") {
            args.config.desc.batch = std::stoul(val);
        } else if (arg == "--nrhs") {
            args.config.desc.nrhs = std::stoul(val);
        } else if (arg == "--solver") {
            args.config.solver_name = argv[i];
            have_solver = true;
        } else if (arg == "--mode") {
            if (val == "profile")
                args.config.mode =
                    profile_mode::profile;
            else if (val == "instrument")
                args.config.mode =
                    profile_mode::instrument;
            else if (val == "instrument_verify")
                args.config.mode =
                    profile_mode::instrument_verify;
            else {
                std::cerr << "error: unknown mode '"
                          << val << "'\n";
                exit(1);
            }
        } else if (arg == "--warmup") {
            args.config.warmup_runs = std::stoul(val);
        } else if (arg == "--runs") {
            args.config.measured_runs = std::stoul(val);
        } else if (arg == "--iters") {
            args.ir_iters = std::stoul(val);
        } else {
            std::cerr << "error: unknown option '"
                      << arg << "'\n";
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
    if (!have_solver) {
        std::cerr << "error: --solver is required\n";
        exit(1);
    }

    return args;
}
