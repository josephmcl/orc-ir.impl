#pragma once

#include "precision.h"
#include <cstddef>
#include <cstdint>

enum class factor_type : uint8_t {
    lu = 0,
    cholesky = 1
};

const char *factor_type_name(factor_type ft);

struct problem_descriptor {
    factor_type factor;
    size_t      n;
    size_t      batch;
    size_t      nrhs;
    precision   working_prec;
};
