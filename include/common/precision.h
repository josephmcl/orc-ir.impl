#pragma once

#include <cstddef>
#include <cstdint>

enum class precision : uint8_t {
    fp64 = 0,
    fp32 = 1,
    fp16 = 2
};

template <precision P> struct precision_type;
template <> struct precision_type<precision::fp64> { using type = double; };
template <> struct precision_type<precision::fp32> { using type = float; };

template <precision P>
using precision_t = typename precision_type<P>::type;

constexpr size_t precision_bytes(precision p) {
    switch (p) {
        case precision::fp64: return 8;
        case precision::fp32: return 4;
        case precision::fp16: return 2;
    }
    return 0;
}

const char *precision_name(precision p);
