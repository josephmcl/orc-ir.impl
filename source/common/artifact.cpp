#include "common/artifact.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

const char *precision_name(precision p) {
    switch (p) {
        case precision::fp64: return "fp64";
        case precision::fp32: return "fp32";
        case precision::fp16: return "fp16";
    }
    return "unknown";
}

const char *factor_type_name(factor_type ft) {
    switch (ft) {
        case factor_type::lu:       return "lu";
        case factor_type::cholesky: return "cholesky";
    }
    return "unknown";
}

std::string artifact_directory(const problem_descriptor &desc) {
    return "artifact/"
         + std::string(factor_type_name(desc.factor))
         + "." + std::to_string(desc.n)
         + "." + std::to_string(desc.batch)
         + "." + std::to_string(desc.nrhs);
}

void write_array(const std::string &path, const void *data, size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "error: could not open " << path << " for writing\n";
        exit(1);
    }
    out.write(static_cast<const char *>(data), bytes);
    if (!out) {
        std::cerr << "error: failed writing " << bytes
                  << " bytes to " << path << "\n";
        exit(1);
    }
}

void write_metadata(const std::string &dir, const problem_descriptor &desc) {
    fs::create_directories(dir);

    std::string path = dir + "/meta.json";
    std::ofstream out(path);
    if (!out) {
        std::cerr << "error: could not open " << path << " for writing\n";
        exit(1);
    }

    size_t matrix_bytes = desc.batch * desc.n * desc.n
                        * precision_bytes(desc.working_prec);
    size_t rhs_bytes    = desc.batch * desc.n * desc.nrhs
                        * precision_bytes(desc.working_prec);

    out << "{\n"
        << "  \"factor_type\": \"" << factor_type_name(desc.factor) << "\",\n"
        << "  \"n\": " << desc.n << ",\n"
        << "  \"batch\": " << desc.batch << ",\n"
        << "  \"nrhs\": " << desc.nrhs << ",\n"
        << "  \"precision\": \"" << precision_name(desc.working_prec) << "\",\n"
        << "  \"precision_bytes\": " << precision_bytes(desc.working_prec) << ",\n"
        << "  \"layout\": \"column_major\",\n"
        << "  \"matrix_bytes\": " << matrix_bytes << ",\n"
        << "  \"rhs_bytes\": " << rhs_bytes << ",\n"
        << "  \"files\": {\n"
        << "    \"a\": \"a.bin\",\n"
        << "    \"l\": \"l.bin\",\n";

    if (desc.factor == factor_type::lu) {
        out << "    \"u\": \"u.bin\",\n"
            << "    \"ipiv\": \"ipiv.bin\",\n";
    }

    out << "    \"b\": \"b.bin\",\n"
        << "    \"x\": \"x.bin\"\n"
        << "  }\n"
        << "}\n";
}
