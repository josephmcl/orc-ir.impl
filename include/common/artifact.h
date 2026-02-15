#pragma once

#include "problem.h"
#include <string>

/* builds: artifact/{factor_type}.{n}.{batch}.{nrhs}/ */
std::string artifact_directory(const problem_descriptor &desc);

void write_array(const std::string &path, const void *data, size_t bytes);

void write_metadata(const std::string &dir, const problem_descriptor &desc);
