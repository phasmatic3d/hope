#pragma once
#include <nanobind/nanobind.h>

#include "csv_entry.hpp"
#include <chrono>

namespace nb = nanobind;

void bind_csv_entry(nb::module_ &m);