#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>  

#include "server.hpp"

namespace nb = nanobind;

void bind_broadcaster(nb::module_ &m);