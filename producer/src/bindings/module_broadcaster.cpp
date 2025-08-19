#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include "csv_entry/csv_entry_bindings.hpp"
#include "server/server_bindings.hpp"

NB_MODULE(broadcaster, m) {
    m.doc() = "Hope Producer broadcaster bindings (server + csv)";
    bind_csv_entry(m);
    bind_broadcaster(m);
}
