#include "csv_entry_bindings.hpp"

void bind_csv_entry(nb::module_ &m) {
    nb::class_<CsvFileEntry> cls(m, "CsvFileEntry");

    // Special-case the time_point as milliseconds
    cls.def_prop_ro("timestamp_ms", [](const CsvFileEntry& e) 
    {
        using namespace std::chrono;
        return duration_cast<milliseconds>(e.timestamp.time_since_epoch()).count();
    });

    #define NB_BIND_FIELD(T, name, label, ...) \
    cls.def_prop_ro(#name, [](const CsvFileEntry& e) -> decltype(auto) { return (e.name); });
    CSV_FIELDS_NO_TS(NB_BIND_FIELD)
    #undef NB_BIND_FIELD
}