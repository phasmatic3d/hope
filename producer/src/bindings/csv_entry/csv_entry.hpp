#include <chrono>
#include <string>
#include <vector>
#include <sstream>
#include <type_traits>

#include "time_types.hpp"


#define CSV_FIELDS_NO_TS(X) \
  X(double,             approximate_rtt_ms,      "approximate_rtt_ms(ping or through client)", std::numeric_limits<double>::quiet_NaN()) \
  X(double,             one_way_ms,              "one_way_ms(through client)",                 std::numeric_limits<double>::quiet_NaN()) \
  X(double,             one_way_plus_processing, "one_way_plus_processing(through client)",    std::numeric_limits<double>::quiet_NaN()) \
  X(double,             wait_in_queue,           "wait_in_queue(through client)",              std::numeric_limits<double>::quiet_NaN()) \
  X(double,             pure_decode_ms,          "pure_decode_ms(through client)",             std::numeric_limits<double>::quiet_NaN()) \
  X(double,             pure_geometry_upload_ms, "pure_geometry_upload_ms(through client)",    std::numeric_limits<double>::quiet_NaN()) \
  X(double,             pure_render_ms,          "pure_render_ms(through client)",             std::numeric_limits<double>::quiet_NaN()) \
  X(double,             pure_processing_ms,      "pure_processing_ms(through client)",         std::numeric_limits<double>::quiet_NaN()) \
  X(std::vector<double>,chunk_decode_times,      "chunk_decode_times(through client)",         std::vector<double>{}) \
  X(std::string,        connection_id,           "connection_id",                              std::string{}) \
  X(std::string,        error,                   "error",                                      std::string{"no error"}) \
  X(size_t,             broadcast_round,         "broadcast_round",                            size_t{0}) \
  X(size_t,             batch_id,                "batch_id",                                   size_t{0}) \
  X(size_t,             message_size,            "message_size",                               size_t{0}) \
  X(size_t,             connections_size,        "connections_size",                           size_t{0})

#define CSV_FIELDS(X) \
  X(hope::Timestamp,          timestamp,               "timestamp_ms_since_epoch",             hope::Timestamp{}) \
  CSV_FIELDS_NO_TS(X)

#define HEADER_ELEM(T, name, label, ...) label ","
static constexpr char header_raw[] = CSV_FIELDS(HEADER_ELEM) "";
#undef HEADER_ELEM

static constexpr std::string_view header_sv = []() constexpr 
{
    std::string_view v{header_raw, sizeof(header_raw) - 1}; // drop null
    return v.substr(0, v.size() - 1);                       // drop trailing comma
}();

template <class T>
using csv_getter_ret_t =
    std::conditional_t<(std::is_trivially_copyable<T>::value && sizeof(T) <= sizeof(void*)), T, const T&>;

template<class T>
using csv_setter_param_t =
    std::conditional_t<(std::is_trivially_copyable_v<T> && sizeof(T) <= sizeof(void*)), T, const T&>;


struct CsvFileEntry
{
    #define DECLARE(T, name, label, def) T name = def;
    CSV_FIELDS(DECLARE)
    #undef DECLARE

    #define GETTER(T, name, label, ...) csv_getter_ret_t<T> get_##name() const noexcept { return name; }
    CSV_FIELDS(GETTER)
    #undef GETTER

    #define SETTER(T, name, label, ...) \
    void set_##name(csv_setter_param_t<T> value) { this->name = value; } \
    void set_##name(T&& value) noexcept { this->name = std::move(value); }
    CSV_FIELDS(SETTER)
    #undef SETTER

    static constexpr std::string_view header() { return header_sv; }

    static std::string cell(const hope::Timestamp& tp)
    {
        auto ms = std::chrono::time_point_cast<std::chrono::milliseconds>(tp).time_since_epoch().count();
        return std::to_string(ms);
    }

    static std::string cell(const std::vector<double>& v)
    {
        std::ostringstream ss;
        ss << '[';
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) ss << '+';
            ss << v[i];
        }
        ss << ']';
        return ss.str();
    }

    static std::string cell(const std::string& s)
    {
        return s;
    }

    template <class T>
    static std::string cell(const T& v)
    {
        std::ostringstream ss;
        ss << v;
        return ss.str();
    }

    std::string create_csv_entry() const
    {
        std::ostringstream row;
        bool first = true;
        #define S(T, name, label, ...) do { if (!first) row << ','; row << cell(this->name); first = false; } while(0);
        CSV_FIELDS(S)
        #undef S
        return row.str();
    }
};
