#include <vector>
#include <limits>
#include <sstream>

#include <nlohmann/json.hpp>

#include "time_types.hpp"

#define CLIENT_SUPPLIED_METRICS(X) \
  X(double,          rtt,                      "rtt",                      std::numeric_limits<double>::quiet_NaN()) \
  X(double,          one_way_ms,               "one_way_ms",               std::numeric_limits<double>::quiet_NaN()) \
  X(double,          one_way_plus_processing,  "one_way_plus_processing",  std::numeric_limits<double>::quiet_NaN()) \
  X(double,          wait_in_queue,            "wait_in_queue",            std::numeric_limits<double>::quiet_NaN()) \
  X(double,          pure_decode_ms,           "pure_decode_ms",           std::numeric_limits<double>::quiet_NaN()) \
  X(double,          pure_geometry_upload_ms,  "pure_geometry_upload_ms",  std::numeric_limits<double>::quiet_NaN()) \
  X(double,          pure_render_ms,           "pure_render_ms",           std::numeric_limits<double>::quiet_NaN()) \
  X(double,          pure_processing_ms,       "pure_processing_ms",       std::numeric_limits<double>::quiet_NaN()) \
  X(std::vector<double>, chunk_decode_times,   "chunk_decode_times",       std::vector<double>{})

// Same list minus the computed `rtt` (so parsing is generated only for JSON fields)
#define CLIENT_SUPPLIED_METRICS_PARSE_FIELDS(X) \
  X(double,               one_way_ms,               "one_way_ms",               0) \
  X(double,               one_way_plus_processing,  "one_way_plus_processing",  0) \
  X(double,               wait_in_queue,            "wait_in_queue",            0) \
  X(double,               pure_decode_ms,           "pure_decode_ms",           0) \
  X(double,               pure_geometry_upload_ms,  "pure_geometry_upload_ms",  0) \
  X(double,               pure_render_ms,           "pure_render_ms",           0) \
  X(double,               pure_processing_ms,       "pure_processing_ms",       0) \
  X(std::vector<double>,  chunk_decode_times,       "chunk_decode_times",       0)

template <class T>
inline void assign_from_json(T& dst, const nlohmann::json& j, const char* key) 
{
    dst = j.at(key).get<T>();
}

template <>
inline void assign_from_json<std::vector<double>>(std::vector<double>& dst, const nlohmann::json& j, const char* key) 
{
    if (j.contains(key) && j[key].is_array()) 
	{
        dst = j[key].get<std::vector<double>>();
    } 
	else 
	{
        dst.clear();
    }
}

inline std::string to_string_helper(const std::vector<double>& v) 
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) 
    {
        if (i) oss << ",";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

template <typename T>
inline std::string to_string_helper(const T& v) 
{
    std::ostringstream oss;
    oss << v;
    return oss.str();
}


struct ClientSuppliedMetrics
{
    #define DECL(T, name, key, def) T name = def;
    CLIENT_SUPPLIED_METRICS(DECL)
    #undef DECL

    #define GET(T, name, key, ...) const T& get_##name() const noexcept { return name; }
    CLIENT_SUPPLIED_METRICS(GET)
    #undef GET

    #define SET(T, name, key, ...) \
    void set_##name(const T& v) { name = v; } \
    void set_##name(T&& v) noexcept { name = std::move(v); }
    CLIENT_SUPPLIED_METRICS(SET)
    #undef SET

    static ClientSuppliedMetrics from_json_and_times(const nlohmann::json& j, const hope::Timestamp& client_ts, const hope::Timestamp& send_ts)
    {
        ClientSuppliedMetrics metrics;
        // computed field:
        metrics.rtt = std::chrono::duration<double, std::milli>(client_ts - send_ts).count();

        // auto-generated parse for all JSON-backed fields:
        #define PARSE(T, name, key, ...) assign_from_json<T>(metrics.name, j, key);
    	CLIENT_SUPPLIED_METRICS_PARSE_FIELDS(PARSE)
        #undef PARSE

        return metrics;
    }

    std::string to_string() const 
    {
        std::ostringstream oss;
        oss << "ClientSuppliedMetrics{";

        // expand over all fields
        #define PRINT(T, name, key, def) \
            oss << #name << "=" << ::to_string_helper(name) << ", ";
        CLIENT_SUPPLIED_METRICS(PRINT)
        #undef PRINT

        std::string s = oss.str();
        if (s.size() >= 2 && s[s.size() - 2] == ',')
            s.erase(s.end() - 2, s.end()); // strip last comma+space
        oss << "}";

        return oss.str();
    }
};