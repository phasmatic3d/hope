#include <string>

#include <nlohmann/json.hpp>

#define BROADCAST_INFO_FIELDS(X) \
  X(std::string, type,                 "type") \
  X(size_t,      broadcast_round,      "broadcast_round") \
  X(size_t,      message_size,         "message_size") \
  X(int64_t,     send_timestamp_ms,    "send_timestamp_ms")

template <class T>
using csv_getter_ret_t =
    std::conditional_t<(std::is_trivially_copyable<T>::value && sizeof(T) <= sizeof(void*)), T, const T&>;

template<class T>
using csv_setter_param_t =
    std::conditional_t<(std::is_trivially_copyable_v<T> && sizeof(T) <= sizeof(void*)), T, const T&>;

struct BroadcastInfo 
{
    #define DECLARE(T, name, label ) T name = {};
    BROADCAST_INFO_FIELDS(DECLARE)
    #undef DECLARE

    #define GETTER(T, name, label) csv_getter_ret_t<T> get_##name() const noexcept { return name; }
    BROADCAST_INFO_FIELDS(GETTER)
    #undef GETTER

    #define SETTER(T, name, label) \
    void set_##name(csv_setter_param_t<T> value) { this->name = value; } \
    void set_##name(T&& value) noexcept { this->name = std::move(value); }
    BROADCAST_INFO_FIELDS(SETTER)
    #undef SETTER


    inline std::string to_json_dump()
    {
        nlohmann::json j;
        #define TO_JSON(T, name, label) j[label] = this->name;
        BROADCAST_INFO_FIELDS(TO_JSON)
        #undef TO_JSON

        return j.dump();
    }
};