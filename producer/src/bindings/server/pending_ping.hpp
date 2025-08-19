#include <string>

#include "time_types.hpp"

struct PendingPing
{
    hope::Timestamp  send_time{};
    std::size_t      round{};
    std::size_t      batch_id{};
    std::size_t      message_size{};
    std::size_t      connections_size{};
    std::string      error;
};
