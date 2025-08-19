#pragma once
#include <chrono>

namespace hope 
{
    using Clock = std::chrono::system_clock;
    using Timestamp = Clock::time_point;
}