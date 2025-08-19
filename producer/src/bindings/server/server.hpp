#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <nlohmann/json.hpp>

#include "csv_entry.hpp"
#include "time_types.hpp"
#include "broadcast_info.hpp"
#include "client_supplied_metrics.hpp"

#include "pending_ping.hpp"
#include "pending_metadata.hpp"
#include "client_metrics_and_csv_entry.hpp"

#ifndef ASIO_STANDALONE
#define ASIO_STANDALONE
#endif

#ifndef USE_TLS
#define USE_TLS
#endif

// Choose TLS / no-TLS config (must be visible here since we own a server<> member)
#ifdef USE_TLS
  #include <websocketpp/config/asio.hpp>
  using asio_config = websocketpp::config::asio_tls;
#else
  #include <websocketpp/config/asio_no_tls.hpp>
  using asio_config = websocketpp::config::asio;
#endif
#include <websocketpp/server.hpp>

// Forward declare spdlog::logger to keep this header light
namespace spdlog { class logger; }

class ProducerServer
{
public:
    ProducerServer
    (
        uint16_t port, 
        bool write_to_csv, 
        bool use_pings_for_rtt,
        spdlog::level::level_enum log_level = spdlog::level::info
    );
    ~ProducerServer();

    void listen();
    void run();
    void stop();
    //void broadcast(const nb::bytes& data);
    void broadcast(const void* data, std::size_t size);

    CsvFileEntry get_entry_for_round(std::size_t broadcast_round);
    std::optional<CsvFileEntry> wait_for_entry(std::size_t broadcast_round);

    void set_redirect(const std::string &url);
    void set_current_batch_id(std::size_t id);
    std::size_t connection_count() const;

private:

    websocketpp::lib::error_code _send_broadcast_info_packet(websocketpp::connection_hdl hdl, const size_t broadcast_round, const size_t message_size);
    //websocketpp::lib::error_code _send_broadcast_data(websocketpp::connection_hdl hdl, const nb::bytes& data);
    websocketpp::lib::error_code _send_broadcast_data(websocketpp::connection_hdl hdl, const void* data, std::size_t size);

    using MetaKey = std::pair<websocketpp::connection_hdl, std::size_t>;
    struct MetaKeyCompare
    {
        bool operator()(MetaKey const &a, MetaKey const &b) const
        {
            std::owner_less<websocketpp::connection_hdl> cmp;
            if (cmp(a.first, b.first)) return true;
            if (cmp(b.first, a.first)) return false;
            return a.second < b.second;
        }
    };

    void start_logging_thread();
    void enqueue_log_entry(const CsvFileEntry &entry);
    void ping_async(std::size_t broadcast_round, std::size_t message_size, std::size_t connections_size);

private:
    uint16_t     m_port               = 0;
    std::size_t m_broadcast_counter  = 0;
    std::size_t m_current_batch_id   = 0;

    std::shared_ptr<spdlog::logger> m_logger;

    websocketpp::server<asio_config> m_server;
    std::set<websocketpp::connection_hdl, std::owner_less<websocketpp::connection_hdl>> m_connections;
    std::mutex m_connection_mutex;

    std::string m_redirect_url = "/";

    hope::Clock::time_point m_last_broadcast_time = hope::Clock::now();
    std::size_t             m_last_bytes_sent     = 0;

    bool write_to_csv   = false;
    bool use_pings_for_rtt = false;
    std::ofstream csv;
    std::mutex log_mutex;
    std::map<std::size_t, CsvFileEntry> broadcast_round_to_entry;
    std::condition_variable log_cv;
    std::condition_variable entry_cv;
    std::queue<CsvFileEntry> log_queue;
    std::thread log_thread;
    bool stop_logging = false;

    std::mutex ping_mutex;
    std::map<websocketpp::connection_hdl, PendingPing, std::owner_less<websocketpp::connection_hdl>> pending_pings;

    std::mutex m_metadata_mutex;
    std::map<MetaKey, PendingMetadata, MetaKeyCompare> m_metadata;

    inline static constexpr std::string_view HEADER = CsvFileEntry::header();

    //websocket handlers
    websocketpp::lib::shared_ptr<websocketpp::lib::asio::ssl::context> tls_init_handler(websocketpp::connection_hdl hdl);
    void close_handler(websocketpp::connection_hdl hdl);
    void open_handler(websocketpp::connection_hdl hdl);
    void pong_handler(websocketpp::connection_hdl hdl, const std::string& _);
    void message_handler(websocketpp::connection_hdl hdl, websocketpp::server<websocketpp::config::asio>::message_ptr message);
    void http_handler(websocketpp::connection_hdl hdl);
};
