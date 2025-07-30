#include <set>
#include <string>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <sstream>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <vector>
#include <map>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <nlohmann/json.hpp>

#ifndef ASIO_STANDALONE
#define ASIO_STANDALONE
#endif

namespace nb = nanobind;
using websocketpp::connection_hdl;

class ProducerServer
{
public:
    ProducerServer(
        int port,
        bool write_to_csv = false)
        : m_port(port),
          stop_logging(false),
          write_to_csv(write_to_csv)
    {
        m_logger = spdlog::stdout_color_mt("broadcaster");
        m_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v");
        m_logger->set_level(spdlog::level::debug);
        m_logger->flush_on(spdlog::level::debug);
        m_logger->info("Initialized Broadcaster");

        if (write_to_csv) startLoggingThread();

        m_server.init_asio();

        // disable logging
        m_server.clear_access_channels
        (
            websocketpp::log::alevel::frame_header   |
            websocketpp::log::alevel::frame_payload  |
            websocketpp::log::alevel::control       // ← disable control-frame logs
        );

        // On new WS connection
        m_server.set_open_handler
        (
            [this](connection_hdl h)
            {
                std::lock_guard<std::mutex> lock(m_connection_mutex);
                m_connections.insert(h);
                m_logger->info("Client connected");
            }
        );

        // On WS close
        m_server.set_close_handler
        (
            [this](connection_hdl h)
            {
                std::lock_guard<std::mutex> lock(m_connection_mutex);
                m_connections.erase(h);
                m_logger->info("Client disconnected");
            }
        );

        m_server.set_pong_handler
        (
            [this](connection_hdl hdl, std::string _) 
            {
                PendingPing meta;
                {
                    std::lock_guard<std::mutex> lg(ping_mutex);
                    auto it = pending_pings.find(hdl);
                    if (it == pending_pings.end()) return;
                    meta = it->second;
                    pending_pings.erase(it);
                }

                std::ostringstream ss;
                ss << hdl.lock().get();

                auto t1 = Clock::now();
                double rtt = std::chrono::duration<double,std::milli>(t1 - meta.t0).count();
                CsvFileEntry entry 
                {
                    meta.t0,
                    rtt,
                    ss.str(),
                    meta.ping_error,
                    meta.round,
                    meta.message_size,
                    meta.connections_size
                };
                enqueueLogEntry(entry);
            }
        );


        m_server.set_message_handler
        (
            [this](connection_hdl hdl,  websocketpp::server<websocketpp::config::asio>::message_ptr message)
            {

                if(!this->write_to_csv) return;

                try
                {
                    auto parsed_json = nlohmann::json::parse(message->get_payload());
                    m_logger->debug("Received message: {}", parsed_json.dump());

                    if (parsed_json.value("type","") != "received-timestamp")
                    {
                        m_logger->error("Message doesn't contain timestamp field...");
                        return;
                    }

                    uint64_t client_ms = parsed_json.at("timestamp").get<uint64_t>();
                    auto client_timestamp = Timestamp
                    (
                        std::chrono::milliseconds(client_ms)
                    );

                    PendingMetadata metadata; 
                    {
                        std::lock_guard<std::mutex> lg(m_metadata_mutex);
                        auto it = m_metadata.find(hdl);
                        if (it == m_metadata.end()) return;
                        metadata = it->second;
                        m_metadata.erase(it);
                    }
                    auto t1 = Clock::now();
                    double rtt = std::chrono::duration<double,std::milli>(client_timestamp - metadata.send_time).count();
                    std::ostringstream ss;
                    ss << hdl.lock().get();
                    CsvFileEntry entry 
                    {
                        metadata.send_time,
                        rtt,
                        ss.str(),
                        metadata.error,
                        metadata.round,
                        metadata.message_size,
                        metadata.connections_size
                    };
                    enqueueLogEntry(entry);
                }
                catch(const std::exception& e)
                {
                    m_logger->error("Bad JSON on timestamp message: {}", e.what());
                }

            }
        );

        // HTTP redirect  client
        m_server.set_http_handler
        (
            [this](connection_hdl hdl)
            {
                auto con = m_server.get_con_from_hdl(hdl);
                con->set_status(websocketpp::http::status_code::found);
                con->replace_header("Location", m_redirect_url);
                m_logger->info("Redirecting HTTP client to {}", m_redirect_url);
            }
        );
    }

    ~ProducerServer()
    {
        if (write_to_csv)
        {
            // Signal logging thread to exit, then join it
            {
                std::lock_guard<std::mutex> lg(log_mutex);
                stop_logging = true;
            }
            log_cv.notify_one();
            if (log_thread.joinable())
                log_thread.join();
        }
    }

    // Start listening on the given port
    void listen()
    {
        m_server.listen(m_port);
        m_server.start_accept();
    }

    // Blocking run (typically spawn in a thread)
    void run()
    {
        m_server.run();
    }

    void stop()
    {
        // Stop accepting any more incoming connections
        m_server.stop_listening();
        // Interrupt the ASIO loop so run() will return
        m_server.stop();
        m_logger->info("Server stopped");
    }


    // Broadcast a binary message to all clients
    void broadcast(const nb::bytes &data)
    {
        // m_logger->debug("Broadcasting {} bytes", data.size());

        std::lock_guard<std::mutex> lock(m_connection_mutex);

        size_t broadcast_round  = 0;
        size_t message_size     = 0;
        size_t connections_size = 0;
        //TODO this assumes the bytes are received by all connections with no errors
        size_t bytes_sent       = 0;

        if (!m_connections.empty())
        {
            broadcast_round  = m_broadcast_counter++;
            message_size     = data.size();
            connections_size = m_connections.size();
            bytes_sent       = message_size * connections_size;
        }

        std::map<connection_hdl, PendingMetadata, std::owner_less<connection_hdl>> map_to_metadata; 

        for (auto &hdl : m_connections)
        {

            websocketpp::lib::error_code ec;
            m_server.send
            (
                hdl,
                (const void *)data.data(),
                data.size(),
                websocketpp::frame::opcode::binary,
                ec
            );

            if (ec)
            {
                m_logger->error("Send Failed: {}", ec.message());
            }

            if(!write_to_csv) continue;

            map_to_metadata[hdl] = PendingMetadata
            {
                Clock::now(),
                broadcast_round,
                message_size,  
                connections_size, 
                bytes_sent,
                ec ? ec.message() : "no error"
            };
        }

        if(!write_to_csv) return;


        {
            std::lock_guard<std::mutex> lg(m_metadata_mutex);
            for(auto& entry: map_to_metadata) 
            {
                m_metadata[entry.first] = entry.second;
            }
        }
        //if(!m_connections.empty() && write_to_csv)
        //{
        //    ping_async(broadcast_round, message_size, connections_size);
        //}
    }

    void set_redirect(const std::string &url)
    {
        // Trim leading / trailing whitespace
        // std::string tmp = url.cast<std::string>();
        std::string tmp = url;
        auto ws_front = tmp.find_first_not_of(" \t\r\n");
        auto ws_back = tmp.find_last_not_of(" \t\r\n");
        if (ws_front == std::string::npos)
        {
            throw std::invalid_argument("Redirect URL cannot be empty or whitespace");
        }
        tmp = tmp.substr(ws_front, ws_back - ws_front + 1);

        // Strip a single trailing slash for parsing
        bool had_slash = (!tmp.empty() && tmp.back() == '/');
        if (had_slash)
            tmp.pop_back();

        // Lower-case prefix to check for scheme
        std::string lower = tmp;
        std::transform
        (
            lower.begin(),
            lower.end(),
            lower.begin(),
            [](unsigned char c)
            {
                return std::tolower(c);
            }
        );

        if (lower.rfind("http://", 0) == 0 || lower.rfind("https://", 0) == 0)
        {
            // tmp has no trailing slash now
            const auto scheme_end = tmp.find("://") + 3;
            const auto path_pos = tmp.find('/', scheme_end);

            // authority = host[:port]
            std::string authority = tmp.substr(
                scheme_end,
                (path_pos == std::string::npos ? tmp.size() : path_pos) - scheme_end);
            // rest = “/…” or empty
            std::string rest =
                (path_pos == std::string::npos
                     ? std::string{}
                     : tmp.substr(path_pos));

            // if no “:port” in authority, append default
            if (authority.find(':') == std::string::npos)
            {
                authority += ":" + std::to_string(m_port);
            }

            // rebuild (and re-append the slash we stripped earlier)
            m_redirect_url = tmp.substr(0, scheme_end) + authority + rest + '/';

            m_logger->info("Constructed URL: {}", m_redirect_url);
            return;
        }

        // Otherwise treat as host[:port]
        std::string host;
        int port = m_port; // default to constructor port
        auto colon = tmp.find(':');
        if (colon != std::string::npos)
        {
            host = tmp.substr(0, colon);
            std::string port_str = tmp.substr(colon + 1);
            try
            {
                port = std::stoi(port_str);
            }
            catch (...)
            {
                m_logger->error("Invalid port in redirect URL: “" + port_str + "”");
                throw std::invalid_argument("Invalid port in redirect URL: “" + port_str + "”");
            }
        }
        else
        {
            host = tmp;
        }

        // Build final URL
        m_redirect_url = "http://" + host + ":" + std::to_string(port) + "/";
        m_logger->info("Constructed URL: {}", m_redirect_url);
    }

private:
    size_t m_port;
    size_t m_broadcast_counter = 0;
    std::shared_ptr<spdlog::logger> m_logger;
    websocketpp::server<websocketpp::config::asio> m_server;
    std::set<connection_hdl, std::owner_less<connection_hdl>> m_connections;
    std::mutex m_connection_mutex;
    std::string m_redirect_url = "/";


    // --- Members for metric calculation ---
    using Clock = std::chrono::system_clock;
    using Timestamp = Clock::time_point;

    // --- Retrieve the data rate for the last broadcast round ---
    Clock::time_point   m_last_broadcast_time = Clock::now();
    size_t              m_last_bytes_sent     = 0;

    struct PendingPing 
    {
        Timestamp          t0;
        size_t             round;
        size_t             message_size;
        size_t             connections_size;
        std::string        ping_error;
    };

    // keep per-connection send metadata
    struct PendingMetadata
    {
        Timestamp send_time;
        size_t round;
        size_t message_size;
        size_t connections_size;
        size_t bytes_sent;
        std::string error;
    };

    struct CsvFileEntry
    {
        Timestamp timestamp;
        double approximate_rtt_ms;
        std::string connection_id;
        std::string error;
        size_t broadcast_round;
        size_t message_size;
        size_t connections_size;
    };

    bool write_to_csv;
    std::ofstream csv;
    std::mutex log_mutex;
    std::condition_variable log_cv;
    std::queue<CsvFileEntry> log_queue;
    std::thread log_thread;
    bool stop_logging;

    std::mutex ping_mutex;
    std::map<connection_hdl, PendingPing, std::owner_less<connection_hdl>> pending_pings;

    std::mutex m_metadata_mutex;
    std::map<connection_hdl, PendingMetadata, std::owner_less<connection_hdl>> m_metadata;

    static constexpr const char *HEADER = "timestamp_ms_since_epoch,approximate_rtt_ms,connection_id,error,broadcast_round,message_size,connections_size";

    void startLoggingThread()
    {
        // open CSV file and write header
        const std::string path = "./broadcaster_wrapper/send_times.csv";

        bool need_header = true;
        {
            std::ifstream in(path);
            if (in.is_open())
            {
                std::string first;
                if (std::getline(in, first) && first == HEADER)
                {
                    need_header = false;
                }
            }
            in.close();
        }

        csv.open(path, std::ios::out | std::ios::app);

        if (!csv.is_open())
        {
            m_logger->error("Failed to open log CSV at {}", path);
            throw std::runtime_error("Failed to open log CSV at " + path);
        }

        if (need_header)
        {
            m_logger->info("Writing header {} to csv", HEADER);
            csv << HEADER << "\n";
        }

        // spawn background thread
        log_thread = std::thread([this]{
            std::unique_lock<std::mutex> lk(log_mutex);
            while (true) 
            {
                // wait until there is data or we should stop
                log_cv.wait
                (
                    lk, [&]
                    {
                        return !log_queue.empty() || stop_logging;
                    }
                );

                // flush all entries
                while (!log_queue.empty()) 
                {
                    auto entry = std::move(log_queue.front());
                    log_queue.pop();

                    // convert timestamp to milliseconds since epoch
                    auto epoch = std::chrono::time_point_cast<std::chrono::milliseconds>(entry.timestamp).time_since_epoch().count();

                    csv 
                        << epoch << "," 
                        << entry.approximate_rtt_ms << "," 
                        << entry.connection_id << "," 
                        << entry.error << ","
                        << entry.broadcast_round << "," 
                        << entry.message_size << "," 
                        << entry.connections_size << "\n";
                }

                csv.flush();

                if (stop_logging) break;
            } });
    }

    void enqueueLogEntry(const CsvFileEntry &entry)
    {
        std::lock_guard<std::mutex> lg(log_mutex);
        log_queue.push(entry);
        log_cv.notify_one();
    }

    void ping_async
    (
        size_t broadcast_round,
        size_t message_size, 
        size_t connections_size
    )
    {

        for(auto& hdl: m_connections)
        {
            websocketpp::lib::error_code ec;
            m_server.ping(hdl, "", ec);
            {
                std::lock_guard<std::mutex> lg(ping_mutex);
                pending_pings[hdl] = PendingPing
                {
                    Clock::now(),
                    broadcast_round,
                    message_size,
                    connections_size,
                    ec ? ec.message() : "no ping error"
                };
            };
        }

    }
};

NB_MODULE(broadcaster, m)
{
    m.doc() = "WebSocket++ Producer server binding";

    nb::class_<ProducerServer>(m, "ProducerServer")
        .def(nb::init<int, bool>(),
             nb::arg("port"),
             nb::arg("write_to_csv"))
        .def("listen", &ProducerServer::listen,
             nb::call_guard<nb::gil_scoped_release>(),
             "Begin listening on the configured port")
        .def("run", &ProducerServer::run,
             nb::call_guard<nb::gil_scoped_release>(),
             "Enter the ASIO event loop (blocking)")
        .def("stop", &ProducerServer::stop,
             "Stop accepting and shut down the server cleanly")
        .def("broadcast", &ProducerServer::broadcast,
             nb::arg("data"),
             "Broadcast raw binary data to all connected WebSocket clients")
        .def(
            "set_redirect",
            [](ProducerServer &s, nb::str url_obj)
            {
                // manually pull UTF-8 chars out of the Python str
                auto py_ptr = (PyObject *)url_obj.ptr();
                const char *cstr = PyUnicode_AsUTF8(py_ptr);
                if (!cstr)
                    throw std::runtime_error("Failed to convert redirect URL to UTF-8");
                s.set_redirect(std::string(cstr));
            },
            nb::arg("url"),
            "Set HTTP redirect target for incoming browser requests");
}
