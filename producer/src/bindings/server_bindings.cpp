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
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>


#ifndef ASIO_STANDALONE
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
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>


#ifndef ASIO_STANDALONE
#define ASIO_STANDALONE
#endif


#define USE_TLS
#ifdef USE_TLS
    #include <websocketpp/config/asio.hpp>    // TLS‐enabled config
    typedef websocketpp::config::asio_tls asio_config;
#else
    #include <websocketpp/config/asio_no_tls.hpp>
    typedef websocketpp::config::asio asio_config;
#endif
#include <websocketpp/server.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <nlohmann/json.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <nlohmann/json.hpp>

namespace nb = nanobind;
using websocketpp::connection_hdl;

class ProducerServer
{
public:
    // --- Members for metric calculation ---
    using Clock = std::chrono::system_clock;
    using Timestamp = Clock::time_point;

    struct CsvFileEntry
    {
        Timestamp   timestamp;
        double      approximate_rtt_ms;
        double      one_way_ms;
        double      one_way_plus_processing;
        std::string connection_id;
        std::string error;
        size_t      broadcast_round;
        size_t      message_size;
        size_t      connections_size;
    };

    ProducerServer
    (
        int port,
        bool write_to_csv,
        bool use_pings_for_rtt
    ):
    m_port(port),
    stop_logging(false),
    write_to_csv(write_to_csv),
    use_pings_for_rtt(use_pings_for_rtt)
    {
        m_logger = spdlog::stdout_color_mt("broadcaster");
        m_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v");
        m_logger->set_level(spdlog::level::debug);
        m_logger->flush_on(spdlog::level::debug);
        m_logger->info("Initialized Broadcaster");


        m_logger->info("Initialized port: {}", m_port);
        m_logger->info("Write to CSV: {}", write_to_csv);
        m_logger->info("Use Pings for RTT: {}", use_pings_for_rtt);

        if (write_to_csv) startLoggingThread();

        m_server.init_asio();

        #ifdef USE_TLS
		m_server.set_tls_init_handler
        (
            [this](websocketpp::connection_hdl h) -> websocketpp::lib::shared_ptr<websocketpp::lib::asio::ssl::context>
            {
                namespace asio = websocketpp::lib::asio;
                websocketpp::lib::shared_ptr<websocketpp::lib::asio::ssl::context> ctx = std::make_shared<asio::ssl::context>(asio::ssl::context::sslv23);

                ctx->set_options
                (
                    asio::ssl::context::default_workarounds |
                    asio::ssl::context::no_sslv2 |
                    asio::ssl::context::no_sslv3 |
                    asio::ssl::context::no_tlsv1 |
                    asio::ssl::context::single_dh_use
                );
                //TODO how to generate keys and use them
                ctx->use_certificate_chain_file("./broadcaster_wrapper/cert/server.crt");
                ctx->use_private_key_file("./broadcaster_wrapper/cert/server.key", asio::ssl::context::pem);

                return ctx;
            }
        );
        #endif


        #ifdef USE_TLS
		m_server.set_tls_init_handler
        (
            [this](websocketpp::connection_hdl h) -> websocketpp::lib::shared_ptr<websocketpp::lib::asio::ssl::context>
            {
                namespace asio = websocketpp::lib::asio;
                websocketpp::lib::shared_ptr<websocketpp::lib::asio::ssl::context> ctx = std::make_shared<asio::ssl::context>(asio::ssl::context::sslv23);

                ctx->set_options
                (
                    asio::ssl::context::default_workarounds |
                    asio::ssl::context::no_sslv2 |
                    asio::ssl::context::no_sslv3 |
                    asio::ssl::context::no_tlsv1 |
                    asio::ssl::context::single_dh_use
                );
                //TODO how to generate keys and use them
                ctx->use_certificate_chain_file("./broadcaster_wrapper/cert/server.crt");
                ctx->use_private_key_file("./broadcaster_wrapper/cert/server.key", asio::ssl::context::pem);

                return ctx;
            }
        );
        #endif

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
        m_server.set_close_handler([this](connection_hdl h){
            std::lock_guard<std::mutex> lock(m_connection_mutex);
            m_connections.erase(h);
            std::cout << "Client disconnected\n";
        });

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
                write_to_csv = false;
            }
            log_cv.notify_one();
            entry_cv.notify_all();
            if (log_thread.joinable())
                log_thread.join();
        }
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
                write_to_csv = false;
            }
            log_cv.notify_one();
            entry_cv.notify_all();
            if (log_thread.joinable())
                log_thread.join();
        }
    }

    // Start listening on the given port
    void listen()
    {
    void listen()
    {
        m_server.listen(m_port);
        m_server.start_accept();
    }

    // Blocking run (typically spawn in a thread)
    void run()
    {
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
        if (write_to_csv)
        {
            // Signal logging thread to exit, then join it
            {
                std::lock_guard<std::mutex> lg(log_mutex);
                stop_logging = true;
                write_to_csv = false;
            }
            log_cv.notify_one();
            entry_cv.notify_all();
            if (log_thread.joinable())
                log_thread.join();
        }
        m_logger->info("Server stopped");
    }


    void stop()
    {
        // Stop accepting any more incoming connections
        m_server.stop_listening();
        // Interrupt the ASIO loop so run() will return
        m_server.stop();
        if (write_to_csv)
        {
            // Signal logging thread to exit, then join it
            {
                std::lock_guard<std::mutex> lg(log_mutex);
                stop_logging = true;
                write_to_csv = false;
            }
            log_cv.notify_one();
            entry_cv.notify_all();
            if (log_thread.joinable())
                log_thread.join();
        }
        m_logger->info("Server stopped");
    }


    // Broadcast a binary message to all clients
    void broadcast(const nb::bytes &data) {
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

        std::map<MetaKey, PendingMetadata, MetaKeyCompare> map_to_metadata; 

        for (auto &hdl : m_connections)
        {

            websocketpp::lib::error_code ec;
            m_server.send(hdl,
                          (const void*)data.data(),
                          data.size(),
                          websocketpp::frame::opcode::binary,
                          ec);
            if (ec) {
                throw std::runtime_error("Send failed: " + ec.message());
            }
        );

        // accept http, https, ws, or wss
        if (lower.rfind("http://",0) == 0 ||
            lower.rfind("https://",0) == 0 ||
            lower.rfind("ws://",0)   == 0 ||
            lower.rfind("wss://",0)  == 0)
        {
            const auto scheme_end = tmp.find("://") + 3;
            const auto path_pos   = tmp.find('/', scheme_end);

            // grab “host[:port]”
            std::string authority = tmp.substr(
                scheme_end,
                (path_pos == std::string::npos ? tmp.size() : path_pos) - scheme_end
            );

            std::string rest = (path_pos == std::string::npos
                                ? std::string{}
                                : tmp.substr(path_pos));

            // if no “:port” add the server’s port
            if (authority.find(':') == std::string::npos) {
                authority += ":" + std::to_string(m_port);
            }

            // rebuild and re‐append the trailing slash
            m_redirect_url = tmp.substr(0, scheme_end)
                           + authority
                           + rest
                           + "/";
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
    websocketpp::server<asio_config> m_server;
    std::set<connection_hdl, std::owner_less<connection_hdl>> m_connections;
    std::mutex m_connection_mutex;
    std::string m_redirect_url = "/";
};

NB_MODULE(broadcaster, m)
{
    m.doc() = "WebSocket++ Producer server binding";

    nb::class_<ProducerServer::CsvFileEntry>(m, "CsvFileEntry")
    // timestamp_ms as a read-only property
    .def_prop_ro("timestamp_ms",
        [](ProducerServer::CsvFileEntry const &e) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                       e.timestamp.time_since_epoch())
                   .count();
        })
    // now all the other members as read-only props:
    .def_prop_ro("approximate_rtt_ms",
        [](ProducerServer::CsvFileEntry const &e) {
            return e.approximate_rtt_ms;
        })
    .def_prop_ro("one_way_ms",
        [](ProducerServer::CsvFileEntry const &e) {
            return e.one_way_ms;
        })
    .def_prop_ro("one_way_plus_processing",
        [](ProducerServer::CsvFileEntry const &e) {
            return e.one_way_plus_processing;
        })
    .def_prop_ro("connection_id",
        [](ProducerServer::CsvFileEntry const &e) {
            return e.connection_id;
        })
    .def_prop_ro("error",
        [](ProducerServer::CsvFileEntry const &e) {
            return e.error;
        })
    .def_prop_ro("broadcast_round",
        [](ProducerServer::CsvFileEntry const &e) {
            return e.broadcast_round;
        })
    .def_prop_ro("message_size",
        [](ProducerServer::CsvFileEntry const &e) {
            return e.message_size;
        })
    .def_prop_ro("connections_size",
        [](ProducerServer::CsvFileEntry const &e) {
            return e.connections_size;
        });

    nb::class_<ProducerServer>(m, "ProducerServer")
        .def(nb::init<int, bool, bool>(),
            nb::arg("port"),
            nb::arg("write_to_csv"),
            nb::arg("use_pings_for_rtt"))
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
        .def("get_entry_for_round",
            &ProducerServer::get_entry_for_round,
            nb::arg("broadcast_round"))
        .def("wait_for_entry",
            &ProducerServer::wait_for_entry,
            nb::arg("broadcast_round"),
            nb::call_guard<nb::gil_scoped_release>())
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