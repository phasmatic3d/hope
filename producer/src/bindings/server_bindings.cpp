#include <set>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#ifndef ASIO_STANDALONE
#define ASIO_STANDALONE
#endif

namespace nb = nanobind;
using websocketpp::connection_hdl;


class ProducerServer {
public:
    ProducerServer(int port)
    : m_port(port)
    {
        m_logger = spdlog::stdout_color_mt("broadcaster");
        m_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v");
        m_logger->set_level(spdlog::level::debug);
        m_logger->flush_on(spdlog::level::info);
        m_logger->info("Initialized Broadcaster");

        m_server.init_asio();
        
        // disable logging
        m_server.clear_access_channels(
        websocketpp::log::alevel::frame_header   
        | websocketpp::log::alevel::frame_payload  
        );

        // On new WS connection
        m_server.set_open_handler([this](connection_hdl h){
            std::lock_guard<std::mutex> lock(m_connection_mutex);
            m_connections.insert(h);
            std::cout << "Client connected\n";
        });

        // On WS close
        m_server.set_close_handler([this](connection_hdl h){
            std::lock_guard<std::mutex> lock(m_connection_mutex);
            m_connections.erase(h);
            std::cout << "Client disconnected\n";
        });

        // HTTP redirect  client
        m_server.set_http_handler([this](connection_hdl hdl){
            auto con = m_server.get_con_from_hdl(hdl);
            con->set_status(websocketpp::http::status_code::found);
            con->replace_header("Location", m_redirect_url);
        });

    }

    // Start listening on the given port
    void listen() {
        m_server.listen(m_port);
        m_server.start_accept();
    }

    // Blocking run (typically spawn in a thread)
    void run() {
        m_server.run();
    }

    // Broadcast a binary message to all clients
    void broadcast(const nb::bytes &data) {
        std::lock_guard<std::mutex> lock(m_connection_mutex);
        for (auto &hdl : m_connections) {
            websocketpp::lib::error_code ec;
            m_server.send(hdl,
                          (const void*)data.data(),
                          data.size(),
                          websocketpp::frame::opcode::binary,
                          ec);
            if (ec) {
                throw std::runtime_error("Send failed: " + ec.message());
            }
        }
    }


    void set_redirect(const std::string &url) {
        // Trim leading / trailing whitespace
        //std::string tmp = url.cast<std::string>();
        std::string tmp = url;
        auto ws_front = tmp.find_first_not_of(" \t\r\n");
        auto ws_back  = tmp.find_last_not_of (" \t\r\n");
        if (ws_front == std::string::npos) {
            throw std::invalid_argument("Redirect URL cannot be empty or whitespace");
        }
        tmp = tmp.substr(ws_front, ws_back - ws_front + 1);

        // Strip a single trailing slash for parsing
        bool had_slash = (!tmp.empty() && tmp.back() == '/');
        if (had_slash) tmp.pop_back();

        // Lower-case prefix to check for scheme
        std::string lower = tmp;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c){ return std::tolower(c); });

        if (lower.rfind("http://", 0) == 0 || lower.rfind("https://", 0) == 0) {
            // tmp has no trailing slash now
            const auto scheme_end = tmp.find("://") + 3;
            const auto path_pos   = tmp.find('/', scheme_end);
        
            // authority = host[:port]
            std::string authority = tmp.substr(
                scheme_end,
                (path_pos == std::string::npos ? tmp.size() : path_pos) - scheme_end
            );
            // rest = “/…” or empty
            std::string rest = (path_pos == std::string::npos
                                ? std::string{}
                                : tmp.substr(path_pos));
            
            // if no “:port” in authority, append default
            if (authority.find(':') == std::string::npos) {
                authority += ":" + std::to_string(m_port);
            }
        
            // rebuild (and re-append the slash we stripped earlier)
            m_redirect_url = tmp.substr(0, scheme_end)
                           + authority
                           + rest
                           + '/';
        
            m_logger->info("Constructed URL: {}", m_redirect_url);
            return;
        }

        // Otherwise treat as host[:port]
        std::string host;
        int port = m_port;  // default to constructor port
        auto colon = tmp.find(':');
        if (colon != std::string::npos) {
            host = tmp.substr(0, colon);
            std::string port_str = tmp.substr(colon + 1);
            try {
                port = std::stoi(port_str);
            } catch (...) {
                m_logger->error("Invalid port in redirect URL: “" + port_str + "”");
                throw std::invalid_argument("Invalid port in redirect URL: “" + port_str + "”");
            }
        } else {
            host = tmp;
        }

        // Build final URL
        m_redirect_url = "http://" + host + ":" + std::to_string(port) + "/";
        m_logger->info("Constructed URL: {}", m_redirect_url);
    }

private:
    int m_port;
    std::shared_ptr<spdlog::logger> m_logger;
    websocketpp::server<websocketpp::config::asio> m_server;
    std::set<connection_hdl, std::owner_less<connection_hdl>> m_connections;
    std::mutex m_connection_mutex;
    std::string m_redirect_url = "/";
};

NB_MODULE(broadcaster, m) {
    m.doc() = "WebSocket++ Producer server binding";

    nb::class_<ProducerServer>(m, "ProducerServer")
        .def(nb::init<int>(), nb::arg("port"))
        .def("listen", &ProducerServer::listen,
         nb::call_guard<nb::gil_scoped_release>(),
         "Begin listening on the configured port")
        .def("run", &ProducerServer::run,
         nb::call_guard<nb::gil_scoped_release>(),
         "Enter the ASIO event loop (blocking)")
        .def("broadcast", &ProducerServer::broadcast,
             nb::arg("data"),
             "Broadcast raw binary data to all connected WebSocket clients")
        .def
        (
            "set_redirect", 
            [](ProducerServer &s, nb::str url_obj) {
                // manually pull UTF-8 chars out of the Python str
                auto py_ptr = (PyObject *) url_obj.ptr();
                const char *cstr = PyUnicode_AsUTF8(py_ptr);
                if (!cstr)
                    throw std::runtime_error("Failed to convert redirect URL to UTF-8");
                s.set_redirect(std::string(cstr));
            },
            nb::arg("url"),
            "Set HTTP redirect target for incoming browser requests"
        );
}
