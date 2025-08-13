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
#define ASIO_STANDALONE
#endif


//#define USE_TLS
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

namespace nb = nanobind;
using websocketpp::connection_hdl;

class ProducerServer {
public:
    ProducerServer(int port):m_port(port) {
        m_logger = spdlog::stdout_color_mt("broadcaster");
        m_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v");
        m_logger->set_level(spdlog::level::debug);
        m_logger->flush_on(spdlog::level::debug);
        m_logger->info("Initialized Broadcaster");


        m_logger->info("Initialized port: {}", m_port);

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
                // no client-cert verification
                ctx->set_verify_mode(asio::ssl::verify_none);
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
        m_server.set_close_handler
        (
            [this](connection_hdl h)
            {
                std::lock_guard<std::mutex> lock(m_connection_mutex);
                m_connections.erase(h);
                m_logger->info("Client disconnected");
            }
        );

        // HTTP redirect  client
        m_server.set_http_handler(
        [this](connection_hdl hdl) {
            auto con = m_server.get_con_from_hdl(hdl);

            auto base = std::filesystem::path("broadcaster_wrapper") / "public";
            m_logger->info("Serving files from {}", base.string());

            std::string req = con->get_resource();  // was get_request_uri()
            if (req == "/") req = "/index.html";                // default

            std::filesystem::path file = base / req.substr(1);
            if (!std::filesystem::exists(file) || std::filesystem::is_directory(file)) {
                con->set_status(websocketpp::http::status_code::not_found);
                con->replace_header("Cross-Origin-Opener-Policy",  "same-origin");
                con->replace_header("Cross-Origin-Embedder-Policy","require-corp");
                con->replace_header("Cross-Origin-Resource-Policy","same-origin");
                return;
            }

            // read the file
            std::ifstream in(file, std::ios::binary);
            std::ostringstream buf;
            buf << in.rdbuf();
            std::string body = buf.str();

            // set some basic headers
            con->set_status(websocketpp::http::status_code::ok);
            con->replace_header("Cross-Origin-Opener-Policy",   "same-origin");
            con->replace_header("Cross-Origin-Embedder-Policy", "require-corp");
            con->replace_header("Cross-Origin-Resource-Policy", "same-origin");

            // very minimal MIME-type mapping
            if (file.extension() == ".html") {
            con->replace_header("Content-Type", "text/html");
            } else if (file.extension() == ".js") {
            con->replace_header("Content-Type", "application/javascript");
            } else if (file.extension() == ".css") {
            con->replace_header("Content-Type", "text/css");
            }
            con->set_body(body);
        }
        );
    }

    ~ProducerServer(){}

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
    // TODO match broadcast round with the buffers of the importance sampling
    void broadcast(const nb::bytes &data)
    {
        std::lock_guard<std::mutex> lock(m_connection_mutex);

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
        }
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
    size_t m_port = 0;
    std::shared_ptr<spdlog::logger> m_logger;
    websocketpp::server<asio_config> m_server;
    std::set<connection_hdl, std::owner_less<connection_hdl>> m_connections;
    std::mutex m_connection_mutex;
    std::string m_redirect_url = "/";

};

NB_MODULE(broadcaster, m)
{
    m.doc() = "WebSocket++ Producer server binding";

    nb::class_<ProducerServer>(m, "ProducerServer")
        .def(nb::init<int>(),
            nb::arg("port"))
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