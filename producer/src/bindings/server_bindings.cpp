#include <nanobind/nanobind.h>
#include <set>
#include <string>

#define ASIO_STANDALONE
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

namespace nb = nanobind;
using websocketpp::connection_hdl;


class ProducerServer {
public:
    ProducerServer(int port)
    : m_port(port)
    {
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
        m_redirect_url = url;
    }

private:
    int m_port;
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
        .def("set_redirect", &ProducerServer::set_redirect,
             nb::arg("url"),
             "Set HTTP redirect target for incoming browser requests");
}
