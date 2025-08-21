#include "server_bindings.hpp"

void bind_broadcaster(nb::module_ &m) 
{
    nb::enum_<spdlog::level::level_enum>(m, "log_level")
    .value("trace",    spdlog::level::trace)
    .value("debug",    spdlog::level::debug)
    .value("info",     spdlog::level::info)
    .value("warn",     spdlog::level::warn)
    .value("err",      spdlog::level::err)
    .value("critical", spdlog::level::critical)
    .value("off",      spdlog::level::off);

    nb::class_<ProducerServer>(m, "ProducerServer")
        .def(nb::init<uint16_t, bool, bool, spdlog::level::level_enum>(),
            nb::arg("port"),
            nb::arg("write_to_csv"),
            nb::arg("use_pings_for_rtt"),
            nb::arg("log_level") = spdlog::level::info) 
        .def("listen", &ProducerServer::listen,
            nb::call_guard<nb::gil_scoped_release>(),
            "Begin listening on the configured port")
        .def("run", &ProducerServer::run,
            nb::call_guard<nb::gil_scoped_release>(),
            "Enter the ASIO event loop (blocking)")
        .def("stop", &ProducerServer::stop,
            "Stop accepting and shut down the server cleanly")
        .def("broadcast",
        [](ProducerServer& s, nb::bytes data) {
            const void* ptr  = data.data();
            std::size_t len  = data.size();
            s.broadcast(ptr, len);
        },
        nb::arg("data"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Broadcast raw binary data to all connected WebSocket clients")
        .def("broadcast_batch",
        [](ProducerServer &s, std::size_t batch_id, nb::bytes data) -> bool {
            if (s.connection_count() == 0) return false;
            s.set_current_batch_id(batch_id);
            const void* ptr  = data.data();
            std::size_t len  = data.size();
            s.broadcast(ptr, len);
            return true;
        },
        nb::arg("batch_id"),
        nb::arg("data"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Broadcast raw binary data and tag it with a user-supplied batch_id")
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