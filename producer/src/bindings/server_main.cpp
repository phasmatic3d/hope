#include "server_bindings_no_nanobind.cpp"
int main() {
    ProducerServer srv(5555, /*write_csv=*/true, /*use_pings=*/true);
    srv.set_redirect("https://localhost");
    srv.listen();
    srv.run();
    return 0;
}