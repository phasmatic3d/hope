#define ASIO_STANDALONE
#define _WEBSOCKETPP_CPP11_TYPE_TRAITS_
//#define ASIO_NO_DEPRECATED
#define USE_TLS

#ifdef USE_TLS
#include <websocketpp/config/asio.hpp>
#else
#include <websocketpp/config/asio_no_tls.hpp>
#endif
#include <websocketpp/server.hpp>

#include <draco/compression/encode.h>
#include <draco/point_cloud/point_cloud.h>
#include <draco/point_cloud/point_cloud_builder.h>
#include <draco/core/encoder_buffer.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp> // glm::value_ptr

#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include <memory>
#include <set>
#ifdef USE_TLS
#include <openssl/ssl.h>
#endif

#ifdef USE_TLS
typedef websocketpp::server<websocketpp::config::asio_tls> server;
#else
typedef websocketpp::server<websocketpp::config::asio> server;
#endif

using websocketpp::connection_hdl;

using namespace std::chrono_literals;

#ifdef USE_TLS
// TLS init function
std::shared_ptr<asio::ssl::context> on_tls_init(websocketpp::connection_hdl hdl) {
	auto ctx = std::make_shared<asio::ssl::context>(asio::ssl::context::tlsv12);

	try {
		ctx->set_options(
			asio::ssl::context::default_workarounds |
			asio::ssl::context::no_sslv2 |
			asio::ssl::context::no_sslv3 |
			asio::ssl::context::single_dh_use
		);

		ctx->use_certificate_chain_file("server.crt");
		ctx->use_private_key_file("server.key", asio::ssl::context::pem);
	}
	catch (std::exception& e) {
		std::cout << "TLS init error: " << e.what() << std::endl;
	}

	return ctx;
}
#endif

std::set<connection_hdl, std::owner_less<connection_hdl>> connections;
std::mutex connection_mutex;

class point_cloud {
public:
	point_cloud(int num_of_points = 1000) : m_num_of_points(num_of_points) {

	}

	void build() {
		m_point_data.resize(m_num_of_points);

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dist(-5.0f, 5.0f); // Range for x, y, z

		for (int i = 0; i < m_num_of_points; ++i) {
			glm::vec3 point(dist(gen), dist(gen), dist(gen));
			//glm::vec3 point(float(i) / float(m_num_of_points), float(i) / float(m_num_of_points), float(i) / float(m_num_of_points));
			m_point_data[i] = point;
		}
		// Create a point cloud builder
		draco::PointCloudBuilder builder;
		builder.Start(m_num_of_points);

		// Add position attribute
		int pos_att_id = builder.AddAttribute(draco::GeometryAttribute::POSITION, 3, draco::DT_FLOAT32);
		builder.SetAttributeValuesForAllPoints(pos_att_id, m_point_data.data(), 0);

		// Build the point cloud
		std::unique_ptr<draco::PointCloud> pc = builder.Finalize(false);
		if (!pc) {
			std::cerr << "Failed to build point cloud.\n";
			return;
		}

		// Set encoding options
		draco::Encoder encoder;
		encoder.SetSpeedOptions(5, 10); // balance between speed and compression
		encoder.SetAttributeQuantization(draco::GeometryAttribute::POSITION, 10);

		// Encode the point cloud
		m_point_cloud_buffer.Clear();
		const draco::Status status = encoder.EncodePointCloudToBuffer(*pc, &m_point_cloud_buffer);

		if (!status.ok()) {
			std::cerr << "Encoding failed: " << status.error_msg_string() << std::endl;
			return;
		}
	}

	draco::EncoderBuffer& GetBuffer() {
		return m_point_cloud_buffer;
	}
private:

	draco::EncoderBuffer m_point_cloud_buffer;
	std::vector<glm::vec3> m_point_data;
	int m_num_of_points;
};

point_cloud pc(4000);

void time_broadcast_loop(server* s) {
	while (true) {
		//std::this_thread::sleep_for(1s);

		auto now = std::chrono::system_clock::now();
		std::time_t now_time = std::chrono::system_clock::to_time_t(now);
		std::string msg = std::ctime(&now_time); // includes newline

		pc.build();
		const auto& buffer = pc.GetBuffer();

		std::lock_guard<std::mutex> lock(connection_mutex);
		for (auto const& hdl : connections) {
			websocketpp::lib::error_code ec;
			//s->send(hdl, msg, websocketpp::frame::opcode::text, ec);
			s->send(hdl, buffer.data(), buffer.size(), websocketpp::frame::opcode::binary, ec);
			if (ec) {
				std::cerr << "Send error: " << ec.message() << std::endl;
			}
		}
	}
}

int main() {
	server echo_server;

	try {
		// Set TLS init handler
#ifdef USE_TLS
		echo_server.set_tls_init_handler(on_tls_init);
#endif

		// Set open handler
		echo_server.set_open_handler([](connection_hdl hdl) {
			std::lock_guard<std::mutex> lock(connection_mutex);
			connections.insert(hdl);
			std::cout << "Client connected\n";
		});
		echo_server.set_close_handler([](connection_hdl hdl) {
			std::lock_guard<std::mutex> lock(connection_mutex);
			connections.erase(hdl);
			std::cout << "Client disconnected\n";
		});

		echo_server.set_http_handler([&echo_server](websocketpp::connection_hdl hdl) {
			server::connection_ptr con = echo_server.get_con_from_hdl(hdl);

			con->set_status(websocketpp::http::status_code::found); // 302
			//con->replace_header("Location", "https://google.com"); // Redirect target
			con->replace_header("Location", "https://192.168.1.169:5173/"); // Redirect target

			//con->set_body("Hello World!");
			//con->set_status(websocketpp::http::status_code::ok);
		});

		echo_server.init_asio();
		//echo_server.set_access_channels(websocketpp::log::alevel::all);
		echo_server.clear_access_channels(websocketpp::log::alevel::all);
		//echo_server.set_error_channels(websocketpp::log::elevel::all);

		std::thread broadcaster([&echo_server]() {
			time_broadcast_loop(&echo_server);
		});

		echo_server.listen(9002);
		echo_server.start_accept();
		echo_server.run();

		broadcaster.join(); // Wait for the broadcaster thread (never exits)
	}
	catch (websocketpp::exception const& e) {
		std::cout << "WebSocket exception: " << e.what() << std::endl;
	}
	catch (std::exception const& e) {
		std::cout << "STD exception: " << e.what() << std::endl;
	}

	return 0;
}