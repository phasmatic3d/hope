#define ASIO_STANDALONE
#define _WEBSOCKETPP_CPP11_TYPE_TRAITS_
//#define ASIO_NO_DEPRECATED

#include <websocketpp/config/asio.hpp>
//#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

#include <draco/compression/encode.h>
#include <draco/point_cloud/point_cloud.h>
#include <draco/point_cloud/point_cloud_builder.h>
#include <draco/core/encoder_buffer.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp> // glm::value_ptr

#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <random>
#include <thread>
#include <set>
#include <mutex>

typedef websocketpp::server<websocketpp::config::asio> server;

using websocketpp::connection_hdl;
using namespace std::chrono_literals;

class point_cloud {
public:
	point_cloud(int num_of_points = 1000) : m_num_of_points(num_of_points){

	}

	void build() {
		m_point_data.resize(m_num_of_points);

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dist(-5.0f, 5.0f); // Range for x, y, z

		for (int i = 0; i < m_num_of_points; ++i) {
			glm::vec3 point(dist(gen), dist(gen), dist(gen));
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
		encoder.SetSpeedOptions(5, 5); // balance between speed and compression
		encoder.SetAttributeQuantization(draco::GeometryAttribute::POSITION, 14);

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

point_cloud pc(10000);

class time_server {
public:
	time_server() {
		m_server.init_asio();

		m_server.set_open_handler([this](connection_hdl hdl) {
			std::lock_guard<std::mutex> lock(m_mutex);
			m_connections.insert(hdl);
			std::cout << "Client connected\n";
			});

		m_server.set_close_handler([this](connection_hdl hdl) {
			std::lock_guard<std::mutex> lock(m_mutex);
			m_connections.erase(hdl);
			std::cout << "Client disconnected\n";
			});
	}

	void run(uint16_t port) {
		m_server.listen(port);
		m_server.start_accept();

		std::thread broadcaster([this]() {
			while (true) {
				send_time();
				//std::this_thread::sleep_for(1s);
			}
			});

		m_server.run();
		broadcaster.join(); // optional, not reached unless server exits
	}

private:
	void send_time() {
		auto now = std::chrono::system_clock::now();
		std::time_t t_now = std::chrono::system_clock::to_time_t(now);
		std::string time_str = std::ctime(&t_now);
		time_str.pop_back(); // remove trailing newline

		pc.build();
		const auto& buffer = pc.GetBuffer();
		std::lock_guard<std::mutex> lock(m_mutex);
		for (auto& hdl : m_connections) {
			websocketpp::lib::error_code ec;
			//m_server.send(hdl, time_str, websocketpp::frame::opcode::text, ec);
			m_server.send(hdl, buffer.data(), buffer.size(), websocketpp::frame::opcode::binary, ec);
			if (ec) {
				//std::cerr << "Send failed: " << ec.message() << std::endl;
			}
		}
	}

	server m_server;
	std::set<connection_hdl, std::owner_less<connection_hdl>> m_connections;
	std::mutex m_mutex;
};

int main() {
	try {
		time_server server;
		server.run(9002);
	}
	catch (const std::exception& e) {
		std::cerr << "Server failed: " << e.what() << std::endl;
	}
}