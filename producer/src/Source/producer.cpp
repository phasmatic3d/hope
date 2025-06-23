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

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp> // Include OpenCV API

#include <draco/compression/encode.h>
#include <draco/point_cloud/point_cloud.h>
#include <draco/point_cloud/point_cloud_builder.h>
#include <draco/core/encoder_buffer.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp> // glm::value_ptr

#include "Detector.h"
#include "Structs.h"

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
std::mutex point_cloud_mutex;

class point_cloud {
public:
	point_cloud(int num_of_points = 1000) : m_num_of_points(num_of_points) {

	}

	void build(const rs2::vertex* vertices,
		const rs2::texture_coordinate* texcoords,
		const rs2::video_frame& color_frame,
		size_t count,
		DracoSettings &dracoSettings,
		EncodingStats& stats) {
		std::lock_guard<std::mutex> lock(point_cloud_mutex);

		stats = {};
		Timer t_prep, t_encode;

		m_point_data.resize(count);
		m_num_of_points = count;

		std::memcpy(m_point_data.data(), vertices, sizeof(*vertices) * count);

		// Wrap the color_frame in an OpenCV Mat for easy lookup
		cv::Mat colMat(
			color_frame.get_height(),
			color_frame.get_width(),
			CV_8UC3,
			(void*)color_frame.get_data(),
			cv::Mat::AUTO_STEP);

		// Create a point cloud builder
		draco::PointCloudBuilder builder;
		builder.Start(m_num_of_points);

		// Position attribute
		int pos_att_id = builder.AddAttribute(draco::GeometryAttribute::POSITION, 3, draco::DT_FLOAT32);

		// Color attribute
		int col_att_id = builder.AddAttribute(draco::GeometryAttribute::COLOR, 3, draco::DT_UINT8);

		// Feed each point’s position & color into Draco
		for (int i = 0; i < m_num_of_points; ++i) {		
			// Position: pack into a small float array
			float pos[3] = {
				static_cast<float>(m_point_data[i].x),
				static_cast<float>(m_point_data[i].y),
				static_cast<float>(m_point_data[i].z)
			};

			// Texture to pixel lookup
			const auto& tc = texcoords[i];
			int u = std::clamp(int(tc.u * (colMat.cols - 1)), 0, colMat.cols - 1);
			int v = std::clamp(int(tc.v * (colMat.rows - 1)), 0, colMat.rows - 1);
			auto bgr = colMat.at<cv::Vec3b>(v, u);
			uint8_t rgb[3] = { bgr[2], bgr[1], bgr[0] };

			// Position
			builder.SetAttributeValueForPoint(pos_att_id, draco::PointIndex(i), pos);

			// BGR to RGB
			builder.SetAttributeValueForPoint(col_att_id, draco::PointIndex(i), rgb);
		}
		stats.prep_ms = t_prep.elapsed_ms(); // Prep time

		// Build the point cloud
		std::unique_ptr<draco::PointCloud> pc = builder.Finalize(false);
		if (!pc) {
			std::cerr << "Failed to build point cloud.\n";
			return;
		}

		// Set encoding options (TODO: IMPORTANT)
		draco::Encoder encoder;
		dracoSettings.applyTo(encoder); // Apply settings to encoder

		// Encode the point cloud
		m_point_cloud_buffer.Clear();
		const draco::Status status = encoder.EncodePointCloudToBuffer(*pc, &m_point_cloud_buffer);

		if (!status.ok()) {
			std::cerr << "Encoding failed: " << status.error_msg_string() << std::endl;
			return;
		}


		stats.encode_ms = t_encode.elapsed_ms(); // Encoding time
		stats.bytes = m_point_cloud_buffer.size(); // Final buffer size

	}

	void build() {
		std::lock_guard<std::mutex> lock(point_cloud_mutex);

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

point_cloud pc(10000);

void time_broadcast_loop(server* s, rs2::pipeline& pipe) {
	rs2::pointcloud rpc;
	// We want the points object to be persistent so we can display the last cloud when a frame drops
	rs2::points points;

	while (true) {
		//std::this_thread::sleep_for(1s);
	
		auto now = std::chrono::system_clock::now();
		std::time_t now_time = std::chrono::system_clock::to_time_t(now);
		std::string msg = std::ctime(&now_time); // includes newline
	
		auto frames = pipe.wait_for_frames();

		auto color = frames.get_color_frame();

		// For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
		if (!color)
			color = frames.get_infrared_frame();

		// Tell pointcloud object to map to this color frame
		rpc.map_to(color);

		auto depth = frames.get_depth_frame();

		// Generate the pointcloud and texture mappings
		points = rpc.calculate(depth);

		const auto vertices = points.get_vertices();
		const auto texcoords = points.get_texture_coordinates();
		const auto vertex_count = points.size();

		//pc.build();
		EncodingStats stats;
		DracoSettings dracoSettings;
		pc.build(vertices, texcoords, color, vertex_count, dracoSettings, stats);

		std::lock_guard<std::mutex> lock(connection_mutex);
		std::lock_guard<std::mutex> lock_pc(point_cloud_mutex);
		const auto& buffer = pc.GetBuffer();
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

void update_and_draw_fps(cv::Mat& frame,
	int& frame_count,
	std::chrono::steady_clock::time_point& last_time,
	double& fps)
{
	frame_count++;
	auto now = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_time).count();
	if (elapsed >= 1)
	{
		fps = frame_count / static_cast<double>(elapsed);
		frame_count = 0;
		last_time = now;
	}

	std::ostringstream oss;
	oss << std::fixed << std::setprecision(1) << fps << " FPS";
	std::string text = oss.str();

	int font = cv::FONT_HERSHEY_SIMPLEX;
	double scale = 0.7;
	int thickness = 2;
	int baseline = 0;
	auto text_size = cv::getTextSize(text, font, scale, thickness, &baseline);
	cv::Point org(frame.cols - text_size.width - 10,
		text_size.height + 10);
	cv::putText(frame, text, org, font, scale,
		cv::Scalar(0, 255, 0), thickness);
}

// Wrap a color frame (RGB8 or BGR8) into an OpenCV Mat in BGR order for display
cv::Mat visualize_color_frame(const rs2::video_frame& frame)
{
	int w = frame.get_width();
	int h = frame.get_height();

	cv::Mat rgb(h, w, CV_8UC3, (void*)frame.get_data(), cv::Mat::AUTO_STEP);

	cv::Mat bgr;
	cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
	return bgr;

}

// Apply a RealSense colorizer to the depth frame, wrap it in a Mat
cv::Mat visualize_depth_frame(const rs2::depth_frame& frame, rs2::colorizer& color_map)
{
	// color_map.process returns a video_frame in RGB8
	rs2::frame colored = color_map.process(frame);
	int  w = colored.as<rs2::video_frame>().get_width();
	int  h = colored.as<rs2::video_frame>().get_height();

	// Wrap & convert
	cv::Mat rgb(h, w, CV_8UC3, (void*)colored.get_data(), cv::Mat::AUTO_STEP);
	cv::Mat bgr;
	cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
	return bgr;
}


int main() {
	server echo_server;

	// Open a window
	const std::string win_name = "RealSense Color";
	cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);

	rs2::pipeline p;
	// Declare pointcloud object, for calculating pointclouds and texture mappings
	rs2::pointcloud pc;
	// We want the points object to be persistent so we can display the last cloud when a frame drops
	rs2::points points;

	// Filters & colorizer
	rs2::colorizer       color_map;
	rs2::align   align_to_color(RS2_STREAM_COLOR);

	// Configure and start the pipeline
	rs2::config cfg;
	cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
	cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
	p.start(cfg);
	//auto profile = p.start();

	// FPS variables
	auto last_time = std::chrono::steady_clock::now();
	int  frame_count = 0;
	double fps = 0.0;

	// Mode toggle: true = color, false = depth
	bool showColor = true;

	// Define a centered 200×200 ROI
	const int ROI_W = 100, ROI_H = 100;
	cv::Rect roi;
	
	// Object Detector
	frame_count = 0;
	Detector det("../models/yolov8n.onnx", "../models/coco.names"); 

	// Encoding Settings and stats
	DracoSettings dracoSettings;
	EncodingStats stats;

#if 1
	while (true)
	{
		// Block program until frames arrive
		rs2::frameset frames = p.wait_for_frames();
		frames = align_to_color.process(frames); // Align depth to color
		auto color = frames.get_color_frame();
		rs2::depth_frame depth = frames.get_depth_frame();

		// For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
		if (!color)
			color = frames.get_infrared_frame();

		// Get the depth frame's dimensions
		auto width = depth.get_width();
		auto height = depth.get_height();

		// Query the distance from the camera to the object in the center of the image
		float dist_to_center = depth.get_distance(width / 2, height / 2);

		pc.map_to(color); // Tell pointcloud object to map to this color frame
		points = pc.calculate(depth); 		// Generate the pointcloud and texture mappings

		// Upload the color frame to OpenGL
		//app_state.tex.upload(color);

		// Print the distance
		//std::cout << "The camera is facing an object " << dist_to_center << " meters away \r";

		//Choose which to display
		cv::Mat output;
		if (showColor) {
			output = visualize_color_frame(color);

			// Object Detection (TODO)
			det.detect_and_draw(output, depth, frame_count, 10);

		}
		else
			output = visualize_depth_frame(depth, color_map);

		// Draw the ROI rectangle
		int cx = output.cols / 2, cy = output.rows / 2;
		roi = cv::Rect(cx - ROI_W / 2, cy - ROI_H / 2, ROI_W, ROI_H);
		cv::rectangle(output, roi, cv::Scalar(0, 255, 0), 2);

		

		// Get a pointer to the vertex data
		auto vertices = points.get_vertices();   // returns a pointer to rs2::vertex[]
		auto texcoords = points.get_texture_coordinates(); // returns a pointer to rs2::texture_coordinate[]
		size_t vertex_count = points.size(); 
		int  w = depth.get_width();
		int  h = depth.get_height();

		// Gather only the vertices inside the ROI
		std::vector<rs2::vertex> roiVerts;
		std::vector<rs2::texture_coordinate> roiTex;
		roiVerts.reserve(roi.area());
		roiTex.reserve(roi.area());


		
		// Iterate over the ROI rectangle

		for (int y = roi.y; y < roi.y + roi.height; ++y)
		{
			for (int x = roi.x; x < roi.x + roi.width; ++x)
			{
				int idx = y * w + x;
				float z = vertices[idx].z;
				if (z <= 0 || !std::isfinite(z))
					continue;   // skip invalid depth holes

				roiVerts.push_back(vertices[idx]);
				roiTex.push_back(texcoords[idx]);
			}
		}
		
		
		//Hand off to custom point_cloud class
		point_cloud myCloud(static_cast<int>(roiVerts.size()));
		

		myCloud.build(
			roiVerts.data(),
			roiTex.data(),
			color,                   
			roiVerts.size(),
			dracoSettings,
			stats
		);
		
		stats.print(); // Print encoding stats to console

		// Update & draw FPS
		update_and_draw_fps(output, frame_count, last_time, fps);

		// Show current draco encoding settings in the corner:
		cv::putText(output,
			dracoSettings.toString(),
			{ 10,20 },
			cv::FONT_HERSHEY_SIMPLEX,
			0.6, { 255,255,255 }, 2);

		cv::imshow(win_name, output);
		// Exit on ESC
		int key = cv::waitKey(1);
		if (key == 'q' || key == 27)  // q or Esc to quit
			break;
		else if (key == 'c')
			showColor = true;
		else if (key == 'd')
			showColor = false;
		else if (key == ' ') {
			// Full Point cloud
			points.export_to_ply("snapshot.ply", color);
			std::cout << "Saved full point cloud to snapshot.ply\n";

			// ROI Point cloud in draco file
			draco::EncoderBuffer& buf = myCloud.GetBuffer();
			std::string filename = "roi_cloud.drc";

			std::ofstream ofs(filename, std::ios::binary);
			if (!ofs) {
				std::cerr << "Failed to open file for writing: " << filename << "\n";
			}
			else {
				ofs.write(buf.data(), buf.size());
				if (!ofs) {
					std::cerr << "Error writing data to: " << filename << "\n";
				}
				else {
					std::cout << "Wrote compressed ROI cloud ("
						<< roiVerts.size() << " pts) to "
						<< filename << " (" << buf.size() << " bytes)\n";
				}
			}
		}

		else if (key == 61) if (dracoSettings.posQuant < 20) dracoSettings.posQuant++;
		else if (key == 45) if (dracoSettings.posQuant > 1) dracoSettings.posQuant--;
		else if (key == 93) if (dracoSettings.colorQuant < 16) dracoSettings.colorQuant++;
		else if (key == 91) if (dracoSettings.colorQuant > 1) dracoSettings.colorQuant--;
		else if (key == 46) if (dracoSettings.speedEncode < 10) dracoSettings.speedEncode++;
		else if (key == 44) if (dracoSettings.speedEncode > 0) dracoSettings.speedEncode--;

		//std::cout << frame_count << std::endl;
	}
#endif
	return 1;

#if 0
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
			//con->replace_header("Location", "https://192.168.1.169:5173/"); // Redirect target
			con->replace_header("Location", "https://localhost:5173/"); // Redirect target

			//con->set_body("Hello World!");
			//con->set_status(websocketpp::http::status_code::ok);
		});

		echo_server.init_asio();
		//echo_server.set_access_channels(websocketpp::log::alevel::all);
		echo_server.clear_access_channels(websocketpp::log::alevel::all);
		//echo_server.set_error_channels(websocketpp::log::elevel::all);

		std::thread broadcaster([&echo_server, &p]() {
			time_broadcast_loop(&echo_server, p);
		});

		/*std::thread pc_builder([]() {
			while (true) {
				pc.build();
			}
		});*/

		echo_server.listen(9002);
		echo_server.start_accept();
		echo_server.run();

		//pc_builder.join(); // Wait for the broadcaster thread (never exits)
		broadcaster.join(); // Wait for the broadcaster thread (never exits)
	}
	catch (websocketpp::exception const& e) {
		std::cout << "WebSocket exception: " << e.what() << std::endl;
	}
	catch (std::exception const& e) {
		std::cout << "STD exception: " << e.what() << std::endl;
	}

	return 0;
#endif
}