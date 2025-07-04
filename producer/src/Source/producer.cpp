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
#include <draco/compression/decode.h>
#include <draco/point_cloud/point_cloud.h>
#include <draco/point_cloud/point_cloud_builder.h>
#include <draco/core/encoder_buffer.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp> // glm::value_ptr
#include "Encoder.h"
#include "Detector.h"
#include "Structs.h"
#include "Util.hpp"




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
	point_cloud(int num_of_points = 1000)
		: m_num_of_points(num_of_points)
	{
	}

	void build(const rs2::vertex* vertices,
		const rs2::texture_coordinate* texcoords,
		const rs2::video_frame& color_frame,
		size_t count,
		DracoSettings& dracoSettings,
		EncodingStats& stats) {
		std::lock_guard<std::mutex> lock(point_cloud_mutex);

		Timer t_prep;

		m_point_data.resize(count);
		m_num_of_points = count;
		stats.pts = count;
		stats.raw_bytes = count * (3 * sizeof(float) + 3 * sizeof(uint8_t));

		std::memcpy(m_point_data.data(), vertices, sizeof(*vertices) * count);

		// Color matrix (OpenCV) for lookup
		cv::Mat colMat(
			color_frame.get_height(),
			color_frame.get_width(),
			CV_8UC3,
			(void*)color_frame.get_data(),
			cv::Mat::AUTO_STEP);

		draco::PointCloudBuilder builder;
		builder.Start(m_num_of_points);

		int pos_att_id = builder.AddAttribute(draco::GeometryAttribute::POSITION, 3, draco::DT_FLOAT32);
		int col_att_id = builder.AddAttribute(draco::GeometryAttribute::COLOR, 3, draco::DT_UINT8);

		// Create points
		for (int i = 0; i < m_num_of_points; ++i) {
			float pos[3] = {
				static_cast<float>(m_point_data[i].x),
				static_cast<float>(m_point_data[i].y),
				static_cast<float>(m_point_data[i].z)
			};

			const auto& tc = texcoords[i];
			int u = std::clamp(int(tc.u * (colMat.cols - 1)), 0, colMat.cols - 1);
			int v = std::clamp(int(tc.v * (colMat.rows - 1)), 0, colMat.rows - 1);
			auto bgr = colMat.at<cv::Vec3b>(v, u);
			uint8_t rgb[3] = { bgr[2], bgr[1], bgr[0] };

			builder.SetAttributeValueForPoint(pos_att_id, draco::PointIndex(i), pos);
			builder.SetAttributeValueForPoint(col_att_id, draco::PointIndex(i), rgb);
		}
		stats.prep_ms = t_prep.elapsed_ms();

		// Build the point cloud
		std::unique_ptr<draco::PointCloud> pc = builder.Finalize(false);
		if (!pc) {
			std::cerr << "Failed to build point cloud." << std::endl;
			return;
		}

		// Encode point cloud
		m_encoder = std::make_unique<DracoEncoder>(dracoSettings);
		if (!m_encoder->encode(*pc, stats)) {
			return;
		}

#ifdef DECODE
		Timer t_decode;
		draco::Decoder decoder;
		m_decoder_buffer.Init(m_encoder->GetBuffer().data(), m_encoder->GetBuffer().size());

		auto decoded = decoder.DecodePointCloudFromBuffer(&m_decoder_buffer);
		stats.decode_ms = t_decode.elapsed_ms();

		if (!decoded.ok()) {
			std::cerr << "Decoding failed: " << decoded.status().error_msg_string() << std::endl;
		}
#endif // DECODE
	}

	const draco::EncoderBuffer& GetBuffer() const {
		return m_encoder->GetBuffer();
	}

private:
	std::mutex point_cloud_mutex;
	std::vector<glm::vec3> m_point_data;
	int m_num_of_points;
	std::unique_ptr<DracoEncoder> m_encoder;
	draco::DecoderBuffer m_decoder_buffer;
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

	cv::Rect roi;
	
	// Object Detector
	frame_count = 0;
	Detector det("../models/yolov8n.onnx", "../models/coco.names"); 
	AsyncDetector asyncDet{ std::move(det), /*interval=*/10 };

	// Encoding Settings and stats
	DracoSettings dracoSettingsROI;
	DracoSettings dracoSettingsOut;
	DracoSettings dracoSettingsFull;
	EncodingStats statsROI;
	EncodingStats statsOut;
	EncodingStats statsFull;

	BuildMode currentMode = BuildMode::ROI;

	// background model
	//cv::Mat bgDepth = compute_background_depth(p, align_to_color);

#if 1
	while (true)
	{
		statsROI = {};
		statsOut = {};
		statsFull = {};

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

		//Choose which to display
		cv::Mat output;
		if (showColor) {
			output = visualize_color_frame(color);
		}
		else
			output = visualize_depth_frame(depth, color_map);

		// Get a pointer to the vertex data
		auto vertices = points.get_vertices();   // returns a pointer to rs2::vertex[]
		auto texcoords = points.get_texture_coordinates(); // returns a pointer to rs2::texture_coordinate[]
		size_t vertex_count = points.size();
		int  w = depth.get_width();
		int  h = depth.get_height();

		// BUILD MODE ROI-OUTSIDE
		if (currentMode == BuildMode::ROI) { 
			Timer t_det;

			// feed the detector whenever frame_count % 10 == 0
			asyncDet.pushFrame(output, depth, frame_count);

			// Get the best ROI from the detector (thread)
			if (auto detRoi = asyncDet.getROI()) {
				if(frame_count % 10 != 0)
					roi = *detRoi;
			}
			else {
				// fallback... center a default ROI
				int cx = output.cols / 2, cy = output.rows / 2;
				roi = cv::Rect(cx - dracoSettingsROI.roiWidth / 2,
					cy - dracoSettingsROI.roiHeight / 2,
					dracoSettingsROI.roiWidth,
					dracoSettingsROI.roiHeight);
			}
			cv::rectangle(output, roi, cv::Scalar(0, 255, 0), 2);
			statsROI.det_ms = t_det.elapsed_ms();

			// Gather only the vertices inside the ROI
			std::vector<rs2::vertex> roiVerts;
			std::vector<rs2::texture_coordinate> roiTex;
			roiVerts.reserve(roi.area());
			roiTex.reserve(roi.area());


			// Create point cloud of points outside of the ROI

			std::vector<rs2::vertex>       outVerts;
			std::vector<rs2::texture_coordinate> outTex;
			outVerts.reserve(width * height - roiVerts.size());
			outTex.reserve(width * height - roiVerts.size());

			Timer t_pc;
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					int idx = y * w + x;
					float z = vertices[idx].z;
					if (z <= 0 || !std::isfinite(z))
						continue;

					if (x >= roi.x && x < roi.x + roi.width &&
						y >= roi.y && y < roi.y + roi.height) {
						roiVerts.push_back(vertices[idx]);
						roiTex.push_back(texcoords[idx]);
						continue;
					}
						
					outVerts.push_back(vertices[idx]);
					outTex.push_back(texcoords[idx]);
				}
			}

			

			statsROI.pc_ms = t_pc.elapsed_ms(); // Time to find outside ROI point cloud

			// ROI Point_cloud (high importance)
			point_cloud cloudROI(static_cast<int>(roiVerts.size()));


			dracoSettingsOut.posQuant = dracoSettingsROI.posQuant / 4;
			dracoSettingsOut.colorQuant = dracoSettingsROI.colorQuant;
			dracoSettingsOut.speedEncode = dracoSettingsROI.speedDecode;
			dracoSettingsOut.speedDecode = dracoSettingsROI.speedDecode;
			// Out-of-ROI point_cloud (low importance)
			point_cloud cloudOut(static_cast<int>(outVerts.size()));


			// Create lambda functions to build the point clouds
			auto buildROI = [&]() {
				if (!roiVerts.empty()) {
					cloudROI.build(
						roiVerts.data(),
						roiTex.data(),
						color,
						roiVerts.size(),
						dracoSettingsROI,
						statsROI
					);
				}
			};

			auto buildOut = [&]() {
				if (!outVerts.empty()) {
					cloudOut.build(
						outVerts.data(),
						outTex.data(),
						color,
						outVerts.size(),
						dracoSettingsOut,
						statsOut
					);
				}
			};

			// Launch each on its own thread
			std::thread tROI(buildROI);
			std::thread tOut(buildOut);

			// Wait for both to finish
			tROI.join();
			tOut.join();


		} // BUILD MODE ROI-OUTSIDE
		// BUILD MODE FULL FRAME
		else if (currentMode == BuildMode::FULL) {

			// Take the Same settings as high quality ROI
			dracoSettingsFull.posQuant = dracoSettingsROI.posQuant;
			dracoSettingsFull.colorQuant = dracoSettingsROI.colorQuant;
			dracoSettingsFull.speedEncode = dracoSettingsROI.speedEncode;
			dracoSettingsFull.speedDecode = dracoSettingsROI.speedDecode;

			std::vector<rs2::vertex>       fullVerts;
			std::vector<rs2::texture_coordinate> fullTex;
			fullVerts.reserve(width* height);
			fullTex.reserve(width* height);

			Timer t_pc_full;
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					int idx = y * width + x;
					float z = vertices[idx].z;
					if (z <= 0 || !std::isfinite(z))
						continue;
					fullVerts.push_back(vertices[idx]);
					fullTex.push_back(texcoords[idx]);
				}
			}
			statsFull.pc_ms = t_pc_full.elapsed_ms(); // Time to find full point cloud

			point_cloud cloudFull(static_cast<int>(fullVerts.size()));
			cloudFull.build(
				fullVerts.data(),
				fullTex.data(),
				color,
				fullVerts.size(),
				dracoSettingsFull,
				statsFull
			);

		} // BUILD MODE FULL FRAME


		// Print Stats
		print_stats(statsROI, statsOut, statsFull);


		std::cout << "\033[0m" << "Total Time : " << std::fixed << std::setprecision(2) << statsROI.total_time_ms + statsOut.total_time_ms + statsFull.total_time_ms << " ms\n";
			

		// Update & draw FPS
		update_and_draw_fps(output, frame_count, last_time, fps);

		// Show current draco encoding settings in the corner:
		cv::putText(output,
			dracoSettingsROI.toString(),
			{ 10,20 },
			cv::FONT_HERSHEY_SIMPLEX,
			0.6, { 255,255,255 }, 2);

		cv::imshow(win_name, output);

		int key = cv::waitKeyEx(1);
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

			/*
			// ROI Point cloud in draco file
			draco::EncoderBuffer& buf = cloud.GetBuffer();
			std::string filename = "encoded.drc";

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
			}*/
		}
		// DRACO SETTINGS
		else if (key == 61) {
			if (dracoSettingsROI.posQuant < 20)
				dracoSettingsROI.posQuant++;
		}
		else if (key == 45) {
			if (dracoSettingsROI.posQuant > 1)
				dracoSettingsROI.posQuant--;
		}
		else if (key == 93) {
			if (dracoSettingsROI.colorQuant < 16)
				dracoSettingsROI.colorQuant++;
		}
		else if (key == 91) {
			if (dracoSettingsROI.colorQuant > 1)
				dracoSettingsROI.colorQuant--;
		}
		else if (key == 46) {
			if (dracoSettingsROI.speedEncode < 10)
				dracoSettingsROI.speedEncode++;
			if (dracoSettingsROI.speedDecode < 10)
				dracoSettingsROI.speedDecode++;
		}
		else if (key == 44) {
			if (dracoSettingsROI.speedEncode > 0)
				dracoSettingsROI.speedEncode--;
			if (dracoSettingsROI.speedDecode > 0)
				dracoSettingsROI.speedDecode--;
		}
		// ARROW KEYS resize ROI
		else if (key == 2490368) {           // Up arrow
			dracoSettingsROI.roiHeight += 20;
			
		}
		else if (key == 2621440) {           // Down arrow
			if (dracoSettingsROI.roiHeight > 1)
				dracoSettingsROI.roiHeight -= 20;
			
		}
		else if (key == 2555904) {           // Right arrow
			dracoSettingsROI.roiWidth += 20;
			
		}
		else if (key == 2424832) {           // Left arrow
			if (dracoSettingsROI.roiWidth > 1)
				dracoSettingsROI.roiWidth -= 20;
		}
		else if (key == 'f') {
			// cycle modes
			if (currentMode == BuildMode::ROI)
				currentMode = BuildMode::FULL;
			else                                     
				currentMode = BuildMode::ROI;
		}

		frame_count++; // Next Frame
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