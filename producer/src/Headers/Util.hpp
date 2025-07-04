#pragma once

#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include <memory>
#include <set>
#include <iomanip>

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp> // Include OpenCV API

inline void update_and_draw_fps(cv::Mat& frame,
	int& frame_count,
	std::chrono::steady_clock::time_point& last_time,
	double& fps)
{

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
inline cv::Mat visualize_color_frame(const rs2::video_frame& frame)
{
	int w = frame.get_width();
	int h = frame.get_height();

	cv::Mat rgb(h, w, CV_8UC3, (void*)frame.get_data(), cv::Mat::AUTO_STEP);

	cv::Mat bgr;
	cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
	return bgr;

}

// Apply a RealSense colorizer to the depth frame, wrap it in a Mat
inline cv::Mat visualize_depth_frame(const rs2::depth_frame& frame, rs2::colorizer& color_map)
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

inline void print_stats(EncodingStats& statsROI, EncodingStats& statsOut, EncodingStats& statsFull) {
	static bool first = true;
	const int LINES_ROI = 7;
	const int LINES_OUT = 7;
	const int LINES_FULL = 7;
	const int TOTAL = LINES_ROI + LINES_OUT + LINES_FULL + 4;

	if (first) {
		// Reserve TOTAL blank lines
		for (int i = 0; i < TOTAL; ++i) std::cout << "\n";
		first = false;
	}

	// move up TOTAL lines, then clear everything below
	std::cout << "\033[" << TOTAL << "A" << "\033[J";

	// --- ROI stats ---
	std::cout << "\033[32m" << "=== ROI Stats ===\n";
	statsROI.printBodyOnly();
	// --- Outside ROI stats ---
	std::cout << "\033[0m" << "\033[31m" << "===Outside-of-ROI Stats ===\n";
	statsOut.printBodyOnly();
	// --- Full Frame stats ---
	std::cout << "\033[0m" << "\033[38;2;255;165;0m" << "=== Full Frame Stats ===\n";
	statsFull.printBodyOnly();

	std::cout << std::flush;
}

// Background Depth matrix
inline cv::Mat compute_background_depth(rs2::pipeline& pipe,
	rs2::align& align_to_color,
	int num_frames = 30) {
	// Collect depth frames
	std::vector<cv::Mat> depthMats;
	depthMats.reserve(num_frames);

	for (int i = 0; i < num_frames; ++i) {
		rs2::frameset frames = pipe.wait_for_frames();
		frames = align_to_color.process(frames);
		rs2::depth_frame depth_frame = frames.get_depth_frame();
		cv::Mat dMat(depth_frame.get_height(),
			depth_frame.get_width(),
			CV_32FC1,
			const_cast<void*>(depth_frame.get_data()));
		depthMats.push_back(dMat.clone());
	}

	// Prepare output median map
	cv::Mat bgDepth(depthMats[0].size(), CV_32FC1);
	int rows = bgDepth.rows;
	int cols = bgDepth.cols;
	std::vector<float> vals;
	vals.reserve(depthMats.size());

	// Compute per-pixel median
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			vals.clear();
			for (const auto& mat : depthMats) {
				vals.push_back(mat.at<float>(y, x));
			}
			auto mid = vals.begin() + vals.size() / 2;
			std::nth_element(vals.begin(), mid, vals.end());
			bgDepth.at<float>(y, x) = *mid;
		}
	}

	return bgDepth;
}
