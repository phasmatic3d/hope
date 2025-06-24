#include "Detector.h"
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <iostream>

#include <librealsense2/rs.hpp>

Detector::Detector(const std::string& modelPath,
    const std::string& classesPath,
    float               confThresh,
    float               nmsThresh,
    cv::Size            inpSz)
    : confThreshold(confThresh)
    , nmsThreshold(nmsThresh)
    , inpSize(inpSz)
{
    // Load class names
    std::ifstream ifs(classesPath);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open class list: " + classesPath);
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty())
            classNames.push_back(line);
    }
    if (classNames.empty())
        throw std::runtime_error("No classes found in: " + classesPath);

    // Load the network
    net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty())
        throw std::runtime_error("Failed to load model: " + modelPath);

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
}

bool Detector::detect_and_draw(cv::Mat& frame,
    const rs2::depth_frame& depth,
    int                     frameCount,
    cv::Rect&               bestROI,
    int                     interval)
{
    // Preprocess


    cv::Mat blob = cv::dnn::blobFromImage(
        frame, 1 / 255.0f, inpSize, cv::Scalar(), true, false);
    net.setInput(blob);

    // Forward pass
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());


    cv::Mat& out = outputs[0];  

    

    // Parse Detections
    std::vector<int>     ids;
    std::vector<float>   confs;
    std::vector<cv::Rect> boxes;
    int rows = out.size[2], dimensions = out.size[1];

    out = out.reshape(1, dimensions); 
    cv::transpose(out, out); 

    float* data = reinterpret_cast<float*>(out.data);

    for (int i = 0; i < rows; ++i, data += dimensions) {

		float* classScores = data + 4; 
        cv::Mat scores(1, 80, CV_32FC1, classScores);

        cv::Point classIdPoint;
        double maxClassScore;
        cv::minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &classIdPoint);

        float conf = static_cast<float>(maxClassScore);
        if (conf < confThreshold) continue;

		float x = data[0], y = data[1], w = data[2], h = data[3];

        int left = x * frame.cols - w * frame.cols / 2;
		int top = y * frame.rows - h * frame.rows / 2;

		int width = w * frame.cols;
		int height = h * frame.rows;

        ids.push_back(classIdPoint.x);
        confs.push_back(conf);
        boxes.emplace_back(left, top, width, height);
    }
    //std::cout << "[DEBUG] kept " << boxes.size() << " boxes before NMS\n";
    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, nmsThreshold, indices);

    if (indices.empty()) {
        // nothing passed the threshold
        return false;
    }

    //Draw Results
    int bestIdx = indices[0];
    float bestConf = confs[bestIdx];
    for (int idx : indices) {

        const auto& b = boxes[idx];
        int            cls = ids[idx];
        float          c = confs[idx];

        /*
		// draw bounding box
        cv::rectangle(frame, b, cv::Scalar(0, 255, 0), 2);

        //label text
        std::ostringstream label;
        label << classNames[cls] << ' ' << int(c * 100) << '%';
        int baseLine;
        auto textSize = cv::getTextSize(
            label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(frame,
            cv::Point(b.x, b.y - textSize.height - baseLine),
            cv::Point(b.x + textSize.width, b.y),
            cv::Scalar(0, 255, 0),
            cv::FILLED);
        cv::putText(frame,
            label.str(),
            cv::Point(b.x, b.y - baseLine),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            1);
        
        // assume `box` is already clamped into [0..W)×[0..H)
        int cx = b.x + b.width / 2;
        int cy = b.y + b.height / 2;

        // clamp into the depth frame
        cx = std::clamp(cx, 0, depth.get_width() - 1);
        cy = std::clamp(cy, 0, depth.get_height() - 1);

        // get distance (returns 0 on holes, throws only if x,y out of range)
        float z = depth.get_distance(cx, cy);

        // treat zero or nan as invalid
        if (z <= 0.f || !std::isfinite(z)) {
            // skip drawing depth or draw “N/A”
        }
        else {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << z << " m";
            cv::putText(frame, oss.str(),
                { b.x, b.y + b.height - 45 },
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 0, 0), 1);
        }
        */
        if (confs[idx] > bestConf) {
            bestConf = confs[idx];
            bestIdx = idx;
        }
    }
    bestROI = boxes[bestIdx];

    
    

    // clamp into frame bounds
    bestROI.x = std::clamp(bestROI.x, 0, frame.cols - 1);
    bestROI.y = std::clamp(bestROI.y, 0, frame.rows - 1);
    bestROI.width = std::clamp(bestROI.width, 1, frame.cols - bestROI.x);
    bestROI.height = std::clamp(bestROI.height, 1, frame.rows - bestROI.y);

    return true;
}