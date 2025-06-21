#include "Detector.h"
#include <fstream>
#include <stdexcept>
#include <iomanip>


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

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

void Detector::detect_and_draw(cv::Mat& frame,
    const rs2::depth_frame& depth,
    int                     frameCount,
    int                     interval)
{
    // Preprocess
    cv::Mat blob = cv::dnn::blobFromImage(
        frame, 1 / 255.0f, inpSize, cv::Scalar(), true, false);
    net.setInput(blob);

    // Only run detection every `interval` frames
    if (frameCount % interval != 0)
        return;

    // Forward pass
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    cv::Mat& out = outputs[0];  

    // Parse Detections
    std::vector<int>     ids;
    std::vector<float>   confs;
    std::vector<cv::Rect> boxes;
    int rows = out.size[1], dims = out.size[2];
    float* data = reinterpret_cast<float*>(out.data);

    for (int i = 0; i < rows; ++i, data += dims) {
        float score = data[4];
        if (score < confThreshold) continue;

        // find best class
        cv::Mat scores(1, (int)classNames.size(), CV_32FC1, data + 5);
        cv::Point classIdPoint;
        double maxClassScore;
        cv::minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &classIdPoint);
        if (maxClassScore < confThreshold) continue;

		//decode bounding box
        float cx = data[0] * frame.cols;
        float cy = data[1] * frame.rows;
        float w = data[2] * frame.cols;
        float h = data[3] * frame.rows;
        int left = static_cast<int>(cx - w / 2);
        int top = static_cast<int>(cy - h / 2);

        ids.push_back(classIdPoint.x);
        confs.push_back(static_cast<float>(maxClassScore));
        boxes.emplace_back(left, top, static_cast<int>(w), static_cast<int>(h));
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, nmsThreshold, indices);

    //Draw Results
    for (int idx : indices) {
        const auto& b = boxes[idx];
        int            cls = ids[idx];
        float          c = confs[idx];

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

        // depth at box center
        int cx = b.x + b.width / 2;
        int cy = b.y + b.height / 2;
        float z = depth.get_distance(cx, cy);
        std::ostringstream distLabel;
        distLabel << std::fixed << std::setprecision(2) << z << " m";
        cv::putText(frame,
            distLabel.str(),
            cv::Point(b.x, b.y + b.height + 15),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255, 0, 0),
            1);
    }
}