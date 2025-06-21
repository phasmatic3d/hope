#pragma once

#include <librealsense2/rs.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>


class Detector {
public:

    Detector(const std::string& modelPath,
        const std::string& classesPath,
        float               confThresh = 0.5f,
        float               nmsThresh = 0.4f,
        cv::Size            inpSz = { 640,640 });

    void detect_and_draw(cv::Mat& frame,
        const rs2::depth_frame& depth,
        int                     frameCount,
        int                     interval = 10);


private:
    cv::dnn::Net            net;
    std::vector<std::string> classNames;
    float                   confThreshold;
    float                   nmsThreshold;
    cv::Size                inpSize;
};