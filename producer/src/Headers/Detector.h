#pragma once

#include <librealsense2/rs.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>


#include <thread>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <atomic>

class Detector {
public:

    Detector(const std::string& modelPath,
        const std::string& classesPath,
        float               confThresh = 0.4f,
        float               nmsThresh = 0.3f,
        cv::Size            inpSz = { 640,640 });

    bool detect_and_draw(cv::Mat& frame,
        const rs2::depth_frame& depth,
        int                     frame_count,
        cv::Rect&               bestROI,
        int                     interval = 10);


private:
    cv::dnn::Net            net;
    std::vector<std::string> classNames;
    float                   confThreshold;
    float                   nmsThreshold;
    cv::Size                inpSize;
};

// For running object detection on threads
class AsyncDetector {
public:
    AsyncDetector(Detector detector, int interval)
        : det_(std::move(detector))
        , interval_(interval)
        , stop_{ false }
    {
        worker_ = std::thread(&AsyncDetector::workerLoop, this);
    }

    ~AsyncDetector() {
        {
            std::lock_guard lk(mtx_);
            stop_ = true;
            cv_.notify_all();
        }
        if (worker_.joinable())
            worker_.join();
    }

    // Called every frame; only enqueues work when frame_count % interval_ == 0
    void pushFrame(cv::Mat frame, rs2::depth_frame depth, int frame_count) {
        if (frame_count % interval_ != 0) return;
        {
            std::lock_guard lk(mtx_);
            pendingFrame_ = std::move(frame);
            pendingDepth_ = depth;
            cv_.notify_one();
        }
    }

    // Non-blocking; returns the most recent ROI if available
    std::optional<cv::Rect> getROI() {
        std::lock_guard lk(roiMtx_);
        return sharedROI_;
    }

private:
    void workerLoop() {
        std::unique_lock lk(mtx_);
        while (!stop_) {
            cv_.wait(lk, [&] { return pendingFrame_.has_value() && pendingDepth_.has_value() || stop_; });
            if (stop_) break;

            // Grab work out of the queue:
            cv::Mat frame = std::move(*pendingFrame_);
            rs2::depth_frame depth = *pendingDepth_;
            pendingFrame_.reset();
            pendingDepth_.reset();

            // Unlock while we run inference:
            lk.unlock();
            cv::Rect roi;
            if (det_.detect_and_draw(frame, depth, 0, roi, interval_)) {
                std::lock_guard roiLk(roiMtx_);
                sharedROI_ = roi;
            }
            lk.lock();
        }
    }

    Detector                 det_;
    int                      interval_;
    std::thread              worker_;

    // for pushing work
    std::mutex                           mtx_;
    std::condition_variable              cv_;
    bool                                 stop_;
    std::optional<cv::Mat>               pendingFrame_;
    std::optional<rs2::depth_frame>      pendingDepth_;

    // for reading results
    std::mutex               roiMtx_;
    std::optional<cv::Rect>  sharedROI_;
};