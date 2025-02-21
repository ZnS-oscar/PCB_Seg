#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "pcbseg_status.hpp"
#include <pybind11/pybind11.h>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>

#include "pcb_seg_io.hpp"
namespace py = pybind11;

namespace PCBSEG {

class SegmentationEvaluator {
public:
    SegmentationEvaluator (const std::string& model_path, const std::string& label_path,int gpu_id, float gpu_mem_gb);
    ~SegmentationEvaluator ();

    Status evaluateSingle(const cv::Mat& bmp_image,  SegResult& cpp_segresult);

    //TODO 未实现
    Status evaluateBatch(const std::vector<cv::Mat>& bmp_images,
                        const std::vector<cv::Mat>& tiff_images,
                        std::vector<cv::Mat>& masks);

    // 线程安全的推理接口
    Status threadSafeInference(const cv::Mat& image,SegResult& cpp_segresult);

    // 用于测试的详细推理接口 未实现
    Status testInference(const cv::Mat& image, int thread_id, 
                        std::atomic<int>& successful_count,
                        std::atomic<int>& failed_count,
                        const std::atomic<bool>& stop_flag,
                        int iterations = 10);

    // GIL 管理方法
    void acquireGIL() {
        if (!gil_acquired_) {
            gil_acquire_ = std::make_unique<py::gil_scoped_acquire>();
            gil_acquired_ = true;
        }
    }

    void releaseGIL() {
        if (gil_acquired_) {
            gil_acquire_.reset();
            gil_acquired_ = false;
        }
    }

private:
    std::string model_path_;
    std::string label_path_;
    py::object model_instance_;
    std::mutex eval_mutex_;
    std::unique_ptr<py::gil_scoped_acquire> gil_acquire_;
    bool gil_acquired_{false};
    std::mutex gil_mutex_;  // 保护 GIL 操作的互斥锁
};

} 