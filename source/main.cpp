#include "pcb_seg.hpp"
#include <pybind11/embed.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <signal.h>

namespace py = pybind11;

// GPU配置常量
constexpr int GPU_ID = 0;          // 使用的GPU ID
constexpr float GPU_MEM_GB = 4.0f; // GPU显存限制（GB）
constexpr int TEST_ITERATIONS = 10; // 每个线程的测试次数

// 用于统计线程执行次数
std::atomic<int> successful_inferences(0);
std::atomic<int> failed_inferences(0);

// 全局停止标志
std::atomic<bool> g_stop_flag(false);

// 信号处理函数
void signal_handler(int signum) {
    if (signum == SIGINT) {
        std::cout << "\nReceived Ctrl+C, stopping..." << std::endl;
        g_stop_flag = true;
    }
}

void thread_func(PCBSEG::SegmentationEvaluator& evaluator, const cv::Mat& image, int thread_id) {
    cv::Mat mask;
    cv::Mat dummy_tiff;
    PCBSEG::SegResult cpp_segresult;
    try {
        for (int i = 0; i < TEST_ITERATIONS && !g_stop_flag; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto status = PCBSEG::Status::SUCCESS;
            
            std::cout << "Thread " << thread_id << " - Iteration " << i + 1 
                     << ": Acquiring GIL..." << std::endl;
            {
                py::gil_scoped_acquire acquire;
                std::cout << "Thread " << thread_id << " - GIL acquired" << std::endl;
                status = evaluator.evaluateSingle(image, cpp_segresult);
                std::cout << "Thread " << thread_id << " - Inference completed" << std::endl;
            }
            std::cout << "Thread " << thread_id << " - GIL released" << std::endl;
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            
            if (status == PCBSEG::Status::SUCCESS) {
                successful_inferences++;
                std::cout << "Thread " << thread_id << " - Iteration " << i + 1 
                         << "/" << TEST_ITERATIONS 
                         << " completed in " << elapsed.count() << "s" << std::endl;
                
                if (i == 0) {
                    cv::imwrite("thread_" + std::to_string(thread_id) + "_result.bmp", cpp_segresult.mask);
                }
            } else {
                failed_inferences++;
                std::cerr << "Thread " << thread_id << " - Iteration " << i + 1 
                         << "/" << TEST_ITERATIONS << " failed with status: " 
                         << static_cast<int>(status) << std::endl;
            }
            
            // 在没有GIL的情况下检查停止标志和休眠
            if (g_stop_flag) {
                std::cout << "Thread " << thread_id << " stopping..." << std::endl;
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } catch (const std::exception& e) {
        std::cerr << "Thread " << thread_id << " error: " << e.what() << std::endl;
    }
}

// 简单的线程使用示例
void simple_thread_test(const std::string& model_path, 
                       const std::string& image_path,
                       int gpu_id = 0, 
                       float gpu_mem_gb = 4.0f) {
    //std::cout << "Initializing evaluator for simple test..." << std::endl;
    PCBSEG::SegmentationEvaluator segmentationEvaluator(model_path, "",gpu_id, gpu_mem_gb);
    
    //std::cout << "Loading image..." << std::endl;
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image for simple test" << std::endl;
        return;
    }
    
    PCBSEG::SegResult cpp_segresult;
    //auto start_time = std::chrono::high_resolution_clock::now();
    
    //std::cout << "Starting single thread inference..." << std::endl;
    bool inference_completed = false;
    
    std::cout << "Releasing GIL before creating thread..." << std::endl;

    /*---------------- 注意，主线程中创建线程前必须释放GIL，否则线程无法获取GIL ---------------- */

    py::gil_scoped_release release;

    /*---------------- 注意，主线程中创建线程前必须释放GIL，否则线程无法获取GIL ---------------- */
    
    std::thread t1([&]() {
        try {
            //std::cout << "Simple test thread: Acquiring GIL..." << std::endl;
            auto status = segmentationEvaluator.threadSafeInference(image, cpp_segresult);
            //std::cout << "Simple test thread: GIL operations completed" << std::endl;
            
            if (status == PCBSEG::Status::SUCCESS) {
                inference_completed = true;
                cv::imwrite("simple_test_result.jpeg", cpp_segresult.mask);
                std::cout << "Single thread inference successful" << std::endl;
            } else {
                std::cerr << "Single thread inference failed with status: " 
                         << static_cast<int>(status) << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in simple test thread: " << e.what() << std::endl;
        }
    });
    
    if (t1.joinable()) {
        t1.join();
        //auto end_time = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> elapsed = end_time - start_time;
        
        if (inference_completed) {
            //std::cout << "Simple test completed in " << elapsed.count() << "s" << std::endl;
            std::cout << "Result saved as 'simple_test_result.bmp'" << std::endl;
        } else {
            std::cerr << "Simple test failed or timed out" << std::endl;
        }
    }
}

// 详细的测试功能
// void detailed_test(const std::string& model_path, 
//                   const std::string& image_path,
//                   int gpu_id = 0, 
//                   float gpu_mem_gb = 4.0f) {
//     std::cout << "Initializing evaluators for detailed test..." << std::endl;
    
//     std::atomic<int> successful_inferences(0);
//     std::atomic<int> failed_inferences(0);
//     std::atomic<bool> stop_flag(false);
    
//     GoldWireSeg::Evaluator evaluator1(model_path, gpu_id, gpu_mem_gb);
//     std::cout << "Evaluator 1 initialized" << std::endl;
    
//     GoldWireSeg::Evaluator evaluator2(model_path, gpu_id, gpu_mem_gb);
//     std::cout << "Evaluator 2 initialized" << std::endl;
    
//     std::cout << "Loading image..." << std::endl;
//     cv::Mat image = cv::imread(image_path);
//     if (image.empty()) {
//         std::cerr << "Failed to load image for detailed test" << std::endl;
//         return;
//     }
    
//     auto start_time = std::chrono::high_resolution_clock::now();
    
//     std::cout << "Starting detailed test threads..." << std::endl;
    
//     std::cout << "Releasing GIL before creating threads for detailed test..." << std::endl;
//     py::gil_scoped_release release;
    
//     std::thread t1([&]() {
//         std::cout << "Detailed test thread 1: Starting..." << std::endl;
//         evaluator1.testInference(image, 1, successful_inferences, 
//                                failed_inferences, stop_flag);
//         std::cout << "Detailed test thread 1: Completed" << std::endl;
//     });
    
//     std::thread t2([&]() {
//         std::cout << "Detailed test thread 2: Starting..." << std::endl;
//         evaluator2.testInference(image, 2, successful_inferences, 
//                                failed_inferences, stop_flag);
//         std::cout << "Detailed test thread 2: Completed" << std::endl;
//     });
    
//     t1.join();
//     t2.join();
    
//     auto end_time = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end_time - start_time;
    
//     // 输出统计信息
//     std::cout << "\nDetailed test statistics:" << std::endl;
//     std::cout << "Total time: " << elapsed.count() << "s" << std::endl;
//     std::cout << "Successful inferences: " << successful_inferences << std::endl;
//     std::cout << "Failed inferences: " << failed_inferences << std::endl;
//     if (successful_inferences + failed_inferences > 0) {
//         std::cout << "Average time per inference: " 
//                   << elapsed.count() / (successful_inferences + failed_inferences) 
//                   << "s" << std::endl;
//     }
// }

int main(int argc, char* argv[]) {
    // std::string model_path = "/root/autodl-tmp/sahi/demo/models/yolov8n-segdy.onnx";
    std::string model_path = "/root/autodl-tmp/sahi/demo/models/synpcbseg.onnx";
    // std::string image_path = "/root/autodl-tmp/sahi/input/input/a1a1.jpeg";
    // std::string model_path = "/root/autodl-tmp/sahi/demo/models/synpcbseg.onnx";
    std::string image_path="/root/autodl-tmp/sahi/input/input/pcb_0002_5768_0_1024_1024_200.png";
    if (argc > 1) {
        image_path = argv[1];
    }

    // 初始化Python解释器
    py::scoped_interpreter guard{};

    // 设置信号处理
    signal(SIGINT, signal_handler);

    try {
        // std::cout << "\n=== Starting Simple Thread Test ===" << std::endl;
        // simple_thread_test(model_path, image_path, GPU_ID, GPU_MEM_GB);
        // std::cout << "Simple thread test completed\n" << std::endl;

        // std::cout << "\n=== Starting Detailed Test ===" << std::endl;
        // detailed_test(model_path, image_path, GPU_ID, GPU_MEM_GB);
        // std::cout << "Detailed test completed\n" << std::endl;

        std::cout << "\n=== Starting Multi-Instance Test ===" << std::endl;
        // 读取图像
        cv::Mat bmp_image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (bmp_image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return 1;
        }
        std::cout << "Image loaded successfully" << std::endl;




// TODO
        // 创建两个评估器实例
        PCBSEG::SegmentationEvaluator evaluator1(model_path, "",0, GPU_MEM_GB);
        std::cout << "Evaluator 1 initialized" << std::endl;
        
        PCBSEG::SegmentationEvaluator evaluator2(model_path, "",0, GPU_MEM_GB);
        std::cout << "Evaluator 2 initialized" << std::endl;
        
        std::cout << "Starting threads with separate evaluators..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Releasing GIL before creating threads for multi-instance test..." << std::endl;
        py::gil_scoped_release release;
        
        std::thread t1(thread_func, std::ref(evaluator1), std::ref(bmp_image), 1);
        std::thread t2(thread_func, std::ref(evaluator2), std::ref(bmp_image), 2);
        
        t1.join();
        t2.join();

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = end_time - start_time;

        // 如果是因为Ctrl+C停止的，输出相应信息
        if (g_stop_flag) {
            std::cout << "\nTest stopped by user." << std::endl;
        }

        // 输出统计信息
        std::cout << "\nMulti-instance test completed!" << std::endl;
        std::cout << "Total time: " << total_time.count() << "s" << std::endl;
        std::cout << "Total successful inferences: " << successful_inferences << std::endl;
        std::cout << "Total failed inferences: " << failed_inferences << std::endl;
        std::cout << "Average time per inference: " 
                  << total_time.count() / (successful_inferences + failed_inferences) 
                  << "s" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}