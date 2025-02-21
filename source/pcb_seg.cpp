#include "pcb_seg.hpp"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace PCBSEG {
Status convert_to_cpp_segresult(const py::object& py_segresult, SegResult& cpp_segresult) {
    try {
        // 获取 mask
        py::array_t<int> py_mask = py_segresult.attr("mask").cast<py::array_t<int>>();
        auto buf = py_mask.request();
        cv::Mat mask(buf.shape[0], buf.shape[1], CV_32S, buf.ptr);
        cpp_segresult.mask = mask.clone();

        // 获取 component_results
        py::list py_component_results = py_segresult.attr("component_results").cast<py::list>();
        for (const auto& py_result : py_component_results) {
            ComponentResult cpp_result;
            cpp_result.confidence = py_result.attr("confidence").cast<float>();
            cpp_result.label_ID = py_result.attr("label_ID").cast<int>();
            cpp_result.center_x = py_result.attr("center_x").cast<int>();
            cpp_result.center_y = py_result.attr("center_y").cast<int>();
            cpp_result.width = py_result.attr("width").cast<int>();
            cpp_result.height = py_result.attr("height").cast<int>();
            cpp_result.class_name = py_result.attr("class_name").cast<std::string>();
            cpp_result.part_name = py_result.attr("part_name").cast<std::string>();
            cpp_result.component_name = py_result.attr("component_name").cast<std::string>();
            cpp_segresult.component_results.push_back(cpp_result);
        }
        return Status::SUCCESS;
    } catch (const py::error_already_set &e) {
        std::cerr << "Error in convert_to_cpp_segresult: " << e.what() << std::endl;
        return Status::ERR_SYSTEM;
    } catch (const std::exception &e) {
        std::cerr << "Error in convert_to_cpp_segresult: " << e.what() << std::endl;
        return Status::ERR_UNKNOWN;
    }
}


SegmentationEvaluator::SegmentationEvaluator(const std::string& model_path, const std::string& label_path,int gpu_id, float gpu_mem_gb) 
    : model_path_(model_path),label_path_(label_path) {
    try {
        // 导入sys模块并修改sys.path
        py::module sys = py::module::import("sys");
        sys.attr("path").cast<py::list>().append("../sahi");
        sys.attr("path").cast<py::list>().append("../");
        
        // std::cout<<"here is the path"<<std::endl;
        // py::list path = sys.attr("path").cast<py::list>();
        // for (const auto& p : path) {
        //     std::cout << p.cast<std::string>() << std::endl;
        // }

        // 导入模块并指定GPU配置
        py::module model_module = py::module::import("segapi");
        py::object model_class = model_module.attr("SEGAPI");

        model_instance_ = model_class(model_path, label_path);
    } catch (const std::exception& e) {
        std::cerr << "Error initializing Evaluator: " << e.what() << std::endl;
        throw;
    }
}

SegmentationEvaluator::~SegmentationEvaluator() {}

Status SegmentationEvaluator::evaluateSingle(const cv::Mat& bmp_image,  SegResult& cpp_segresult) {
    try {
        if (bmp_image.empty()) {
            return Status::ERR_INPUT;
        }

        if (!bmp_image.isContinuous()) {
            return Status::ERR_INPUT;
        }

        if (bmp_image.channels() != 3) {
            return Status::ERR_INPUT;
        }

        // Python 代码执行需要 GIL
        {
            py::gil_scoped_acquire acquire;

            // 转换图像为numpy数组
            py::array_t<uint8_t> array;
            array = py::array_t<uint8_t>(
                {bmp_image.rows, bmp_image.cols, bmp_image.channels()}, 
                bmp_image.data
            );
            // 调用Python模型进行预测
           
            py::object py_result = model_instance_.attr("infer")(array);

            //转换结果为SegResult
            return convert_to_cpp_segresult(py_result, cpp_segresult);
            // // 转换结果为cv::Mat
            // if (py::isinstance<py::array>(py_result)) {
            //     py::array py_array = py_result.cast<py::array>();
            //     if (py_array.ndim() != 2) {
            //         return Status::ERR_LOGIC;
            //     }

            //     const ssize_t* shape = py_array.shape();
            //     mask = cv::Mat(shape[0], shape[1], CV_8UC1, py_array.mutable_data());
            //     mask = mask.clone(); // 创建深拷贝以确保数据所有权
            // } else {
            //     return Status::ERR_LOGIC;
            // }
        }

        return Status::SUCCESS;
    } catch (const py::error_already_set &e) {
        std::cerr << "Error in evaluateSingle: " << e.what() << std::endl;
        return Status::ERR_SYSTEM;
    } catch (const std::exception &e) {
        std::cerr << "Error in evaluateSingle: " << e.what() << std::endl;
        return Status::ERR_UNKNOWN;
    }
}

Status SegmentationEvaluator::evaluateBatch(const std::vector<cv::Mat>& bmp_images,
                               const std::vector<cv::Mat>& tiff_images,
                               std::vector<cv::Mat>& masks) {
    try {
        if (bmp_images.empty()) {
            return Status::ERR_INPUT;
        }

        masks.clear();

        // Python 代码执行需要 GIL
        {
            py::gil_scoped_acquire acquire;

            // 将所有图像转换为numpy数组列表
            py::list image_list;
            for (const auto& bmp_image : bmp_images) {
                if (bmp_image.empty() || !bmp_image.isContinuous() || bmp_image.channels() != 3) {
                    return Status::ERR_INPUT;
                }

                py::array_t<uint8_t> array({bmp_image.rows, bmp_image.cols, bmp_image.channels()}, 
                                         bmp_image.data);
                image_list.append(array);
            }

            // 调用Python的批处理方法
            py::object py_result = model_instance_.attr("predict_batch")(image_list);

            // 检查返回结果
            if (!py::isinstance<py::list>(py_result)) {
                return Status::ERR_LOGIC;
            }

            // 转换结果
            py::list result_list = py_result.cast<py::list>();
            for (size_t i = 0; i < result_list.size(); ++i) {
                py::array py_array = result_list[i].cast<py::array>();
                if (py_array.ndim() != 2) {
                    return Status::ERR_LOGIC;
                }

                const ssize_t* shape = py_array.shape();
                cv::Mat mask(shape[0], shape[1], CV_8UC1, py_array.mutable_data());
                masks.push_back(mask.clone());
            }
        }

        return Status::SUCCESS;
    } catch (const py::error_already_set &e) {
        std::cerr << "Error in evaluateBatch: " << e.what() << std::endl;
        return Status::ERR_SYSTEM;
    } catch (const std::exception &e) {
        std::cerr << "Error in evaluateBatch: " << e.what() << std::endl;
        return Status::ERR_UNKNOWN;
    }
}

Status SegmentationEvaluator::threadSafeInference(const cv::Mat& image,SegResult& cpp_segresult) {
    try {
        // 使用 RAII 方式管理 GIL
        {
            std::lock_guard<std::mutex> lock(gil_mutex_);  // 保护 GIL 操作
            acquireGIL();
        }
        
        // 执行推理
        auto status = evaluateSingle(image,cpp_segresult);
        
        {
            std::lock_guard<std::mutex> lock(gil_mutex_);
            releaseGIL();
        }
        
        return status;
    } catch (const std::exception& e) {
        std::cerr << "Thread inference error: " << e.what() << std::endl;
        {
            std::lock_guard<std::mutex> lock(gil_mutex_);
            releaseGIL();
        }
        return Status::ERR_UNKNOWN;
    }
}


Status SegmentationEvaluator::testInference(const cv::Mat& image, int thread_id, 
                              std::atomic<int>& successful_count,
                              std::atomic<int>& failed_count,
                              const std::atomic<bool>& stop_flag,
                              int iterations) {
    SegResult cpp_segresult;
    
    try {
        for (int i = 0; i < iterations && !stop_flag; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            
            auto status = threadSafeInference(image, cpp_segresult);
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            
            if (status == Status::SUCCESS) {
                successful_count++;
                std::cout << "Thread " << thread_id << " - Iteration " << i + 1 
                         << "/" << iterations 
                         << " completed in " << elapsed.count() << "s" << std::endl;
                
                if (i == 0) {
                    cv::imwrite("thread_" + std::to_string(thread_id) + "_result.bmp", cpp_segresult.mask);
                }
            } else {
                failed_count++;
                std::cerr << "Thread " << thread_id << " - Iteration " << i + 1 
                         << "/" << iterations << " failed" << std::endl;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Test inference error: " << e.what() << std::endl;
        return Status::ERR_UNKNOWN;
    }
}

} 