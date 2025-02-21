#pragma once
#include <vector>
#include <string>
namespace PCBSEG{
class SegmentationEvalConfig {
public:
    // Attributes
    std::vector<std::string> modelPaths; // Array of model paths
    float confThreshold; // Confidence threshold for segmentation
    float iouThreshold;  // IOU threshold for NMS (if applicable)
    bool useCudaGraph;

     // 新增默认构造函数
    SegmentationEvalConfig()
        : modelPaths({}), confThreshold(0.25), iouThreshold(0.65), useCudaGraph(false) {}
    // Constructor
    SegmentationEvalConfig(const std::vector<std::string>& modelPaths, float confThreshold = 0.25, float iouThreshold = 0.65,bool useCudaGraph=false)
        : modelPaths(modelPaths), confThreshold(confThreshold), iouThreshold(iouThreshold),useCudaGraph(useCudaGraph) {}
    
};
// 用于进行单个元件信息的存储
struct ComponentResult {
    float confidence;  // 当前元件的识别的置信度
    int label_ID;      // 当前元件的大类的类别 ID
    int center_x;      // 当前识别的元件在原图的中心点 X
    int center_y;      // 当前识别的元件在原图的中心点 Y
    int width;         // 当前识别的元件的宽度
    int height;        // 当前识别的元件的高度
    std::string class_name;  // 当前识别的元件的大类  例如：电感、电容、电阻、引脚、焊盘等
    std::string part_name;   // 当前识别的元件的料号名称  例如：电阻下面有：R0201、R01005等
    std::string component_name;  // 当前识别的元件的名称 例如R0201下面是板子上有 R0201-1号电阻 R0201-2号电阻
};

// 用于存放总的结果类
class SegResult {
public:
    cv::Mat mask;  // cv mat 格式的推理时候放进去原图大小一致的mask图，对应的单个分割区域到元件名称级别，主要用于进行检测时候的区域分割和显示，方便查看分割结果是否准确
    std::vector<ComponentResult> component_results;  // 存放每个元件的推理结果
};
}

