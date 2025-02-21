# PCB_Seg api文档
### 模型类
```cpp
class SegmentationEvaluator {
public:
    //初始化
    SegmentationEvaluator (const std::string& model_path, const std::string& label_path,int gpu_id, float gpu_mem_gb);

    //推理函数,返回PCBSEG Status为程序运行状态, 分割结果保存在形式参数cpp_segresult之中
    Status evaluateSingle(const cv::Mat& bmp_image,  SegResult& cpp_segresult);
}
```

### 输出类
```cpp

// 用于进行单个元件信息的存储
struct ComponentResult {
    float confidence;  // 当前元件的识别的置信度
    
    int center_x;      // 当前识别的元件在原图的中心点 X
    int center_y;      // 当前识别的元件在原图的中心点 Y
    int width;         // 当前识别的元件的宽度
    int height;        // 当前识别的元件的高度

    // 原始的模型的分类方案不够好需要另外设计, 故下面各个成员暂未实现, 其中三个string使用"dummy"来填充, label_ID暂时使用打标签时设定的编号
    int label_ID;      // 当前元件的大类的类别 ID
    std::string class_name;  // 当前识别的元件的大类  例如：电感、电容、电阻、引脚、焊盘等
    std::string part_name;   // 当前识别的元件的料号名称  例如：电阻下面有：R0201、R01005等
    std::string component_name;  // 当前识别的元件的名称 例如R0201下面是板子上有 R0201-1号电阻 R0201-2号电阻
};

// 用于存放总的分割结果类,是一张PCB的FOV的结果
class SegResult {
public:
    cv::Mat mask;  // cv mat 格式的推理时候放进去原图大小一致的mask图，对应的单个分割区域到元件名称级别，主要用于进行检测时候的区域分割和显示，方便查看分割结果是否准确
    std::vector<ComponentResult> component_results;  // 存放每个元件的推理结果
};

```
### 状态类
```cpp
namespace PCBSEG {
    enum class Status {
        SUCCESS = 0,        // 成功
        ERR_INPUT,         // 输入错误
        ERR_SYSTEM,        // 系统错误
        ERR_LOGIC,         // 逻辑错误
        ERR_PROTECTION,    // 保护错误
        ERR_INITIALIZE,    // 初始化错误
        ERR_TIMEOUT,       // 超时错误
        ERR_FATAL,         // 致命错误
        ERR_UNKNOWN        // 未知错误
    };
} 
```

