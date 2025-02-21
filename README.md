# PCB实例分割

## 环境配置

### CUDA & Cudnn 测试可运行版本
| CUDA Version | Python Version | Cudnn Version | GPU      |
| ------------ | -------------- | ------------- | -------- |
| CUDA 11.8    | 3.8            | 8.x           | RTX 3080 |


### Python 依赖安装

```bash

pip install onnx==1.17.0
pip install onnxruntime-gpu==1.18.1
pip install torch==2.0.0+cu118
pip install opencv-python
```

### 编译项目

```bash
cd Goldwire_Segmentation
rm -rf build  
mkdir build && cd build
cmake ..
make -j
```

```


clear
cd ..
rm -r build

mkdir build && cd build
cmake ..
make -j

./GoldWireSegmentation


```

### 运行项目

```bash
./GoldWireSegmentation
```



## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License - see the [LICENSE](LICENSE) file for details.

This means:
- ✅ You can freely use this code for non-commercial purposes
- ✅ You can modify and share this code
- ❌ You cannot use this code for commercial purposes
- ❌ You cannot sublicense or sell this code

