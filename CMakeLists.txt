cmake_minimum_required(VERSION 3.12)
project(PCB_Seg)

set(CMAKE_CXX_STANDARD 14)

# 添加线程支持
find_package(Threads REQUIRED)

# 添加pybind11子模块
add_subdirectory(third_party/pybind11)

# 查找Python安装
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

include_directories(include)
file(GLOB SOURCES "source/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

# 链接库时添加 Threads::Threads
target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
    pybind11::embed 
    ${Python3_LIBRARIES} 
    ${OpenCV_LIBS}
    Threads::Threads
)