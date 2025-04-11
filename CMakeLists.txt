cmake_minimum_required(VERSION 3.14)

project(TRT_Engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(TENSORRT_ROOT "/usr/local")
set(TENSORRT_INCLUDE_DIRS "${TENSORRT_ROOT}/include")
set(TENSORRT_LIBRARIES
    "${TENSORRT_ROOT}/lib/libnvinfer.so"
    "${TENSORRT_ROOT}/lib/libnvinfer_plugin.so"
    "${TENSORRT_ROOT}/lib/libnvonnxparser.so"
    "${TENSORRT_ROOT}/lib/libnvparsers.so"
)

find_package(OpenCV REQUIRED)

include_directories(
    ${TENSORRT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB_RECURSE CPP_SOURCES
    "src/*.cpp"
    "src/utils/*.cpp"
    "src/model/*.cpp"
)

file(GLOB_RECURSE CUDA_SOURCES
    "src/*.cu"
)

add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

target_link_libraries(${PROJECT_NAME}
    ${TENSORRT_LIBRARIES}
    ${OpenCV_LIBS}
    cudart
    cudnn
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)
