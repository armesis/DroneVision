cmake_minimum_required(VERSION 3.10)
project(LiveCameraOpenCV_VSCode)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- OpenCV ---
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# --- ONNX Runtime ---
# IMPORTANT: replace this with your actual extraction path if it differs
set(ONNXRUNTIME_ROOTDIR "$ENV{HOME}/Downloads/onnxruntime_cpp/onnxruntime-linux-x64-gpu-1.20.0" CACHE PATH "Root directory of ONNX Runtime")

if(NOT EXISTS ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime_cxx_api.h)
    message(FATAL_ERROR "ONNX Runtime include directory not found at ${ONNXRUNTIME_ROOTDIR}/include. Please set ONNXRUNTIME_ROOTDIR correctly.")
endif()
if(NOT EXISTS ${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so)
    message(FATAL_ERROR "ONNX Runtime library not found at ${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so. Please set ONNXRUNTIME_ROOTDIR correctly.")
endif()

include_directories(${ONNXRUNTIME_ROOTDIR}/include)

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so")

add_library(onnxruntime_providers_cuda SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime_providers_cuda PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime_providers_cuda.so")

# --- Project static library with camera & drawing helpers ---
add_library(uavlib STATIC
    src/camera.cpp
    src/draw.cpp
    src/classnames.cpp
    src/print_shape.cpp
    src/onnx_model.cpp
    src/model_info.cpp
    src/preprocess.cpp
    src/infer.cpp)

# Public include path so anything that links against uavlib sees the headers
# They live under <root>/lib according to your folder tree
target_include_directories(uavlib
    PUBLIC
        ${PROJECT_SOURCE_DIR}/lib)

# uavlib itself depends only on OpenCV modules used in camera/draw code
# (extend this list if you start using more OpenCV modules later)
target_link_libraries(uavlib
    PUBLIC
        ${OpenCV_LIBS})

# --- Main executable ---
add_executable(live_camera_app src/main.cpp)

target_link_libraries(live_camera_app
    PRIVATE
        uavlib            # brings headers & OpenCV
        ${OpenCV_LIBS}
        onnxruntime
        onnxruntime_providers_cuda)

# --- Diagnostics ---
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "ONNX Runtime include path: ${ONNXRUNTIME_ROOTDIR}/include")
message(STATUS "ONNX Runtime library path: ${ONNXRUNTIME_ROOTDIR}/lib")
