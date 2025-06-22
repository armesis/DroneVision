cmake_minimum_required(VERSION 3.10)
project(LiveCameraOpenCV_VSCode)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- OpenCV ---
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# --- ONNX Runtime ---
# Set this path to where you extracted ONNX Runtime
# For example: /home/your_username/onnxruntime_cpp/onnxruntime-linux-x64-1.17.0
# IMPORTANT: Replace with your actual path!
set(ONNXRUNTIME_ROOTDIR "$ENV{HOME}/Downloads/onnxruntime_cpp/onnxruntime-linux-x64-gpu-1.20.0" CACHE PATH "Root directory of ONNX Runtime")

# Check if the directory exists
if(NOT EXISTS ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime_cxx_api.h)
    message(FATAL_ERROR "ONNX Runtime include directory not found at ${ONNXRUNTIME_ROOTDIR}/include. Please set ONNXRUNTIME_ROOTDIR correctly.")
endif()
if(NOT EXISTS ${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so)
    message(FATAL_ERROR "ONNX Runtime library not found at ${ONNXRUNTIME_ROOTDIR}/lib. Please set ONNXRUNTIME_ROOTDIR correctly.")
endif()

# Include ONNX Runtime headers
include_directories(${ONNXRUNTIME_ROOTDIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Define the ONNX Runtime library
# The actual .so file might have version numbers, but libonnxruntime.so is usually a symlink
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so"
)
# If using a specific versioned .so file, e.g., libonnxruntime.so.1.17.0:
# set_target_properties(onnxruntime PROPERTIES
#     IMPORTED_LOCATION "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so.1.17.0"
# )

# Example if libonnxruntime_providers_cuda.so is needed
add_library(onnxruntime_providers_cuda SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime_providers_cuda PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime_providers_cuda.so"
)
# # ... and then add it to target_link_libraries
# target_link_libraries(live_camera_app
#     ${OpenCV_LIBS}
#     onnxruntime
#     # onnxruntime_providers_cuda # If needed
# )


# --- Executable ---
add_executable(live_camera_app main.cpp)

add_executable(inference_service services/inference_service.cpp)
add_executable(capture_service services/capture_service.cpp)

# Link libraries
target_link_libraries(live_camera_app
    ${OpenCV_LIBS}
    onnxruntime # Link against the ONNX Runtime library we defined
    onnxruntime_providers_cuda
)

target_link_libraries(inference_service
    ${OpenCV_LIBS}
    onnxruntime
    onnxruntime_providers_cuda
)

target_link_libraries(capture_service
    ${OpenCV_LIBS}
)

# Optional: Print messages
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "ONNX Runtime status:")
message(STATUS "    include path: ${ONNXRUNTIME_ROOTDIR}/include")
message(STATUS "    library path: ${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so")