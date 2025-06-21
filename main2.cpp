#include <opencv2/opencv.hpp>       // Main OpenCV header
#include <iostream>                 // For standard I/O
#include <vector>                   // For std::vector
#include <onnxruntime_cxx_api.h>    // ONNX Runtime C++ API

int main() {
    // --- ONNX Runtime Setup ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO-NAS-App"); // Initialize ONNX Runtime environment
    Ort::SessionOptions session_options; // Create session options
    session_options.SetIntraOpNumThreads(1); // Example: configure threads

    // Optional: Enable CUDA, DirectML, etc. For now, we'll use CPU.
    // session_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;

    try {
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    std::cout << "INFO: Attempting to use CUDA execution provider." << std::endl;
} catch (const Ort::Exception& e) {
    std::cerr << "WARNING: Could not append CUDA execution provider: " << e.what() << std::endl;
    std::cerr << "INFO: Will fall back to CPU or other available providers." << std::endl;
    // If CUDA is not available or there's an issue, it might fall back.
    // You might want to make this a fatal error if CUDA is strictly required.
}

    // Path to your ONNX model file (should be in the same directory as the executable or provide full path)
    
    const char* model_path = "./yolonas_s.onnx";
    Ort::Session session(nullptr); // Declare session variable before try-catch

    try {
        session = Ort::Session(env, model_path, session_options); // Create a session and load the model
        std::cout << "ONNX model loaded successfully: " << model_path << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "ERROR loading ONNX model: " << e.what() << std::endl;
        return -1;
    }

    // Get model input node details (optional for now, but useful later)
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<std::vector<int64_t>> input_node_dims;

    std::cout << "Number of inputs: " << num_input_nodes << std::endl;
    for (size_t i = 0; i < num_input_nodes; i++) {
        // C-style string from Ort::AllocatedStringPtr
        Ort::AllocatedStringPtr name_alloc_str = session.GetInputNameAllocated(i, allocator);
        input_node_names[i] = name_alloc_str.get(); // name_alloc_str will release memory on destruction
        std::cout << "Input " << i << " : name = " << input_node_names[i];

        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> current_dims = tensor_info.GetShape();
        input_node_dims.push_back(current_dims);

        std::cout << " dims = {";
        for (size_t j = 0; j < current_dims.size(); ++j) {
            std::cout << current_dims[j] << (j == current_dims.size() - 1 ? "" : ", ");
        }
        std::cout << "}" << std::endl;
    }
    // We'll stop here for model loading. Camera loop will be re-added later.


    // --- OpenCV Camera Setup (from previous step) ---
    int camera_index = 0;
    cv::VideoCapture cap(camera_index);

    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open camera with index " << camera_index << std::endl;
        if (session) { /* session is valid */ } // just to use session to avoid unused variable warning for now
        return -1;
    }

    double frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Camera resolution: " << frame_width << "x" << frame_height << std::endl;

    std::string window_name = "Live Camera Feed (VS Code with ONNX)";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::Mat frame;
    std::cout << "Press 'q' in the camera window to quit." << std::endl;

    while (true) {
        bool success = cap.read(frame);
        if (!success) {
            std::cerr << "ERROR: Could not read a frame from camera." << std::endl;
            break;
        }
        if (frame.empty()) {
            std::cerr << "WARNING: Captured an empty frame." << std::endl;
            continue;
        }

        // FOR NOW: We are not doing any inference yet, just displaying the frame
        cv::imshow(window_name, frame);

        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 27) {
            std::cout << "Quitting..." << std::endl;
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}