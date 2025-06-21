#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <numeric> // For std::iota and std::max_element
#include <algorithm> // For std::max_element
#include <sstream>   // For std::stringstream
#include <iomanip>   // For std::fixed and std::setprecision

#include <onnxruntime_cxx_api.h>

// CLASS_NAMES for LeNet-5 (digits 0-9)
const std::vector<std::string> CLASS_NAMES = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

// Helper function to pretty print tensor shapes
std::string print_shape(const std::vector<int64_t>& v) {
    std::stringstream ss;
    ss << "{";
    for (size_t i = 0; i < v.size(); ++i) {
        ss << v[i] << (i == v.size() - 1 ? "" : ", ");
    }
    ss << "}";
    return ss.str();
}

int main() {
    // --- ONNX Runtime Setup ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "LeNet-5-App"); // Updated App Name
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Optional: Enable CUDA, if available and desired
    OrtCUDAProviderOptions cuda_options{};
    try {
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "INFO: Attempting to use CUDA execution provider." << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "WARNING: Could not append CUDA execution provider: " << e.what() << std::endl;
        std::cerr << "INFO: Will fall back to CPU or other available providers." << std::endl;
    }

    const char* model_path = "lenet5_emnist_digits.onnx"; // User specified model path
    Ort::Session session(nullptr);
    try {
        session = Ort::Session(env, model_path, session_options);
        std::cout << "ONNX model loaded successfully: " << model_path << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "ERROR loading ONNX model: " << e.what() << std::endl;
        return -1;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    // Get input node details
    size_t num_input_nodes = session.GetInputCount();
    if (num_input_nodes != 1) {
        std::cerr << "ERROR: Expected 1 input node, but got " << num_input_nodes << std::endl;
        return -1;
    }
    Ort::AllocatedStringPtr input_name_alloc = session.GetInputNameAllocated(0, allocator);
    const char* input_node_names[] = {input_name_alloc.get()};
    std::cout << "Input Name: " << input_node_names[0] << std::endl;

    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = input_tensor_info.GetShape(); // e.g. {-1, 1, 32, 32}
    std::cout << "Expected Model Input Dims: " << print_shape(input_dims) << std::endl;

    // Determine batch_size, channels, height, width for inference
    // For LeNet, input_dims[0] is often -1 (dynamic batch). We'll use batch_size = 1 for real-time inference.
    const int64_t infer_batch_size = 1; // We process one frame/ROI at a time
    const int64_t model_channels = (input_dims.size() > 1 && input_dims[1] > 0) ? input_dims[1] : 1; // Should be 1 for LeNet
    const int64_t model_height = (input_dims.size() > 2 && input_dims[2] > 0) ? input_dims[2] : 32; // Should be 32
    const int64_t model_width = (input_dims.size() > 3 && input_dims[3] > 0) ? input_dims[3] : 32;  // Should be 32

    std::cout << "Using for Inference: Batch=" << infer_batch_size << ", Channels=" << model_channels
              << ", Height=" << model_height << ", Width=" << model_width << std::endl;

    if (model_channels != 1) {
        std::cerr << "WARNING: Model expects " << model_channels << " channels, but LeNet-5 typically uses 1 (grayscale). Ensure preprocessing is correct." << std::endl;
    }


    // Get output node details
    size_t num_output_nodes = session.GetOutputCount();
    if (num_output_nodes != 1) {
        std::cerr << "WARNING: Expected 1 output node for classification, but got " << num_output_nodes << std::endl;
        // Continue if possible, assuming the first output is the classification scores
    }
    std::vector<std::string> output_node_names_str(num_output_nodes);
    std::vector<const char*> output_node_names_ptr(num_output_nodes);
    std::cout << "Number of output nodes: " << num_output_nodes << std::endl;

    Ort::AllocatedStringPtr output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    output_node_names_str[0] = output_name_alloc.get(); // Store it
    output_node_names_ptr[0] = output_node_names_str[0].c_str(); // Get C-string pointer

    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_dims = output_tensor_info.GetShape(); // e.g. {-1, 10} or {1,10}
    std::cout << "Output 0 Name: " << output_node_names_ptr[0] << " Dims: " << print_shape(output_dims) << std::endl;

    if (output_dims.size() != 2 || (output_dims[0] != -1 && output_dims[0] != infer_batch_size) || output_dims[1] != CLASS_NAMES.size()) {
        std::cerr << "WARNING: Output dimensions " << print_shape(output_dims)
                  << " might not match expected {-1, " << CLASS_NAMES.size() << "} or {" << infer_batch_size << ", " << CLASS_NAMES.size()
                  << "} for LeNet-5 classification." << std::endl;
    }


    // --- OpenCV Camera Setup ---
    cv::VideoCapture cap(0); // Open default camera
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame;
    std::vector<float> input_tensor_values(infer_batch_size * model_channels * model_height * model_width);

    std::string window_name = "LeNet-5 Digit Recognition ONNX C++";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    double fps = 0.0;
    int64 tick_start;
    int frame_counter_fps = 0;
    double total_time_fps = 0.0;

    std::cout << "Press 'q' in the camera window to quit." << std::endl;

    while (true) {
        tick_start = cv::getTickCount();
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "ERROR: Captured empty frame" << std::endl;
            break;
        }

        // --- Define ROI in the center of the frame ---
        int roi_edge_length = std::min(frame.cols, frame.rows) / 2; // ROI will be square
        // Ensure roi_edge_length is reasonable, e.g., at least model_width and model_height.
        roi_edge_length = std::max(roi_edge_length, (int)std::max(model_width, model_height) * 2); 
        roi_edge_length = std::min({roi_edge_length, frame.cols, frame.rows}); // Cap by frame dimensions


        int roi_x = (frame.cols - roi_edge_length) / 2;
        int roi_y = (frame.rows - roi_edge_length) / 2;
        cv::Rect roi_definition(roi_x, roi_y, roi_edge_length, roi_edge_length);
        cv::Mat frame_roi = frame(roi_definition).clone(); // Clone ROI to avoid issues if frame is modified

        // --- 1. Preprocessing for LeNet-5 ---
        cv::Mat gray_roi, resized_input_img, preprocessed_frame;
        cv::cvtColor(frame_roi, gray_roi, cv::COLOR_BGR2GRAY);

        // Resize to model's expected input size (e.g., 32x32)
        cv::resize(gray_roi, resized_input_img, cv::Size(model_width, model_height));

        // Normalize to [0, 1]. LeNet models might also expect other normalizations (e.g. specific mean/std).
        // This is a common simple normalization. Adjust if your model needs something else.
        resized_input_img.convertTo(preprocessed_frame, CV_32F, 1.0 / 255.0);

        // Populate input_tensor_values (data is already [H][W] for a single channel image)
        // The model expects {batch_size, channels, height, width} = {1, 1, 32, 32}
        if (preprocessed_frame.isContinuous()) {
            memcpy(input_tensor_values.data(), preprocessed_frame.data, preprocessed_frame.total() * preprocessed_frame.elemSize());
        } else {
            // If not continuous, copy row by row (slower)
            for (int r = 0; r < preprocessed_frame.rows; ++r) {
                memcpy(input_tensor_values.data() + r * preprocessed_frame.cols,
                       preprocessed_frame.ptr<float>(r),
                       preprocessed_frame.cols * sizeof(float));
            }
        }


        // --- 2. Create Input Tensor ---
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> current_input_dims = {infer_batch_size, model_channels, model_height, model_width};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                  input_tensor_values.data(),
                                                                  input_tensor_values.size(),
                                                                  current_input_dims.data(),
                                                                  current_input_dims.size());
        
        // --- 3. Run Inference ---
        std::vector<Ort::Value> output_tensors;
        try {
             output_tensors = session.Run(Ort::RunOptions{nullptr},
                                         input_node_names, &input_tensor, 1, // 1 input tensor
                                         output_node_names_ptr.data(), num_output_nodes); // Use the actual num_output_nodes
        } catch (const Ort::Exception& e) {
            std::cerr << "ERROR during inference: " << e.what() << std::endl;
            // Draw ROI box even if inference fails, for visual feedback
            cv::rectangle(frame, roi_definition, cv::Scalar(0, 0, 255), 2); // Red box for error
            cv::imshow(window_name, frame);
            if (cv::waitKey(1) == 'q') break;
            continue;
        }

        // --- 4. Post-processing for LeNet-5 (Classification) ---
        std::string predicted_label_text = "Pred: N/A";
        if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
            const float* score_data = output_tensors[0].GetTensorData<float>();
            auto current_output_dims_info = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape(); // e.g., {1, 10}

            if (current_output_dims_info.size() == 2 && current_output_dims_info[0] == infer_batch_size) {
                int num_classes_from_model = current_output_dims_info[1];
                
                if (num_classes_from_model == CLASS_NAMES.size()) {
                    float max_score_val = -std::numeric_limits<float>::infinity();
                    int predicted_idx = -1;

                    for(int i = 0; i < num_classes_from_model; ++i) {
                        if(score_data[i] > max_score_val) {
                            max_score_val = score_data[i];
                            predicted_idx = i;
                        }
                    }

                    if (predicted_idx != -1) {
                        // Softmax output interpretation (optional, if output is not already probabilities)
                        // For raw logits, the argmax is usually sufficient for the predicted class.
                        // If you need probabilities:
                        // std::vector<float> probabilities(num_classes_from_model);
                        // float sum_exp = 0.0f;
                        // for(int i=0; i<num_classes_from_model; ++i) {
                        //     probabilities[i] = std::exp(score_data[i]);
                        //     sum_exp += probabilities[i];
                        // }
                        // for(int i=0; i<num_classes_from_model; ++i) {
                        //     probabilities[i] /= sum_exp;
                        // }
                        // max_score_val = probabilities[predicted_idx]; // update score to be probability

                        std::stringstream ss_label;
                        ss_label << "Pred: " << CLASS_NAMES[predicted_idx]
                                 << " (" << std::fixed << std::setprecision(2) << max_score_val << ")";
                        predicted_label_text = ss_label.str();
                    }
                } else {
                     predicted_label_text = "Output class mismatch";
                }
            } else {
                 predicted_label_text = "Unexpected output shape";
            }
        } else {
            predicted_label_text = "Output tensor error";
        }
        
        // Draw the ROI box on the original frame
        cv::rectangle(frame, roi_definition, cv::Scalar(0, 255, 0), 2); // Green box for ROI
        // Display the prediction text
        cv::putText(frame, predicted_label_text, cv::Point(roi_x, std::max(0, roi_y - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);


        // Calculate and Display FPS
        double frame_time = (cv::getTickCount() - tick_start) / cv::getTickFrequency();
        total_time_fps += frame_time;
        frame_counter_fps++;
        if (frame_counter_fps >= 10) {
            fps = frame_counter_fps / total_time_fps;
            frame_counter_fps = 0;
            total_time_fps = 0.0;
        }
        std::string fps_text = "FPS: " + cv::format("%.2f", fps);
        cv::putText(frame, fps_text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                    
        cv::imshow(window_name, frame);

        if (cv::waitKey(1) == 'q') {
            std::cout << "Quitting..." << std::endl;
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    // Ort::AllocatedStringPtr will be deallocated automatically.
    return 0;
}