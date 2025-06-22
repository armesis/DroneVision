

// YOLO_NAS runner ----------------------------------------------







#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <cstdint>

#include <onnxruntime_cxx_api.h>

// COCO Class Names
const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"};

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

    // --- ONNX Runtime Setup -------------------------------------------------------------------------------------------------------------------------------------------

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO-NAS-App-GPU");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    OrtCUDAProviderOptions cuda_options{};
    try {
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "INFO: Attempting to use CUDA execution provider." << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "WARNING: Could not append CUDA execution provider: " << e.what() << std::endl;
        std::cerr << "INFO: Will fall back to CPU or other available providers." << std::endl;
    }

    const char* model_path = "yolo11n.onnx";
    Ort::Session session(nullptr);
    try {
        session = Ort::Session(env, model_path, session_options);
        std::cout << "ONNX model loaded successfully: " << model_path << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "ERROR loading ONNX model: " << e.what() << std::endl;
        return -1;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    // --- ONNX Runtime Setup -------------------------------------------------------------------------------------------------------------------------------------------

    // --- Get input node details ---------------------------------------------------------------------------------------------------------------------------------------
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
    std::vector<int64_t> input_dims = input_tensor_info.GetShape();
    std::cout << "Model Input Dims Reported: " << print_shape(input_dims) << std::endl;

    // --- Resolve Model Input Dimensions ---
    // Define the default size you used during the Python export (e.g., imgsz=640)
    const int64_t DEFAULT_MODEL_WIDTH = 640;
    const int64_t DEFAULT_MODEL_HEIGHT = 640;

    // Use reported dimension if it's positive, otherwise use the default.
    const int64_t batch_size = (input_dims[0] == -1 || input_dims[0] == 0) ? 1 : input_dims[0];
    const int64_t channels = input_dims[1];
    const int64_t height = (input_dims[2] == -1 || input_dims[2] == 0) ? DEFAULT_MODEL_HEIGHT : input_dims[2];
    const int64_t width = (input_dims[3] == -1 || input_dims[3] == 0) ? DEFAULT_MODEL_WIDTH : input_dims[3];
    
    std::cout << "Using Resolved Input Dims: { Batch: " << batch_size << ", Channels: " << channels
              << ", Height: " << height << ", Width: " << width << " }" << std::endl;


    // --- Get output node details -------------------------------------------------------------------------------------------------------------------------------------
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<std::string> output_node_names_str(num_output_nodes);
    std::vector<const char*> output_node_names_ptr(num_output_nodes);
    std::cout << "Number of output nodes: " << num_output_nodes << std::endl;
    for (size_t i = 0; i < num_output_nodes; ++i) {
        Ort::AllocatedStringPtr output_name_alloc_loop = session.GetOutputNameAllocated(i, allocator);
        output_node_names_str[i] = output_name_alloc_loop.get();
        output_node_names_ptr[i] = output_node_names_str[i].c_str();

        Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(i);
        auto output_tensor_info_loop = output_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_dims_loop = output_tensor_info_loop.GetShape();
        std::cout << "Output " << i << " Name: " << output_node_names_ptr[i] << " Dims: " << print_shape(output_dims_loop) << std::endl;
    }

    // --- OpenCV Camera Setup ------------------------------------------------------------------------------------------------------------------------------------------
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return -1;
    }
    // Request camera resolution, but actual resolution will be in frame.cols/rows
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640); 
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame, resized_frame_rgb, preprocessed_frame;
    std::vector<float> input_tensor_values(batch_size * channels * height * width);

    std::string window_name = "YOLO-NAS ONNX C++ (Debug Pre-NMS Boxes)";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    double fps = 0.0;
    int64_t tick_start; // For storing start tick count
    int frame_counter_fps = 0; // To average FPS over a few frames for stability
    double total_time_fps = 0.0;

    std::cout << "Press 'q' in the camera window to quit." << std::endl;

    int frame_count_for_debug = 0; // For limiting debug prints
    // ... (after cv::namedWindow and before the while loop) ...

    

    while (true) {
        
        tick_start = cv::getTickCount(); // Start timer for this frame
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "ERROR: Captured empty frame" << std::endl;
            break;
        }
        // Get original frame dimensions for this specific frame
        float frame_width_orig = static_cast<float>(frame.cols);
        float frame_height_orig = static_cast<float>(frame.rows);

        // 1. Preprocessing
        // 'width' and 'height' here are model input dimensions (e.g., 640x640)
        cv::resize(frame, resized_frame_rgb, cv::Size(width, height));
        cv::cvtColor(resized_frame_rgb, resized_frame_rgb, cv::COLOR_BGR2RGB);
        resized_frame_rgb.convertTo(preprocessed_frame, CV_32F, 1.0 / 255.0);

        for (int c = 0; c < channels; ++c) {
            for (int h_img = 0; h_img < height; ++h_img) {
                for (int w_img = 0; w_img < width; ++w_img) {
                    input_tensor_values[c * height * width + h_img * width + w_img] =
                        preprocessed_frame.at<cv::Vec3f>(h_img, w_img)[c];
                }
            }
        }

        // 2. Create Input Tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> current_input_dims = {batch_size, channels, height, width};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                  input_tensor_values.data(),
                                                                  input_tensor_values.size(),
                                                                  current_input_dims.data(),
                                                                  current_input_dims.size());
        
        // 3. Run Inference
        std::vector<Ort::Value> output_tensors;
        try {
             output_tensors = session.Run(Ort::RunOptions{nullptr},
                                         input_node_names, &input_tensor, 1,
                                         output_node_names_ptr.data(), num_output_nodes);
        } catch (const Ort::Exception& e) {
            std::cerr << "ERROR during inference: " << e.what() << std::endl;
            cv::imshow(window_name, frame);
            if (cv::waitKey(1) == 'q') break;
            continue;
        }

        // 4. Post-processing
        const float obj_threshold = 0.25f; // Objectness threshold for YOLOv11
        const float conf_threshold = 0.25f; // Final confidence threshold
        const float nms_threshold = 0.45f;  // Your NMS threshold

        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        // Check if we have the expected single output tensor
        if (output_tensors.size() == 1 && output_tensors[0].IsTensor()) {
            const float* all_data_ptr = output_tensors[0].GetTensorData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

            // output_shape should be [batch_size, num_potential_detections, num_attributes]
            // For Ultralytics YOLO export without postprocessing the last dimension is
            // (num_classes + 5) -> [objectness, cx, cy, w, h, class scores]
            const int64_t num_potential_detections = output_shape[1];
            const int64_t attributes_per_detection = output_shape[2];

            // Infer number of classes from the output shape
            const int inferred_num_classes = static_cast<int>(attributes_per_detection - 5);

            if (inferred_num_classes != CLASS_NAMES.size()) {
                std::cerr << "WARNING: Model's inferred number of classes (" << inferred_num_classes
                          << ") does not match CLASS_NAMES size (" << CLASS_NAMES.size()
                          << "). Check your CLASS_NAMES vector or model output structure." << std::endl;
                // Decide how to handle this: proceed with inferred_num_classes and risk label mismatch,
                // or use CLASS_NAMES.size() and risk reading out of bounds if inferred_num_classes is smaller.
                // For safety, it's often better to use the smaller of the two for loops,
                // but ensure CLASS_NAMES is correct for the model.
            }
            // Let's proceed assuming CLASS_NAMES.size() is the true number of classes the model was trained for,
            // and the model output reflects this.
            const int num_classes_to_iterate = static_cast<int>(CLASS_NAMES.size());


            float model_input_width_float = static_cast<float>(width);  // Model input width (e.g., 640)
            float model_input_height_float = static_cast<float>(height); // Model input height (e.g., 640)

            // Each detection row is [obj, cx, cy, w, h, class_scores...]
            for (int i = 0; i < num_potential_detections; ++i) {
                const float* current_detection_data = all_data_ptr + i * attributes_per_detection;

                float objectness = current_detection_data[0];
                if (objectness < obj_threshold) {
                    continue;
                }

                float max_class_score = 0.0f;
                int best_class_id = -1;
                const float* class_scores_ptr = current_detection_data + 5;
                for (int j = 0; j < num_classes_to_iterate && j < inferred_num_classes; ++j) {
                    if (class_scores_ptr[j] > max_class_score) {
                        max_class_score = class_scores_ptr[j];
                        best_class_id = j;
                    }
                }

                float conf = objectness * max_class_score;
                if (conf < conf_threshold) {
                    continue;
                }

                float cx_model = current_detection_data[1];
                float cy_model = current_detection_data[2];
                float w_model  = current_detection_data[3];
                float h_model  = current_detection_data[4];

                float x1_model = cx_model - w_model / 2.0f;
                float y1_model = cy_model - h_model / 2.0f;

                float frame_width_orig = static_cast<float>(frame.cols);
                float frame_height_orig = static_cast<float>(frame.rows);

                int x1_orig = static_cast<int>(x1_model * (frame_width_orig / model_input_width_float));
                int y1_orig = static_cast<int>(y1_model * (frame_height_orig / model_input_height_float));
                int box_width_orig = static_cast<int>(w_model * (frame_width_orig / model_input_width_float));
                int box_height_orig = static_cast<int>(h_model * (frame_height_orig / model_input_height_float));

                cv::Rect box_orig(x1_orig, y1_orig, box_width_orig, box_height_orig);
                box_orig &= cv::Rect(0, 0, static_cast<int>(frame_width_orig), static_cast<int>(frame_height_orig));

                if (box_orig.width > 0 && box_orig.height > 0) {
                    bboxes.push_back(box_orig);
                    scores.push_back(conf);
                    class_ids.push_back(best_class_id);
                }
            }

            // --- Per-Class NMS (Your existing logic should still work if bboxes, scores, class_ids are filled correctly) ---
            std::vector<int> final_kept_indices;
            std::vector<int> unique_class_ids = class_ids;
            std::sort(unique_class_ids.begin(), unique_class_ids.end());
            unique_class_ids.erase(std::unique(unique_class_ids.begin(), unique_class_ids.end()), unique_class_ids.end());

            for (int current_class_id : unique_class_ids) {
                if (current_class_id < 0) continue;

                std::vector<cv::Rect> bboxes_for_class;
                std::vector<float> scores_for_class;
                std::vector<int> original_indices_for_class;

                for (size_t i = 0; i < bboxes.size(); ++i) {
                    if (class_ids[i] == current_class_id) {
                        bboxes_for_class.push_back(bboxes[i]);
                        scores_for_class.push_back(scores[i]);
                        original_indices_for_class.push_back(static_cast<int>(i));
                    }
                }

                if (!bboxes_for_class.empty()) {
                    std::vector<int> nms_result_indices_for_class;
                    cv::dnn::NMSBoxes(bboxes_for_class, scores_for_class, conf_threshold, nms_threshold, nms_result_indices_for_class);

                    for (int temp_idx : nms_result_indices_for_class) {
                        final_kept_indices.push_back(original_indices_for_class[temp_idx]);
                    }
                }
            }
            
            // --- Draw the final Detections using final_kept_indices ---
            // std::cout << "Frame " << frame_count_for_debug << ": Num boxes AFTER PER-CLASS NMS: " << final_kept_indices.size() << std::endl;
            for (int idx : final_kept_indices) { 
                cv::Rect box = bboxes[idx]; // Use original bboxes vector with the filtered indices
                float score = scores[idx];
                int class_id = class_ids[idx];
            
                if (class_id >= 0 && class_id < CLASS_NAMES.size()) { 
                    cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2); 
                    std::string label = CLASS_NAMES[class_id] + " " + cv::format("%.2f", score);
                    // ... (putText code) ...
                }
            }
            
            // --- Draw the final Detections using final_kept_indices ---
            // This section remains largely the same, drawing from bboxes, scores, class_ids using final_kept_indices
            for (int idx : final_kept_indices) {
                cv::Rect box = bboxes[idx];
                float score = scores[idx];
                int class_id = class_ids[idx];

                if (class_id >= 0 && class_id < CLASS_NAMES.size()) {
                    cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2); // Green boxes for final NMS
                    std::string label = CLASS_NAMES[class_id] + " " + cv::format("%.2f", score);
                    int baseLine;
                    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
                    int top = std::max(box.y, labelSize.height);
                    cv::putText(frame, label, cv::Point(box.x, top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 2); // Black text for better visibility
                    cv::putText(frame, label, cv::Point(box.x, top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1); // Green text
                }
            }
            // --- END Section A ---


            // --- Section B: Perform Non-Maximum Suppression (NMS) (Currently Commented Out) ---
            /*
            std::vector<int> nms_indices;
            if (!bboxes.empty()) {
                 cv::dnn::NMSBoxes(bboxes, scores, conf_threshold, nms_threshold, nms_indices);
            }
            std::cout << "Frame " << frame_count_for_debug << ": Num boxes AFTER NMS: " << nms_indices.size() << std::endl;
            for (int idx : nms_indices) {
                cv::Rect box = bboxes[idx];
                float score = scores[idx];
                int class_id = class_ids[idx];

                if (class_id >= 0 && class_id < CLASS_NAMES.size()) {
                    cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2); // Green boxes for NMS
                    std::string label = CLASS_NAMES[class_id] + " " + cv::format("%.2f", score);
                    int baseLine;
                    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
                    int top = std::max(box.y, labelSize.height);
                    cv::putText(frame, label, cv::Point(box.x, top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                }
            }
            */
            // --- END Section B ---

        } else {
             cv::putText(frame, "Output tensor issue or no detections", cv::Point(10, 60), // Moved down
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }

        // Calculate FPS
        double frame_time = (cv::getTickCount() - tick_start) / cv::getTickFrequency();

        // Averaged FPS for more stability
        total_time_fps += frame_time;
        frame_counter_fps++;
        if (frame_counter_fps >= 10) { // Average over 10 frames
            fps = frame_counter_fps / total_time_fps;
            frame_counter_fps = 0;
            total_time_fps = 0.0;
        }

        // Display FPS on the frame
        std::string fps_text = "FPS: " + cv::format("%.2f", fps);
        cv::Point fps_position(10, 30); // Top-left corner
        cv::putText(frame, fps_text, fps_position, 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                    

        cv::imshow(window_name, frame);
        if (frame_count_for_debug < 2) {
            frame_count_for_debug++;
        }

        if (cv::waitKey(1) == 'q') {
            std::cout << "Quitting..." << std::endl;
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}