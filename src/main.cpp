
// YOLO_NAS runner ----------------------------------------------

#include "camera.hpp"
#include "draw.hpp"
#include "classnames.hpp"
#include "print_shape.hpp"
#include "onnx_model.hpp"
#include "model_info.hpp"
#include "preprocess.hpp"
#include "infer.hpp"
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
const auto& CLASS_NAMES = cocoClassNames();

int main() {

    // --- ONNX Runtime Setup -------------------------------------------------------------------------------------------------------------------------------------------

    OnnxModel model("./../models/yolo11n.onnx"); 
    auto& session   = model.session();               // if you need direct access
    auto& allocator = model.allocator();

    // --- Get input node details ---------------------------------------------------------------------------------------------------------------------------------------

    modelutil::InputInfo input = modelutil::inspectInput(session, allocator);

    // the resolved dimensions are now in input.dims

    int64_t width     = input.dims.width;
    int64_t height    = input.dims.height;
    int64_t channels  = input.dims.channels;
    int64_t batch     = input.dims.batch;

    // --- Get output node details -------------------------------------------------------------------------------------------------------------------------------------

    auto output = modelutil::reportOutputs(session, allocator);

    // --- OpenCV Camera Setup ------------------------------------------------------------------------------------------------------------------------------------------

    Camera cam(0, 640, 480);
    cv::Mat frame;

    const std::string window_name = "Live camera feed";

    double fps = 0.0;
    int64_t tick_start; // For storing start tick count
    int frame_counter_fps = 0; // To average FPS over a few frames for stability
    double total_time_fps = 0.0;

    std::cout << "Press 'q' in the camera window to quit." << std::endl;

    int frame_count_for_debug = 0; // For limiting debug prints

    // ... (after cv::namedWindow and before the while loop) ...
    
    while (true) {
        if (!cam.grabFrame(frame)) break;
        tick_start = cv::getTickCount(); // Start timer for this frame
        if (frame.empty()) {
            std::cerr << "ERROR: Captured empty frame" << std::endl;
            break;
        }
        // Get original frame dimensions for this specific frame

        float frame_width_orig = static_cast<float>(frame.cols);
        float frame_height_orig = static_cast<float>(frame.rows);

        // 1. Preprocessing
        // 'width' and 'height' here are model input dimensions (e.g., 640x640)

        auto input_tensor_values = preprocess::toTensor(frame, width, height);

        // 2. Create Input Tensor

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> current_input_dims = {batch, channels, height, width};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                  input_tensor_values.data(),
                                                                  input_tensor_values.size(),
                                                                  current_input_dims.data(),
                                                                  current_input_dims.size());
        
        // 3. Run Inference

        std::vector<Ort::Value> output_tensors;
        try {
            output_tensors = infer::run(session,
                                input.name,        // from InputInfo
                                input_tensor,
                                output);           // vector<string> from reportOutputs
        } catch (const Ort::Exception& e) {
            std::cerr << "ERROR during inference: " << e.what() << std::endl;
            cv::imshow(window_name, frame);
            if (cv::waitKey(1) == 'q') break;
            continue;
        }

        // 4. Post-processing

        // const float obj_threshold = 1.0f; // Objectness threshold for YOLOv11
        const float conf_threshold = 0.75f; // Final confidence threshold
        const float nms_threshold = 1.0f;  // Your NMS threshold

        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        // Check if we have the expected single output tensor

        if (output_tensors.size() == 1 && output_tensors[0].IsTensor()) {
            
            auto shape   = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            const float* base = output_tensors[0].GetTensorData<float>();
            
            int64_t ATTR = shape[1];          // 84
            int64_t DETS = shape[2];          // 8400

            int det_id = 0;                   // ← pick the column you want for debugging

            // Example debug print for a single detection column
            /*
            std::cout << "Detection " << det_id << " (" << ATTR << " numbers):\n";
            for (int a = 0; a < ATTR; ++a) {
                float v = base[a * DETS + det_id];   // [attr , det_id]
                std::cout << std::setw(10) << v;
                if ((a + 1) % 8 == 0) std::cout << '\n';
            }
            std::cout << std::endl;
            */

            int64_t B = shape[0];
            int64_t D = 0;                      // will hold the true detection count
            int64_t STRIDE;                     // 85 in the normal case
            int64_t NC = CLASS_NAMES.size();

            // Debug: print raw output tensor shape
            // std::cout << "Raw output shape = [ ";
            // for (auto v : shape) std::cout << v << ' ';
            // std::cout << "]\n";

            const int EXP_ROW   = NC + 5;          // 85 for COCO
            const int EXP_ROW_N = NC + 4;          // 84 when no obj score

            bool layout_B84D = false;
            if (shape.size() == 3)
            {
                if (shape[2] == EXP_ROW) {         // [B, D, 85]
                    D      = shape[1];
                    STRIDE = EXP_ROW;
                    // std::cout << "[B, D, 85]" << std::endl;
                }
                else if (shape[1] == EXP_ROW) {    // [B, 85, D]
                    D      = shape[2];
                    STRIDE = EXP_ROW;
                    // std::cout << "[B, 85, D]" << std::endl;
                }
                else if (shape[1] == 1 && shape[2] % EXP_ROW == 0) { // [B,1,flat]
                    D      = shape[2] / EXP_ROW;
                    STRIDE = EXP_ROW;
                    // std::cout << "[B,1,flat]" << std::endl;
                }
                else if (shape[1] == EXP_ROW_N) {  // [B, 84, D]
                    D      = shape[2];
                    STRIDE = EXP_ROW_N;
                    layout_B84D = true;
                    std::cout << "[B, 84, D]" << std::endl;
                }
                else {
                    std::cerr << "Unhandled 3-D output shape\n";
                    return 0;
                }
            }


            float model_input_width_float = static_cast<float>(width);  // Model input width (e.g., 640)
            float model_input_height_float = static_cast<float>(height); // Model input height (e.g., 640)

            // output_shape should be [batch_size, num_potential_detections, num_attributes]
            // For Ultralytics YOLO export without postprocessing the last dimension is
            // (num_classes + 5) -> [objectness, cx, cy, w, h, class scores]

            for (int i = 0; i < D; ++i) {
                const float* row = nullptr;
                std::vector<float> row_buf;
                if (layout_B84D) {
                    row_buf.resize(STRIDE);
                    for (int a = 0; a < STRIDE; ++a)
                        row_buf[a] = base[a * D + i];       // gather attribute a for det i
                    row = row_buf.data();
                } else {
                    row = base + i * STRIDE;
                }

                // float objectness = 1.0f;               // default when no explicit obj score
                int   cls_offset = 4;                  // cx,cy,w,h then classes

                /* if (!layout_B84D && STRIDE == EXP_ROW) { // full 85 layout
                    objectness = row[0];
                    cls_offset = 5;
                }*/
                // if (objectness < obj_threshold) continue;
                // std::cout << "objectness = " << objectness << std::endl;

                const float* cls = row + cls_offset;
                int best_id      = std::max_element(cls, cls + NC) - cls;
                float conf       = cls[best_id];
                
                if (conf < conf_threshold) continue;
                std::cout << "conf = " << conf << std::endl;

                float cx_model = row[0];
                float cy_model = row[1];
                float w_model  = row[2];
                float h_model  = row[3];

                std::cout << "center x = " << cx_model << " center y = " << cy_model << " width = " << w_model << " height = " << h_model << std::endl;

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
                    std::cout << " we have a box ! " << std::endl;
                    bboxes.push_back(box_orig);
                    scores.push_back(conf);
                    class_ids.push_back(best_id);
                }
            }
            // Debug marker
            // std::cout << "Ha" << std::endl;

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
                const int TOP_K = 1;                // keep at most 1 per class
                
                if (!bboxes_for_class.empty()) {
                    std::vector<int> nms_result_indices_for_class;
                    cv::dnn::NMSBoxes(bboxes_for_class, scores_for_class, conf_threshold, nms_threshold, nms_result_indices_for_class);

                    for (int temp_idx : nms_result_indices_for_class) {
                        final_kept_indices.push_back(original_indices_for_class[temp_idx]);
                    }
                    if (final_kept_indices.size() > TOP_K)
                        final_kept_indices.resize(TOP_K);
                }
            }
            
            // --- Draw the final Detections using final_kept_indices ---
            draw::detections(frame, bboxes, scores, class_ids, CLASS_NAMES);
            // --- END Section A ---

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

    cv::destroyAllWindows();
    return 0;
}