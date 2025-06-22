#include "../third_party/httplib.h"
#include "../third_party/json.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

using json = nlohmann::json;

// Simple base64 decoding (from stackoverflow)
static inline std::string base64_decode(const std::string &in) {
    std::string out;
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++)
        T["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i]] = i;
    int val = 0, valb = -8;
    for (uint8_t c : in) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

int main() {
    // Set up ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    OrtCUDAProviderOptions cuda_options{};
    try {
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    } catch (const Ort::Exception&) {
        std::cerr << "CUDA provider not available, using CPU" << std::endl;
    }

    const char* model_path = "yolo11n.onnx";
    Ort::Session session(env, model_path, session_options);

    httplib::Server svr;

    svr.Post("/infer", [&session](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);
            std::string img_b64 = j.at("image");
            std::string img_bytes = base64_decode(img_b64);
            std::vector<uchar> buf(img_bytes.begin(), img_bytes.end());
            cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
            if (img.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"decode failed\"}", "application/json");
                return;
            }

            cv::Mat resized;
            cv::resize(img, resized, cv::Size(640, 640));
            resized.convertTo(resized, CV_32F, 1.0/255);
            std::vector<int64_t> dims = {1, 3, 640, 640};
            std::vector<float> input_tensor(3 * 640 * 640);
            std::vector<cv::Mat> channels(3);
            for (int i = 0; i < 3; ++i) {
                channels[i] = cv::Mat(640, 640, CV_32F, input_tensor.data() + i * 640 * 640);
            }
            cv::split(resized, channels);

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            Ort::Value input = Ort::Value::CreateTensor<float>(memory_info, input_tensor.data(), input_tensor.size(), dims.data(), dims.size());
            Ort::AllocatorWithDefaultOptions alloc;
            const char* in_name = session.GetInputName(0, alloc);
            std::vector<const char*> out_names;
            for (size_t i=0;i<session.GetOutputCount();++i) out_names.push_back(session.GetOutputName(i, alloc));
            auto output = session.Run(Ort::RunOptions{nullptr}, &in_name, &input, 1, out_names.data(), out_names.size());
            for (const char* n : out_names) alloc.Free((void*)n);
            alloc.Free((void*)in_name);
            // Serialize first output to JSON (truncated)
            float* out_data = output[0].GetTensorMutableData<float>();
            size_t out_size = output[0].GetTensorTypeAndShapeInfo().GetElementCount();
            size_t sample = std::min<size_t>(out_size, 10);
            json out_json = json::array();
            for (size_t i = 0; i < sample; ++i) out_json.push_back(out_data[i]);
            json resp;
            resp["sample"] = out_json;
            res.set_content(resp.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json");
        }
    });

    std::cout << "Inference service listening on http://0.0.0.0:5000" << std::endl;
    svr.listen("0.0.0.0", 5000);
}
