#include "onnx_model.hpp"
#include <iostream>

OnnxModel::OnnxModel(const std::string& model_path,
                     OrtLoggingLevel log_level,
                     int intra_threads)
    : env_(log_level, "YOLO-NAS-App-GPU"),
      opts_(),
      session_(nullptr)
{
    opts_.SetIntraOpNumThreads(intra_threads);

    OrtCUDAProviderOptions cuda_opts{};
    try {
        opts_.AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "INFO: Attempting to use CUDA execution provider.\n";
    } catch (const Ort::Exception& e) {
        std::cerr << "WARNING: Could not append CUDA execution provider: " << e.what() << '\n'
                  << "INFO: Will fall back to CPU or other available providers.\n";
    }

    session_ = Ort::Session(env_, model_path.c_str(), opts_);
    std::cout << "ONNX model loaded successfully: " << model_path << '\n';
}
