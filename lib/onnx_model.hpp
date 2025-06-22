#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>

class OnnxModel {
public:
    // throws Ort::Exception on failure
    explicit OnnxModel(const std::string& model_path,
                       OrtLoggingLevel log_level = ORT_LOGGING_LEVEL_WARNING,
                       int intra_threads = 1);

    // accessors the rest of the program might need
    Ort::Session&        session()            { return session_;   }
    Ort::AllocatorWithDefaultOptions&      allocator()          { return allocator_; }

private:
    Ort::Env                  env_;
    Ort::SessionOptions       opts_;
    Ort::Session              session_;
    Ort::AllocatorWithDefaultOptions allocator_;
};
