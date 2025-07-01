#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

namespace infer
{
    // Runs one forward pass and returns the output tensors.
    // Throws Ort::Exception on any runtime error.
    std::vector<Ort::Value> run(Ort::Session&               session,
                                const std::string&          input_name,
                                Ort::Value&                 input_tensor,
                                const std::vector<std::string>& output_names);
}
