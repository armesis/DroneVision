#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <cstdint>
#include <ostream>
#include <iostream>

namespace modelutil
{
    /* ----------  small helper types ---------- */

    struct InputDims {
        int64_t batch;
        int64_t channels;
        int64_t height;
        int64_t width;
    };

    struct InputInfo {
        std::string name;   // e.g. "images"
        InputDims   dims;   // resolved values
    };

    /* ----------  constants ---------- */

    constexpr int64_t DEFAULT_MODEL_WIDTH  = 640;
    constexpr int64_t DEFAULT_MODEL_HEIGHT = 640;

    /* ----------  API ---------- */

    // Inspects the first (and only) input node, prints everything, and
    // throws Ort::Exception if the model does not have exactly one input.
    InputInfo  inspectInput (Ort::Session&   session,
                             Ort::AllocatorWithDefaultOptions&  alloc,
                             std::ostream&   os = std::cout);

    // Walks every output node, prints its shape, and returns their names.
    std::vector<std::string> reportOutputs(Ort::Session&   session,
                                           Ort::AllocatorWithDefaultOptions&  alloc,
                                           std::ostream&   os = std::cout);
}
