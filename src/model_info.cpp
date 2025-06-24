#include "model_info.hpp"
#include "print_shape.hpp"
#include <sstream>

#include <iostream>

namespace modelutil
{
    /* ---------- helpers ---------- */

    static InputDims resolveInputDims(const std::vector<int64_t>& raw)
    {
        InputDims r;
        r.batch    = (raw[0] <= 0) ? 1                    : raw[0];
        r.channels =          raw[1];
        r.height   = (raw[2] <= 0) ? DEFAULT_MODEL_HEIGHT : raw[2];
        r.width    = (raw[3] <= 0) ? DEFAULT_MODEL_WIDTH  : raw[3];
        return r;
    }

    /* ---------- public API ---------- */

    InputInfo inspectInput(Ort::Session& session,
                           Ort::AllocatorWithDefaultOptions&  alloc,
                           std::ostream& os)
    {
        size_t n_inputs = session.GetInputCount();
        if (n_inputs != 1)
        {
            std::ostringstream msg;
            msg << "Model expected 1 input node but has " << n_inputs;
            throw std::runtime_error(msg.str());
        }

        // name
        Ort::AllocatedStringPtr name_alloc = session.GetInputNameAllocated(0, alloc);
        std::string name = name_alloc.get();
        os << "Input Name: " << name << '\n';

        // raw dims
        auto type_info     = session.GetInputTypeInfo(0);
        auto tensor_info   = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> raw_dims = tensor_info.GetShape();

        os << "Model Input Dims Reported: ";
        util::printShape(raw_dims, os);
        os << '\n';

        // resolved dims
        InputDims dims = resolveInputDims(raw_dims);

        os << "Using Resolved Input Dims: { Batch: "  << dims.batch
           << ", Channels: " << dims.channels
           << ", Height: "   << dims.height
           << ", Width: "    << dims.width  << " }\n";

        return { name, dims };
    }

    std::vector<std::string> reportOutputs(Ort::Session& session,
                                           Ort::AllocatorWithDefaultOptions& alloc,
                                           std::ostream& os)
    {
        size_t count = session.GetOutputCount();
        os << "Number of output nodes: " << count << '\n';

        std::vector<std::string> names(count);

        for (size_t i = 0; i < count; ++i)
        {
            Ort::AllocatedStringPtr name_alloc = session.GetOutputNameAllocated(i, alloc);
            names[i] = name_alloc.get();

            Ort::TypeInfo  ti         = session.GetOutputTypeInfo(i);
            auto shape_info           = ti.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> dims = shape_info.GetShape();

            os << "Output " << i << " Name: " << names[i] << " Dims: ";
            util::printShape(dims, os);
            os << '\n';
        }
        return names;
    }
}
