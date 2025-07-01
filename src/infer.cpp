#include "infer.hpp"

namespace infer
{
    std::vector<Ort::Value> run(Ort::Session&               session,
                                const std::string&          input_name,
                                Ort::Value&                 input_tensor,
                                const std::vector<std::string>& output_names)
    {
        /* ---- build the raw C-string arrays expected by ORT ---- */
        const char* in_name_c = input_name.c_str();

        std::vector<const char*> out_names_c;
        out_names_c.reserve(output_names.size());
        for (auto& s : output_names) out_names_c.push_back(s.c_str());

        /* ---- call ONNX Runtime ---- */
        Ort::RunOptions run_opts{nullptr};          // default: no special flags
        return session.Run(run_opts,
                           &in_name_c, &input_tensor, 1,
                           out_names_c.data(), out_names_c.size());
    }
}
