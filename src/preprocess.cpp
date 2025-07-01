#include "preprocess.hpp"

namespace preprocess
{
    std::vector<float> toTensor(const cv::Mat& frame,
                                int width,
                                int height,
                                int channels)
    {
        cv::Mat resized_rgb;
        cv::resize(frame, resized_rgb, cv::Size(width, height));
        cv::cvtColor(resized_rgb, resized_rgb, cv::COLOR_BGR2RGB);

        cv::Mat f32;
        resized_rgb.convertTo(f32, CV_32F, 1.0 / 255.0);   // scale to 0-1

        std::vector<float> tensor(static_cast<size_t>(channels) * height * width);

        for (int c = 0; c < channels; ++c)
            for (int h = 0; h < height; ++h)
                for (int w = 0; w < width; ++w)
                    tensor[c * height * width + h * width + w] =
                        f32.at<cv::Vec3f>(h, w)[c];

        return tensor;
    }
}
