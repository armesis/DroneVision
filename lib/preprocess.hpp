#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace preprocess
{
    // Convert a BGR frame to a normalized RGB float tensor (NCHW order).
    // • frame   : original camera image (BGR 8-bit)
    // • width   : target model width   (e.g. 640)
    // • height  : target model height  (e.g. 640)
    // Returns a std::vector<float> sized channels*height*width, ready to feed
    // into Ort::Value::CreateTensor.
    std::vector<float> toTensor(const cv::Mat& frame,
                                int width,
                                int height,
                                int channels = 3);
}
