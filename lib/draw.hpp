#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace draw {
    void detections(cv::Mat& frame,
                    const std::vector<cv::Rect>& boxes,
                    const std::vector<float>& scores,
                    const std::vector<int>& ids,
                    const std::vector<std::string>& classNames);
}
