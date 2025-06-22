#pragma once
#include <opencv2/opencv.hpp>

class Camera {
public:
    Camera(int index = 0, int w = 640, int h = 480);
    bool grabFrame(cv::Mat& out);
private:
    cv::VideoCapture cap_;
};
