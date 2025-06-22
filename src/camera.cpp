#include "camera.hpp"
#include <iostream>

Camera::Camera(int index, int w, int h) : cap_(index)
{
    if (!cap_.isOpened()) {
        std::cerr << "ERROR: Could not open camera\n";
        std::exit(EXIT_FAILURE);
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH,  w);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, h);
}

bool Camera::grabFrame(cv::Mat& out)
{
    cap_ >> out;
    return !out.empty();
}
