#include "../third_party/httplib.h"
#include "../third_party/json.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using json = nlohmann::json;

int main() {
    const char* url = "http://localhost:5000/infer";
    httplib::Client cli("localhost", 5000);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera" << std::endl;
        return 1;
    }

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) break;
        std::vector<uchar> buf;
        cv::imencode(".jpg", frame, buf);
        std::string b64 = httplib::detail::base64_encode(std::string(buf.begin(), buf.end()));
        json j; j["image"] = b64;
        auto res = cli.Post("/infer", j.dump(), "application/json");
        if (res && res->status == 200) {
            std::cout << res->body << std::endl;
        } else {
            std::cerr << "Inference request failed" << std::endl;
        }
        cv::imshow("capture", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    cap.release();
    cv::destroyAllWindows();
}
