#include "draw.hpp"

void draw::detections(cv::Mat& frame,
                      const std::vector<cv::Rect>& boxes,
                      const std::vector<float>& scores,
                      const std::vector<int>& ids,
                      const std::vector<std::string>& names)
{
    for (size_t i = 0; i < boxes.size(); ++i) {
        int id = ids[i];
        if (id < 0 || id >= static_cast<int>(names.size())) continue;
        const cv::Rect& box = boxes[i];
        cv::rectangle(frame, box, {0,255,0}, 2);
        std::string label = names[id] + " " + cv::format("%.2f", scores[i]);
        int baseLine = 0;
        cv::Size sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        int top = std::max(box.y, sz.height);
        cv::putText(frame, label, {box.x, top - 5},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,0}, 2);
        cv::putText(frame, label, {box.x, top - 5},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 1);
    }
}
