#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>
#include <iterator>

namespace tensor_utils {

struct Detection {
    float objectness;
    float cx, cy, w, h;            // relative to model input
    const float* class_scores;     // pointer to NC floats
};

class DetectionView {
public:
    DetectionView(const Ort::Value& tensor, int num_classes)
        : tensor_(tensor), NC_(num_classes)
    {
        if (!tensor_.IsTensor())
            throw std::runtime_error("DetectionView: value is not a tensor");

        shape_ = tensor_.GetTensorTypeAndShapeInfo().GetShape();
        if (shape_.size() < 2 || shape_.size() > 3)
            throw std::runtime_error("DetectionView: unsupported tensor rank");

        EXP_ROW_   = NC_ + 5;          // cx,cy,w,h,obj + cls scores
        EXP_ROW_N_ = NC_ + 4;          // cx,cy,w,h + cls scores (no obj)

        resolveLayout();
    }

    /* ---------------- iterator ---------------- */
    class iterator {
    public:
        using difference_type   = std::ptrdiff_t;
        using value_type        = Detection;
        using pointer           = const Detection*;
        using reference         = const Detection&;
        using iterator_category = std::forward_iterator_tag;

        iterator(const float* base, int stride, int idx, int total, bool has_obj, int nc)
            : base_(base), stride_(stride), idx_(idx), total_(total), has_obj_(has_obj), nc_(nc) {}

        reference operator*() {
            det_.objectness   = has_obj_ ? base_[idx_*stride_] : 1.0f;
            int off           = has_obj_ ? 1 : 0;             // skip obj if present
            const float* row  = base_ + idx_*stride_;
            det_.cx = row[off + 0];
            det_.cy = row[off + 1];
            det_.w  = row[off + 2];
            det_.h  = row[off + 3];
            det_.class_scores = row + off + 4;
            return det_;
        }
        pointer operator->() { return &(**this); }

        iterator& operator++() { ++idx_; return *this; }
        iterator  operator++(int){ iterator tmp=*this; ++(*this); return tmp; }

        bool operator==(const iterator& other) const { return idx_ == other.idx_; }
        bool operator!=(const iterator& other) const { return !(*this==other); }

    private:
        const float* base_;
        int stride_;
        int idx_;
        int total_;
        bool has_obj_;
        int nc_;
        Detection det_{};
    };

    iterator begin() const { return iterator(base_, stride_, 0, D_, has_obj_, NC_); }
    iterator end()   const { return iterator(base_, stride_, D_, D_, has_obj_, NC_); }

    int detections() const { return D_; }
    int stride()     const { return stride_; }
    bool hasObjectness() const { return has_obj_; }

private:
    void resolveLayout() {
        const float* data = tensor_.GetTensorData<float>();
        base_ = data;

        if (shape_.size() == 3) {
            if (shape_[2] == EXP_ROW_) {
                D_ = shape_[1];
                stride_ = EXP_ROW_;
                has_obj_ = true;
            } else if (shape_[1] == EXP_ROW_) {
                D_ = shape_[2];
                stride_ = EXP_ROW_;
                has_obj_ = true;
            } else if (shape_[1] == 1 && shape_[2] % EXP_ROW_ == 0) {
                D_ = shape_[2] / EXP_ROW_;
                stride_ = EXP_ROW_;
                has_obj_ = true;
            } else if (shape_[1] == EXP_ROW_N_) {
                D_ = shape_[2];
                stride_ = EXP_ROW_N_;
                has_obj_ = false;
            } else {
                throw std::runtime_error("DetectionView: unhandled 3D layout");
            }
        } else if (shape_.size() == 2) {
            if (shape_[1] % EXP_ROW_ == 0) {
                D_ = shape_[1] / EXP_ROW_;
                stride_ = EXP_ROW_;
                has_obj_ = true;
            } else if (shape_[1] % EXP_ROW_N_ == 0) {
                D_ = shape_[1] / EXP_ROW_N_;
                stride_ = EXP_ROW_N_;
                has_obj_ = false;
            } else {
                throw std::runtime_error("DetectionView: unhandled 2D layout");
            }
        }
    }

    const Ort::Value& tensor_;
    std::vector<int64_t> shape_;
    int NC_;

    int EXP_ROW_;    // 85
    int EXP_ROW_N_;  // 84

    const float* base_ = nullptr;
    int stride_  = 0;
    int D_       = 0;
    bool has_obj_ = true;
};

} // namespace tensor_utils
