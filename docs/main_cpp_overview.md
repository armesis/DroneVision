# main.cpp Overview

This document summarizes the functionality of `main.cpp`, which implements a real-time object detection pipeline.

## Purpose
- Capture frames from a camera.
- Run YOLO object detection using the `yolo11n.onnx` model with ONNX Runtime (GPU if available).
- Display the resulting bounding boxes with labels and frames-per-second (FPS) information.

## Key Components

- **COCO class names**: Lines 22–34 define the array of 80 class labels corresponding to the model outputs.
- **ONNX Runtime initialization**: The program sets up `Ort::Env` and `Ort::SessionOptions`, enabling CUDA if possible, otherwise falling back to CPU execution. The model is loaded from `yolo11n.onnx`.
- **Model input handling**: Input node names and dimensions are retrieved, with default input size 640×640 used when dynamic dimensions are encountered.
- **OpenCV camera setup**: A camera capture is initialized at 640×480, and inference buffers are prepared.
- **Frame processing loop**: Each iteration performs:
  - Reading and timing a frame.
  - Resizing and normalizing the image for the model.
  - Building an input tensor and running inference.
  - Parsing detections and mapping coordinates back to the original frame, with optional debug output for early frames.
  - Applying per-class Non-Max Suppression and drawing boxes with labels.
  - Displaying FPS and exiting when `q` is pressed.

Overall, `main.cpp` demonstrates preprocessing, inference, and postprocessing steps for YOLO-based object detection using OpenCV and ONNX Runtime.
