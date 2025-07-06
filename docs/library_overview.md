# Library Overview

This repository exposes a small C++ library named `uavlib` that is used by the
`live_camera_app` example. The library provides helper classes and functions for
working with OpenCV and ONNX Runtime.

## Components

- **camera.[hpp/cpp]** – lightweight wrapper around `cv::VideoCapture` for
  grabbing frames from a webcam.
- **draw.[hpp/cpp]** – routines for drawing bounding boxes and labels on a frame.
- **classnames.[hpp/cpp]** – returns the list of COCO object labels used by the
  example model.
- **onnx_model.[hpp/cpp]** – convenience wrapper that creates an ORT environment,
  configures CUDA if available and loads an ONNX model.
- **model_info.[hpp/cpp]** – utilities to inspect model input/output nodes and
  print their shapes.
- **preprocess.[hpp/cpp]** – converts an OpenCV `cv::Mat` into a normalized float
  tensor in NCHW layout.
- **infer.[hpp/cpp]** – thin wrapper around `Ort::Session::Run`.
- **tensor_utils.hpp** – helper types to iterate over detection tensors.
- **print_shape.[hpp/cpp]** – helper used by the model inspection utilities.

These pieces are compiled into the `uavlib` static library by the main
`CMakeLists.txt`. They keep the application code in `src/main.cpp` concise while
exposing reusable functionality for future experiments.
