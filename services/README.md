# DroneVision Microservices

This folder contains a minimal example of splitting the original monolithic code into
distinct services. Each service can be developed and run separately.

## Services

- **inference_service** – C++ application that loads the YOLO model with ONNX
  Runtime and exposes an HTTP endpoint `/infer`. Clients send base64‑encoded
  images and receive a small portion of the raw model output as JSON.
- **capture_service** – Captures frames from the webcam and posts them to the
  inference service. Press `q` to quit the viewer.

### Build

Both programs are built with CMake. From the repository root run:

```bash
cmake -S . -B build
cmake --build build --target inference_service capture_service
```

### Run

In separate terminals execute:

```bash
# Terminal 1
./build/inference_service

# Terminal 2
./build/capture_service
```

This architecture keeps inference logic isolated from image capture, making it
simpler to replace or extend each component independently.
