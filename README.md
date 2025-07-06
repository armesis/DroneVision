# DroneVision Project

This repository contains preliminary experiments for running YOLO models with OpenCV and ONNX Runtime. The current layout mixes build artifacts and source files in the top level. The suggestions below provide a cleaner organization for future development.

## Suggested Folder Layout

```
DroneVision/
├── CMakeLists.txt
├── src/            # C++ source files
├── lib/            # C++ header files (if any)
├── models/         # Pretrained model files
├── data/           # Example images or videos
├── scripts/        # Utility scripts for setup or dataset preparation
├── tests/          # Test cases
├── build/          # CMake build output (should be git‑ignored)
└── docs/           # Project documentation
```

Key points:

- Keep build artifacts out of version control by adding `build/` to `.gitignore`.
- Place application code inside `src/` and headers in `include/` for clarity.
- Use `models/` or `data/` to store external assets rather than mixing them with source files.
- Document design decisions and usage instructions in `docs/`.

These changes make it easier to navigate the repository and separate generated files from source code.

## Building the Example

The project uses CMake. From the repository root run:

```bash
cmake -S . -B build
cmake --build build
```

The resulting executable `live_camera_app` will appear inside the `build/`
directory. Run it with:

```bash
./build/live_camera_app
```

Make sure the `ONNXRUNTIME_ROOTDIR` variable in `CMakeLists.txt` points to your
local ONNX Runtime installation.

Additional documentation can be found under the `docs/` directory, including a
walkthrough of `src/main.cpp` and an overview of the helper library.
